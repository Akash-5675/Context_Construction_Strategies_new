"""
Stage 8: descriptive statistics and figures.

Reads the results set in analysis_config.py and writes to its output folder:
  - summary.csv         : per-condition means, SDs, answer rates
  - summary_anova.csv   : omnibus ANOVA per metric
  - hypotheses.txt      : descriptive H1/H2/H3 tables and tests
  - figures/fig1..fig6  : publication figures

These tests treat every run as independent, so they overstate the evidence.
Use robust_stats.py for the inferential results.

Two analysis decisions are baked in, both forced by the data:

1. Refusals ("Insufficient context.") score near zero on most metrics. A raw mean therefore mostly measures *how often a condition
   refused*, not *how good its answers were*. Every comparison is reported as
   two separate outcomes: answer_rate (did it answer at all) and quality
   conditional on answering.

2. Strategies differ systematically in how much context they deliver at the
   same k — sentence chunks are far larger than 128-word chunks. Since context
   volume itself affects the outcome (H3), strategy comparisons are also run
   with context_words as a covariate (ANCOVA) so the strategy effect is
   estimated at matched context size.

Usage:
    python src/analyze.py
"""

from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.insert(0, os.path.dirname(__file__))
from analysis_config import FIGURES as FIGURES_DIR, OUTPUT as OUTPUTS_DIR, RESULTS as RESULTS_PATH, refusal_flags

SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "summary.csv")
HYPOTHESES_PATH = os.path.join(OUTPUTS_DIR, "hypotheses.txt")

PRIMARY = "rouge_l"
METRICS = ["answer_relevance", "rouge_l", "faithfulness",
           "retrieval_recall", "context_words", "latency_ms"]

SENTENCE_STRATEGIES = ["sent_nltk", "sent_scispacy"]
ALPHA = 0.05

PLT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}

_report_lines: list[str] = []


def say(line: str = ""):
    """Print to console and capture for outputs/hypotheses.txt."""
    print(line)
    _report_lines.append(line)


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #

def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}\nRun experiment_runner.py first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_PATH)

    # A refusal is a genuine model answer, not a failure — see validate.py.
    df["refused"] = refusal_flags(df["answer"])
    df["answered"] = ~df["refused"]

    is_fixed = df["condition"].str.startswith("fixed_")
    df["chunk_size"] = df["condition"].str.extract(r"fixed_(\d+)_")[0].astype("float")
    df["overlap_pct"] = df["condition"].str.extract(r"fixed_\d+_(\d+)pct")[0].astype("float")
    df["is_fixed"] = is_fixed

    print(f"Loaded {len(df):,} rows across {df['condition'].nunique()} conditions "
          f"({df['refused'].mean()*100:.1f}% refusals)\n")
    return df


def best_fixed_condition(df: pd.DataFrame) -> str:
    """Highest mean primary metric among fixed-token conditions."""
    fixed = df[df["is_fixed"]]
    return fixed.groupby("condition")[PRIMARY].mean().idxmax()


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        ans = sub[sub["answered"]]
        row = {"condition": cond, "n": len(sub),
               "answer_rate": round(sub["answered"].mean(), 4),
               f"{PRIMARY}_answered": round(ans[PRIMARY].mean(), 4) if len(ans) else np.nan}
        for metric in METRICS:
            if metric in sub.columns:
                row[f"{metric}_mean"] = round(sub[metric].mean(), 4)
                row[f"{metric}_std"] = round(sub[metric].std(), 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    anova = {}
    for metric in METRICS:
        if metric not in df.columns:
            continue
        groups = [g[metric].dropna().values
                  for _, g in df.groupby("condition") if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            anova[metric] = {"f_stat": round(f_stat, 4), "p_value": p_val}

    (pd.DataFrame(anova).T.reset_index().rename(columns={"index": "metric"})
       .to_csv(SUMMARY_PATH.replace(".csv", "_anova.csv"), index=False))
    return summary_df


# --------------------------------------------------------------------------- #
# hypothesis tests
# --------------------------------------------------------------------------- #

def verdict(supported: bool, detail: str) -> str:
    label = "CONSISTENT WITH HYPOTHESIS" if supported else "NOT CONSISTENT WITH HYPOTHESIS"
    return f"{label} (descriptive; confirm with robust_stats.py) — {detail}"


def test_h1(df: pd.DataFrame):
    """H1: overlap adds no benefit for fixed-token chunking."""
    say("=" * 70)
    say("H1 — Overlap adds no benefit (fixed-token conditions)")
    say("=" * 70)

    fixed = df[df["is_fixed"]].copy()

    tbl = fixed.groupby("overlap_pct").agg(
        n=("run_key", "size"),
        answer_rate=("answered", "mean"),
        rouge_all=(PRIMARY, "mean"),
    )
    tbl["rouge_answered"] = fixed[fixed["answered"]].groupby("overlap_pct")[PRIMARY].mean()
    say("\nBy overlap level:")
    say(tbl.round(4).to_string())

    # Two-way ANOVA: does overlap explain variance once chunk size is accounted for?
    model = ols(f"{PRIMARY} ~ C(chunk_size) * C(overlap_pct)", data=fixed).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    say("\nTwo-way ANOVA (chunk size x overlap):")
    say(aov.round(6).to_string())

    p_overlap = aov.loc["C(overlap_pct)", "PR(>F)"]

    tukey = pairwise_tukeyhsd(fixed[PRIMARY], fixed["overlap_pct"].astype(int), alpha=ALPHA)
    say("\nTukey HSD across overlap levels:")
    say(str(tukey))

    # Direction: compare 0% against the highest overlap level.
    zero = fixed[fixed["overlap_pct"] == 0][PRIMARY]
    forty = fixed[fixed["overlap_pct"] == 40][PRIMARY]
    t_stat, p_dir = stats.ttest_ind(zero, forty, equal_var=False)

    say(f"\n0% overlap mean = {zero.mean():.4f}   40% overlap mean = {forty.mean():.4f}")
    say(f"Welch t-test 0% vs 40%: t = {t_stat:.3f}, p = {p_dir:.6g}")

    if p_overlap >= ALPHA:
        v = verdict(True, f"overlap has no significant effect on {PRIMARY} "
                          f"(p = {p_overlap:.4g}) once chunk size is controlled")
    elif zero.mean() > forty.mean():
        v = verdict(True, f"overlap significantly *harms* {PRIMARY} (p = {p_overlap:.4g}); "
                          f"0% outperforms 40%, so added overlap is never justified")
    else:
        v = verdict(False, f"overlap significantly improves {PRIMARY} (p = {p_overlap:.4g})")
    say(f"\nH1 VERDICT: {v}\n")


def test_h2(df: pd.DataFrame):
    """H2: sentence chunking matches semantic chunking."""
    say("=" * 70)
    say("H2 — Sentence chunking matches semantic chunking")
    say("=" * 70)

    present = [c for c in SENTENCE_STRATEGIES + ["semantic"]
               if c in set(df["condition"])]
    if "semantic" not in present or not any(c in present for c in SENTENCE_STRATEGIES):
        say("\nSkipped: need both a sentence strategy and semantic to compare.\n")
        return

    sub = df[df["condition"].isin(present)].copy()

    tbl = sub.groupby("condition").agg(
        n=("run_key", "size"),
        answer_rate=("answered", "mean"),
        rouge_all=(PRIMARY, "mean"),
        ctx_words=("context_words", "mean"),
    )
    tbl["rouge_answered"] = sub[sub["answered"]].groupby("condition")[PRIMARY].mean()
    say("\nBy strategy:")
    say(tbl.round(4).to_string())

    f_stat, p_val = stats.f_oneway(
        *[g[PRIMARY].values for _, g in sub.groupby("condition")])
    say(f"\nOne-way ANOVA across strategies: F = {f_stat:.4f}, p = {p_val:.6g}")

    tukey = pairwise_tukeyhsd(sub[PRIMARY], sub["condition"], alpha=ALPHA)
    say("\nTukey HSD:")
    say(str(tukey))

    # Context volume differs by strategy, so also estimate the effect at matched context.
    model = ols(f"{PRIMARY} ~ C(condition) + context_words", data=sub).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    say("\nANCOVA (strategy, controlling for context_words):")
    say(aov.round(6).to_string())
    p_adj = aov.loc["C(condition)", "PR(>F)"]
    say(f"Strategy effect controlling for context size: p = {p_adj:.6g}")

    if p_val >= ALPHA:
        v = verdict(True, f"no significant difference between sentence and semantic "
                          f"chunking (p = {p_val:.4g})")
    else:
        best = tbl["rouge_all"].idxmax()
        v = verdict(False, f"strategies differ significantly (p = {p_val:.4g}); "
                           f"'{best}' scores highest. Controlling for context size, "
                           f"p = {p_adj:.4g}")
    say(f"\nH2 VERDICT: {v}\n")


def test_h3(df: pd.DataFrame):
    """H3: quality degrades once context grows past roughly 2500 tokens."""
    say("=" * 70)
    say("H3 — Context cliff (quality falls as context grows)")
    say("=" * 70)

    d = df.dropna(subset=["context_words", PRIMARY]).copy()
    # ~0.75 words per token for English text; the hypothesis is stated in tokens.
    d["approx_tokens"] = d["context_words"] / 0.75

    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, np.inf]
    labels = ["<500", "500-1k", "1k-1.5k", "1.5k-2k", "2k-2.5k", "2.5k-3k", ">3k"]
    d["bin"] = pd.cut(d["approx_tokens"], bins=bins, labels=labels)

    tbl = d.groupby("bin", observed=True).agg(
        n=("run_key", "size"),
        answer_rate=("answered", "mean"),
        rouge_all=(PRIMARY, "mean"),
    )
    tbl["rouge_answered"] = d[d["answered"]].groupby("bin", observed=True)[PRIMARY].mean()
    say("\nBy approximate context tokens:")
    say(tbl.round(4).to_string())

    r, p_r = stats.pearsonr(d["approx_tokens"], d[PRIMARY])
    say(f"\nPearson correlation (context tokens vs {PRIMARY}): "
        f"r = {r:.4f}, p = {p_r:.6g}")

    ans = d[d["answered"]]
    r_a, p_a = stats.pearsonr(ans["approx_tokens"], ans[PRIMARY])
    say(f"Same, answered rows only: r = {r_a:.4f}, p = {p_a:.6g}")

    # Where does the largest drop between adjacent bins occur?
    means = tbl["rouge_all"].dropna()
    if len(means) >= 2:
        drops = means.diff()
        cliff = drops.idxmin()
        say(f"\nLargest drop between adjacent bins: at {cliff} "
            f"(change {drops.min():+.4f})")

    if p_r < ALPHA and r < 0:
        v = verdict(True, f"{PRIMARY} declines significantly as context grows "
                          f"(r = {r:.3f}, p = {p_r:.3g})")
    elif p_r < ALPHA and r > 0:
        v = verdict(False, f"{PRIMARY} *increases* with context (r = {r:.3f}, "
                           f"p = {p_r:.3g})")
    else:
        v = verdict(False, f"no significant relationship between context size and "
                           f"{PRIMARY} (p = {p_r:.3g})")
    say(f"\nH3 VERDICT: {v}\n")


def test_embeddings(df: pd.DataFrame):
    """Secondary: does the biomedical encoder beat the general-purpose one?"""
    say("=" * 70)
    say("Secondary — MiniLM (general) vs PubMedBERT (biomedical)")
    say("=" * 70)

    tbl = df.groupby("embedding_model").agg(
        n=("run_key", "size"),
        answer_rate=("answered", "mean"),
        rouge_all=(PRIMARY, "mean"),
        recall=("retrieval_recall", "mean"),
    )
    tbl["rouge_answered"] = df[df["answered"]].groupby("embedding_model")[PRIMARY].mean()
    say("\n" + tbl.round(4).to_string())

    a = df[df["embedding_model"] == "minilm"][PRIMARY]
    b = df[df["embedding_model"] == "pubmedbert"][PRIMARY]
    if len(a) and len(b):
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        say(f"\nWelch t-test: t = {t_stat:.3f}, p = {p_val:.6g}")
        better = "MiniLM" if a.mean() > b.mean() else "PubMedBERT"
        sig = "significantly" if p_val < ALPHA else "not significantly"
        say(f"{better} scores higher ({sig} different at alpha={ALPHA}).\n")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def fig1_overlap(df: pd.DataFrame):
    """H1: overlap ratio vs primary metric, split by chunk size."""
    fixed = df[df["is_fixed"]]
    with plt.rc_context(PLT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for size, grp in fixed.groupby("chunk_size"):
            ax1.plot(*_agg(grp, "overlap_pct", PRIMARY), marker="o",
                     label=f"{int(size)} words")
            ax2.plot(*_agg(grp, "overlap_pct", "answered"), marker="s",
                     label=f"{int(size)} words")
        ax1.set_xlabel("Overlap (%)"); ax1.set_ylabel(f"Mean {PRIMARY}")
        ax1.set_title("Answer quality"); ax1.set_xticks([0, 10, 20, 40]); ax1.legend()
        ax2.set_xlabel("Overlap (%)"); ax2.set_ylabel("Answer rate")
        ax2.set_title("Answer rate"); ax2.set_xticks([0, 10, 20, 40]); ax2.legend()
        fig.suptitle("H1: Overlap ratio and performance", fontsize=13)
        _save(fig, "fig1_overlap.png")


def fig2_chunk_size(df: pd.DataFrame):
    """Chunk size effect, the dominant factor in Part A."""
    fixed = df[df["is_fixed"]]
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(*_agg(fixed, "chunk_size", PRIMARY), marker="o",
                linewidth=2, label=f"{PRIMARY} (all runs)")
        ax.plot(*_agg(fixed[fixed["answered"]], "chunk_size", PRIMARY), marker="s",
                linewidth=2, label=f"{PRIMARY} (answered only)")
        ax.plot(*_agg(fixed, "chunk_size", "answered"), marker="^",
                linewidth=2, label="answer rate")
        ax.set_xlabel("Chunk size (words)"); ax.set_ylabel("Score")
        ax.set_title("Chunk size vs performance (fixed-size chunks)")
        ax.set_xticks([128, 256, 512]); ax.legend()
        _save(fig, "fig2_chunk_size.png")


def fig3_strategies(df: pd.DataFrame, anchor: str):
    """H2: strategy comparison against the best fixed-token anchor."""
    keep = [anchor] + [c for c in SENTENCE_STRATEGIES + ["semantic"]
                       if c in set(df["condition"])]
    sub = df[df["condition"].isin(keep)]
    agg = sub.groupby("condition")[["answer_relevance", PRIMARY,
                                    "faithfulness", "retrieval_recall"]].mean()
    agg = agg.reindex([c for c in keep if c in agg.index])

    x = np.arange(len(agg)); width = 0.2
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (col, label) in enumerate([("answer_relevance", "Answer relevance"),
                                          (PRIMARY, "ROUGE-L"),
                                          ("faithfulness", "Faithfulness"),
                                          ("retrieval_recall", "Retrieval recall")]):
            ax.bar(x + (i - 1.5) * width, agg[col], width, label=label)
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"H2: Chunking strategies vs best fixed-token ({anchor})")
        ax.legend()
        _save(fig, "fig3_strategies.png")


def fig4_context_cliff(df: pd.DataFrame):
    """H3: primary metric against context size, binned."""
    d = df.dropna(subset=["context_words", PRIMARY]).copy()
    d["approx_tokens"] = d["context_words"] / 0.75
    d["bin"] = pd.cut(d["approx_tokens"],
                      [0, 500, 1000, 1500, 2000, 2500, 3000, np.inf],
                      labels=["<500", "500-1k", "1k-1.5k", "1.5k-2k",
                              "2k-2.5k", "2.5k-3k", ">3k"])
    g = d.groupby("bin", observed=True)
    mean, sem = g[PRIMARY].mean(), g[PRIMARY].sem()

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots()
        ax.errorbar(range(len(mean)), mean.values, yerr=sem.values,
                    marker="o", capsize=4, linewidth=2, label=f"mean {PRIMARY}")
        ax.plot(range(len(mean)), g["answered"].mean().values, marker="^",
                linewidth=2, label="answer rate")
        ax.set_xticks(range(len(mean))); ax.set_xticklabels(mean.index, rotation=20)
        ax.set_xlabel("Approximate context size (tokens)"); ax.set_ylabel("Score")
        ax.set_title("H3: Performance vs context size")
        ax.legend()
        _save(fig, "fig4_context_cliff.png")


def fig5_embeddings(df: pd.DataFrame):
    """Embedding model comparison across conditions."""
    agg = df.groupby(["condition", "embedding_model"])[PRIMARY].mean().unstack()
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(agg)); width = 0.35
        for i, (col, label) in enumerate([("minilm", "MiniLM (general)"),
                                          ("pubmedbert", "PubMedBERT (biomedical)")]):
            if col in agg.columns:
                ax.bar(x + (i - 0.5) * width, agg[col], width, label=label)
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=35, ha="right")
        ax.set_ylabel(f"Mean {PRIMARY}")
        ax.set_title("Embedding model x chunking condition")
        ax.legend()
        _save(fig, "fig5_embeddings.png")


def fig6_k_effect(df: pd.DataFrame, anchor: str):
    """Retrieval depth k against quality and answer rate."""
    keep = [anchor] + [c for c in SENTENCE_STRATEGIES + ["semantic"]
                       if c in set(df["condition"])]
    sub = df[df["condition"].isin(keep)]
    with plt.rc_context(PLT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for cond, grp in sub.groupby("condition"):
            ax1.plot(*_agg(grp, "k", PRIMARY), marker="o", label=cond)
            ax2.plot(*_agg(grp, "k", "answered"), marker="s", label=cond)
        ax1.set_xlabel("k (retrieval depth)"); ax1.set_ylabel(f"Mean {PRIMARY}")
        ax1.set_title("Quality"); ax1.set_xticks([3, 5, 8, 10]); ax1.legend(fontsize=8)
        ax2.set_xlabel("k (retrieval depth)"); ax2.set_ylabel("Answer rate")
        ax2.set_title("Answer rate"); ax2.set_xticks([3, 5, 8, 10]); ax2.legend(fontsize=8)
        fig.suptitle("Retrieval depth k across strategies", fontsize=13)
        _save(fig, "fig6_k_effect.png")


def _agg(frame: pd.DataFrame, by: str, col: str):
    s = frame.groupby(by)[col].mean()
    return s.index.values, s.values


def _save(fig, filename: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #

def main():
    df = load_results()
    anchor = best_fixed_condition(df)
    print(f"Best fixed-token condition (anchor): {anchor}\n")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    summary = compute_summary(df)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved to {SUMMARY_PATH}\n")

    test_h1(df)
    test_h2(df)
    test_h3(df)
    test_embeddings(df)

    print("\nGenerating figures...")
    fig1_overlap(df)
    fig2_chunk_size(df)
    fig3_strategies(df, anchor)
    fig4_context_cliff(df)
    fig5_embeddings(df)
    fig6_k_effect(df, anchor)

    with open(HYPOTHESES_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_report_lines))
    print(f"\nHypothesis report saved to {HYPOTHESES_PATH}")
    print(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
