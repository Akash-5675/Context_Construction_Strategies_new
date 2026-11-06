"""Stage 8: main inferential tests, treating the question as the unit of analysis.

Runs on one question are repeated measures, so tests that count every run as
independent overstate the evidence. Here, paired tests use one mean per question,
and mixed models give each question its own random intercept.

Writes robust_stats.txt and robust_stats.json to the analysis output folder.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
from analysis_config import OUTPUT, load_results

warnings.filterwarnings("ignore")
lines, report = [], {}


def say(text=""):
    print(text)
    lines.append(text)


def paired(frame, factor, pairs, metric):
    """Paired t-tests on per-question means, Holm-corrected within this family."""
    means = frame.groupby(["question_id", factor])[metric].mean().unstack()
    rows = []
    for a, b in pairs:
        if a not in means or b not in means:
            continue
        diff = (means[a] - means[b]).dropna()
        ci = stats.t.interval(0.95, len(diff) - 1, diff.mean(), stats.sem(diff)) if diff.std() > 0 else (0.0, 0.0)
        p = stats.ttest_rel(means.loc[diff.index, a], means.loc[diff.index, b]).pvalue if diff.std() > 0 else 1.0
        rows.append({"a": str(a), "b": str(b), "metric": metric, "n_questions": int(len(diff)),
                     "mean_a": float(means.loc[diff.index, a].mean()), "mean_b": float(means.loc[diff.index, b].mean()),
                     "diff": float(diff.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(p)})
    if rows:
        adjusted = multipletests([r["p"] for r in rows], method="holm")[1]
        for row, value in zip(rows, adjusted):
            row["p_holm"] = float(value)
            say(f"  {row['metric']}: {row['a']} vs {row['b']}: diff {row['diff']:+.4f} "
                f"(95% CI {row['ci_low']:+.4f} to {row['ci_high']:+.4f}), Holm p = {row['p_holm']:.4g}, "
                f"n = {row['n_questions']}")
    return rows


def lr_test(frame, full, reduced):
    """Likelihood-ratio test between nested models, with question random intercepts."""
    fit_full = mixedlm(full, frame, groups=frame["question_id"]).fit(reml=False)
    fit_reduced = mixedlm(reduced, frame, groups=frame["question_id"]).fit(reml=False)
    chi2 = max(0.0, 2 * (fit_full.llf - fit_reduced.llf))
    df = len(fit_full.fe_params) - len(fit_reduced.fe_params)
    return {"chi2": float(chi2), "df": int(df), "p": float(stats.chi2.sf(chi2, df))}


df = load_results()
df["answered_num"] = df["answered"].astype(float)
df["chunk_size"] = df["condition"].str.extract(r"fixed_(\d+)_")[0].astype(float)
df["overlap_pct"] = df["condition"].str.extract(r"fixed_\d+_(\d+)pct")[0].astype(float)
say(f"Rows: {len(df):,} | questions: {df.question_id.nunique()} | refusal rate: {df.refused.mean():.1%}")
say("Read a non-significant test as 'no effect detected', and use the confidence interval for its plausible size.\n")

# ---------------------------------------------------------------- H1
say("H1: overlap versus no overlap (fixed-size chunks)")
fixed = df[df.chunk_size.notna()].copy()
overlap_pairs = [(0.0, 10.0), (0.0, 20.0), (0.0, 40.0)]
report["h1_paired"] = {}
for size in sorted(fixed.chunk_size.unique()):
    say(f" Chunk size {int(size)} words:")
    report["h1_paired"][str(int(size))] = []
    for metric in ("rouge_l", "answered_num"):
        report["h1_paired"][str(int(size))] += paired(fixed[fixed.chunk_size == size], "overlap_pct", overlap_pairs, metric)
base = "rouge_l ~ C(chunk_size) + C(k) + C(embedding_model)"
report["h1_overlap_lr"] = lr_test(fixed, base + " + C(overlap_pct)", base)
report["h1_interaction_lr"] = lr_test(fixed, base + " + C(overlap_pct) + C(chunk_size):C(overlap_pct)", base + " + C(overlap_pct)")
say(f" Mixed model, overlap: {report['h1_overlap_lr']}")
say(f" Mixed model, size x overlap: {report['h1_interaction_lr']}")
say(" Abstracts average about 238 words, so overlap can change little at 512 words.\n")

# ---------------------------------------------------------------- H2
say("H2: sentence versus semantic chunking")
three = df[df.condition.isin(["sent_nltk", "sent_scispacy", "semantic"])].copy()
report["h2_paired"] = []
if three.condition.nunique() == 3:
    pairs = [("sent_nltk", "semantic"), ("sent_scispacy", "semantic"), ("sent_nltk", "sent_scispacy")]
    for metric in ("rouge_l", "answered_num"):
        report["h2_paired"] += paired(three, "condition", pairs, metric)
    three["context_k"] = three.context_words / 1000
    base = "rouge_l ~ C(k) + C(embedding_model)"
    report["h2_condition_lr"] = lr_test(three, base + " + C(condition)", base)
    report["h2_condition_given_context_lr"] = lr_test(three, base + " + context_k + C(condition)", base + " + context_k")
    say(f" Mixed model, condition: {report['h2_condition_lr']}")
    say(f" Mixed model, condition given context size: {report['h2_condition_given_context_lr']}")
    say(" Context size and semantic chunking overlap heavily, so the adjusted result relies on few comparable runs.")
    say(" NLTK and scispaCy produce nearly identical chunks, so their similarity is expected by construction.\n")
else:
    say(" Skipped: all three conditions are required.\n")

# ---------------------------------------------------------------- H3
say("H3: quality against context size, with a change in slope at 2,500 estimated tokens")
d = df.copy()
d["tokens_k"] = d.context_words / 750
d["after_cliff"] = np.maximum(0, d.tokens_k - 2.5)
base = "rouge_l ~ C(embedding_model) + tokens_k"
fit = mixedlm(base + " + after_cliff", d, groups=d["question_id"]).fit(reml=False)
report["h3"] = {"n_runs_past_2500": int((d.after_cliff > 0).sum()),
                "slope_before_per_1k_tokens": float(fit.fe_params["tokens_k"]),
                "slope_change_after_2500": float(fit.fe_params["after_cliff"]),
                "slope_change_lr": lr_test(d, base + " + after_cliff", base)}
say(f" {report['h3']}")
say(" A cliff needs a significantly negative slope change. Context size is tied to k and chunk size, so this is association only.\n")

# ---------------------------------------------------------------- encoders
say("Secondary: MiniLM versus PubMedBERT")
report["encoder_paired"] = []
for metric in ("rouge_l", "answered_num", "retrieval_recall"):
    report["encoder_paired"] += paired(df, "embedding_model", [("minilm", "pubmedbert")], metric)

os.makedirs(OUTPUT, exist_ok=True)
with open(os.path.join(OUTPUT, "robust_stats.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUTPUT, "robust_stats.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print(f"\nSaved to {OUTPUT}")
