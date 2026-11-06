"""Command-line entry point for the project.

    python main.py status      check that every required file is present
    python main.py validate    run data-quality checks on the 12,000 results
    python main.py results     show the main results by condition, k and encoder
    python main.py demo        trace one question through retrieval, answer and scoring
    python main.py analyze     regenerate statistics, tables and figures
"""

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "all_results.csv")
QUESTIONS = os.path.join(ROOT, "data", "questions.csv")
METRICS = ["answered", "rouge_l", "answer_relevance", "faithfulness", "retrieval_recall", "context_words"]

sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def heading(text):
    print(f"\n{text}\n{'=' * len(text)}", flush=True)


def run_script(path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    code = subprocess.call([sys.executable, os.path.join(ROOT, path)], cwd=ROOT, env=env)
    if code:
        raise SystemExit(f"FAILED: {path} (exit code {code})")


def status(_args):
    import pandas as pd

    heading("Project status")
    checks = [
        ("Corpus", len(pd.read_csv(os.path.join(ROOT, "data", "pubmed_corpus.csv"))), 981, "abstracts"),
        ("Questions", len(pd.read_csv(QUESTIONS)), 100, "questions"),
        ("Chunk files", len(glob.glob(os.path.join(ROOT, "data", "chunks", "*.pkl"))), 15, "conditions"),
        ("FAISS indexes", len(glob.glob(os.path.join(ROOT, "indexes", "*.index"))), 30, "indexes"),
        ("Experiment results", len(pd.read_csv(RESULTS, usecols=["run_key"])), 12000, "runs"),
        ("Figures", len(glob.glob(os.path.join(ROOT, "results", "analysis", "figures", "*.png"))), 6, "figures"),
    ]
    failed = False
    for name, found, expected, unit in checks:
        ok = found == expected
        failed |= not ok
        print(f"  [{'OK' if ok else 'MISSING'}] {name:20s} {found:>6,} of {expected:,} {unit}")
    if failed:
        raise SystemExit("\nProject is incomplete.")
    print("\nAll required files are present.")


def validate(_args):
    heading("Validating results")
    run_script("src/validate.py")


def results(_args):
    from analysis_config import load_results
    import pandas as pd

    df = load_results()
    pd.set_option("display.width", 200)
    heading(f"Results: {len(df):,} runs")
    for title, column in [("By chunking condition", "condition"), ("By retrieval depth k", "k"),
                          ("By embedding model", "embedding_model")]:
        print(f"\n{title}")
        table = df.groupby(column)[METRICS].mean().round(3)
        print(table.sort_values("rouge_l", ascending=False).to_string() if column == "condition" else table.to_string())
    print("\nHypothesis tests: results/analysis/robust_stats.txt")
    print("Figures:          results/analysis/figures/")


def demo(args):
    import pandas as pd
    from pipeline.retriever import retrieve

    questions = pd.read_csv(QUESTIONS, dtype=str).set_index("question_id")
    if args.question not in questions.index:
        raise SystemExit(f"Unknown question ID: {args.question}")
    question = questions.loc[args.question]

    heading("1. Question")
    print(f"{args.question} ({question.topic}): {question.question}")
    print(f"Gold PMID: {question.gold_doc_ids}")
    print(f"Reference answer: {question.reference_answer}")

    heading(f"2. Retrieval: {args.condition}, {args.encoder}, k={args.k}")
    for rank, chunk in enumerate(retrieve(question.question, args.condition, args.encoder, k=args.k), 1):
        gold = "  <-- gold abstract" if chunk["doc_id"] == question.gold_doc_ids else ""
        print(f"{rank}. PMID {chunk['doc_id']}, score {chunk['retrieval_score']:.3f}{gold}")
        print(f"   {chunk['text'][:150]}...")

    heading("3. Answer saved from the experiment run (Llama 3.1 8B)")
    run_key = f"{args.question}__{args.condition}__{args.encoder}__k{args.k}"
    rows = pd.read_csv(RESULTS)
    row = rows[rows.run_key == run_key]
    if row.empty:
        raise SystemExit(f"No saved run for {run_key}")
    row = row.iloc[0]
    print(row.answer)

    heading("4. Scores for this run")
    for metric in ["rouge_l", "answer_relevance", "faithfulness", "retrieval_recall", "context_words"]:
        print(f"  {metric:18s} {row[metric]}")
    print(f"  {'latency':18s} {row.latency_ms / 1000:.2f} s")


def analyze(_args):
    for title, script in [("Descriptive statistics and figures", "src/analyze.py"),
                          ("Hypothesis tests", "src/robust_stats.py"),
                          ("Passage-level retrieval", "src/evidence_recall.py")]:
        heading(title)
        run_script(script)
    print("\nDone. Outputs are in results/analysis/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in [("status", status), ("validate", validate), ("results", results), ("analyze", analyze)]:
        commands.add_parser(name).set_defaults(function=function)
    demo_parser = commands.add_parser("demo")
    demo_parser.add_argument("--question", default="COV_019")
    demo_parser.add_argument("--condition", default="fixed_256_20pct")
    demo_parser.add_argument("--encoder", default="pubmedbert", choices=["minilm", "pubmedbert"])
    demo_parser.add_argument("--k", type=int, default=5, choices=[3, 5, 8, 10])
    demo_parser.set_defaults(function=demo)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
