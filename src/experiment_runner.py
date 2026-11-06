"""
Stage 6: experiment runner.

Runs every combination of chunking condition, question, k and embedding model:

    Part A: 12 fixed-size conditions x 100 x 4 x 2 = 9,600 runs
    Part B: 3 other strategies x 100 x 4 x 2 = 2,400 runs, plus an anchor
            fixed-size condition whose runs are reused from Part A if present

Rows are saved every CHECKPOINT_EVERY runs. Finished runs are skipped, so an
interrupted experiment can resume. The final two-GPU run called this module from
kaggle/run_worker.py.

Usage:
    python src/experiment_runner.py --part A
    python src/experiment_runner.py --part B --best-fixed fixed_128_0pct
    python src/experiment_runner.py --part A --dry-run
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.retriever import retrieve
from pipeline.generator import generate
from pipeline.evaluator import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_PATH = os.path.join(ROOT, "results", "all_results.csv")
QUESTIONS_PATH = os.path.join(ROOT, "data", "questions.csv")
CHUNKS_DIR = os.path.join(ROOT, "data", "chunks")

K_VALUES = [3, 5, 8, 10]
EMBEDDING_MODELS = ["minilm", "pubmedbert"]
CHECKPOINT_EVERY = 50  # rows still in memory are lost if the process is killed

FIXED_CONDITIONS = [
    f"fixed_{size}_{pct}pct"
    for size in [128, 256, 512]
    for pct in [0, 10, 20, 40]
]
OTHER_STRATEGIES = ["sent_nltk", "sent_scispacy", "semantic"]


def load_questions(dry_run: bool = False, path: str = QUESTIONS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"question_id", "question", "gold_doc_ids", "reference_answer", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questions.csv is missing columns: {missing}")
    if dry_run:
        df = df.head(5)
    return df


def get_completed_keys(results_path: str) -> set[str]:
    if not os.path.exists(results_path):
        return set()
    df = pd.read_csv(results_path, usecols=["run_key"])
    return set(df["run_key"].tolist())


def run_single(
    question_id: str,
    question: str,
    gold_doc_ids: list[str],
    reference_answer: str,
    topic: str,
    condition_key: str,
    model_key: str,
    k: int,
) -> dict:
    chunks = retrieve(question, condition_key, model_key, k=k)
    gen_result = generate(question, chunks)
    metrics = evaluate(
        query=question,
        answer=gen_result["answer"],
        reference_answer=reference_answer,
        gold_doc_ids=gold_doc_ids,
        retrieved_chunks=chunks,
        context_word_count=gen_result["context_word_count"],
        latency_ms=gen_result["latency_ms"],
        model_key=model_key,
    )
    return {
        "run_key": f"{question_id}__{condition_key}__{model_key}__k{k}",
        "question_id": question_id,
        "topic": topic,
        "condition": condition_key,
        "embedding_model": model_key,
        "k": k,
        "answer": gen_result["answer"],
        "prompt_tokens": gen_result["prompt_tokens"],
        "completion_tokens": gen_result["completion_tokens"],
        **metrics,
    }


def run_experiment(
    conditions: list[str],
    questions_df: pd.DataFrame,
    results_path: str = RESULTS_PATH,
    max_runs: int | None = None,
    max_seconds: int | None = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    completed = get_completed_keys(results_path)
    started = time.time()
    new_runs = 0
    stop = False
    print(f"Already completed: {len(completed)} runs. Resuming...")

    buffer = []
    total = len(conditions) * len(questions_df) * len(K_VALUES) * len(EMBEDDING_MODELS)
    pbar = tqdm(total=total, desc="runs")

    try:
        for cond in conditions:
            for _, row in questions_df.iterrows():
                gold_ids = [g.strip() for g in str(row["gold_doc_ids"]).split("|")]
                for model_key in EMBEDDING_MODELS:
                    for k in K_VALUES:
                        run_key = f"{row['question_id']}__{cond}__{model_key}__k{k}"
                        if stop:
                            continue
                        pbar.update(1)
                        if run_key in completed:
                            continue
                        if (max_runs is not None and new_runs >= max_runs) or (
                            max_seconds is not None and time.time() - started >= max_seconds
                        ):
                            stop = True
                            continue
                        try:
                            result = run_single(
                                question_id=str(row["question_id"]),
                                question=str(row["question"]),
                                gold_doc_ids=gold_ids,
                                reference_answer=str(row["reference_answer"]),
                                topic=str(row["topic"]),
                                condition_key=cond,
                                model_key=model_key,
                                k=k,
                            )
                            buffer.append(result)
                            completed.add(run_key)
                            new_runs += 1
                        except SystemExit:
                            raise
                        except Exception as e:
                            tqdm.write(f"ERROR {run_key}: {e}")

                        if len(buffer) >= CHECKPOINT_EVERY:
                            _flush(buffer, results_path)
                            buffer = []
    finally:
        # Save buffered rows even after an abort or Ctrl+C.
        if buffer:
            _flush(buffer, results_path)
        pbar.close()

    elapsed = time.time() - started
    status = "Stopped at limit" if stop else "Done"
    print(f"{status}. New runs: {new_runs} in {elapsed / 60:.1f} min. Results at {results_path}")


def _flush(rows: list[dict], results_path: str = RESULTS_PATH):
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(results_path)
    df.to_csv(results_path, mode="a", index=False, header=write_header)
    tqdm.write(f"  Checkpoint: {len(rows)} rows written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B"], required=True,
                        help="A = fixed-size conditions; B = other strategies plus an anchor")
    parser.add_argument("--best-fixed", default="fixed_128_0pct",
                        help="Fixed-size anchor condition for Part B")
    parser.add_argument("--dry-run", action="store_true", help="Use the first 5 questions")
    parser.add_argument("--questions", default=QUESTIONS_PATH, help="Questions CSV to use")
    parser.add_argument("--results", default=RESULTS_PATH, help="Results CSV to append to")
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after this many new runs")
    parser.add_argument("--max-seconds", type=int, default=None, help="Stop cleanly after this long")
    args = parser.parse_args()

    questions_df = load_questions(dry_run=args.dry_run, path=args.questions)
    print(f"Running {'DRY RUN' if args.dry_run else 'FULL'} with {len(questions_df)} questions.")

    conditions = FIXED_CONDITIONS if args.part == "A" else [args.best_fixed] + OTHER_STRATEGIES
    print(f"Part {args.part}: {conditions}")

    missing = [c for c in conditions if not os.path.exists(os.path.join(CHUNKS_DIR, f"{c}.pkl"))]
    if missing:
        raise SystemExit(f"Missing chunk files: {missing}. Run src/run_chunkers.py first.")

    run_experiment(conditions, questions_df, args.results, args.max_runs, args.max_seconds)


if __name__ == "__main__":
    main()
