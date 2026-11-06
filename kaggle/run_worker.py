"""Run one GPU's share of the experiment. The notebook starts one per GPU.

    python run_worker.py WORKER RESULTS_CSV [--max-runs N] [--max-seconds N]

Set GENERATOR_BACKEND=ollama and OLLAMA_URL before starting. Conditions alternate
between workers so each gets a similar mix of chunk sizes; the split is fixed, so
resuming in a later session continues the same work.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiment_runner import (  # noqa: E402
    EMBEDDING_MODELS,
    FIXED_CONDITIONS,
    K_VALUES,
    get_completed_keys,
    load_questions,
    run_experiment,
)

CONDITIONS = FIXED_CONDITIONS + ["sent_nltk", "sent_scispacy", "semantic"]
MAX_PASSES = 3  # later passes retry only runs that failed in an earlier pass

parser = argparse.ArgumentParser()
parser.add_argument("worker", type=int, choices=[0, 1])
parser.add_argument("results")
parser.add_argument("--max-runs", type=int, default=None)
parser.add_argument("--max-seconds", type=int, default=None)
args = parser.parse_args()

conditions = CONDITIONS[args.worker :: 2]
questions = load_questions(path=os.path.join(ROOT, "data", "questions.csv"))
expected = {
    f"{q}__{c}__{m}__k{k}"
    for q in questions.question_id
    for c in conditions
    for m in EMBEDDING_MODELS
    for k in K_VALUES
}
print(f"Worker {args.worker}: {conditions} ({len(expected)} runs)", flush=True)

started = time.time()
for attempt in range(1 if args.max_runs is not None else MAX_PASSES):
    remaining = None
    if args.max_seconds is not None:
        remaining = int(args.max_seconds - (time.time() - started))
        if remaining <= 0:
            break
    run_experiment(conditions, questions, args.results, args.max_runs, remaining)
    if not expected - get_completed_keys(args.results):
        break

missing = len(expected - get_completed_keys(args.results))
print(f"Worker {args.worker} finished: {missing} runs still missing", flush=True)
