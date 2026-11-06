"""Build the Kaggle dataset zip. Run from the project root:

    python kaggle/make_kaggle_bundle.py data/questions.csv

Refuses to build if the semantic chunks or questions are not ready.
Never includes .env or any other secrets file.
"""

import glob
import os
import pickle
import sys
import zipfile

import pandas as pd

OUT = "kaggle/rag_bundle.zip"
REQUIRED = {"question_id", "topic", "question", "gold_doc_ids", "reference_answer"}

questions_path = sys.argv[1]
questions = pd.read_csv(questions_path, dtype=str)
missing = REQUIRED - set(questions.columns)
if missing:
    sys.exit(f"Questions file is missing columns: {sorted(missing)}")
if len(questions) != 100 or not (questions.groupby("topic").size() == 20).all():
    sys.exit("Questions file must contain exactly 20 questions for each of 5 topics.")
if questions.question_id.duplicated().any():
    sys.exit("Questions file has duplicate question IDs.")

semantic = pickle.load(open("data/chunks/semantic.pkl", "rb"))
scispacy = {c["text"] for c in pickle.load(open("data/chunks/sent_scispacy.pkl", "rb"))}
if sum(c["text"] in scispacy for c in semantic) > len(semantic) * 0.5:
    sys.exit("Semantic chunks still mostly match scispaCy: fix the chunker and rebuild first.")

conditions = [os.path.basename(p)[:-4] for p in glob.glob("data/chunks/*.pkl")]
if len(conditions) != 15:
    sys.exit(f"Expected 15 chunk files, found {len(conditions)}.")

files = [
    "pipeline/__init__.py", "pipeline/embedder.py", "pipeline/evaluator.py",
    "pipeline/generator.py", "pipeline/retriever.py",
    "src/experiment_runner.py", "src/validate.py", "src/analyze.py", "src/analysis_config.py",
] + sorted(glob.glob("data/chunks/*.pkl"))
for condition in conditions:
    for model in ("minilm", "pubmedbert"):
        files += [f"indexes/{condition}__{model}.index", f"indexes/{condition}__{model}.meta.pkl"]

absent = [f for f in files if not os.path.exists(f)]
if absent:
    sys.exit(f"Missing files: {absent[:5]}")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, path)
    z.write(questions_path, "data/questions.csv")
print(f"Built {OUT}: {len(files) + 1} files, {os.path.getsize(OUT) / 1e6:.0f} MB")
