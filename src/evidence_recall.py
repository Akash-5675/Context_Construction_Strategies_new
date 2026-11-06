"""Stage 8: passage-level retrieval: did the top-k chunks contain the evidence sentence?

retrieval_recall only asks whether a chunk from the gold abstract came back. This
script repeats retrieval, which needs no language model, and measures how much of
the question's evidence text those gold-abstract chunks cover.

  evidence_coverage  share of the evidence's distinct words found in them (0-1)
  evidence_hit       the whole evidence text is inside one retrieved chunk

Word coverage ignores word order, and evidence_hit is strict when evidence spans
sentences that a chunker separated, so report both. Writes evidence_recall.csv
and evidence_recall_summary.csv to the analysis output folder.
"""

import os
import pickle
import re
import sys

import faiss
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis_config import OUTPUT, QUESTIONS, ROOT
from pipeline.embedder import embed

K_VALUES = [3, 5, 8, 10]
MODELS = ["minilm", "pubmedbert"]
FIXED = [f"fixed_{s}_{o}pct" for s in (128, 256, 512) for o in (0, 10, 20, 40)]
CONDITIONS = FIXED + ["sent_nltk", "sent_scispacy", "semantic"]


def norm(text):
    text = re.sub(r"<[^>]+>", "", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return set(re.findall(r"[a-z0-9]+", text))


questions = pd.read_csv(QUESTIONS, dtype=str)
assert "evidence" in questions, "Questions file needs an evidence column."
rows = []
for model in MODELS:
    vectors = embed(questions.question.tolist(), model)
    for condition in CONDITIONS:
        stem = os.path.join(ROOT, "indexes", f"{condition}__{model}")
        index = faiss.read_index(stem + ".index")
        with open(stem + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        _, positions = index.search(vectors, max(K_VALUES))
        for i, q in enumerate(questions.itertuples()):
            evidence = norm(q.evidence)
            evidence_words = words(evidence)
            for k in K_VALUES:
                gold = [norm(meta[p]["text"]) for p in positions[i][:k] if meta[p]["doc_id"] == q.gold_doc_ids]
                covered = words(" ".join(gold)) & evidence_words
                rows.append({
                    "run_key": f"{q.question_id}__{condition}__{model}__k{k}",
                    "question_id": q.question_id, "condition": condition, "embedding_model": model, "k": k,
                    "doc_recall": float(bool(gold)),
                    "evidence_coverage": round(len(covered) / max(1, len(evidence_words)), 4),
                    "evidence_hit": float(any(evidence in text for text in gold)),
                })

result = pd.DataFrame(rows)
assert not result.run_key.duplicated().any()
os.makedirs(OUTPUT, exist_ok=True)
result.to_csv(os.path.join(OUTPUT, "evidence_recall.csv"), index=False)
summary = result.groupby(["condition", "k"])[["doc_recall", "evidence_coverage", "evidence_hit"]].mean().round(4)
summary.to_csv(os.path.join(OUTPUT, "evidence_recall_summary.csv"))
print(result.groupby("condition")[["doc_recall", "evidence_coverage", "evidence_hit"]].mean().round(3).to_string())
print(f"\n{len(result):,} rows saved to {OUTPUT}")
