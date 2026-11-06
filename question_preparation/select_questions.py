"""Pick the final 100 questions. Run from the project root:

    python question_preparation/select_questions.py

Drops the two excluded PMIDs, then samples 20 per topic with a fixed seed, so
the choice does not depend on how any question performs.
"""

import pandas as pd

SEED = 42
EXCLUDED = {"42173528", "42176342"}
TOPICS = ["hypertension", "diabetes", "lung_cancer", "covid19", "alzheimers"]

PREFIX = {'hypertension': 'HYP', 'diabetes': 'DIA', 'lung_cancer': 'LCA', 'covid19': 'COV', 'alzheimers': 'ALZ'}

draft = pd.read_csv("question_preparation/draft_checked.csv", dtype=str).fillna("")
draft = draft[~draft.pmid.isin(EXCLUDED) & (draft["flags"] == "")]

rows = []
for topic in TOPICS:
    pool = draft[draft.topic == topic].sort_values("pmid")
    chosen = pool.sample(n=20, random_state=SEED).sort_values("pmid")
    print(f"{topic}: chose 20 of {len(pool)}")
    for i, row in enumerate(chosen.itertuples(), 1):
        rows.append({
            "question_id": f"{PREFIX[topic]}_{i:03d}",
            "topic": topic,
            "question": row.question,
            "gold_doc_ids": row.pmid,
            "reference_answer": row.reference_answer,
            "evidence": row.evidence,
        })

final = pd.DataFrame(rows)
final.to_csv("data/questions.csv", index=False, encoding="utf-8")
print(f"saved {len(final)} questions, {final.gold_doc_ids.nunique()} unique PMIDs")
