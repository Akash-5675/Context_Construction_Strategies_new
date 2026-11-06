"""Check rebuilt questions against the corpus.

Run from the project root:
    python question_preparation/check_questions.py draft.csv

draft.csv columns: topic,pmid,question,reference_answer,evidence
Flags are prompts for human review; passing here does not prove an answer right.
"""

import re
import sys

import pandas as pd

CORPUS = "data/pubmed_corpus.csv"
TOPICS = ["hypertension", "diabetes", "lung_cancer", "covid19", "alzheimers"]


def norm(text):
    text = re.sub(r"<[^>]+>", "", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", str(text)))


draft = pd.read_csv(sys.argv[1], dtype=str).fillna("")
corpus = pd.read_csv(CORPUS, dtype=str).set_index("pmid")
flags = []

for i, row in draft.iterrows():
    problems = []
    if row.pmid not in corpus.index:
        problems.append("PMID not in corpus")
    else:
        source = corpus.loc[row.pmid]
        abstract = norm(source.abstract)
        if row.topic != source.topic:
            problems.append(f"topic should be {source.topic}")
        if norm(row.evidence) not in abstract:
            problems.append("evidence is not an exact quote")
        extra = numbers(row.reference_answer) - numbers(source.abstract)
        if extra:
            problems.append(f"numbers not in abstract: {sorted(extra)}")
    if len(row.reference_answer.split()) < 5:
        problems.append("answer under 5 words")
    if re.search(r"\b(this|the) (study|article|paper|abstract|review|trial)\b",
                 row.question, re.I):
        problems.append("question may not stand alone")
    flags.append("; ".join(problems))

draft["flags"] = flags
output = sys.argv[1].rsplit(".", 1)[0] + "_checked.csv"
draft.to_csv(output, index=False)

clean = draft[draft["flags"] == ""]
print(f"questions: {len(draft)} | no flags: {len(clean)} | flagged: {len(draft) - len(clean)}")
print("no-flag questions per topic:")
print(clean.topic.value_counts().reindex(TOPICS, fill_value=0).to_string())
print(f"duplicate questions: {draft.question.str.lower().duplicated().sum()}")
print(f"PMIDs used more than once: {draft.pmid.duplicated().sum()}")
print(f"saved: {output}")
