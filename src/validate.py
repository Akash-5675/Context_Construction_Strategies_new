"""Stage 7: data-quality check on the results file: failures, blanks, duplicates, out-of-range metrics.

Note on ranges: answer_relevance is a cosine similarity, so its valid range is
[-1, 1]; negative values are legitimate (answer semantically opposed to the
question, common on refusals). The other three metrics are bounded [0, 1].

Note on refusals: a terse "Insufficient context." is a genuine model answer.
The only reliable signature of a failed API call is latency_ms == 0, since
latency is assigned only inside the successful branch of generator.generate().
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analysis_config import RESULTS, refusal_flags

df = pd.read_csv(RESULTS)

print(f"file: {RESULTS}")
print(f"rows: {len(df)} of 12000\n")
per_condition = df.groupby("condition").size()
print(f"conditions: {len(per_condition)} of 15; not at 800 runs: {per_condition[per_condition != 800].to_dict()}\n")

corrupt = 0

failed = (df["answer"].astype(str).str.strip() == "GENERATION_FAILED").sum()
blank = df["answer"].isna().sum() + (df["answer"].astype(str).str.strip() == "").sum()
zero_lat = (df["latency_ms"] == 0).sum()
dupes = df.duplicated("run_key").sum()

print("corrupt-row checks (all should be 0):")
for label, n in [("GENERATION_FAILED rows", failed),
                 ("blank/null answers", blank),
                 ("latency_ms == 0 (no API response)", zero_lat),
                 ("duplicate run_keys", dupes)]:
    print("  %-38s %d" % (label, n))
    corrupt += n

print("\nmetric ranges:")
for col, lo_ok in [("rouge_l", 0.0), ("answer_relevance", -1.0),
                   ("faithfulness", 0.0), ("retrieval_recall", 0.0)]:
    bad = ((df[col] < lo_ok) | (df[col] > 1)).sum()
    corrupt += bad
    print("  %-18s min %6.3f  max %.3f  valid [%.0f, 1]  out-of-range %d"
          % (col, df[col].min(), df[col].max(), lo_ok, bad))

nulls = df.isna().sum()
nulls = nulls[nulls > 0]
print("\nnull values:", "none" if nulls.empty else f"\n{nulls.to_string()}")
corrupt += int(nulls.sum())

refused = refusal_flags(df["answer"])
terse = (df["answer"].astype(str).str.strip() == "Insufficient context.").sum()
print(f"\nrefusals: {refused.sum()} ({100*refused.mean():.1f}%) "
      f"-- {terse} terse, {refused.sum()-terse} with explanation")
print(f"  median latency on terse refusals: "
      f"{df.loc[df['answer'].astype(str).str.strip()=='Insufficient context.','latency_ms'].median():.0f} ms")

print(f"\nrows needing re-run: {corrupt}")
