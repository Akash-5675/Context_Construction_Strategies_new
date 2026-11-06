"""Shared paths and refusal rule for every analysis script."""

import os
import re

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results", "all_results.csv")
QUESTIONS = os.path.join(ROOT, "data", "questions.csv")
OUTPUT = os.path.join(ROOT, "results", "analysis")
FIGURES = os.path.join(OUTPUT, "figures")

_EXPLICIT = re.compile(r"insufficient context", re.I)
# Only an opening statement counts: "does not mention X" later in a real answer
# is not a refusal.
_SOFT_OPENING = re.compile(
    r"^\s*(?:(?:based on|according to) the (?:provided |given )?context,?\s*)?"
    r"(?:the |this )?(?:provided |given )?(?:context|sources?|text|passages?)\s+"
    r"(?:does|do)(?: not|n't)\s+"
    r"(?:contain|provide|mention|state|specify|include|discuss|address|report|give)\b",
    re.I,
)


def is_refusal(answer) -> bool:
    text = "" if pd.isna(answer) else str(answer)
    return bool(_EXPLICIT.search(text) or _SOFT_OPENING.match(text))


def refusal_flags(answers: pd.Series) -> pd.Series:
    return answers.map(is_refusal).astype(bool)


def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS):
        raise SystemExit(f"Results file not found: {RESULTS}")
    df = pd.read_csv(RESULTS)
    df["refused"] = refusal_flags(df["answer"])
    df["answered"] = ~df["refused"]
    return df
