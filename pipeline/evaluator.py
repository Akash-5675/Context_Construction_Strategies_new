"""
Stage 5: evaluation metrics.

Six values computed for every (question, condition, k, embedding_model) run:

1. answer_relevance  — cosine sim(query_embedding, answer_embedding)
                       Measures topical alignment between question and answer.

2. rouge_l           — ROUGE-L F1 between generated answer and gold reference.
                       Measures longest common subsequence overlap; standard NLP
                       metric for generation quality against a reference string.

3. faithfulness      — token-level precision overlap between answer tokens and
                       all retrieved chunk tokens. Detects hallucination: a low
                       score means the answer contains tokens not in the context.

4. retrieval_recall  — proportion of gold_doc_ids found in retrieved chunk
                       doc_ids. Measures whether the correct source was returned.

5. context_words     — total word count of concatenated retrieved chunks
                       (the H3 variable for context-cliff analysis).

6. latency_ms        — wall-clock milliseconds from query to answer.
"""

from __future__ import annotations
import re
import numpy as np
from rouge_score import rouge_scorer

from pipeline.embedder import embed_single

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def answer_relevance(query: str, answer: str, model_key: str) -> float:
    q_vec = embed_single(query, model_key)
    a_vec = embed_single(answer, model_key)
    return float(np.dot(q_vec, a_vec))


def rouge_l(answer: str, reference: str) -> float:
    scores = _scorer.score(reference, answer)
    return round(scores["rougeL"].fmeasure, 4)


def faithfulness(answer: str, chunks: list[dict]) -> float:
    answer_tokens = _tokenise(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _tokenise(" ".join(c["text"] for c in chunks))
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def retrieval_recall(gold_doc_ids: list[str], retrieved_chunks: list[dict]) -> float:
    if not gold_doc_ids:
        return 0.0
    retrieved_doc_ids = {c["doc_id"] for c in retrieved_chunks}
    found = sum(1 for gid in gold_doc_ids if gid in retrieved_doc_ids)
    return found / len(gold_doc_ids)


def evaluate(
    query: str,
    answer: str,
    reference_answer: str,
    gold_doc_ids: list[str],
    retrieved_chunks: list[dict],
    context_word_count: int,
    latency_ms: float,
    model_key: str,
) -> dict:
    return {
        "answer_relevance": round(answer_relevance(query, answer, model_key), 4),
        "rouge_l": rouge_l(answer, reference_answer),
        "faithfulness": round(faithfulness(answer, retrieved_chunks), 4),
        "retrieval_recall": round(retrieval_recall(gold_doc_ids, retrieved_chunks), 4),
        "context_words": context_word_count,
        "latency_ms": latency_ms,
    }
