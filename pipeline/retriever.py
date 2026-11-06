"""
Stage 5: retrieval.

Loads a pre-built FAISS index and metadata, embeds the query, and returns
the top-k most similar chunks with their similarity scores and doc IDs.

K values tested in experiments: 3, 5, 8, 10.
"""

from __future__ import annotations
import os
import pickle

import faiss
import numpy as np

from pipeline.embedder import embed_single

INDEXES_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

_index_cache: dict[str, faiss.Index] = {}
_meta_cache: dict[str, list[dict]] = {}


def _load(condition_key: str, model_key: str):
    cache_key = f"{condition_key}__{model_key}"
    if cache_key not in _index_cache:
        index_path = os.path.join(INDEXES_DIR, f"{cache_key}.index")
        meta_path = os.path.join(INDEXES_DIR, f"{cache_key}.meta.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: python pipeline/indexer.py"
            )
        _index_cache[cache_key] = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            _meta_cache[cache_key] = pickle.load(f)
    return _index_cache[cache_key], _meta_cache[cache_key]


def retrieve(
    query: str,
    condition_key: str,
    model_key: str,
    k: int = 5,
) -> list[dict]:
    index, meta = _load(condition_key, model_key)
    k = min(k, index.ntotal)
    query_vec = embed_single(query, model_key).reshape(1, -1)
    scores, positions = index.search(query_vec, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        chunk = dict(meta[pos])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results
