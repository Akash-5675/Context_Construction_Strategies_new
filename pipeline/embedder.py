"""
Stage 4: embedding.

Loads and caches the two embedding models used in the study:
  - all-MiniLM-L6-v2     (384-dim, ~90 MB, general-domain)
  - NeuML/pubmedbert-base-embeddings  (768-dim, ~420 MB, biomedical)

Both models can run on a CPU or GPU.

The embed() function normalises all vectors to unit length so that
FAISS IndexFlatIP computes cosine similarity correctly.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "pubmedbert": "NeuML/pubmedbert-base-embeddings",
}

_cache: dict[str, SentenceTransformer] = {}


def load_model(model_key: str) -> SentenceTransformer:
    if model_key not in MODELS:
        raise ValueError(f"model_key must be one of {list(MODELS)}")
    if model_key not in _cache:
        print(f"Loading embedding model: {MODELS[model_key]}")
        _cache[model_key] = SentenceTransformer(MODELS[model_key])
    return _cache[model_key]


def embed(texts: list[str], model_key: str, batch_size: int = 64) -> np.ndarray:
    model = load_model(model_key)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,  # L2-normalise for cosine via dot product
    )
    return vectors.astype(np.float32)


def embed_single(text: str, model_key: str) -> np.ndarray:
    return embed([text], model_key)[0]
