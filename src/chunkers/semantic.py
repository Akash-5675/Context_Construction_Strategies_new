"""
Semantic chunker, part of stage 3.

Uses scispaCy for sentence segmentation, then embeds each sentence with
all-MiniLM-L6-v2. A new chunk boundary is inserted whenever the cosine
similarity between adjacent sentences falls below SIMILARITY_THRESHOLD (0.75).
This means topically coherent sentences are grouped together regardless of
word count.

Condition key: semantic

Reference: Kamradt (2023), SemanticChunker concept applied to biomedical text.
"""

from __future__ import annotations
import numpy as np
from .sentence_scispacy import split_sentences

SIMILARITY_THRESHOLD = 0.75

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticChunker:
    condition_key = "semantic"

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = similarity_threshold

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        # Use individual scispaCy sentences as atomic units
        return self.chunk_sentences(split_sentences(text), doc_id)

    def chunk_sentences(self, sentences: list[str], doc_id: str) -> list[dict]:
        if not sentences:
            return []

        model = get_embed_model()
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        chunks: list[dict] = []
        buffer: list[str] = [sentences[0]]
        idx = 0

        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            if sim < self.threshold and buffer:
                # semantic boundary detected — flush current buffer
                chunk_text = " ".join(buffer)
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}__{self.condition_key}_{idx}",
                        "doc_id": doc_id,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "strategy": self.condition_key,
                        "chunk_index": idx,
                        "boundary_similarity": round(sim, 4),
                    }
                )
                idx += 1
                buffer = [sentences[i]]
            else:
                buffer.append(sentences[i])

        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}__{self.condition_key}_{idx}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "strategy": self.condition_key,
                    "chunk_index": idx,
                    "boundary_similarity": None,
                }
            )
        return chunks
