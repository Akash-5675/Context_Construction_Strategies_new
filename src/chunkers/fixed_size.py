"""
Fixed-size chunker, part of stage 3.

Splits text into chunks of exactly `chunk_size` words with a configurable
word overlap. Produces 12 conditions: 3 sizes (128/256/512) × 4 overlaps
(0 / 10 / 20 / 40 percent of chunk_size).

The condition key format is: fixed_{size}_{overlap_pct}pct
  e.g. fixed_256_20pct
"""

from __future__ import annotations


class FixedSizeChunker:
    SIZES = [128, 256, 512]
    OVERLAP_PCTS = [0, 10, 20, 40]

    def __init__(self, chunk_size: int = 256, overlap_pct: int = 20):
        if chunk_size not in self.SIZES:
            raise ValueError(f"chunk_size must be one of {self.SIZES}")
        if overlap_pct not in self.OVERLAP_PCTS:
            raise ValueError(f"overlap_pct must be one of {self.OVERLAP_PCTS}")
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_pct / 100)
        self.overlap_pct = overlap_pct

    @property
    def condition_key(self) -> str:
        return f"fixed_{self.chunk_size}_{self.overlap_pct}pct"

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    "chunk_id": f"{doc_id}__{self.condition_key}_{idx}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "word_count": end - start,
                    "strategy": self.condition_key,
                    "chunk_index": idx,
                }
            )
            if end == len(words):
                break
            start += step
            idx += 1
        return chunks

    @classmethod
    def all_conditions(cls) -> list["FixedSizeChunker"]:
        return [
            cls(size, pct)
            for size in cls.SIZES
            for pct in cls.OVERLAP_PCTS
        ]
