"""
Stage 3: apply all chunking conditions to the corpus.

Saves each condition to data/chunks/{condition}.pkl and writes statistics to
data/chunk_stats.csv. Existing chunk files are reused; delete one to rebuild it.

Usage:
    python src/run_chunkers.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from chunkers import (
    FixedSizeChunker,
    SentenceNLTKChunker,
    SentenceScispaCyChunker,
    SemanticChunker,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "pubmed_corpus.csv")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
STATS_PATH = os.path.join(DATA_DIR, "chunk_stats.csv")


def build_chunker_list() -> list:
    return FixedSizeChunker.all_conditions() + [
        SentenceNLTKChunker(),
        SentenceScispaCyChunker(),
        SemanticChunker(),
    ]


def run_chunker(chunker, df: pd.DataFrame) -> list[dict]:
    all_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=chunker.condition_key, leave=False):
        chunks = chunker.chunk(str(row["abstract"]), str(row["pmid"]))
        for c in chunks:
            c["topic"] = row.get("topic", "")
        all_chunks.extend(chunks)
    return all_chunks


def compute_stats(condition_key: str, chunks: list[dict]) -> dict:
    word_counts = [c["word_count"] for c in chunks]
    return {
        "condition": condition_key,
        "total_chunks": len(chunks),
        "mean_words": round(float(np.mean(word_counts)), 1),
        "std_words": round(float(np.std(word_counts)), 1),
        "min_words": int(np.min(word_counts)),
        "max_words": int(np.max(word_counts)),
        "median_words": round(float(np.median(word_counts)), 1),
    }


def main():
    df = pd.read_csv(CORPUS_PATH)
    print(f"Loaded {len(df)} abstracts from corpus.")

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    stats_rows = []

    for chunker in build_chunker_list():
        key = chunker.condition_key
        out_path = os.path.join(CHUNKS_DIR, f"{key}.pkl")

        if os.path.exists(out_path):
            print(f"  [skip] {key} already exists.")
            with open(out_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            print(f"  Chunking: {key}")
            chunks = run_chunker(chunker, df)
            with open(out_path, "wb") as f:
                pickle.dump(chunks, f)
            print(f"    -> {len(chunks)} chunks saved to {out_path}")

        stats_rows.append(compute_stats(key, chunks))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(STATS_PATH, index=False)
    print(f"\nChunk statistics saved to {STATS_PATH}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
