"""
Stage 4: indexing.

Builds a FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine
similarity) for every combination of chunking condition × embedding model.
Each index is saved alongside a metadata list mapping vector position → chunk dict.

Index files:
    indexes/{condition_key}__{model_key}.index
    indexes/{condition_key}__{model_key}.meta.pkl

Usage:
    python pipeline/indexer.py
    python pipeline/indexer.py --condition fixed_256_20pct --model minilm
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys

import faiss
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.embedder import embed

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chunks")
INDEXES_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")
MODEL_KEYS = ["minilm", "pubmedbert"]


def list_conditions() -> list[str]:
    return [
        f.replace(".pkl", "")
        for f in os.listdir(CHUNKS_DIR)
        if f.endswith(".pkl")
    ]


def build_index(condition_key: str, model_key: str, batch_size: int = 256):
    chunks_path = os.path.join(CHUNKS_DIR, f"{condition_key}.pkl")
    index_path = os.path.join(INDEXES_DIR, f"{condition_key}__{model_key}.index")
    meta_path = os.path.join(INDEXES_DIR, f"{condition_key}__{model_key}.meta.pkl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"  [skip] {condition_key}__{model_key} already exists")
        return

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks for {condition_key}__{model_key} ...")

    all_vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="batches", leave=False):
        batch = texts[i : i + batch_size]
        vecs = embed(batch, model_key)
        all_vectors.append(vecs)

    matrix = np.vstack(all_vectors).astype(np.float32)
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    os.makedirs(INDEXES_DIR, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved: {index_path} ({index.ntotal} vectors, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=None)
    parser.add_argument("--model", default=None, choices=MODEL_KEYS + [None])
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else list_conditions()
    models = [args.model] if args.model else MODEL_KEYS

    print(f"Building {len(conditions)} condition(s) × {len(models)} model(s) indexes...")
    for cond in sorted(conditions):
        for model in models:
            build_index(cond, model)
    print("Done.")


if __name__ == "__main__":
    main()
