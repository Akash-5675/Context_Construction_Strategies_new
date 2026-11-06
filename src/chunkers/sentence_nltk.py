"""
Sentence-boundary chunker using the NLTK punkt tokeniser, part of stage 3.

Accumulates sentences until the running word count reaches ~TARGET_WORDS,
then carries forward the last sentence(s) as overlap into the next chunk.
This avoids cutting mid-sentence while keeping a roughly consistent chunk size.

Condition key: sent_nltk
"""

from __future__ import annotations
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

TARGET_WORDS = 256
CARRY_FORWARD_SENTENCES = 1  # sentences to overlap between chunks


class SentenceNLTKChunker:
    condition_key = "sent_nltk"

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks: list[dict] = []
        buffer: list[str] = []
        word_count = 0
        idx = 0

        for sent in sentences:
            sent_words = len(sent.split())
            buffer.append(sent)
            word_count += sent_words

            if word_count >= TARGET_WORDS:
                chunk_text = " ".join(buffer)
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}__{self.condition_key}_{idx}",
                        "doc_id": doc_id,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "strategy": self.condition_key,
                        "chunk_index": idx,
                    }
                )
                idx += 1
                # carry forward last N sentences as overlap
                carry = buffer[-CARRY_FORWARD_SENTENCES:]
                buffer = carry
                word_count = sum(len(s.split()) for s in carry)

        # flush remaining sentences
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
                }
            )
        return chunks
