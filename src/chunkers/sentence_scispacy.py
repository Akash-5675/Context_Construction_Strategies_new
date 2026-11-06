"""
Sentence-boundary chunker using the scispaCy biomedical tokeniser, part of stage 3.

Identical accumulation logic to SentenceNLTKChunker but uses scispaCy's
en_core_sci_sm model, which correctly handles biomedical abbreviations like
i.v., p.o., Dr., Fig., and gene/protein notation that NLTK punkt splits wrongly.

Condition key: sent_scispacy

Install dependency:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
"""

from __future__ import annotations

TARGET_WORDS = 256
CARRY_FORWARD_SENTENCES = 1


def _load_nlp():
    try:
        import spacy
        return spacy.load("en_core_sci_sm", disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        raise RuntimeError(
            "scispaCy model not found. Install with:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
        )


_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_nlp()
    return _nlp


def split_sentences(text: str) -> list[str]:
    """Return the individual scispaCy sentences of text."""
    doc = get_nlp()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


class SentenceScispaCyChunker:
    condition_key = "sent_scispacy"

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sentences = split_sentences(text)
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
                carry = buffer[-CARRY_FORWARD_SENTENCES:]
                buffer = carry
                word_count = sum(len(s.split()) for s in carry)

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
