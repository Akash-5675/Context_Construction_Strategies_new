"""
Stage 5: answer generation.

Sends the question and retrieved chunks to Llama 3.1 8B through a local Ollama
server. Set OLLAMA_URL to use a server on another port, such as a second GPU.
"""

from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
# Large enough for k=10 at 512 words; a smaller window would silently truncate.
OLLAMA_NUM_CTX = 16384
OLLAMA_SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 300
RETRY_BACKOFF = [2, 5, 10, 20]

# Stop instead of spending the session on calls that keep failing.
MAX_CONSECUTIVE_FAILURES = 10
_consecutive_failures = 0

SYSTEM_PROMPT = (
    "You are a precise biomedical assistant. Answer the question using ONLY "
    "the provided context. If the context does not contain enough information "
    "to answer, say 'Insufficient context.' Do not use external knowledge."
)


def generate(query: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(
        f"[Source {i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    context_word_count = len(context.split())
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "num_ctx": OLLAMA_NUM_CTX,
            "seed": OLLAMA_SEED,
        },
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    global _consecutive_failures

    answer = "GENERATION_FAILED"
    latency_ms = 0.0

    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                result = json.loads(resp.read().decode("utf-8"))
                answer = result["message"]["content"].strip()
            break
        except urllib.error.HTTPError as e:
            # 429 and 5xx responses are temporary, so they are worth retrying.
            if e.code == 429 or e.code >= 500:
                if attempt < 4:
                    time.sleep(RETRY_BACKOFF[attempt])
            else:
                raise
        except Exception:
            if attempt < 4:
                time.sleep(RETRY_BACKOFF[attempt])
            else:
                raise

    if not answer or answer == "GENERATION_FAILED":
        _consecutive_failures += 1
        if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise SystemExit(
                f"\nAborting: {_consecutive_failures} consecutive generation failures. "
                "Completed rows are saved, so rerun the same command to resume."
            )
        raise RuntimeError("Generation failed after 5 attempts: row skipped, will retry next run")

    _consecutive_failures = 0

    return {
        "answer": answer,
        "context_word_count": context_word_count,
        "latency_ms": latency_ms,
        # Token counts were not recorded in this study.
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
