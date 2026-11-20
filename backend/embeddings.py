# shared embedding helper: Google Gemini Embedding API (gemini-embedding-001) via the official google-genai SDK
# used in batch at build time (backend/build_index.py) and for a single query at runtime (backend/rag_pipeline.py)
# no local model weights, no torch, no sentence-transformers

import time

import numpy as np
from google import genai
from google.genai import errors, types

from backend.config import GEMINI_API_KEYS, EMBEDDING_MODEL_NAME, EMBEDDING_DIM

# gemini-embedding-001 returns unit-length vectors only at its full 3072 dims;
# truncated (Matryoshka) outputs such as 768 must be re-normalized manually
FULL_DIM = 3072

BUILD_BATCH_SIZE = 50  # texts per embed_content request when building the index
MAX_RETRIES = 6

_clients: dict[str, genai.Client] = {}
_active_key_idx = 0


def _ensure_keys() -> list[str]:
    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "GEMINI_API_KEY / GEMINI_API_KEYS is not set. Add it to backend/.env locally, "
            "or to the Vercel project's Environment Variables."
        )
    return GEMINI_API_KEYS


def iter_gemini_clients() -> list[tuple[int, genai.Client]]:
    """Return (key_index, client) pairs starting from the currently active key."""
    keys = _ensure_keys()
    n = len(keys)
    ordered = []
    for offset in range(n):
        idx = (_active_key_idx + offset) % n
        key = keys[idx]
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key)
        ordered.append((idx, _clients[key]))
    return ordered


def advance_gemini_key(exhausted_idx: int) -> None:
    """Move the active key pointer past a rate-limited key."""
    global _active_key_idx
    n = len(GEMINI_API_KEYS)
    if n > 1 and _active_key_idx == exhausted_idx:
        _active_key_idx = (exhausted_idx + 1) % n


def get_client() -> genai.Client:
    return iter_gemini_clients()[0][1]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    # row-wise vec / np.linalg.norm(vec); guards against an all-zero row
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _is_daily_quota_error(err: errors.APIError) -> bool:
    return "PerDay" in str(err)


def _embed_request(texts: list[str], task_type: str) -> np.ndarray:
    delay = 10.0
    for attempt in range(MAX_RETRIES):
        clients = iter_gemini_clients()
        last_err = None
        all_daily_exhausted = True

        for key_idx, client in clients:
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM,
                        task_type=task_type,
                    ),
                )
                vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
                if EMBEDDING_DIM != FULL_DIM:
                    vectors = l2_normalize(vectors)
                return vectors
            except errors.APIError as e:
                # 429 = free-tier rate limit, 5xx = transient; anything else is fatal
                retryable = e.code == 429 or (e.code is not None and e.code >= 500)
                if not retryable:
                    raise
                last_err = e
                if e.code == 429:
                    advance_gemini_key(key_idx)
                    if not _is_daily_quota_error(e):
                        all_daily_exhausted = False
                    if len(clients) > 1:
                        print(f"[embeddings] Gemini key #{key_idx + 1} hit 429; switching to next key...")
                else:
                    all_daily_exhausted = False

        if all_daily_exhausted:
            raise RuntimeError(
                f"All {len(clients)} Gemini API key(s) exceeded their daily embed_content quota. "
                "Add another key to GEMINI_API_KEYS or wait for the daily quota reset."
            ) from last_err
        if attempt == MAX_RETRIES - 1:
            raise last_err
        print(f"[embeddings] All keys hit transient rate limit; retrying in {delay:.0f}s "
              f"(attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(delay)
        delay = min(delay * 2, 90.0)
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str], batch_size: int = BUILD_BATCH_SIZE) -> np.ndarray:
    # batched embedding for build time; returns an (n, EMBEDDING_DIM) float32 L2-normalized matrix
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        out.append(_embed_request(batch, task_type="RETRIEVAL_DOCUMENT"))
        print(f"[embeddings] embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    if not out:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.vstack(out)


def embed_query(text: str) -> np.ndarray:
    # single-query embedding at runtime; returns an (EMBEDDING_DIM,) float32 L2-normalized vector
    return _embed_request([text], task_type="RETRIEVAL_QUERY")[0]
