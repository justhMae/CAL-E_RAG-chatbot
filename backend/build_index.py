import json
from pathlib import Path

import numpy as np

from backend.config import (
    DATA_DIR, DOCUMENTS_PATH, VECTOR_STORE_PATH, CHUNKS_META_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
)
from backend.preprocess import preprocess_document
from backend.embeddings import embed_texts, BUILD_BATCH_SIZE

CHECKPOINT_PATH = DATA_DIR / ".build_index_checkpoint.npz"


def load_chunks() -> list[dict]:
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    seen = set()
    for doc in documents:
        for chunk in preprocess_document(doc):
            if chunk["chunk_id"] in seen:
                raise ValueError(f"duplicate chunk_id {chunk['chunk_id']!r}; fix ids in documents.json")
            seen.add(chunk["chunk_id"])
            all_chunks.append(chunk)

    print(f"Loaded {len(documents)} documents -> {len(all_chunks)} chunks.")
    return all_chunks


def _load_checkpoint(chunk_ids: list[str]) -> dict[str, np.ndarray]:
    if not CHECKPOINT_PATH.is_file():
        return {}
    try:
        with np.load(CHECKPOINT_PATH, allow_pickle=False) as npz:
            saved_ids = list(npz["chunk_ids"])
            saved_vecs = npz["embeddings"]
        if saved_vecs.ndim != 2 or saved_vecs.shape[1] != EMBEDDING_DIM:
            return {}
        valid = set(chunk_ids)
        return {
            str(cid): vec
            for cid, vec in zip(saved_ids, saved_vecs)
            if str(cid) in valid
        }
    except Exception:
        return {}


def _save_checkpoint(cached: dict[str, np.ndarray], ordered_ids: list[str]) -> None:
    done_ids = [cid for cid in ordered_ids if cid in cached]
    if not done_ids:
        return
    vecs = np.vstack([cached[cid] for cid in done_ids]).astype(np.float32)
    np.savez(CHECKPOINT_PATH, chunk_ids=np.array(done_ids), embeddings=vecs)


def main():
    chunks = load_chunks()
    ordered_ids = [c["chunk_id"] for c in chunks]
    cached = _load_checkpoint(ordered_ids)
    if cached:
        print(f"Resuming from checkpoint: {len(cached)}/{len(chunks)} chunks already embedded.")

    pending = [c for c in chunks if c["chunk_id"] not in cached]
    if pending:
        print(f"Embedding {len(pending)} remaining chunks with {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM} dims)...")
        for start in range(0, len(pending), BUILD_BATCH_SIZE):
            batch = pending[start:start + BUILD_BATCH_SIZE]
            vecs = embed_texts([c["text"] for c in batch], batch_size=len(batch)).astype(np.float32)
            for chunk, vec in zip(batch, vecs):
                cached[chunk["chunk_id"]] = vec
            _save_checkpoint(cached, ordered_ids)
            print(f"[build_index] checkpointed {len(cached)}/{len(chunks)} chunks")

    vectors = np.vstack([cached[cid] for cid in ordered_ids]).astype(np.float32)
    assert vectors.shape == (len(chunks), EMBEDDING_DIM), vectors.shape

    meta = {
        "model": EMBEDDING_MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "count": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "category": c["category"],
                "title": c["title"],
                "office": c.get("office") or None,
                "text": c["text"],
            }
            for c in chunks
        ],
    }

    np.savez_compressed(VECTOR_STORE_PATH, embeddings=vectors)
    with open(CHUNKS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if CHECKPOINT_PATH.is_file():
        CHECKPOINT_PATH.unlink()

    print(f"Wrote {VECTOR_STORE_PATH} and {CHUNKS_META_PATH}.")
    print("Done. Commit both files so the deployment can load them read-only.")


if __name__ == "__main__":
    main()
