import chromadb
from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    MIN_SIMILARITY,
)

_model = None


_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    # embed a single piece of text (used for the user's question)
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    # embed many texts at once (used when building the index)
    model = get_embedding_model()
    vectors = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    return vectors.tolist()


def get_chroma_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.Client()  # in-memory
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chunks(chunks: list[dict]) -> None:
    # embed a list of preprocessed chunks (in one batch) and store them in ChromaDB
    collection = get_chroma_collection()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME} ...")
    vectors = embed_texts(texts)

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"doc_id": c["doc_id"], "category": c["category"], "title": c["title"],
         # ChromaDB rejects None metadata values, so store "" for docs with no office
         "office": c.get("office") or ""}
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )


def retrieve(question: str, top_k: int = 3, category_filter: str | None = None) -> list[dict]:
    # embed the user's question, run a similarity search against ChromaDB, and drop any hit below MIN_SIMILARITY
    
    collection = get_chroma_collection()
    query_vector = embed_text(question)

    where = {"category": category_filter} if category_filter else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )

    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1 - dist
        if similarity < MIN_SIMILARITY:
            continue
        hits.append({
            "chunk_id": chunk_id,
            "text": doc,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "office": meta.get("office") or None,
            "doc_id": meta.get("doc_id"),
            "similarity": similarity,
        })
    return hits


def get_chunks_by_doc_id(doc_id: str) -> list[dict]:
    # fetch every chunk belonging to one document directly by doc_id, bypassing similarity search
    # used only when the caller already knows exactly which document is needed

    collection = get_chroma_collection()
    result = collection.get(where={"doc_id": doc_id})
    hits = []
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    for chunk_id, doc, meta in zip(ids, docs, metas):
        hits.append({
            "chunk_id": chunk_id,
            "text": doc,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "office": meta.get("office") or None,
            "doc_id": meta.get("doc_id"),
            "similarity": None,  # not from a similarity search; fetched directly
        })
    return hits


def get_chunk_by_id(chunk_id: str) -> dict | None:
    # fetch one specific chunk directly by its ID, bypassing similarity search
    # used to deterministically pull in a document's short summary (total fee or total processing time) 
    
    collection = get_chroma_collection()
    result = collection.get(ids=[chunk_id])
    docs = result.get("documents", [])
    if not docs:
        return None
    meta = result.get("metadatas", [{}])[0]
    return {
        "chunk_id": chunk_id,
        "text": docs[0],
        "title": meta.get("title"),
        "category": meta.get("category"),
        "office": meta.get("office") or None,
        "doc_id": meta.get("doc_id"),
        "similarity": None,
    }


def retrieve_all_in_category(category: str) -> list[dict]:
    # returns every chunk tagged with this category, no similarity ranking, no top_k cap
    # used for enumeration queries like for example: "list all barangays"
    
    collection = get_chroma_collection()
    results = collection.get(where={"category": category})

    hits = []
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    for doc, meta in zip(docs, metas):
        hits.append({
            "text": doc,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "office": meta.get("office") or None,
            "doc_id": meta.get("doc_id"),
            "similarity": None,
        })
    return hits