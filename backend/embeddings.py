# all-MiniLM-L6-v2 (sentence-transformers) -> vectors -> ChromaDB.
import chromadb
from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    MIN_SIMILARITY,
)

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text (used for the user's question)."""
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts at once (used when building the index)."""
    model = get_embedding_model()
    vectors = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    return vectors.tolist()


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(chunks: list[dict]) -> None:
    """Embed a list of preprocessed chunks (in one batch) and store them in ChromaDB."""
    collection = get_chroma_collection()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME} ...")
    vectors = embed_texts(texts)

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"doc_id": c["doc_id"], "category": c["category"], "title": c["title"]}
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )


def retrieve(question: str, top_k: int = 3, category_filter: str | None = None) -> list[dict]:
    """
    Embed the user's question, run a similarity search against ChromaDB,
    and drop any hit below MIN_SIMILARITY -- ChromaDB always returns its
    top_k nearest neighbors even when none of them are actually related
    to the query, so without this a greeting or off-topic message would
    still get handed 3 "closest by chance" chunks as if they were relevant.
    """
    collection = get_chroma_collection()
    query_vector = embed_text(question)

    where = {"category": category_filter} if category_filter else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist
        if similarity < MIN_SIMILARITY:
            continue
        hits.append({
            "text": doc,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "doc_id": meta.get("doc_id"),
            "similarity": similarity,
        })
    return hits


def retrieve_all_in_category(category: str) -> list[dict]:
    """
    Returns every chunk tagged with this category, no similarity ranking,
    no top_k cap. Used for enumeration queries ("list all X") where a
    vector similarity search would silently drop chunks that are
    correct but not among the top-k nearest to the query embedding.
    """
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
            "doc_id": meta.get("doc_id"),
            "similarity": None,
        })
    return hits