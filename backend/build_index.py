import json
from backend.config import DOCUMENTS_PATH
from backend.preprocess import preprocess_document
from backend.embeddings import index_chunks


def main():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(preprocess_document(doc))

    print(f"Loaded {len(documents)} documents -> {len(all_chunks)} chunks.")
    print("Embedding and indexing into ChromaDB...")
    index_chunks(all_chunks)
    print("Done. Index is ready for queries.")


if __name__ == "__main__":
    main()
