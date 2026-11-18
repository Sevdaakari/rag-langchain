"""
PHASE 1 — INDEXING.  Run this ONCE (or whenever docs change):  python ingest.py

Load docs -> split into chunks -> embed -> store in pgvector (Postgres).
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_postgres import PGVector

from config import get_models, CONNECTION, COLLECTION, PROVIDER


def main():
    # 1. LOAD
    documents = DirectoryLoader(
        "docs", glob="**/*.txt", loader_cls=TextLoader
    ).load()

    # 2. SPLIT
    chunks = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=200,
        chunk_overlap=40,
    ).split_documents(documents)

    # 3. EMBED + STORE in small batches (so the local Ollama runner doesn't crash)
    _, embeddings = get_models()
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION,
    )

    batch_size = 16
    total = len(chunks)
    for i in range(0, total, batch_size):
        store.add_documents(chunks[i:i + batch_size])
        print(f"  indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"Done. Indexed {total} chunks into '{COLLECTION}' "
          f"using {PROVIDER} embeddings.")


if __name__ == "__main__":
    main()