"""
Central config — this is where the SWAPPABLE part lives.

Change one env var (PROVIDER) to switch the whole system between:
  - "ollama"  -> local qwen2:7b + nomic-embed-text   (your original setup)
  - "gemini"  -> Google Gemini API + Google embeddings (free tier, no GPU)

This file is the whole point of the "abstraction" story: nothing downstream
(the chain, the retriever) changes when you swap providers — only this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

# Postgres / pgvector connection string.
CONNECTION = os.getenv(
    "PG_CONNECTION",
    "postgresql+psycopg://rag:rag@localhost:5432/rag",
)

# IMPORTANT: the collection name includes the provider.
# Different embedding models produce different (incompatible) vector spaces,
# so Ollama vectors and Gemini vectors must NOT share a collection. Namespacing
# by provider keeps them separate — re-run ingest.py after switching PROVIDER.
COLLECTION = f"rag_docs_{PROVIDER}"


def get_models():
    """Return (chat_model, embedding_model) for the selected provider."""
    if PROVIDER == "gemini":
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        # Model names change over time. If either 404s, run list_models.py to see
        # what your API key currently supports, and update these strings.
        chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:  # default: ollama (your original local setup)
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        chat = ChatOllama(model="qwen2:7b")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return chat, embeddings
