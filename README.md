# rag-langchain

The same local RAG agent as [rag-from-scratch](https://github.com/Sevdaakari/rag-from-scratch),
rebuilt on **LangChain** with a **pgvector** (Postgres) store and a **swappable LLM**
(local Ollama ↔ Gemini API) — to show what the framework abstracts away.

## What changed vs from-scratch

| from-scratch                     | this repo                                      |
| -------------------------------- | ---------------------------------------------- |
| manual `docs/*.txt` read          | `DirectoryLoader` + `TextLoader`               |
| `chunking.py` (tiktoken 200/40)   | `TokenTextSplitter(200, 40)`                   |
| `embedding.py`                    | `OllamaEmbeddings` / `GoogleGenerativeAIEmbeddings` |
| `vector_store.py` (numpy cosine)  | **pgvector** via `PGVector`                    |
| `persistence.py` (disk cache)     | pgvector persists in Postgres (no re-embed)    |
| `search(..., top_k=3)`            | `vectorstore.as_retriever(k=3)`                |
| `generate.py` prompt + qwen2      | `ChatPromptTemplate \| chat_model \| StrOutputParser` |

## Files

- `config.py` — **the swappable part.** Picks model + embeddings from `PROVIDER`.
- `ingest.py` — Phase 1: load → split → embed → store in pgvector. Run once.
- `rag.py` — Phase 2: retrieve + the RAG chain + chat loop.
- `docker-compose.yml` — Postgres with pgvector.

## Setup

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Configure
cp .env.example .env         # set PROVIDER=ollama or gemini (+ GOOGLE_API_KEY if gemini)

# 4. Index the docs (once), then chat
python ingest.py
python rag.py
```

For `PROVIDER=ollama`, pull the models first: `ollama pull qwen2:7b` and
`ollama pull nomic-embed-text`. For `PROVIDER=gemini`, get a free key at
https://aistudio.google.com/apikey.

## The one thing to understand (and say in an interview)

Swapping the **chat model** (Ollama ↔ Gemini) is free — the chain, the retriever,
the prompt all stay identical, because LangChain hides each model behind the same
interface. That's abstraction in action.

But swapping the **embedding model** is NOT free: different embedding models produce
**different, incompatible vector spaces**, so you must **re-index** (`python ingest.py`)
after switching providers. That's why the collection name is namespaced per provider
in `config.py` (`rag_docs_ollama` vs `rag_docs_gemini`) — the two vector sets never mix.

## Scope

A straight RAG **chain** (fixed retrieve → generate), not an agent loop. The
from-scratch version's query-rewrite / context-sufficiency step is left out here;
turning retrieval into a tool the model calls would be the LangGraph (agent) version.
