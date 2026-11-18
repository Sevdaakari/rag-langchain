# rag-langchain

The same RAG agent as my [rag-from-scratch](https://github.com/Sevdaakari/rag-from-scratch) project, rebuilt on LangChain. I built the first one without a framework to learn the internals; this one shows how much LangChain takes off your hands — the hand-written cosine search and prompt plumbing collapse into a few lines.

Ask questions in the terminal about a folder of documents (I used US mining-safety publications). It retrieves the most relevant chunks and answers only from those.

It comes in two versions:

- **`rag.py` — a chain.** Fixed flow: it always retrieves, then answers. Simple and predictable.
- **`agent.py` — an agent (LangGraph).** Retrieval is a tool the model decides when to call. It can skip retrieval for small talk, or search when a question needs the docs.

Two other changes from the first version:

- Vectors are stored in **pgvector** (Postgres), so they persist and aren't re-embedded on every run.
- The model is **swappable** — local Ollama by default, or the Gemini API by changing one setting. Neither version's logic changes when you swap.

## Files

- `config.py` — picks the model/embeddings from `PROVIDER`. Switch providers here.
- `ingest.py` — load, split, embed, store. Run once.
- `rag.py` — the chain version.
- `agent.py` — the agent version (LangGraph).
- `docker-compose.yml` — Postgres + pgvector.

## LangChain vs LangGraph

LangChain gives the building blocks and simple pipelines (`prompt | model | parser`) — good for a fixed flow. LangGraph adds the loop an agent needs, where the model decides whether to call a tool, maybe more than once, before answering. So `agent.py` uses LangGraph on top of LangChain: LangChain provides the model and the tool, LangGraph runs the reason-act loop.

## Run

```bash
docker compose up -d
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set PROVIDER (+ Gemini key if used)
python ingest.py            # index once
python rag.py               # the chain
python agent.py             # the agent
```

For Ollama: `ollama pull qwen2:7b` and `ollama pull nomic-embed-text`. For Gemini: a free key from Google AI Studio.

## Notes

Swapping the chat model is free. Swapping the embedding model isn't — different models produce incompatible vector spaces, so you must re-run `ingest.py`. The two sets of vectors live in separate collections so they don't mix.

The agent needs a model that supports tool calling (`qwen2:7b` and Gemini both do).
