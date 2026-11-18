"""
RAG AGENT (LangGraph).  Run:  python agent.py

Same models + same pgvector store as rag.py — but instead of a fixed
retrieve-then-answer chain, the MODEL decides when to search.

Retrieval is exposed as a TOOL. On each question the model either:
  - calls search_docs (then reads the result and answers), or
  - answers directly if no search is needed (e.g. "hello").

LangGraph's create_react_agent provides the reason -> act -> observe loop.
"""

from langchain_core.tools import tool
from langchain_postgres import PGVector
from langchain.agents import create_agent

from config import get_models, CONNECTION, COLLECTION

model, embeddings = get_models()

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION,
    connection=CONNECTION,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def search_docs(query: str) -> str:
    """Search the mining-safety knowledge base and return relevant passages."""
    docs = retriever.invoke(query)
    result = "\n\n".join(d.page_content for d in docs)
    print(f"\n--- RETRIEVED ---\n{result}\n--- END ---\n")   # debug line
    return result


# The agent: a model + its tools + the built-in reason/act loop.
agent = create_agent(
    model,
    tools=[search_docs],
    system_prompt="You answer questions about mining safety. Use the search_docs tool "
           "to look things up, and answer based on what it returns. If the tool "
           "has no answer, say you don't know.",
)


if __name__ == "__main__":
    print("RAG agent (LangGraph). Ask a question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.invoke({"messages": [("human", question)]})
        # The final message in the list is the agent's answer.
        print(f"\nAgent: {result['messages'][-1].content}\n")
