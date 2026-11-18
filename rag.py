"""
PHASE 2 — RETRIEVAL + GENERATION.  Run the chat:  python rag.py
(Make sure you ran `python ingest.py` first so pgvector has the chunks.)

Connects to the existing pgvector store, retrieves top-3 chunks, and runs the
RAG chain:  {context, question} | prompt | model | parser.
"""

from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_models, CONNECTION, COLLECTION

# Model + embeddings for the selected provider (see config.py).
model, embeddings = get_models()

# Connect to the pgvector collection that ingest.py already filled.
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION,
    connection=CONNECTION,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Same "answer only from context" prompt as generate.py.
prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the context doesn't contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

parser = StrOutputParser()


def format_docs(docs):
    return "\n".join(f"- {d.page_content}" for d in docs)


# THE CHAIN — identical shape regardless of provider or vector store.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)


if __name__ == "__main__":
    print("RAG (LangChain + pgvector) ready. Ask a question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(f"\nBot: {rag_chain.invoke(question)}\n")
