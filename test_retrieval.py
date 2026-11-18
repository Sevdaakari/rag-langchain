from langchain_postgres import PGVector
from config import get_models, CONNECTION, COLLECTION

_, embeddings = get_models()
store = PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION)

# search with a strong, specific query
results = store.similarity_search("22% lost time claims mobile plant and transport", k=5)
for i, d in enumerate(results):
    print(f"\n--- result {i} ---\n{d.page_content[:400]}")