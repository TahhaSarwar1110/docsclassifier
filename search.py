import json
import os
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from query_encoder import encode_query

os.environ["HF_HUB_OFFLINE"] = "1"

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = vector_store.as_retriever(search_kwargs={"k": 8})

with open("output.json") as f:
    structured = json.load(f)

while True:
    raw_query = input("\nSearch: ")

    encoded_query = encode_query(raw_query)

    docs = retriever.invoke(encoded_query)

    pairs = [(encoded_query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    best_doc = ranked[0][1]
    name = best_doc.metadata["name"]

    print(f"\nBest match: {name}\n")

    if name in structured:
        print(json.dumps(structured[name], indent=2))
    else:
        print("No structured data available.")
