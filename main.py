import json
import pickle

from ingestion import ingest_documents
from classifier import classify
from extractor import run_extraction
from retrieval import build_retriever

docs = ingest_documents()

print("Loaded docs:", docs.keys())
print("Doc lengths:", {k: len(v) if v else 0 for k, v in docs.items()})


results = {}

for name, text in docs.items():
    doc_class, conf = classify(text)

    data = {
        "class": doc_class,
        "confidence": round(conf, 3)
    }

    extracted = run_extraction(doc_class, text or "")
    data.update(extracted)

    results[name] = data

with open("output.json", "w") as f:
    json.dump(results, f, indent=2)

print("output.json generated")

retriever = build_retriever(docs, results)


print("retriever saved")
