from pathlib import Path
import fitz  # PyMuPDF

def ingest_documents(folder="documents"):
    docs = {}

    paths = list(Path(folder).glob("*.pdf"))
    print("Found PDFs:", paths)

    for path in paths:
        try:
            text = ""
            doc = fitz.open(path)

            for page in doc:
                text += page.get_text()

            text = text.strip()

            print(f"Loaded {path.name} → length:", len(text))

            docs[path.name] = text if text else None

        except Exception as e:
            print(f"Failed {path.name}:", e)
            docs[path.name] = None

    return docs
