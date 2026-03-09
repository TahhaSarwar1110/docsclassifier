from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def build_retriever(docs, results):
    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    documents = []

    for name, text in docs.items():
        if not text:
            continue

        doc_class = results[name]["class"]

        chunks = splitter.split_text(text)

        for chunk in chunks:
            canonical = f"""
            document_class: {doc_class}
            filename: {name}
            content: {chunk}
            """

            documents.append(
                Document(
                    page_content=canonical,
                    metadata={"name": name, "class": doc_class}
                )
            )


    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents,
        embedding,
        persist_directory="./chroma_db"
    )

    return vector_store.as_retriever(search_kwargs={"k": 5})
