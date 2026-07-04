"""
ingest.py
Loads all PDFs and text files from the documents/ folder, splits them into
chunks, embeds them, and stores them in a persistent Chroma DB.

Run this once whenever you add or change files in documents/.
"""

import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "my_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents():
    """Load all supported files from the documents folder."""
    docs = []
    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"Folder '{DOCUMENTS_DIR}' not found. Creating it now.")
        os.makedirs(DOCUMENTS_DIR)
        return docs

    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = filename
            docs.extend(loaded)
            print(f"Loaded PDF: {filename} ({len(loaded)} pages)")
        elif filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
            loader = TextLoader(filepath, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = filename
            docs.extend(loaded)
            print(f"Loaded text file: {filename}")
        else:
            print(f"Skipping unsupported file: {filename}")

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def store_chunks(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Clear any old data so re-running ingest.py doesn't duplicate chunks
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Cleared {len(existing_ids)} old chunks from the collection.")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Batch upserts to avoid overly large single requests
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"Stored chunks {i} to {min(i + batch_size, len(documents))}")

    print(f"\nDone. {len(documents)} chunks stored in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    documents = load_documents()
    if not documents:
        print("No documents found. Add PDFs or .txt/.md files to the 'documents' folder and re-run.")
    else:
        chunks = chunk_documents(documents)
        store_chunks(chunks)
