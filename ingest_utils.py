"""
ingest_utils.py
Shared document loading/chunking/storage logic used by both ingest.py
(batch folder ingestion) and app.py (live in-app uploads).
"""

import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 100


def load_single_file(filepath, filename):
    """Load a single PDF, TXT, or MD file into LangChain Document objects."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs


def chunk_documents(docs):
    """Split loaded documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def add_chunks_to_collection(collection, chunks):
    """Upsert chunks into an existing Chroma collection without clearing it.
    Uses random ids so this never collides with or overwrites existing data.
    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    ids = [f"upload_{uuid.uuid4().hex}" for _ in chunks]
    documents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    for i in range(0, len(documents), BATCH_SIZE):
        collection.upsert(
            ids=ids[i:i + BATCH_SIZE],
            documents=documents[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )

    return len(documents)


def ingest_file(collection, filepath, filename):
    """Convenience wrapper: load, chunk, and store a single file. Returns chunk count."""
    docs = load_single_file(filepath, filename)
    chunks = chunk_documents(docs)
    return add_chunks_to_collection(collection, chunks)