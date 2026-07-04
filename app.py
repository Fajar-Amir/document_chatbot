"""
app.py
Streamlit web UI for the RAG chatbot. Reuses the same Chroma DB and
retrieval logic as chat.py, wrapped in a browser-based chat interface.

Run with: streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "my_docs"
TOP_K = 4
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful, conversational assistant with access to the user's
own documents as additional context.

- For greetings, small talk, casual chat, or general conversation, respond exactly
  like a normal, friendly assistant would. NEVER mention "the context," "the
  documents," or that you couldn't find something for these — just chat normally.
- For questions about YOURSELF (your name, what you are, your general capabilities,
  what you can help with) — answer naturally as an AI assistant. Do NOT search the
  documents or describe document content to answer these, even if a retrieved chunk
  seems loosely related. Your identity and capabilities are not defined by the
  user's documents.
- For factual or informational questions about a specific topic: if the provided
  context is relevant, prioritize it and answer using it, mentioning which source
  file the information came from.
- If the context isn't relevant to a factual question, or doesn't fully answer it,
  answer using your own general knowledge instead, and briefly note that the answer
  comes from general knowledge rather than the user's documents.
- You can blend both when useful: use the documents for what they cover, and general
  knowledge to fill gaps, being clear about which is which.

Examples of how to handle casual and meta messages (ignore any retrieved context
entirely for these types of messages):
User: "hi" -> "Hey! How can I help you today?"
User: "how are you" -> "I'm doing well, thanks for asking! What can I help you with?"
User: "what is your name" -> "I'm your AI assistant — happy to help with anything,
  including questions about the documents you've added. What do you need?"
User: "how can you help me" -> "I can chat, answer general questions, and also look
  through any documents you've uploaded if you want specific info from them. What
  are you working on?"
User: "i want some info" -> "Sure, happy to help! What would you like to know more
  about?"

Always be concise, natural, and accurate."""

st.set_page_config(page_title="My Document Chatbot", page_icon="💬", layout="wide")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


@st.cache_resource
def get_groq_client():
    # Check Streamlit Secrets first (for cloud hosting), fallback to .env (for local testing)
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def retrieve_context(collection, query, top_k=TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return documents, metadatas, distances


def build_prompt(query, documents, metadatas):
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        context_blocks.append(f"[Source: {source}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context from documents:
{context}

Question: {query}

Answer the question using only the context above."""


MAX_HISTORY_TURNS = 6  # number of past user/assistant exchanges to remember


def ask_llm(groq_client, query, documents, metadatas, history):
    prompt = build_prompt(query, documents, metadatas)

    # Build plain role/content history (drop sources, keep only role+content)
    recent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-(MAX_HISTORY_TURNS * 2):]
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": prompt})

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 Document Chatbot")
    st.markdown(
        "Ask questions about the documents you've ingested with `ingest.py`."
    )
    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=TOP_K)
    show_sources = st.checkbox("Show retrieved source chunks", value=True)

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "To add new documents: drop files into the `documents/` folder, "
        "run `python ingest.py`, then restart this app."
    )


# ---------- Main chat interface ----------
st.title("💬 Chat With Your Documents")

collection = get_collection()
groq_client = get_groq_client()

if collection is None:
    st.error(
        "No document collection found. Run `python ingest.py` first to load "
        "your documents, then restart this app."
    )
    st.stop()

if groq_client is None:
    st.error(
        "GROQ_API_KEY not found. Add it to your `.env` file. "
        "Get a free key at https://console.groq.com"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📄 Sources used"):
                for i, (doc, meta, dist) in enumerate(message["sources"]):
                    source = meta.get("source", "unknown")
                    st.markdown(f"**{i + 1}. {source}** (relevance: {1 - dist:.2f})")
                    st.text(doc[:300] + ("..." if len(doc) > 300 else ""))

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            documents, metadatas, distances = retrieve_context(collection, query, top_k=top_k)
            # Exclude the current message (already appended above) from history
            past_history = st.session_state.messages[:-1]
            answer = ask_llm(groq_client, query, documents, metadatas, past_history)
            sources = list(zip(documents, metadatas, distances)) if documents else []

            st.markdown(answer)
            if sources and show_sources:
                with st.expander("📄 Sources used"):
                    for i, (doc, meta, dist) in enumerate(sources):
                        source = meta.get("source", "unknown")
                        st.markdown(f"**{i + 1}. {source}** (relevance: {1 - dist:.2f})")
                        st.text(doc[:300] + ("..." if len(doc) > 300 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )