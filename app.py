"""
app.py
Streamlit web UI for the Agentic AI Info RAG chatbot. Supports:
- Retrieval-augmented answers from your own documents
- General knowledge fallback + natural casual conversation
- In-conversation memory (recent turns fed back to the model)
- Persistent chat history saved to disk, with a sidebar to browse and
  switch between past conversations

Run with: streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from groq import Groq

import memory
import ingest_utils

load_dotenv()

DOCUMENTS_DIR = "documents"

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "my_docs"
TOP_K = 4
MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 6  # number of past user/assistant exchanges to remember

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
User: "what is your name" -> "I'm Agentic AI Info RAG — happy to help with anything,
  including questions about the documents you've added. What do you need?"
User: "how can you help me" -> "I can chat, answer general questions, and also look
  through any documents you've uploaded if you want specific info from them. What
  are you working on?"
User: "i want some info" -> "Sure, happy to help! What would you like to know more
  about?"

Always be concise, natural, and accurate."""

st.set_page_config(page_title="Agentic AI Info RAG", page_icon="🤖", layout="wide")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
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

Answer the question using only the context above where relevant."""


def ask_llm(groq_client, query, documents, metadatas, history):
    prompt = build_prompt(query, documents, metadatas)

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


# ---------- Load persistent sessions on first run ----------
if "sessions" not in st.session_state:
    st.session_state.sessions = memory.load_all_sessions()

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        latest_id = max(
            st.session_state.sessions,
            key=lambda sid: st.session_state.sessions[sid].get("created", ""),
        )
        st.session_state.current_session_id = latest_id
    else:
        new_id, new_session = memory.create_new_session()
        st.session_state.sessions[new_id] = new_session
        st.session_state.current_session_id = new_id
        memory.save_all_sessions(st.session_state.sessions)


def current_messages():
    return st.session_state.sessions[st.session_state.current_session_id]["messages"]


def persist():
    memory.save_all_sessions(st.session_state.sessions)


# ---------- Initialize collection and LLM client early (needed by sidebar upload) ----------
collection = get_collection()
groq_client = get_groq_client()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 Agentic AI Info RAG")
    st.caption("Chat with your documents, with memory.")
    st.divider()

    if st.button("➕ New chat", use_container_width=True):
        new_id, new_session = memory.create_new_session()
        st.session_state.sessions[new_id] = new_session
        st.session_state.current_session_id = new_id
        persist()
        st.rerun()

    st.divider()
    st.subheader("Chat history")

    sorted_sessions = sorted(
        st.session_state.sessions.items(),
        key=lambda item: item[1].get("created", ""),
        reverse=True,
    )

    for session_id, session in sorted_sessions:
        is_current = session_id == st.session_state.current_session_id
        label = ("🟢 " if is_current else "") + session.get("title", "New chat")
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"select_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{session_id}"):
                del st.session_state.sessions[session_id]
                if not st.session_state.sessions:
                    new_id, new_session = memory.create_new_session()
                    st.session_state.sessions[new_id] = new_session
                    st.session_state.current_session_id = new_id
                elif session_id == st.session_state.current_session_id:
                    st.session_state.current_session_id = next(iter(st.session_state.sessions))
                persist()
                st.rerun()

    st.divider()
    st.subheader("📤 Upload a document")
    uploaded_files = st.file_uploader(
        "Add a PDF, TXT, or MD file to chat with",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Ingest uploaded file(s)", use_container_width=True):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        total_chunks = 0
        with st.spinner("Reading and embedding your document(s)..."):
            for uploaded_file in uploaded_files:
                save_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    count = ingest_utils.ingest_file(collection, save_path, uploaded_file.name)
                    total_chunks += count
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
        st.success(f"Added {total_chunks} chunks from {len(uploaded_files)} file(s). Ask away!")
        st.rerun()

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=TOP_K)
    show_sources = st.checkbox("Show retrieved source chunks", value=True)

    st.divider()
    st.caption(
        "To add new documents: drop files into the `documents/` folder, "
        "run `python ingest.py`, then restart this app."
    )


# ---------- Main chat interface ----------
st.title("💬 Agentic AI Info RAG")

if collection is None:
    st.error("Could not connect to the document database. Check your setup and restart.")
    st.stop()

if collection.count() == 0:
    st.info(
        "No documents loaded yet. Upload a file using the sidebar, or add files to "
        "the `documents/` folder and run `python ingest.py`, then restart this app. "
        "You can still chat normally in the meantime — I'll just use general knowledge."
    )

if groq_client is None:
    st.error(
        "GROQ_API_KEY not found. Add it to your `.env` file. "
        "Get a free key at https://console.groq.com"
    )
    st.stop()

messages = current_messages()

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            with st.expander("📄 Sources used"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**{i + 1}. {src['source']}** (relevance: {src['relevance']:.2f})")
                    st.text(src["preview"])

if query := st.chat_input("Ask a question, or just say hi..."):
    messages.append({"role": "user", "content": query, "sources": []})
    with st.chat_message("user"):
        st.markdown(query)

    session = st.session_state.sessions[st.session_state.current_session_id]
    if session["title"] == "New chat":
        session["title"] = memory.make_title_from_message(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            documents, metadatas, distances = retrieve_context(collection, query, top_k=top_k)
            past_history = messages[:-1]
            answer = ask_llm(groq_client, query, documents, metadatas, past_history)

            sources = []
            if documents:
                for doc, meta, dist in zip(documents, metadatas, distances):
                    sources.append({
                        "source": meta.get("source", "unknown"),
                        "relevance": 1 - dist,
                        "preview": doc[:300] + ("..." if len(doc) > 300 else ""),
                    })

            st.markdown(answer)
            if sources and show_sources:
                with st.expander("📄 Sources used"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**{i + 1}. {src['source']}** (relevance: {src['relevance']:.2f})")
                        st.text(src["preview"])

    messages.append({"role": "assistant", "content": answer, "sources": sources})
    persist()