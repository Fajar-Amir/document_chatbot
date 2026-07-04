"""
chat.py
Interactive command-line chatbot that answers questions using your own
documents as context (RAG), powered by the free Groq API.

Before running: set your GROQ_API_KEY in a .env file (see .env.example).
"""

import os
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "my_docs"
TOP_K = 4
MODEL = "llama-3.3-70b-versatile"  # free tier on Groq

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


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("No collection found. Run 'python ingest.py' first to load your documents.")
        exit(1)


def retrieve_context(collection, query, top_k=TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return documents, metadatas


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


def ask_llm(client, query, documents, metadatas, history):
    prompt = build_prompt(query, documents, metadatas)

    # Include recent conversation history so the model has memory of the chat
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key_here")
        print("Get a free key at https://console.groq.com")
        return

    groq_client = Groq(api_key=api_key)
    collection = get_collection()

    print("RAG Chatbot ready. Ask a question about your documents (type 'exit' to quit).\n")

    history = []

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        documents, metadatas = retrieve_context(collection, query)
        answer = ask_llm(groq_client, query, documents, metadatas, history)
        print(f"\nBot: {answer}\n")

        # Store the plain exchange (not the retrieved-context version) in history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()