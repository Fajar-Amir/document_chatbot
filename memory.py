"""
memory.py
Handles saving and loading chat sessions to/from a local JSON file, so
conversations persist across app restarts.
"""

import json
import os
import uuid
from datetime import datetime

MEMORY_FILE = "chat_history.json"


def load_all_sessions():
    """Load all saved chat sessions from disk. Returns a dict of session_id -> session."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_all_sessions(sessions):
    """Persist all chat sessions to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def create_new_session():
    """Create a new empty session and return its id and data."""
    session_id = str(uuid.uuid4())
    session = {
        "title": "New chat",
        "created": datetime.now().isoformat(),
        "messages": [],  # list of {"role", "content", "sources"} dicts
    }
    return session_id, session


def make_title_from_message(message, max_len=40):
    """Generate a short title for a session based on its first user message."""
    title = message.strip().replace("\n", " ")
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0] + "..."
    return title or "New chat"