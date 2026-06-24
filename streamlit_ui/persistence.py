"""Conversation persistence helpers for the Streamlit UI."""

import json
import sqlite3
from datetime import timezone, datetime
from typing import Any, Dict, List, Tuple


DEFAULT_SESSION_ID = "demo_session"
UI_DB_FILE = "checkpoints.db"


def get_ui_conn() -> sqlite3.Connection:
    """Use the LangGraph SQLite file while keeping UI data in separate tables."""
    return sqlite3.connect(UI_DB_FILE, check_same_thread=False)


def init_ui_tables() -> None:
    with get_ui_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_conversations (
                user_id TEXT NOT NULL,
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_conversations_user_updated
            ON ui_conversations(user_id, updated_at DESC)
            """
        )


def make_chat_title(messages: List[Dict[str, str]]) -> str:
    """Short title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user" and str(msg.get("content", "")).strip():
            title = str(msg.get("content", "")).replace("\n", " ").strip()
            return title[:46] + ("..." if len(title) > 46 else "")
    return "New chat"


def save_conversation_to_db(user_id: str, conversation_id: str, messages: List[Dict[str, str]]) -> None:
    UTC = timezone.utc
    now = datetime.now(UTC).isoformat(timespec="seconds")
    title = make_chat_title(messages)
    payload = json.dumps(messages, ensure_ascii=False)
    with get_ui_conn() as conn:
        conn.execute(
            """
            INSERT INTO ui_conversations(user_id, conversation_id, title, messages_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                user_id=excluded.user_id,
                title=excluded.title,
                messages_json=excluded.messages_json,
                updated_at=excluded.updated_at
            """,
            (user_id, conversation_id, title, payload, now, now),
        )


def load_user_conversations_from_db(user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    with get_ui_conn() as conn:
        rows = conn.execute(
            """
            SELECT conversation_id, title, messages_json, updated_at
            FROM ui_conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    conversations = []
    for conversation_id, title, messages_json, updated_at in rows:
        try:
            messages = json.loads(messages_json or "[]")
        except Exception:
            messages = []
        conversations.append(
            {
                "conversation_id": conversation_id,
                "title": title or "New chat",
                "messages": messages,
                "updated_at": updated_at,
            }
        )
    return conversations


def load_conversation_messages_from_db(conversation_id: str) -> List[Dict[str, str]]:
    with get_ui_conn() as conn:
        row = conn.execute(
            "SELECT messages_json FROM ui_conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    if not row:
        return []
    try:
        return json.loads(row[0] or "[]")
    except Exception:
        return []


def bootstrap_user_conversations(user_id: str) -> Tuple[Dict[str, List[Dict[str, str]]], List[str], str]:
    """Load recent UI conversations from DB.

    Returns: conversation_chats, recent_ids, active_id.
    """
    rows = load_user_conversations_from_db(user_id)
    if not rows:
        default_conversation_id = user_id or DEFAULT_SESSION_ID
        return {default_conversation_id: []}, [default_conversation_id], default_conversation_id

    conversation_chats = {
        row["conversation_id"]: list(row.get("messages") or [])
        for row in rows
    }
    recent_ids = [row["conversation_id"] for row in rows]
    return conversation_chats, recent_ids, recent_ids[0]
