"""Conversation persistence helpers for the Streamlit UI."""

import json
import re
import sqlite3
from datetime import UTC, datetime
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
                cards_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(ui_conversations)").fetchall()
        }
        if "cards_json" not in columns:
            conn.execute(
                "ALTER TABLE ui_conversations ADD COLUMN cards_json TEXT NOT NULL DEFAULT '{}'"
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


def save_conversation_to_db(
    user_id: str,
    conversation_id: str,
    messages: List[Dict[str, str]],
    cards: Dict[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    title = make_chat_title(messages)
    payload = json.dumps(messages, ensure_ascii=False)
    cards_payload = json.dumps(cards or {}, ensure_ascii=False)
    with get_ui_conn() as conn:
        conn.execute(
            """
            INSERT INTO ui_conversations(
                user_id, conversation_id, title, messages_json, cards_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                user_id=excluded.user_id,
                title=excluded.title,
                messages_json=excluded.messages_json,
                cards_json=excluded.cards_json,
                updated_at=excluded.updated_at
            """,
            (user_id, conversation_id, title, payload, cards_payload, now, now),
        )


def load_user_conversations_from_db(user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    with get_ui_conn() as conn:
        rows = conn.execute(
            """
            SELECT conversation_id, title, messages_json, cards_json, updated_at
            FROM ui_conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    conversations = []
    for conversation_id, title, messages_json, cards_json, updated_at in rows:
        try:
            messages = json.loads(messages_json or "[]")
        except Exception:
            messages = []
        try:
            cards = json.loads(cards_json or "{}")
        except Exception:
            cards = {}
        if not isinstance(cards, dict):
            cards = {}
        conversations.append(
            {
                "conversation_id": conversation_id,
                "title": title or "New chat",
                "messages": messages,
                "cards": cards,
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


def load_conversation_cards_from_db(conversation_id: str) -> Dict[str, Any]:
    with get_ui_conn() as conn:
        row = conn.execute(
            "SELECT cards_json FROM ui_conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    if not row:
        return {}
    try:
        cards = json.loads(row[0] or "{}")
    except Exception:
        return {}
    return cards if isinstance(cards, dict) else {}


def bootstrap_user_conversations(
    user_id: str,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[str], str, Dict[str, Dict[str, Any]]]:
    """Load recent UI conversations from DB.

    Returns: conversation_chats, recent_ids, active_id, conversation_cards.
    """
    rows = load_user_conversations_from_db(user_id)
    if not rows:
        default_conversation_id = user_id or DEFAULT_SESSION_ID
        return (
            {default_conversation_id: []},
            [default_conversation_id],
            default_conversation_id,
            {},
        )

    conversation_chats = {
        row["conversation_id"]: list(row.get("messages") or [])
        for row in rows
    }
    conversation_cards = {
        row["conversation_id"]: dict(row.get("cards") or {})
        for row in rows
    }
    recent_ids = [row["conversation_id"] for row in rows]
    return conversation_chats, recent_ids, recent_ids[0], conversation_cards


def is_new_trip_request(message: str) -> bool:
    """True when the user is starting a fresh trip plan (replace cards, don't merge)."""
    req = (message or "").lower().strip()
    if not req:
        return False

    def has_phrase(phrases):
        return any(phrase in req for phrase in phrases)

    def has_word(words):
        return any(re.search(rf"\b{re.escape(word)}\b", req) for word in words)

    trip_words = [
        "trip",
        "travel plan",
        "full trip",
        "vacation",
        "holiday",
        "itinerary",
        "plan me",
        "plan a trip",
        "plan my",
        "plan a",
        "travel to",
        "go to",
        "visit",
        "weekend in",
        "weekend trip",
        "city break",
        "days in",
        "nights in",
        "week in",
        "טיול",
        "חופשה",
        "תכנון טיול",
        "מסלול",
        "ימים ב",
        "לילות ב",
        "שבוע ב",
    ]
    flight_words = ["flight", "flights", "fly", "flying", "טיסה", "טיסות"]
    hotel_words = ["hotel", "hotels", "stay", "accommodation", "מלון", "מלונות"]
    activity_words = [
        "activity",
        "activities",
        "things to do",
        "attraction",
        "attractions",
        "פעילות",
        "אטרקציות",
    ]
    restaurant_words = ["restaurant", "restaurants", "מסעדה", "מסעדות"]
    car_phrases = ["car rental", "rent a car", "rental car", "השכרת רכב"]

    bookable_count = sum(
        [
            has_word(flight_words),
            has_word(hotel_words),
            has_word(activity_words),
            has_word(restaurant_words),
            has_phrase(car_phrases) or has_word(["car", "רכב"]),
        ]
    )

    if has_phrase(trip_words):
        return True

    if re.search(r"\b\d+\s*(?:days?|nights?|weeks?)\s+(?:in|to)\b", req):
        return True

    if re.search(r"\b(?:go|travel|fly|head)\s+to\b", req):
        return True

    if bookable_count >= 2:
        return True

    if bookable_count >= 1 and re.search(r"\b(?:to|in)\s+[a-z\u0590-\u05ff]", req):
        return True

    return False


def _destination_changed(existing: Dict[str, Any] | None, new: Dict[str, Any] | None) -> bool:
    old = str((existing or {}).get("destination", "")).strip().lower()
    new_dest = str((new or {}).get("destination", "")).strip().lower()
    return bool(old and new_dest and old != new_dest)


def merge_structured_cards(existing: Dict[str, Any] | None, new: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge incremental card updates into the live trip board."""
    existing = dict(existing or {})
    new = dict(new or {})
    if not new:
        return existing
    if not existing:
        return new

    merged = dict(existing)

    def unique_records(records, key):
        out = []
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            value = str(record.get(key, "")).strip().lower()
            if value and value not in seen:
                seen.add(value)
                out.append(record)
            elif not value:
                out.append(record)
        return out

    for list_key, dedupe_key in (
        ("flights", "flight"),
        ("hotels", "name"),
        ("activities", "name"),
        ("car_rentals", "company"),
        ("restaurants", "name"),
    ):
        if new.get(list_key):
            combined = list(merged.get(list_key) or []) + list(new.get(list_key) or [])
            merged[list_key] = unique_records(combined, dedupe_key)

    for scalar_key in (
        "destination",
        "visa",
        "time_difference",
        "currency_exchange",
        "seasonal_recommendations",
        "transport_info",
        "warning",
        "estimated_cost",
    ):
        if new.get(scalar_key):
            merged[scalar_key] = new[scalar_key]

    if new.get("notes"):
        old_lines = [line.strip() for line in str(merged.get("notes") or "").splitlines() if line.strip()]
        new_lines = [line.strip() for line in str(new["notes"]).splitlines() if line.strip()]
        deduped = []
        for line in old_lines + new_lines:
            if line not in deduped:
                deduped.append(line)
        merged["notes"] = "\n".join(deduped)

    return {
        key: value
        for key, value in merged.items()
        if value
    }


def apply_trip_cards_update(
    existing: Dict[str, Any] | None,
    new: Dict[str, Any] | None,
    user_message: str = "",
    force_replace: bool = False,
) -> Dict[str, Any]:
    """Replace cards for a new trip request; merge for follow-ups in the same trip."""
    if not new:
        return dict(existing or {})
    if (
        force_replace
        or is_new_trip_request(user_message)
        or _destination_changed(existing, new)
    ):
        return dict(new)
    return merge_structured_cards(existing, new)
