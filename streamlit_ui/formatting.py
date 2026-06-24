"""Formatting and answer-rendering helpers for the Streamlit UI."""

import html
import re
from typing import Any, List, Optional, Tuple

import mistune


def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def format_inline_markdown(text: str) -> str:
    escaped = esc(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)
    return escaped


def star_icons(value: Any) -> str:
    """Render any star-like value consistently.

    Supports:
    - 5, 4.0
    - "5-star hotel"
    - "⭐⭐⭐⭐⭐" / "★★★★☆"
    """
    raw = str(value or "").strip()
    if not raw:
        return ""

    # Already contains star characters from the model: normalize to the same span.
    if "⭐" in raw or "★" in raw or "☆" in raw:
        filled = raw.count("⭐") + raw.count("★")
        empty = raw.count("☆")
        if filled or empty:
            filled = max(0, min(5, filled))
            return '<span class="answer-stars">' + ("★" * filled) + ("☆" * (5 - filled)) + "</span>"

    # Numeric rating / star count. Also handles text like "5-star hotel".
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return esc(raw)

    try:
        count = max(0, min(5, int(round(float(m.group(1))))))
    except (TypeError, ValueError):
        return esc(raw)

    return '<span class="answer-stars">' + ("★" * count) + ("☆" * (5 - count)) + "</span>"


def normalize_label_value_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\*\*([^*:\n]+):\*\*\s*$", r"\1:", cleaned)
    cleaned = re.sub(r"^\*\*([^*:\n]+)\*\*:\s*$", r"\1:", cleaned)
    cleaned = re.sub(r"^\*\*([^*:\n]+):\*\*\s*(.+)$", r"\1: \2", cleaned)
    cleaned = re.sub(r"^\*\*([^*:\n]+)\*\*:\s*(.+)$", r"\1: \2", cleaned)
    cleaned = re.sub(r"^\*\*(.+?)\*\*$", r"\1", cleaned)
    return cleaned


DETAIL_LABELS = {
    "airline",
    "flight",
    "flight number",
    "price",
    "price per night",
    "duration",
    "departure",
    "arrival",
    "stars",
    "amenities",
    "rating",
    "room type",
    "category",
    "suitability",
    "suitable for",
    "cuisine",
    "price level",
    "special features",
    "location",
    "type",
    "seats",
    "visa requirements",
    "currency",
    "time difference",
    "travel warnings",
    "note",
    "notes",
    "activity",
    "activities",
    "accommodation",
    "estimated food expenses",
    "food expenses",
    "total estimated cost",
    "estimated cost",
    "total cost",
    "cost",
    "budget option",
    "luxury option",
    "features",
}


DETAIL_EMOJIS = {
    "airline": "✈️",
    "flight": "✈️",
    "flight number": "🔢",
    "price": "💰",
    "price per night": "💰",
    "duration": "⏱️",
    "departure": "🛫",
    "arrival": "🛬",
    "stars": "🌟",
    "amenities": "🛎️",
    "rating": "📊",
    "room type": "🛏️",
    "category": "🏷️",
    "cuisine": "🍽️",
    "price level": "💵",
    "special features": "✨",
    "features": "✨",
    "location": "📍",
    "type": "🏷️",
    "seats": "💺",
    "transmission": "⚙️",
    "breakfast": "🥐",
    "note": "📝",
    "notes": "📝",
    "amenity": "🛎️",
    "suitable for": "👥",
    "feature": "✨",
    "description": "📝",
}

PRICE_LEVELS = frozenset(
    {
        "expensive",
        "moderate",
        "cheap",
        "budget",
        "affordable",
        "luxury",
        "mid-range",
        "inexpensive",
        "pricey",
        "low",
        "high",
    }
)


def format_md_label(label: str) -> str:
    clean = label.strip().strip("*")
    if not clean:
        return clean
    low = clean.lower()
    emoji = DETAIL_EMOJIS.get(low, "")
    if not emoji:
        for key, candidate in sorted(DETAIL_EMOJIS.items(), key=lambda item: len(item[0]), reverse=True):
            if key in low:
                emoji = candidate
                break
    if emoji and not clean.startswith(emoji):
        return f"{emoji} {clean}"
    return clean


def format_labeled_md(label: str, value: str) -> str:
    return f"**{format_md_label(label)}:** {value}"


def format_star_rating(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    filled = raw.count("⭐") + raw.count("★")
    empty = raw.count("☆")
    if filled or empty:
        filled = max(0, min(5, filled))
        return "★" * filled + "☆" * (5 - filled)
    match = re.search(r"\b([1-5](?:\.\d+)?)\b", raw)
    if not match:
        return raw
    try:
        count = max(0, min(5, int(round(float(match.group(1))))))
    except (TypeError, ValueError):
        return raw
    return "★" * count + "☆" * (5 - count)


def _is_detail_field_name(name: str) -> bool:
    low = name.strip().strip("*").lower()
    if low in DETAIL_LABELS:
        return True
    hints = (
        "airline",
        "flight",
        "price",
        "duration",
        "departure",
        "arrival",
        "time",
        "number",
        "rating",
        "amenities",
        "stars",
        "room",
        "breakfast",
        "cuisine",
        "features",
        "location",
        "seats",
        "transmission",
        "type",
        "note",
    )
    if len(low.split()) > 7:
        return False
    return any(re.search(rf"\b{re.escape(hint)}\b", low) for hint in hints)


def _format_detail_line(label: str, value: str) -> str:
    if label.lower() == "stars":
        value = format_star_rating(value)
    return format_labeled_md(label, value)


def _parse_bullet_kv(line: str) -> Optional[Tuple[str, str, str]]:
    match = re.match(r"^(\s*)-\s+(.+)$", line)
    if not match:
        return None
    indent, body = match.groups()
    body = normalize_label_value_text(body).strip()
    body = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", body)
    kv = re.match(r"^([^:]{2,60}):\s+(.+)$", body)
    if not kv:
        return None
    return indent, kv.group(1).strip().strip("*"), kv.group(2).strip()


def is_real_section_heading(title: str) -> bool:
    clean = title.strip().strip(":").strip()
    low = clean.lower()
    if not clean:
        return False
    if len(clean.split()) > 7:
        return False
    return any(
        marker in low
        for marker in (
            "weather",
            "best time",
            "flight",
            "accommodation",
            "hotel",
            "stay",
            "thing",
            "activity",
            "tour",
            "dining",
            "restaurant",
            "food",
            "getting around",
            "transport",
            "car rental",
            "rental",
            "essential",
            "visa",
            "currency",
            "cost",
            "budget",
            "warning",
            "warnings",
            "notes",
            "summary",
            "seasonal",
            "recommendation",
            "recommendations",
            "agencies",
            "agency",
            "museum",
            "museums",
            "itinerary",
            "overview",
            "highlights",
            "packing",
            "insurance",
            "health",
            "language",
            "airport",
            "layover",
            "nightlife",
            "shopping",
            "beach",
            "culture",
            "sightseeing",
        )
    )


def _looks_like_section_heading(title: str) -> bool:
    clean = title.strip().strip(":").strip("*").strip()
    low = clean.lower()
    if not clean:
        return False
    if re.match(r"^(here'?s|i was|i am|this is|please|note that)\b", low):
        return False
    if _is_section_intro_line(clean):
        return False
    if len(clean.split()) > 12:
        return False

    if _is_detail_field_name(clean):
        return False

    section_markers = (
        "weather",
        "best time",
        "flight",
        "flights from",
        "accommodation",
        "hotel",
        "hotels in",
        "activities in",
        "activities",
        "activity",
        "things to do",
        "dining",
        "restaurant",
        "food",
        "getting around",
        "transport",
        "transportation",
        "car rental",
        "rental agenc",
        "essential",
        "visa",
        "currency",
        "time difference",
        "cost",
        "budget",
        "warning",
        "alert",
        "seasonal",
        "recommendation",
        "summary",
        "trip",
        "itinerary",
        "museum",
        "overview",
        "highlights",
        "packing",
        "insurance",
        "health",
        "language",
        "airport",
        "nightlife",
        "shopping",
        "beach",
        "culture",
        "sightseeing",
        "stay",
        "tour",
    )
    if any(marker in low for marker in section_markers):
        return True
    return is_real_section_heading(clean)


def _title_has_section_marker(title: str) -> bool:
    low = title.strip().lower()
    return any(
        marker in low
        for marker in (
            "visa",
            "flight",
            "hotel",
            "activities",
            "car rental",
            "seasonal",
            "travel warning",
            "time difference",
            "accommodation",
            "travel alert",
            "recommendation",
            "warning",
        )
    )


def _is_section_intro_line(text: str) -> bool:
    clean = text.strip().strip(":").strip("*").strip()
    low = clean.lower()
    if not clean:
        return False
    if re.match(r"^[-*•]\s", text.strip()):
        return False
    if re.match(r"^\*\*[^*]+\*\*:\s*$", text.strip()):
        return False
    if re.match(r"^(current|important)\s+", low):
        return False
    intro_patterns = (
        r"^you have\b",
        r"^here are\b",
        r"^here is\b",
        r"^there are\b",
        r"\boffers a wealth\b",
        r"^please be aware\b",
        r"\bgood options\b",
        r"\bpopular choices\b",
        r"\bhotel options\b",
        r"\bcouple of good\b",
        r"\bto consider\b",
        r"\bcatering to different\b",
    )
    if any(re.search(pattern, low) for pattern in intro_patterns):
        return True
    if "!" in clean and len(clean.split()) > 5:
        return True
    if ":" in clean:
        title = clean.split(":", 1)[0].strip()
        if _title_has_section_marker(title):
            return False
    if len(clean.split()) > 9:
        if _title_has_section_marker(clean):
            return False
        return True
    return False


def _strip_md_bold(value: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", str(value or "")).strip()


def _is_prose_description(text: str) -> bool:
    clean = text.strip()
    if _is_compact_detail_list(clean):
        return False
    if len(clean) > 90:
        return True
    if clean.count(".") >= 2:
        return True
    if re.search(
        r"\b(?:It costs|It's priced|offering|has an|has a|Explore |Allow about|Please take|Vulnerable travelers)\b",
        clean,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\bincludes\b", clean, flags=re.IGNORECASE):
        return True
    return False


def _is_option_card_name(name: str) -> bool:
    clean = name.strip().strip("*")
    low = clean.lower()
    info_labels = (
        "visa",
        "time difference",
        "currency",
        "seasonal",
        "recommendation",
        "caution",
        "local measures",
        "important",
        "note",
        "alert",
        "warning",
        "measure",
    )
    if any(label in low for label in info_labels):
        return False
    if any(
        hint in low
        for hint in (
            "total estimated",
            "estimated cost",
            "total cost",
            "estimated trip",
            "exchange rate",
        )
    ):
        return False
    if _is_detail_field_name(clean):
        return False
    return True


def _parse_flight_narrative_details(text: str) -> List[Tuple[str, str]]:
    details: List[Tuple[str, str]] = []
    dep = re.search(r"Departs?\s+at\s+([^,]+)", text, flags=re.IGNORECASE)
    if dep:
        details.append(("Departure", re.sub(r"^at\s+", "", dep.group(1).strip(), flags=re.IGNORECASE)))

    arr = re.search(r"arrives?\s+at\s+([^,]+)", text, flags=re.IGNORECASE)
    if arr:
        details.append(("Arrival", re.sub(r"^at\s+", "", arr.group(1).strip(), flags=re.IGNORECASE)))

    dur = re.search(r"duration\s+([^.,]+?)(?:\.|\s*Price:|\s*$)", text, flags=re.IGNORECASE)
    if dur:
        details.append(("Duration", dur.group(1).strip()))

    price = re.search(r"Price:\s*\*\*([^*]+)\*\*", text, flags=re.IGNORECASE)
    if not price:
        price = re.search(r"Price:\s*(\$[\d,]+)", text, flags=re.IGNORECASE)
    if price:
        details.append(("Price", _strip_md_bold(price.group(1))))

    return details


def _parse_narrative_details(text: str) -> List[Tuple[str, str]]:
    text = text.strip().rstrip(".")
    details: List[Tuple[str, str]] = []

    star_match = re.search(r"\b([1-5])[- ]star\b", text, flags=re.IGNORECASE)
    if star_match:
        details.append(("Stars", format_star_rating(star_match.group(1))))

    price_patterns = (
        r"\*\*(\$[\d,]+(?:\s+per\s+night)?)\*\*",
        r"(?:costs?|priced at|priced|for)\s+\*\*(\$[\d,]+(?:\s+per\s+night)?)\*\*",
        r"(?:costs?|priced at|for)\s+(\$[\d,]+(?:\s+per\s+night)?)",
        r"for \*\*(\$[\d,]+)\*\*",
        r"for \*\*(\$[\d,]+)\*\*",
    )
    for pattern in price_patterns:
        price_match = re.search(pattern, text, flags=re.IGNORECASE)
        if price_match:
            details.append(("Price", _strip_md_bold(price_match.group(1))))
            break

    rating_match = re.search(
        r"(?:has an|has a|with an|with a)?\s*(?:excellent |good )?rating of\s+([\d.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if rating_match:
        details.append(("Rating", rating_match.group(1)))

    room_match = re.search(
        r"\b(Deluxe|Standard|Superior|Executive|Premium|Suite)\b(?:\s+room)?",
        text,
        flags=re.IGNORECASE,
    )
    if room_match:
        details.append(("Room Type", room_match.group(1)))

    if re.search(r"breakfast is not included", text, flags=re.IGNORECASE):
        details.append(("Breakfast", "not included"))
    elif re.search(r"includes? breakfast|breakfast included", text, flags=re.IGNORECASE):
        details.append(("Breakfast", "included"))

    duration_match = re.search(
        r"(?:Allow about|typically lasts?|tour typically lasts?|lasts?)\s+(\d+\s+hours?(?:\s+\d+\s+minutes?)?)",
        text,
        flags=re.IGNORECASE,
    )
    if duration_match:
        details.append(("Duration", duration_match.group(1).strip()))
    elif re.search(r"\bfull day\b", text, flags=re.IGNORECASE):
        details.append(("Duration", "Full day"))

    suitable_match = re.search(r"\(?(?:Suitable for|Perfect for)\s+([^).]+)\)?", text, flags=re.IGNORECASE)
    if suitable_match:
        details.append(("Suitable for", suitable_match.group(1).strip()))

    amenities: List[str] = []
    if re.search(r"\bspa\b", text, flags=re.IGNORECASE):
        amenities.append("Spa")
    if re.search(r"fine dining", text, flags=re.IGNORECASE):
        amenities.append("Fine Dining")
    if re.search(r"\bWiFi\b", text, flags=re.IGNORECASE):
        amenities.append("WiFi")
    if amenities:
        details.append(("Amenities", ", ".join(amenities)))

    if not details:
        details.append(("Description", text))
    else:
        first_sentence = re.split(r"[.!]", text)[0].strip()
        has_price = any(label == "Price" for label, _ in details)
        if (
            first_sentence
            and len(first_sentence) < len(text) * 0.75
            and not (has_price and re.search(r"\*\*\$", first_sentence))
        ):
            details.insert(0, ("Description", first_sentence))

    return details


def _is_compact_detail_list(text: str) -> bool:
    """Detect short comma-separated tool rows like '20, Culture, 3h, Suitable for Couples'."""
    parts = _split_comma_parts(text.strip())
    if len(parts) < 2:
        return False
    if any(len(part.split()) > 6 for part in parts):
        return False
    compact_signals = 0
    for part in parts:
        clean = part.strip()
        low = clean.lower()
        if re.match(r"^\d+(?:\.\d+)?$", clean):
            compact_signals += 1
        elif re.match(r"^[1-5]\s*-?\s*stars?$", clean, flags=re.IGNORECASE):
            compact_signals += 1
        elif re.match(r"^rating\b", low):
            compact_signals += 1
        elif re.match(r"^breakfast\b|^no breakfast$", low):
            compact_signals += 1
        elif re.match(r"^(category|duration|rating|departure|arrival|amenities|price|flight)\s*:", low):
            compact_signals += 1
        elif re.match(r"^(price|flight|flight number|duration|departure|arrival)\b", low):
            compact_signals += 1
        elif re.match(r"^\d+h(?:\s+\d+m)?$", clean, flags=re.IGNORECASE):
            compact_signals += 1
        elif low in {"full day", "half day"}:
            compact_signals += 1
        elif low.startswith("suitable for"):
            compact_signals += 1
        elif low in {
            "culture",
            "sightseeing",
            "family",
            "adventure",
            "food",
            "nightlife",
            "nature",
            "shopping",
            "sports",
            "entertainment",
            "museum",
        }:
            compact_signals += 1
    return compact_signals >= 2


def _format_compact_fragment(part: str) -> Optional[str]:
    """Parse compact tool fragments like 'category: Culture' or 'duration: 3h'."""
    clean = part.strip().strip(".").strip()
    kv = re.match(r"^([A-Za-z][A-Za-z\s]{1,38}):\s*(.+)$", clean)
    if not kv:
        return None

    label = kv.group(1).strip().strip("*")
    value = kv.group(2).strip()
    low = label.lower()
    if low not in DETAIL_LABELS and not _is_detail_field_name(label):
        return None

    display_label = "Suitable for" if low == "suitable for" else label.title()
    if low == "flight number":
        display_label = "Flight Number"

    if low == "stars":
        value = format_star_rating(value)
    elif low == "duration" and value.lower() in {"full day", "half day"}:
        value = value.title()

    return format_labeled_md(display_label, value)


def _split_comma_parts(value: str) -> List[str]:
    """Split comma-separated detail fragments, ignoring commas inside parentheses."""
    value = value.strip().rstrip(".")
    if not value:
        return []

    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for i, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            prev_digit = bool(current) and current[-1].isdigit()
            next_digit = i + 1 < len(value) and value[i + 1].isdigit()
            if prev_digit and next_digit:
                current.append(char)
                continue
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_flight_prose_bullets(body: str, indent: str = "") -> List[str]:
    """Turn a narrative flight sentence into structured option bullets."""
    text = body.strip().rstrip(".")
    if not text:
        return []

    lines: List[str] = []
    sub_indent = f"{indent}  "

    airline_match = re.match(r"^A\s+(.+?)\s+flight\b", text, flags=re.IGNORECASE)
    if airline_match:
        lines.append(f"{indent}- **{airline_match.group(1).strip()}**")
        detail_prefix = sub_indent
    else:
        detail_prefix = indent

    price_match = re.search(r"(?:available\s+for|for)\s+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if price_match:
        lines.append(f"{detail_prefix}- {format_labeled_md('Price', price_match.group(1))}")

    flight_no_match = re.search(r"flight\s+number\s+([A-Z0-9]+)", text, flags=re.IGNORECASE)
    if flight_no_match:
        lines.append(f"{detail_prefix}- {format_labeled_md('Flight', flight_no_match.group(1))}")

    dep_match = re.search(r"departing\s+at\s+([^,]+)", text, flags=re.IGNORECASE)
    if dep_match:
        lines.append(f"{detail_prefix}- {format_labeled_md('Departure', dep_match.group(1).strip())}")

    arr_match = re.search(r"arriving\s+at\s+([^,]+)", text, flags=re.IGNORECASE)
    if arr_match:
        lines.append(f"{detail_prefix}- {format_labeled_md('Arrival', arr_match.group(1).strip())}")

    dur_match = re.search(r"duration\s+of\s+([^,.]+)", text, flags=re.IGNORECASE)
    if dur_match:
        lines.append(f"{detail_prefix}- {format_labeled_md('Duration', dur_match.group(1).strip())}")

    if len(lines) > (1 if airline_match else 0):
        return lines
    return [f"{indent}- {text}"]


def normalize_answer_markdown(text: str) -> str:
    """Small, safe Markdown cleanup before rendering.

    Mistune does the real Markdown parsing. This function only:
    - removes terminal/report decoration lines
    - adds friendly emojis to common travel headings
    - normalizes star ratings
    - keeps bullets, times like 07:15, and normal sentences untouched
    """
    raw_text = str(text or "").replace("\r\n", "\n").strip()
    if not raw_text:
        return ""

    raw_text = re.sub(r"(?im)^\s*preferences?\s+on\s+file\s*:.*$", "", raw_text).strip()

    heading_icons = {
        "trip summary": "🧳",
        "travel plan": "🧳",
        "weather": "☁️",
        "current weather": "☁️",
        "best time": "🌤️",
        "best time to visit": "🌤️",
        "flight": "✈️",
        "flights": "✈️",
        "flights from": "✈️",
        "accommodation": "🛏️",
        "accommodation options": "🛏️",
        "hotel": "🏨",
        "hotels": "🏨",
        "things to do": "📍",
        "activities": "📍",
        "activities & recommendations": "📍",
        "dining": "🍽️",
        "dining recommendations": "🍽️",
        "restaurants": "🍽️",
        "getting around": "🚇",
        "transport": "🚇",
        "transportation": "🚇",
        "car rental": "🚗",
        "visa": "🛂",
        "visa information": "🛂",
        "travel essentials": "🧭",
        "currency": "💱",
        "time difference": "🕒",
        "travel warnings": "⚠️",
        "safety": "🛡️",
        "estimated trip cost": "💰",
        "estimated cost": "💰",
        "currency exchange": "💱",
        "cost": "💰",
        "budget": "💰",
        "recommendation": "✨",
        "bond recommendation": "✨",
        "summary": "✨",
        "seasonal recommendation": "🌸",
        "seasonal recommendations": "🌸",
        "travel warning": "⚠️",
        "activities in": "🎯",
        "hotels in": "🏨",
        "flights from": "✈️",
        "car rental agencies": "🚗",
        "car rental": "🚗",
        "city transportation": "🚇",
        "important travel warnings": "⚠️",
        "travel warning": "⚠️",
        "travel alert": "⚠️",
        "current travel alert": "⚠️",
        "visa requirements": "🛂",
        "time difference": "🕒",
        "museum": "🏛️",
        "museums": "🏛️",
        "itinerary": "🗓️",
        "overview": "👀",
        "highlights": "⭐",
        "packing": "🎒",
        "insurance": "🩺",
        "health": "🏥",
        "language": "🗣️",
        "airport": "🛫",
        "nightlife": "🌃",
        "shopping": "🛍️",
        "beach": "🏖️",
        "culture": "🎭",
        "sightseeing": "📸",
    }

    def normalize_stars(value: str) -> str:
        value = (value or "").strip()
        if not value:
            return value

        filled = value.count("⭐") + value.count("★")
        empty = value.count("☆")
        if filled or empty:
            filled = max(0, min(5, filled))
            return "★" * filled + "☆" * (5 - filled)

        match = re.search(r"\b([1-5](?:\.\d+)?)\b", value)
        if not match:
            return value

        try:
            count = max(0, min(5, int(round(float(match.group(1))))))
        except ValueError:
            return value
        return "★" * count + "☆" * (5 - count)

    def add_heading_icon(title: str) -> str:
        clean = title.strip()
        if not clean:
            return clean

        # Already has an icon or flag at the beginning.
        if re.match(r"^[^\w\s#*]", clean):
            return clean

        lowered = clean.lower().strip(":")
        chosen = ""
        for key, icon in sorted(heading_icons.items(), key=lambda item: len(item[0]), reverse=True):
            if lowered.startswith(key) or key in lowered:
                chosen = icon
                break
        return f"{chosen} {clean}" if chosen else f"📌 {clean}"

    def clean_time(value: str) -> str:
        return re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", value.strip())

    def split_detail_parts(value: str) -> List[str]:
        value = value.strip().rstrip(".")
        if not _is_compact_detail_list(value) and _is_prose_description(value):
            return [value]
        value = re.sub(r"\.\s+", ", ", value)
        raw_parts = _split_comma_parts(value)
        parts: List[str] = []
        idx = 0
        while idx < len(raw_parts):
            part = raw_parts[idx]
            if re.match(r"^amenities\s*(?::|(?:include|includes)\s+)", part, flags=re.IGNORECASE):
                amenities = [part]
                idx += 1
                while idx < len(raw_parts):
                    next_part = raw_parts[idx]
                    if re.match(
                        r"^(?:rating\s*:?|breakfast\b|[1-5](?:\.\d+)?\s+stars?$|[$£€₪]|\d+\s+seats?$|automatic$|manual$|departure\b|arrival\b|duration\b|category\b)",
                        next_part,
                        flags=re.IGNORECASE,
                    ):
                        break
                    amenities.append(next_part)
                    idx += 1
                parts.append(", ".join(amenities))
                continue

            parts.append(part)
            idx += 1

        return parts

    def compact_detail_row(part: str) -> str:
        clean = clean_time(part.strip().strip("."))
        low = clean.lower()

        labeled = _format_compact_fragment(part)
        if labeled:
            return labeled

        star_match = re.match(r"^([1-5])\s*-?\s*stars?$", clean, flags=re.IGNORECASE)
        if star_match:
            return format_labeled_md("Stars", format_star_rating(star_match.group(1)))

        star_hotel_match = re.match(r"^([1-5])\s*-\s*star(?:\s+hotel)?$", clean, flags=re.IGNORECASE)
        if star_hotel_match:
            return format_labeled_md("Stars", format_star_rating(star_hotel_match.group(1)))

        rating_match = re.match(r"^rating\s*:?\s*(.+)$", clean, flags=re.IGNORECASE)
        if rating_match:
            return format_labeled_md("Rating", rating_match.group(1).strip())

        rating_value_match = re.match(r"^rating\s+([\d.]+)$", clean, flags=re.IGNORECASE)
        if rating_value_match:
            return format_labeled_md("Rating", rating_value_match.group(1))

        amenities_match = re.match(r"^amenities\s*:?\s*(.+)$", clean, flags=re.IGNORECASE)
        if amenities_match:
            return format_labeled_md("Amenities", amenities_match.group(1).strip())

        price_match = re.match(r"^([$£€₪]\s?[\d,]+(?:\.\d+)?(?:\s*/\s?(?:night|day))?)$", clean)
        if price_match:
            return format_labeled_md("Price", price_match.group(1).replace(" /", "/"))

        seats_match = re.match(r"^(\d+\s+seats?)$", clean, flags=re.IGNORECASE)
        if seats_match:
            return format_labeled_md("Seats", seats_match.group(1))

        if low in {"automatic", "manual"}:
            return format_labeled_md("Transmission", clean.title())

        if low in {"automatic transmission", "manual transmission"}:
            return format_labeled_md("Transmission", clean.split()[0].title())

        if low.startswith("breakfast"):
            val = re.sub(r"^breakfast\s+", "", clean, flags=re.IGNORECASE).strip()
            return format_labeled_md("Breakfast", val)

        if low == "no breakfast":
            return format_labeled_md("Breakfast", "not included")

        price_per_night_match = re.match(r"^price\s+per\s+night\s+(.+)$", clean, flags=re.IGNORECASE)
        if price_per_night_match:
            return format_labeled_md("Price", f"{price_per_night_match.group(1).strip()} per night")

        price_per_day_match = re.match(r"^price\s+per\s+day\s+(.+)$", clean, flags=re.IGNORECASE)
        if price_per_day_match:
            return format_labeled_md("Price", f"{price_per_day_match.group(1).strip()} per day")

        if low.startswith("price"):
            val = re.sub(r"^price\s*:?\s*", "", clean, flags=re.IGNORECASE).strip()
            return format_labeled_md("Price", val)

        flight_number_match = re.match(r"^flight\s+number\s+(.+)$", clean, flags=re.IGNORECASE)
        if flight_number_match:
            return format_labeled_md("Flight Number", flight_number_match.group(1).strip())

        flight_match = re.match(r"^flight\s+(.+)$", clean, flags=re.IGNORECASE)
        if flight_match:
            return format_labeled_md("Flight", flight_match.group(1).strip())

        duration_match = re.match(r"^duration\s*:?\s*(.+)$", clean, flags=re.IGNORECASE)
        if duration_match:
            value = duration_match.group(1).strip()
            if value.lower() in {"full day", "half day"}:
                value = value.title()
            return format_labeled_md("Duration", value)

        departs_match = re.match(r"^depart(?:s|ure(?:s)?)?\s+(.+)$", clean, flags=re.IGNORECASE)
        if departs_match:
            return format_labeled_md("Departure", clean_time(departs_match.group(1).strip()))

        arrives_match = re.match(r"^arriv(?:e|es|al(?:s)?)?\s+(.+)$", clean, flags=re.IGNORECASE)
        if arrives_match:
            return format_labeled_md("Arrival", clean_time(arrives_match.group(1).strip()))

        per_night_match = re.match(r"^(\d+(?:\.\d+)?)\s+per\s+night$", clean, flags=re.IGNORECASE)
        if per_night_match:
            return format_labeled_md("Price", f"{per_night_match.group(1)} per night")

        per_day_match = re.match(r"^(\d+(?:\.\d+)?)\s+per\s+day$", clean, flags=re.IGNORECASE)
        if per_day_match:
            return format_labeled_md("Price", f"{per_day_match.group(1)} per day")

        duration_short_match = re.match(r"^(\d+h(?:\s+\d+m)?)\s+duration$", clean, flags=re.IGNORECASE)
        if duration_short_match:
            return format_labeled_md("Duration", duration_short_match.group(1).strip())

        if re.match(r"^\d+h(?:\s+\d+m)?$", clean, flags=re.IGNORECASE):
            return format_labeled_md("Duration", clean)

        if low in {"full day", "half day"}:
            return format_labeled_md("Duration", clean.title())

        suitable_match = re.match(r"^(?:suitable\s+for|for)\s+(.+)$", clean, flags=re.IGNORECASE)
        if suitable_match:
            return format_labeled_md("Suitable for", suitable_match.group(1).strip())

        cuisine_match = re.match(r"^(.+?)\s+cuisine$", clean, flags=re.IGNORECASE)
        if cuisine_match:
            return format_labeled_md("Cuisine", cuisine_match.group(1).strip())

        if low in PRICE_LEVELS:
            return format_labeled_md("Price level", clean)

        with_match = re.match(r"^with\s+(.+)$", clean, flags=re.IGNORECASE)
        if with_match:
            return format_labeled_md("Amenities", with_match.group(1).strip())

        if re.match(r"^\d+(?:\.\d+)?$", clean):
            return format_labeled_md("Price", clean)

        car_type_match = re.match(
            r"^(economy|compact|suv|sedan|luxury|premium|standard|minivan|coupe)$",
            low,
        )
        if car_type_match:
            return format_labeled_md("Type", clean)

        car_model_match = re.match(
            r"^(economy|compact|suv|sedan|luxury|premium|standard|minivan|coupe)\s+car$",
            low,
        )
        if car_model_match:
            return format_labeled_md("Type", clean.title())

        activity_categories = frozenset(
            {
                "culture",
                "sightseeing",
                "family",
                "adventure",
                "food",
                "nightlife",
                "nature",
                "shopping",
                "sports",
                "entertainment",
                "museum",
            }
        )
        if low in activity_categories:
            return format_labeled_md("Category", clean)

        room_type_match = re.match(
            r"^(standard|deluxe|suite|superior|executive|premium)\s+room$",
            low,
        )
        if room_type_match:
            return format_labeled_md("Room Type", clean.title())

        feature_hints = ("view", "romantic", "trendy", "nightlife", "rooftop", "historic", "cozy", "lively")
        if any(hint in low for hint in feature_hints):
            return format_labeled_md("Feature", clean)

        if re.match(r"^(suite|deluxe|standard|superior|executive|premium)(\s+room)?$", clean, flags=re.IGNORECASE):
            return format_labeled_md("Room Type", clean)

        if re.match(r"^[A-Za-z][A-Za-z0-9\s&'-]{1,40}$", clean) and not re.search(r"\d", clean):
            return format_labeled_md("Amenity", clean)

        return format_labeled_md("Feature", clean)

    def _child_detail_lines(indent: str, parts: List[str]) -> List[str]:
        child_prefix = indent if indent else "  "
        return [f"{child_prefix}- {compact_detail_row(part)}" for part in parts]

    def parse_compact_travel_bullet(line: str) -> Optional[List[str]]:
        bullet_match = re.match(r"^(\s*)-\s+(.+)$", line)
        if not bullet_match:
            return None

        indent, body = bullet_match.groups()
        body = normalize_label_value_text(body).strip()
        body = re.sub(r"\s+:\s+", ": ", body)
        body = clean_time(body)

        flight_match = re.match(
            r"^(.+?)\s+departing\s+at\s+(.+?),\s*arriving\s+at\s+(.+?),\s*for\s+(.+?)\.?$",
            body,
            flags=re.IGNORECASE,
        )
        if flight_match:
            name, departure, arrival, price = flight_match.groups()
            return [
                f"{indent}- **{name.strip()}**",
                f"{indent}  - {format_labeled_md('Departure', clean_time(departure))}",
                f"{indent}  - {format_labeled_md('Arrival', clean_time(arrival))}",
                f"{indent}  - {format_labeled_md('Price', price.strip())}",
            ]

        named_detail = re.match(r"^\*\*([^:*]+)\*\*:\s*(.+)$", body)
        if not named_detail:
            named_detail = re.match(r"^([^:,\n]{2,80}?):\s*(.+)$", body)
        if named_detail:
            name, details_text = named_detail.groups()
            name = name.strip().strip("*")
            details_text = details_text.strip()

            if not _is_option_card_name(name):
                return [f"{indent}- **{name}:** {details_text}"]

            detail_indent = f"{indent}  " if indent else "  "

            if re.search(r"Departs?\s+at", details_text, flags=re.IGNORECASE):
                fields = _parse_flight_narrative_details(details_text)
            elif _is_compact_detail_list(details_text):
                return [f"{indent}- **{name}**"] + [
                    f"{detail_indent}- {compact_detail_row(part)}" for part in split_detail_parts(details_text)
                ]
            elif _is_prose_description(details_text) or re.search(
                r"(?:Perfect for|for \*\*\$)",
                details_text,
                flags=re.IGNORECASE,
            ):
                fields = _parse_narrative_details(details_text)
            elif "," in details_text:
                return [f"{indent}- **{name}**"] + [
                    f"{detail_indent}- {compact_detail_row(part)}" for part in split_detail_parts(details_text)
                ]
            else:
                return [f"{indent}- **{name}:** {details_text}"]

            if fields:
                return [f"{indent}- **{name}**"] + [
                    f"{detail_indent}- {format_labeled_md(label, value)}" for label, value in fields
                ]

        if "," in body and not _is_prose_description(body):
            detail_parts = split_detail_parts(body)
            if detail_parts:
                return _child_detail_lines(indent, detail_parts)

        return None

    lines = raw_text.split("\n")
    normalized: List[str] = []
    in_console_title = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        # Remove CLI decoration lines like ===== or ----.
        if stripped and set(stripped) <= {"=", "-"} and len(stripped) >= 4:
            continue

        upper = stripped.upper()
        if upper.startswith("TRIP SUMMARY FOR:"):
            destination = stripped.split(":", 1)[1].strip()
            normalized.append(f"# 🧳 Trip Summary: {destination.title()}")
            in_console_title = True
            continue

        if upper.startswith("ESTIMATED TOTAL COST:"):
            amount = stripped.split(":", 1)[1].strip()
            normalized.append("")
            normalized.append("### 💰 Estimated total cost")
            normalized.append("")
            normalized.append(f"**{amount}**")
            continue

        if upper.startswith("BUDGET ALERT:"):
            alert = stripped.split(":", 1)[1].strip()
            normalized.append(f"> ⚠️ **Budget alert:** {alert}")
            continue

        if upper.startswith("BUDGET ON FILE:"):
            budget = stripped.split(":", 1)[1].strip()
            normalized.append(f"- **Budget on file:** {budget}")
            continue

        if upper.startswith("PREFERENCES ON FILE:"):
            continue

        if re.match(r"^preferences?\s+on\s+file\s*:", stripped, flags=re.IGNORECASE):
            continue

        # Avoid duplicating immediate "### Your London Travel Plan".
        if in_console_title and stripped.startswith("#"):
            in_console_title = False
            continue
        if stripped:
            in_console_title = False

        if re.match(r"^here'?s\b", stripped, flags=re.IGNORECASE) and re.search(
            r"\b(?:plan|trip|travel|itinerary)\b", stripped, flags=re.IGNORECASE
        ):
            normalized.append("### 🧳 Trip Overview")
            normalized.append(f"- {stripped}")
            continue

        if re.match(r"^i was unable\b", stripped, flags=re.IGNORECASE) and re.search(
            r"time\s+difference", stripped, flags=re.IGNORECASE
        ):
            normalized.append("### 🕒 Time Difference")
            normalized.append(f"- {stripped}")
            continue

        if re.match(r"^\*([^*]+)\*$", stripped):
            normalized.append(f"> 💡 {stripped.strip('*').strip()}")
            continue

        # Normalize common bullet markers, preserving indentation.
        left_stripped = line.lstrip()
        indent = line[: len(line) - len(left_stripped)]
        for marker in ("*   ", "* ", "• "):
            if left_stripped.startswith(marker):
                line = indent + "- " + left_stripped[len(marker):]
                stripped = line.strip()
                break

        if re.match(r"^i hope this\b", stripped, flags=re.IGNORECASE):
            normalized.append(stripped)
            continue

        if _is_section_intro_line(stripped.strip(":")):
            normalized.append(f"- {stripped}")
            continue

        if (
            normalized
            and normalized[-1].startswith("### ")
            and stripped
            and not stripped.startswith(("#", "-", "*", ">", "`"))
            and not re.match(r"^\*\*", stripped)
        ):
            normalized.append(f"- {stripped}")
            continue

        empty_option = re.match(r"^(\s*)-\s+\*\*([^*]+):\*\*\s*$", stripped) or re.match(
            r"^(\s*)-\s+\*\*([^*]+)\*\*:\s*$", stripped
        )
        if empty_option:
            normalized.append(f"{empty_option.group(1)}- **{empty_option.group(2).strip()}**")
            continue

        section_only_bullet = re.match(r"^-\s+([^:\n]{3,100}):\s*$", stripped)
        if section_only_bullet and _looks_like_section_heading(section_only_bullet.group(1).strip()):
            normalized.append(f"### {add_heading_icon(section_only_bullet.group(1).strip())}")
            continue

        section_bullet_body = re.match(r"^-\s+([^:\n]{3,80}):\s+(.+)$", stripped)
        if section_bullet_body:
            raw_title = section_bullet_body.group(1).strip()
            body = section_bullet_body.group(2).strip()
            if _looks_like_section_heading(raw_title) and not _is_option_card_name(raw_title):
                normalized.append(f"### {add_heading_icon(raw_title)}")
                normalized.append(f"- {body}")
                continue

        # Add emojis to Markdown headings only.
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            hashes, title = heading_match.groups()
            line = f"{hashes} {add_heading_icon(title)}"
            normalized.append(line)
            continue

        # If the model writes a bold-only section title, make it look like a heading.
        bold_title = re.match(r"^\*\*([^:*]{3,120}):?\*\*\s*$", stripped)
        if bold_title:
            title = add_heading_icon(bold_title.group(1).strip())
            normalized.append(f"### {title}")
            continue

        bold_section = re.match(r"^\*\*([^:*]{3,80}):\*\*\s+(.+)$", stripped)
        if bold_section:
            raw_title, body = bold_section.group(1).strip(), bold_section.group(2).strip()
            if _looks_like_section_heading(raw_title):
                normalized.append(f"### {add_heading_icon(raw_title)}")
                if re.search(r"\bflights?\b", raw_title, flags=re.IGNORECASE) and re.search(
                    r"\bflight\b", body, flags=re.IGNORECASE
                ):
                    normalized.extend(_parse_flight_prose_bullets(body))
                else:
                    normalized.append(f"- {body}")
                continue

        # Convert plain travel section labels to real Markdown headings.
        plain_title = re.match(r"^([^:\n]{3,100}):\s*$", stripped)
        if plain_title and not stripped.startswith(("-", "*", "•")):
            raw_title = plain_title.group(1).strip()
            if _looks_like_section_heading(raw_title):
                normalized.append(f"### {add_heading_icon(raw_title)}")
                continue

        section_with_body = re.match(r"^([^:\n]{3,100}):\s+(.+)$", stripped)
        if section_with_body and not re.match(r"^[-*•]\s", stripped):
            raw_title, body = section_with_body.group(1).strip(), section_with_body.group(2).strip()
            if _looks_like_section_heading(raw_title):
                normalized.append(f"### {add_heading_icon(raw_title)}")
                normalized.append(f"- {body}")
                continue

        # Normalize "Stars: 5", "**Stars:** 5", etc.
        star_line = re.match(r"^(\s*(?:-\s*)?)(?:\*\*)?Stars(?:\*\*)?:\s*(.+)$", line, flags=re.IGNORECASE)
        if star_line:
            prefix, value = star_line.groups()
            normalized.append(f"{prefix}{format_labeled_md('Stars', normalize_stars(value))}")
            continue

        detail_kv = _parse_bullet_kv(line)
        if detail_kv:
            indent, label, value = detail_kv
            if _is_detail_field_name(label):
                normalized.append(f"{indent}- {_format_detail_line(label, value)}")
                continue

        compact_bullet = parse_compact_travel_bullet(line)
        if compact_bullet:
            normalized.extend(compact_bullet)
            continue

        # Add star glyphs after "5-star hotel" / "4 star" text, without changing ratings like 9.6.
        def _star_repl(match: re.Match) -> str:
            full = match.group(0)
            count = int(match.group(1))
            return f"{full} {'★' * count}{'☆' * (5 - count)}"

        line = re.sub(r"\b([1-5])\s*[- ]star(?:\s+hotel)?\b", _star_repl, line, flags=re.IGNORECASE)

        normalized.append(line)

    return "\n".join(normalized).strip()


_MARKDOWN_RENDERER = mistune.create_markdown(
    renderer=mistune.HTMLRenderer(escape=True),
    plugins=["strikethrough", "table", "url"],
)


def _enhance_travel_answer_html(html_out: str) -> str:
    """Add presentation classes to common travel-report HTML patterns."""
    html_out = re.sub(
        r"<li>\s*<p>((?:(?!</p>).)*)</p>\s*</li>",
        r'<li class="travel-detail-row">\1</li>',
        html_out,
        flags=re.DOTALL,
    )

    html_out = re.sub(
        r'(<ul class="travel-option-details">\s*)<li>([^<]+)</li>',
        r'\1<li class="travel-detail-row">\2</li>',
        html_out,
        flags=re.IGNORECASE,
    )

    html_out = re.sub(
        r"<li>(\s*<strong>[^<:]+</strong>)\s*<ul>",
        r'<li class="travel-option-card">\1<ul class="travel-option-details">',
        html_out,
        flags=re.DOTALL | re.IGNORECASE,
    )

    html_out = re.sub(
        r"<li>\s*<strong>([^<]*:[^<]*)</strong>\s*(.*?)</li>",
        r'<li class="travel-detail-row"><span class="travel-kv-label">\1</span><span class="travel-kv-value">\2</span></li>',
        html_out,
        flags=re.DOTALL | re.IGNORECASE,
    )

    html_out = re.sub(
        r"(<h3 class=\"travel-answer-subsection\">.*?</h3>)\s*(<ul>)",
        r'\1<div class="travel-section-block">\2',
        html_out,
        flags=re.DOTALL,
    )
    html_out = re.sub(
        r"(</ul>)(\s*<h3 class=\"travel-answer-subsection\">)",
        r"\1</div>\2",
        html_out,
    )
    if '<div class="travel-section-block">' in html_out and not html_out.rstrip().endswith("</div>"):
        html_out = html_out.rstrip() + "</div>"

    html_out = re.sub(
        r"<blockquote>\s*<p>(.*?)</p>\s*</blockquote>",
        r'<div class="travel-alert">⚠️ \1</div>',
        html_out,
        flags=re.DOTALL | re.IGNORECASE,
    )

    html_out = re.sub(
        r'<h3 class="travel-answer-subsection">💰 Estimated total cost</h3>\s*<p><strong>(.*?)</strong></p>',
        r'<div class="travel-cost-banner"><span class="travel-cost-label">💰 Estimated total cost</span><span class="travel-cost-value">\1</span></div>',
        html_out,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return html_out


def markdown_to_html(text: str) -> str:
    """Render assistant messages as styled Markdown HTML."""
    clean_text = normalize_answer_markdown(text)
    if not clean_text:
        return ""

    html_out = _MARKDOWN_RENDERER(clean_text)

    # Normalize star strings after Markdown rendering.
    star_patterns = {
        "★★★★★": "★★★★★",
        "★★★★☆": "★★★★☆",
        "★★★☆☆": "★★★☆☆",
        "★★☆☆☆": "★★☆☆☆",
        "★☆☆☆☆": "★☆☆☆☆",
        "⭐⭐⭐⭐⭐": "★★★★★",
        "⭐⭐⭐⭐": "★★★★☆",
        "⭐⭐⭐": "★★★☆☆",
        "⭐⭐": "★★☆☆☆",
    }
    for raw, pretty in star_patterns.items():
        html_out = html_out.replace(raw, f'<span class="answer-stars">{pretty}</span>')

    html_out = re.sub(
        r"\((\d+)\s*stars?\)",
        lambda m: f'<span class="answer-stars">{format_star_rating(m.group(1))}</span>',
        html_out,
        flags=re.IGNORECASE,
    )

    # Light styling hooks without custom parsing.
    html_out = re.sub(
        r"<h1>(.*?)</h1>",
        r'<h1 class="travel-answer-title">\1</h1>',
        html_out,
        flags=re.DOTALL,
    )
    html_out = re.sub(
        r"<h2>(.*?)</h2>",
        r'<h2 class="travel-answer-section">\1</h2>',
        html_out,
        flags=re.DOTALL,
    )
    html_out = re.sub(
        r"<h3>(.*?)</h3>",
        r'<h3 class="travel-answer-subsection">\1</h3>',
        html_out,
        flags=re.DOTALL,
    )

    html_out = _enhance_travel_answer_html(html_out)

    return f'<div class="travel-answer">{html_out}</div>'


def message_to_html(msg: Any) -> str:
    try:
        if not isinstance(msg, dict):
            return f"<p>{format_inline_markdown(str(msg)).replace(chr(10), '<br>')}</p>"

        content = msg.get("content", "")
        if msg.get("role") == "assistant":
            return markdown_to_html(str(content))
        return f"<p>{format_inline_markdown(str(content)).replace(chr(10), '<br>')}</p>"
    except Exception as error:
        raw_content = msg.get("content", msg) if isinstance(msg, dict) else msg
        return (
            f"<p>{esc(raw_content).replace(chr(10), '<br>')}</p>"
            f'<div class="answer-muted">Display formatting fallback: {esc(error)}</div>'
        )


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default
