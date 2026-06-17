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


def format_label_value(text: str, as_row: bool = True) -> Optional[str]:
    """Format only real key:value rows.

    Important:
    Do NOT treat regular sentences that contain times like 07:15 as labels.
    The previous parser split sentences such as:
    "British Airways ... Departs at 07:15"
    into a fake label/value pair. This guard keeps the output readable.
    """
    cleaned = normalize_label_value_text(text)
    match = re.match(r"^\**([^:*]{2,54}?)\**:\s*(.+)$", cleaned)
    if not match:
        return None

    label, value = match.groups()
    label = label.strip().strip("*").strip()
    value = value.strip().strip("*").strip()
    label_low = label.lower()

    if not label or not value:
        return None

    # Reject sentence fragments and time-like false positives.
    if any(ch in label for ch in ".!?;,"):
        return None
    if re.search(r"\b(?:at|by|from|to|depart|arriv|flight)\b", label_low) and label_low not in DETAIL_LABELS:
        return None
    if len(label.split()) > 5 and label_low not in DETAIL_LABELS:
        return None

    # Prefer known labels. Allow short clean labels for generic summaries.
    if label_low not in DETAIL_LABELS and not re.match(r"^[A-Za-z][A-Za-z /&()-]{1,32}$", label):
        return None

    formatted_value = star_icons(value) if label_low == "stars" else format_inline_markdown(value)
    formatted = (
        f'<span class="answer-label">{format_inline_markdown(label)}:</span> '
        f"{formatted_value}"
    )
    if not as_row:
        return formatted
    return f'<span class="answer-row">{formatted}</span>'




def format_bullet_text(item_text: str) -> str:
    """Format one bullet without over-parsing it.

    Keeps the answer as readable text, but improves common travel structures:
    - Luxury Option: The Savoy
    - Stars: 5 / ⭐⭐⭐⭐⭐
    - 5-star hotel
    """
    clean = normalize_label_value_text(item_text).strip()

    # Option headers inside a bullet: make them prominent, not a weird accordion/tab.
    option_match = re.match(
        r"^([^:]{2,50}?(?:option|hotel|inn|flight|restaurant|rental|choice)[^:]*):\s*(.+)$",
        clean,
        flags=re.IGNORECASE,
    )
    if option_match:
        label, title = option_match.groups()
        if label.strip().lower() not in DETAIL_LABELS:
            return (
                '<span class="answer-option-line">'
                f'<span class="answer-label">{format_inline_markdown(label.strip())}:</span> '
                f'{format_inline_markdown(title.strip())}'
                '</span>'
            )

    # e.g. "5-star hotel" / "3 star hotel"
    star_hotel_match = re.match(r"^(\d+(?:\.\d+)?)\s*-?\s*stars?\s+(?:hotel|accommodation|option)$", clean, flags=re.IGNORECASE)
    if star_hotel_match:
        return f'<span class="answer-label">Stars:</span> {star_icons(star_hotel_match.group(1))}'

    label_row = format_label_value(clean, as_row=False)
    if label_row:
        return label_row

    return format_inline_markdown(clean)

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
}


def is_option_section(section_title: str) -> bool:
    low = section_title.lower()
    if any(marker in low for marker in ("estimate", "estimated", "summary", "cost", "budget")):
        return False
    return any(
        marker in low
        for marker in (
            "accommodation",
            "hotel",
            "stay",
            "thing",
            "activity",
            "tour",
            "dining",
            "restaurant",
            "food",
            "getting",
            "transport",
            "car",
            "rental",
            "flight",
        )
    )


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
            "notes",
            "summary",
        )
    )


def parse_option_line(text: str, section_title: str = "") -> Optional[Tuple[str, str]]:
    cleaned = normalize_label_value_text(text)
    match = re.match(
        r"^\**([^:*]*(?:option|hotel|activity|restaurant|flight|tour|choice|pick|rental)[^:*]*)\**:\s*(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        generic_match = re.match(r"^\**([^:*]{2,64}?)\**:\s*(.+)$", cleaned)
        bare_option_match = re.match(r"^\**([^:*]{2,64}?)\**:\s*$", cleaned)
        if not is_option_section(section_title):
            return None
        if generic_match:
            label, title = generic_match.groups()
        elif bare_option_match:
            label = bare_option_match.group(1)
            title = ""
        else:
            return None
        label = label.strip().strip("*").strip()
        title = title.strip().strip("*").strip()
        if label.lower() in DETAIL_LABELS:
            return None
        return label, title

    label, title = match.groups()
    return label.strip().strip("*").strip(), title.strip().strip("*").strip()


def looks_compact_option_details(title: str) -> bool:
    low = title.lower()
    return "," in title or any(
        marker in low
        for marker in (
            "departing",
            "arriving",
            "stars",
            "rating",
            "amenities",
            "breakfast",
            "automatic",
            "manual",
            "seats",
            "/day",
            "/night",
        )
    )


def split_compact_details(title: str) -> List[str]:
    clean = title.strip().rstrip(".")
    if not clean:
        return []

    flight_match = re.match(
        r"departing\s+at\s+(.+?),\s*arriving\s+at\s+(.+?),\s*for\s+(.+)$",
        clean,
        flags=re.IGNORECASE,
    )
    if flight_match:
        dep, arr, price = flight_match.groups()
        return [
            f'<span class="answer-label">Departure:</span> {format_inline_markdown(dep.strip())}',
            f'<span class="answer-label">Arrival:</span> {format_inline_markdown(arr.strip())}',
            f'<span class="answer-label">Price:</span> {format_inline_markdown(price.strip())}',
        ]

    details: List[str] = []
    for part in [chunk.strip() for chunk in clean.split(",") if chunk.strip()]:
        star_match = re.match(r"(\d+(?:\.\d+)?)\s+stars?$", part, flags=re.IGNORECASE)
        rating_match = re.match(r"rating\s+(.+)$", part, flags=re.IGNORECASE)
        amenities_match = re.match(r"amenities\s+include\s+(.+)$", part, flags=re.IGNORECASE)
        breakfast_match = re.match(r"(breakfast\s+.+)$", part, flags=re.IGNORECASE)

        if star_match:
            details.append(f'<span class="answer-label">Stars:</span> {star_icons(star_match.group(1))}')
        elif rating_match:
            details.append(f'<span class="answer-label">Rating:</span> {format_inline_markdown(rating_match.group(1).strip())}')
        elif amenities_match:
            details.append(f'<span class="answer-label">Amenities:</span> {format_inline_markdown(amenities_match.group(1).strip())}')
        elif breakfast_match:
            details.append(f'<span class="answer-label">Breakfast:</span> {format_inline_markdown(breakfast_match.group(1).strip())}')
        else:
            details.append(format_inline_markdown(part))
    return details


def format_compact_option_tab(label: str, title: str) -> str:
    details = split_compact_details(title)
    if not details:
        return format_option_tab(label, "", [])
    return format_option_tab(label, "", details)


def format_option_tab(label: str, title: str, details: List[str]) -> str:
    detail_items = "".join(f"<li>{detail}</li>" for detail in details)
    details_html = f'<ul class="answer-option-detail-list">{detail_items}</ul>' if detail_items else ""
    title_html = f": {format_inline_markdown(title)}" if title else ""
    return (
        '<li class="answer-option-tab">'
        '<details open>'
        f'<summary><span class="answer-label">{format_inline_markdown(label)}</span>{title_html}</summary>'
        f'{details_html}'
        '</details>'
        '</li>'
    )


def section_icon(title: str) -> str:
    low = title.lower()
    if "flight" in low:
        return "✈️"
    if "accommodation" in low or "hotel" in low or "stay" in low:
        return "🛏️"
    if "thing" in low or "activity" in low or "tour" in low:
        return "📸"
    if "dining" in low or "restaurant" in low or "food" in low:
        return "🍴"
    if "getting" in low or "transport" in low or "car" in low:
        return "🚗"
    if "essential" in low or "visa" in low or "currency" in low:
        return "🧳"
    if "weather" in low:
        return "☁️"
    if "best time" in low:
        return "☀️"
    if "cost" in low or "budget" in low:
        return "💰"
    return "◆"


def format_section_heading(title: str) -> str:
    icon = section_icon(title)
    return (
        '<div class="answer-subheading">'
        f'<span class="answer-section-icon">{icon}</span>'
        f'<span>{format_inline_markdown(title)}</span>'
        '</div>'
    )


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
        "cost": "💰",
        "budget": "💰",
        "recommendation": "✨",
        "bond recommendation": "✨",
        "summary": "✨",
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
        return f"{chosen} {clean}" if chosen else clean

    def clean_time(value: str) -> str:
        return re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", value.strip())

    def split_detail_parts(value: str) -> List[str]:
        value = re.sub(r"\.\s+", ", ", value.strip().rstrip("."))
        raw_parts = [part.strip() for part in value.split(",") if part.strip()]
        parts: List[str] = []
        idx = 0
        while idx < len(raw_parts):
            part = raw_parts[idx]
            if re.match(r"^amenities\s+(?:include|includes)\s+", part, flags=re.IGNORECASE):
                amenities = [part]
                idx += 1
                while idx < len(raw_parts):
                    next_part = raw_parts[idx]
                    if re.match(
                        r"^(?:rating\s+|breakfast\b|[1-5](?:\.\d+)?\s+stars?$|[$£€₪]|\d+\s+seats?$|automatic$|manual$)",
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

        star_match = re.match(r"^([1-5](?:\.\d+)?)\s+stars?$", clean, flags=re.IGNORECASE)
        if star_match:
            return f"  - **Stars:** {normalize_stars(star_match.group(1))}"

        rating_match = re.match(r"^rating\s+(.+)$", clean, flags=re.IGNORECASE)
        if rating_match:
            return f"  - **Rating:** {rating_match.group(1).strip()}"

        amenities_match = re.match(r"^amenities\s+(?:include|includes)\s+(.+)$", clean, flags=re.IGNORECASE)
        if amenities_match:
            return f"  - **Amenities:** {amenities_match.group(1).strip()}"

        price_match = re.match(r"^([$£€₪]\s?[\d,]+(?:\.\d+)?(?:\s*/\s?(?:night|day))?)$", clean)
        if price_match:
            return f"  - **Price:** {price_match.group(1).replace(' /', '/')}"

        seats_match = re.match(r"^(\d+\s+seats?)$", clean, flags=re.IGNORECASE)
        if seats_match:
            return f"  - **Seats:** {seats_match.group(1)}"

        if low in {"automatic", "manual"}:
            return f"  - **Transmission:** {clean}"

        if low.startswith("breakfast"):
            return f"  - **Breakfast:** {clean}"

        if low.startswith("price"):
            return f"  - **Price:** {clean.split(':', 1)[-1].strip()}"

        return f"  - {clean}"

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
                f"{indent}  - **Departure:** {clean_time(departure)}",
                f"{indent}  - **Arrival:** {clean_time(arrival)}",
                f"{indent}  - **Price:** {price.strip()}",
            ]

        named_detail = re.match(r"^([^:]{2,80}?):\s*(.+)$", body)
        if not named_detail:
            return None

        name, details_text = named_detail.groups()
        name = name.strip().strip("*")
        details_text = details_text.strip()

        if name.lower() in {
            "price",
            "rating",
            "stars",
            "amenities",
            "duration",
            "departure",
            "arrival",
            "cuisine",
            "note",
        }:
            return None

        detail_parts = split_detail_parts(details_text)
        if len(detail_parts) < 2:
            return None

        return [f"{indent}- **{name}**"] + [indent + compact_detail_row(part) for part in detail_parts]

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

        # Avoid duplicating immediate "### Your London Travel Plan".
        if in_console_title and stripped.startswith("#"):
            in_console_title = False
            continue
        if stripped:
            in_console_title = False

        # Normalize common bullet markers, preserving indentation.
        left_stripped = line.lstrip()
        indent = line[: len(line) - len(left_stripped)]
        for marker in ("*   ", "* ", "• "):
            if left_stripped.startswith(marker):
                line = indent + "- " + left_stripped[len(marker):]
                stripped = line.strip()
                break

        # Add emojis to Markdown headings only.
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            hashes, title = heading_match.groups()
            line = f"{hashes} {add_heading_icon(title)}"
            normalized.append(line)
            continue

        # If the model writes a bold-only section title, make it look like a heading.
        bold_title = re.match(r"^\*\*([^:*]{3,80}):?\*\*\s*$", stripped)
        if bold_title:
            title = add_heading_icon(bold_title.group(1).strip())
            normalized.append(f"### {title}")
            continue

        # Convert plain travel section labels to real Markdown headings.
        # This is intentionally conservative: it only applies to non-bullet lines
        # ending with ':' whose text matches one of our travel section keywords.
        # It avoids detail rows like "Rating: 9.6" or "Price: $510".
        plain_title = re.match(r"^([^:\n]{3,100}):\s*$", stripped)
        if plain_title and not stripped.startswith(("-", "*", "•")):
            raw_title = plain_title.group(1).strip()
            lowered_title = raw_title.lower()
            if any(key in lowered_title for key in heading_icons):
                normalized.append(f"### {add_heading_icon(raw_title)}")
                continue

        # Normalize "Stars: 5", "**Stars:** 5", etc.
        star_line = re.match(r"^(\s*(?:-\s*)?)(?:\*\*)?Stars(?:\*\*)?:\s*(.+)$", line, flags=re.IGNORECASE)
        if star_line:
            prefix, value = star_line.groups()
            line = f"{prefix}**Stars:** {normalize_stars(value)}"
            normalized.append(line)
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
        "⭐": "★☆☆☆☆",
    }
    for raw, pretty in star_patterns.items():
        html_out = html_out.replace(raw, f'<span class="answer-stars">{pretty}</span>')

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
