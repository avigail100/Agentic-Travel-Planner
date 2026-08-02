"""Trip card rendering helpers for the Streamlit UI."""

from typing import List, Optional

import streamlit as st

from plan_and_execute_agent import get_session_preferences
from streamlit_ui.formatting import esc, star_icons


def card_html(title: str, lines: List[str], note: str = "") -> str:
    body = "<br>".join(lines)
    note_html = f'<div class="small">{esc(note)}</div>' if note else ""
    return f"""
<div class="travel-card">
  <h3>{title}</h3>
  {body}
  {note_html}
</div>
"""


def render_card_grid(items: List[dict], renderer, columns: int = 3) -> None:
    if not items:
        return
    # columns is controlled by the parent area. The right rail passes 1, while
    # wider desktop sections can pass 2 or 3.
    safe_columns = max(1, min(int(columns or 1), len(items)))
    cols = st.columns(safe_columns, gap="small")
    for idx, item in enumerate(items):
        with cols[idx % safe_columns]:
            st.markdown(renderer(item), unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="card-section-title">{title}</div>', unsafe_allow_html=True)


def render_cards(data: Optional[dict], columns: int = 3, detail_columns: int = 2) -> None:
    if not data:
        st.markdown(
            """
<div class="empty-state">
  <h2>🗂️ Travel cards</h2>
  <p>Flights, hotels, activities, transport, visa, warnings and costs will appear here.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    if data.get("flights"):
        render_section_title("✈️ Flights")
        render_card_grid(data["flights"], lambda f: card_html(f'✈️ {esc(f.get("airline"))}', [f'<b>Flight:</b> {esc(f.get("flight"))}', f'<b>Destination:</b> {esc(f.get("destination"))}', f'<b>Price:</b> ${esc(f.get("price"))}', f'<b>Duration:</b> {esc(f.get("duration"))}', f'<b>Departure:</b> {esc(f.get("departure"))}', f'<b>Arrival:</b> {esc(f.get("arrival"))}']), columns=columns)

    if data.get("hotels"):
        render_section_title("🏨 Places to stay")
        render_card_grid(data["hotels"], lambda h: card_html(f'🏨 {esc(h.get("name"))}', [f'<b>Stars:</b> {star_icons(h.get("stars"))}', f'<b>Price per night:</b> ${esc(h.get("price"))}', f'<b>Rating:</b> {esc(h.get("rating"))}'], ", ".join(h.get("amenities") or [])), columns=columns)

    if data.get("activities"):
        render_section_title("📸 Things to do")
        render_card_grid(data["activities"], lambda a: card_html(f'📸 {esc(a.get("name"))}', [f'<b>Category:</b> {esc(a.get("category"))}', f'<b>Price:</b> ${esc(a.get("price"))}', f'<b>Duration:</b> {esc(a.get("duration"))}'], a.get("suitability") or ""), columns=columns)

    if data.get("restaurants"):
        render_section_title("🍽️ Restaurants")
        render_card_grid(data["restaurants"], lambda r: card_html(f'🍽️ {esc(r.get("name"))}', [f'<b>Cuisine:</b> {esc(r.get("cuisine"))}', f'<b>Price level:</b> {esc(r.get("price_level"))}', f'<b>Rating:</b> {esc(r.get("rating"))}'], r.get("special_features") or ""), columns=columns)

    if data.get("car_rentals"):
        render_section_title("🚗 Car rentals")
        render_card_grid(data["car_rentals"], lambda c: card_html(f'🚗 {esc(c.get("company"))}', [f'<b>Location:</b> {esc(c.get("location"))}', f'<b>Type:</b> {esc(c.get("type"))}', f'<b>Price per day:</b> ${esc(c.get("price"))}', f'<b>Seats:</b> {esc(c.get("seats"))}']), columns=columns)

    detail_map = [("visa", "🛂 Visa"), ("time_difference", "🕒 Time difference"), ("currency_exchange", "💱 Currency"), ("transport_info", "🚇 Transport"), ("seasonal_recommendations", "🌤️ Seasonal notes"), ("warning", "⚠️ Warning"), ("estimated_cost", "💰 Estimated cost"), ("notes", "📝 Notes")]
    detail_items = [(key, title) for key, title in detail_map if data.get(key)]
    if detail_items:
        render_section_title("🧭 Trip details")
        safe_detail_columns = max(1, min(int(detail_columns or 1), len(detail_items)))
        cols = st.columns(safe_detail_columns, gap="small")
        for idx, (key, title) in enumerate(detail_items):
            with cols[idx % safe_detail_columns]:
                st.markdown(card_html(title, [esc(data[key])]), unsafe_allow_html=True)


def render_saved_panel(thread_id: str) -> None:
    try:
        prefs = get_session_preferences(thread_id)
    except Exception:
        prefs = {}

    if not prefs:
        st.markdown("""
<div class="info-card">
  <h3>Currently saved</h3>
  <div class="small">No saved preferences yet.</div>
</div>
""", unsafe_allow_html=True)
        return

    rows = ""
    for key, value in prefs.items():
        rows += f"🔹 <b>{esc(key.replace('_', ' ').title())}</b>: {esc(value)}<br>"

    st.markdown(f"""
<div class="info-card">
  <h3>Currently saved</h3>
  {rows}
</div>
""", unsafe_allow_html=True)
