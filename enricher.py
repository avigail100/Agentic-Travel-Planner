"""
Web enrichment for already-selected travel items (The Last Mile).

This module runs AFTER the agent has selected concrete items from the database
and the Critic has approved the answer.  It uses Tavily to fetch a real-world
URL and short highlights summary for every bookable item the agent found, then
appends an "Explore Further" block to the agent's response text.

Design constraints enforced here:
- Never invoked by the LLM planner — called deterministically as a graph node.
- Capped at MAX_ENRICH_ITEMS Tavily calls per query to control cost.
  The cap is sized to cover a full-trip query (flights + hotels + activities).
- Covers flights, hotels, activities, and restaurants — ALL returned options,
  not a single selected one, so the user can follow a link for any listing.
- Fails silently on any network or API error — enrichment is supplemental
  and must never break the main agent flow.
- Does not feed back into LLM reasoning; result is appended to the final
  response text only, after Critic approval.
"""

import json
import os
import re

import requests
from langchain_core.messages import ToolMessage

# Cap sized to cover a full-trip query (≈3 flights + 3 hotels + 3 activities + 3 restaurants).
# All options in every category get a link; the cap prevents runaway cost if the
# DB unexpectedly returns a very large result set.
MAX_ENRICH_ITEMS = 12

# Maps tool names to (category_label, name_field_in_result_row).
_ENRICHABLE_TOOLS: dict[str, tuple[str, str]] = {
    "fetch_flights":            ("flight",     "airline"),
    "find_connecting_flights":  ("flight",     "airline_1"),   # primary leg airline
    "fetch_hotels":             ("hotel",      "name"),
    "find_hotels_by_amenity":   ("hotel",      "name"),
    "fetch_activities":         ("activity",   "name"),
    "fetch_restaurants":        ("restaurant", "name"),
}

# Review / booking aggregator domains — we prefer the item's own site over these.
_AGGREGATOR_DOMAINS = frozenset({
    "tripadvisor.com",
    "booking.com",
    "expedia.com",
    "hotels.com",
    "yelp.com",
    "google.com",
    "maps.google.com",
    "trustpilot.com",
    "agoda.com",
    "airbnb.com",
    "lonelyplanet.com",
    "kayak.com",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _destination_from_state(state: dict) -> str:
    """Extract the first destination city mentioned in past_steps (best-effort)."""
    for step in state.get("past_steps", []):
        m = re.search(
            r"(?:city|destination)['\"]?\s*[:=]\s*['\"]?([A-Za-z ]+)",
            step,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
    return ""


def _collect_candidates(messages: list) -> list[tuple[str, str]]:
    """
    Walk the session's ToolMessages and return (category, name) pairs for
    every enrichable item found, in order of first appearance.

    All options from every category are included — flights, hotels, activities,
    and restaurants — so that every listing the agent presented to the user can
    receive an "Explore Further" link.  Items are deduplicated by name so the
    same entry is never fetched twice even if it appears in multiple tool calls.
    """
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = getattr(msg, "name", "")
        if tool_name not in _ENRICHABLE_TOOLS:
            continue

        category, name_key = _ENRICHABLE_TOOLS[tool_name]
        try:
            rows = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except Exception:
            continue
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get(name_key) or "").strip()
            if name and name not in seen:
                seen.add(name)
                candidates.append((category, name))

    return candidates


def _pick_best_url(results: list) -> str:
    """
    Choose the most official-looking URL from Tavily results.
    Prefers the item's own domain over review/booking aggregators.
    Falls back to the first result if all are aggregators.
    """
    for r in results:
        raw_url = (r.get("url") or "").lower()
        hostname = raw_url.replace("https://", "").replace("http://", "").split("/")[0]
        is_aggregator = any(
            hostname == d or hostname.endswith("." + d)
            for d in _AGGREGATOR_DOMAINS
        )
        if not is_aggregator:
            return r.get("url", "")

    return results[0].get("url", "") if results else ""


def _enrich_one(name: str, category: str, destination: str) -> dict:
    """
    Single Tavily call for a named DB item.

    Uses include_answer=True so Tavily synthesizes a concise, meaningful summary
    from its search results rather than returning raw page fragments.  The query
    is crafted per category to surface both the official site URL (for booking)
    and genuinely useful live information (what the place is known for, ratings,
    visitor tips, etc.).

    Returns {"url": str, "highlights": str} on success,
    or {} if Tavily is unavailable, the key is missing, or any error occurs.
    All exceptions are swallowed — this function must never raise.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {}

    location_suffix = f" {destination}" if destination else ""

    # Each query is written to elicit a useful synthesized answer AND surface
    # the official website among the top results (for URL extraction).
    if category == "flight":
        query = f"{name} airline official site destinations routes overview"
    elif category == "hotel":
        query = f"{name} hotel{location_suffix} official site highlights amenities guest reviews"
    elif category == "activity":
        query = f"{name}{location_suffix} official site visitor info highlights tickets"
    else:  # restaurant
        query = f"{name}{location_suffix} restaurant official site cuisine highlights reviews"

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,   # ask Tavily to synthesize a meaningful summary
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
    except Exception:
        return {}

    if not results:
        return {}

    url = _pick_best_url(results)

    # Prefer Tavily's synthesized answer — it is coherent and informative.
    # Fall back to joining raw snippets only when the answer field is absent.
    highlights = str(data.get("answer") or "").strip()
    if not highlights:
        snippets = [
            str(r.get("content") or "").strip()[:200]
            for r in results[:2]
            if r.get("content")
        ]
        highlights = " | ".join(s for s in snippets if s)

    return {"url": url, "highlights": highlights[:400]}


# ---------------------------------------------------------------------------
# Public entry point — called by the enricher_node in plan_and_execute_agent
# ---------------------------------------------------------------------------

def enrich_items(state: dict) -> dict:
    """
    Enrich every bookable item the agent found (flights, hotels, activities,
    restaurants) with a Tavily-sourced URL and highlights, and return a state delta:

        {"enrichments": {"<category>:<name>": {"url": ..., "highlights": ...}}}

    All options in every category receive a link so the user can follow any
    listing they are interested in, not just the one the agent highlighted.
    Items with no Tavily result are omitted.  An empty dict is returned when
    Tavily is unavailable or no enrichable items exist in the session.
    """
    candidates = _collect_candidates(state.get("messages", []))
    if not candidates:
        return {"enrichments": {}}

    destination = _destination_from_state(state)

    # Hard cap to control cost if the DB returns an unusually large result set.
    to_enrich = candidates[:MAX_ENRICH_ITEMS]

    enrichments: dict = {}
    for category, name in to_enrich:
        result = _enrich_one(name, category, destination)
        if result:
            enrichments[f"{category}:{name}"] = result

    if enrichments:
        print(f"[Enricher] Enriched {len(enrichments)} item(s): {list(enrichments.keys())}")
    else:
        print("[Enricher] No enrichment data returned (Tavily unavailable or no results).")

    return {"enrichments": enrichments}
