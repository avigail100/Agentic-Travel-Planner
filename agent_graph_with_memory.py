#!/usr/bin/env python3
"""
Travel Agent with SQLite Memory — Session 4
- SqliteSaver: full cross-session persistence via checkpoints.db
- save_preference tool: LLM explicitly saves user preferences to state
- preference_persist_node: reads save_preference results and merges into state
- personalization_node: formats stored prefs and injects into the system prompt
- should_personalize: conditional edge — skips personalization when no prefs exist
"""

import json
import re
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


import os

from tools import (
    fetch_flights, fetch_hotels, find_hotels_by_amenity, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, find_destinations_by_preference, convert_time_to_destination_timezone,
    lookup_location_options, find_connecting_flights, save_preference,
    fetch_restaurants, fetch_beaches, fetch_city_transport_info

)

load_dotenv()

MAX_STEPS = 10
MAX_REPLANS = 4
VERBOSE_DEBUG = False
# Each turn can run up to: intent(1) + personalization(1) + agent(MAX_STEPS) +
# tools+persist(MAX_STEPS*2) + replanner loops + validator/cost/budget/formatter.
RECURSION_LIMIT = MAX_STEPS * 6

SYSTEM_PROMPT = """You are a proactive travel planning assistant. Use the available tools to fetch real data and answer the user directly.

Rules:
- CHIT-CHAT & INTRODUCTIONS: If the user is just greeting, introducing themselves, or making small talk, 
  YOU MUST NOT CALL ANY TOOLS. Just say hello, use their name, acknowledge their location, and ask how you can help them plan a trip.
ANY request mentioning a place, food, activity, transport, or travel info 
  MUST call the relevant tool — even if it seems casual.
- TRIP PLANNING: ONLY when the user explicitly asks to travel, book, or search for travel information, you should call tools.
- When you DO plan a trip, the very first tool you call MUST be lookup_location_options to verify the destination, 
  and use the verified location for the subsequent tools immediately and explain the substitution.
- Step-by-Step Logic for the lookup_location_options tool:
    1. Use lookup_location_options to see if the requested cities/countries exist or have synonyms in our database.
    2. If you find a logical match or geographic match to the requested location in our DB (e.g., 'Israel' -> 'TLV', 'Liverpool' -> 'London'), 
       you MUST silently substitute it and IMMEDIATELY call the target tools with the substitute. Do NOT ask the user for permission. 
       Just explain the substitution naturally in your final response (e.g., "Since X isn't available, I searched flights from nearby B to C").
    3. IF NO LOGICAL MATCH IS FOUND, do not guess. Clearly state that the specific destination is unavailable.
- If fetch_flights returns no direct flights, IMMEDIATELY call find_connecting_flights to search for a flight with a layover. 
  Dont ask the user for permission, just do it and explain in your final response.
- When a trip requires both flight and hotel data, call fetch_flights AND fetch_hotels in the
  same response so they run in parallel — do not wait for one before calling the other.
- Once you have tool results, give a direct answer. Do not ask follow-up questions if you
  already have enough data to respond.
- If a city or route is not found in the database, say so clearly instead of retrying.
- If a complex request (e.g., flight, hotel, car rental, and activities) is only partially fulfillable
  due to missing data, proceed with the available information and note what could not be found.
- Flights are the only mandatory component. All other items are optional.
- When the user mentions any travel preference (airline, food, travel style, seat type, or cabin
  class), call save_preference(key, value) immediately — before answering — to store it for
  future sessions. Use keys: preferred_airline, food_preference, travel_style, seat_preference,
  class_preference.
"""

BANNER = r"""
____   ____  _   _  ____
| __ ) / __ \| \ | ||  _ \
|  _ \| |  | |  \| || | | |
| |_) | |__| | |\  || |_| |
|____/ \____/|_| \_||____/
"""

            # Progress message map based on node names in your graph

PROGRESS_MAP = {
        "intent_classifier": "⚙️  Decoding travel intent...",
        "context_extractor": "🔍  Extracting budget and context settings...",
        "personalization":   "✨  Personalizing based on saved preferences...",
        "planner":           "📝  Planning the full task before executing...",
        "executor_context":  "▶️  Executing the next planned step...",
        "tools":             "🧳  Accessing database and searching real-time information...",
        "alternatives_injector": "💡  Direct destination not found, searching for suitable alternatives...",
        "replanner":         "🔁  Reviewing progress and updating the remaining plan...",
        "cost_calc":         "🧮  Calculating costs and summarizing data...",
        "budget_check":      "⚖️  Checking compliance with your budget...",
}

# ============================================================================
# 0. Plan-and-Execute Schemas
# ============================================================================
class Plan(BaseModel):
    """A high-level plan consisting of sequential, non-repeating steps."""

    steps: list[str] = Field(
        description="Ordered execution steps. Each step should be concrete and non-redundant."
    )


class ReplanDecision(BaseModel):
    """The replanner decision after observing execution results."""

    response: str = Field(
        default="",
        description="Final user-facing answer if the original request is complete. Empty if more steps are needed.",
    )
    remaining_steps: list[str] = Field(
        default_factory=list,
        description="Updated remaining plan if more work is needed. Do not include completed steps.",
    )
    rationale: str = Field(
        default="",
        description="Short developer-facing reason for continuing, updating, or finishing.",
    )

# ============================================================================
# 1. Shared State
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    current_city: str        # extracted per-turn from user message
    total_budget: float      # extracted per-turn from user message
    calculated_total: float
    over_budget: bool
    intent: str
    missing_data: list
    user_preferences: dict   # persisted across sessions via SqliteSaver
    personalization_context: str
    plan: list[str]          # remaining plan-and-execute steps for this turn
    completed_steps: list[str]
    current_plan_step: str
    executor_instruction: str
    response: str
    replan_count: int
    awaiting_origin_for_destination: str

# ============================================================================
# 2. Model and Tools
# ============================================================================

tools = [
    fetch_flights, fetch_hotels,find_hotels_by_amenity, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, find_destinations_by_preference, convert_time_to_destination_timezone,
    lookup_location_options, find_connecting_flights, save_preference,
    fetch_restaurants, fetch_beaches, fetch_city_transport_info
]

base_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0, max_retries=2
)
planner_model = base_model.with_structured_output(Plan)
replanner_model = base_model.with_structured_output(ReplanDecision)
model = base_model.bind_tools(tools)


# model = ChatGroq(
#   api_key=os.getenv("GROQ_API_KEY"),
#   model="llama-3.3-70b-versatile",
#   temperature=0,
#   max_retries=2
# ).bind_tools(tools)

# ============================================================================
# 3. Helpers
# ============================================================================

def _current_turn_tool_messages(state: AgentState):
    """Returns all ToolMessages from the current turn (since the last HumanMessage)."""
    messages = state["messages"]
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].__class__.__name__ == "HumanMessage":
            cut = i
            break
    return [m for m in messages[cut:] if m.__class__.__name__ == "ToolMessage"]


def _current_turn_messages(state: AgentState):
    """Returns all messages from the current user turn."""
    messages = state["messages"]
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].__class__.__name__ == "HumanMessage":
            cut = i
            break
    return messages[cut:]


def _last_human_content(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "HumanMessage":
            return str(msg.content)
    return ""


def _dedupe_steps(steps: list[str]) -> list[str]:
    """Keeps planner/replanner output compact and prevents repeated steps."""
    clean = []
    seen = set()
    for step in steps or []:
        normalized = re.sub(r"\s+", " ", str(step)).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(normalized)
    return clean


def _format_steps(steps: list[str]) -> str:
    if not steps:
        return "  (no remaining steps)"
    return "\n".join(f"  {idx}. {step}" for idx, step in enumerate(steps, start=1))


def _safe_tool_call_signature(tool_call: dict) -> str:
    name = tool_call.get("name", "unknown_tool")
    args = tool_call.get("args", {})
    try:
        args_text = json.dumps(args, sort_keys=True)
    except TypeError:
        args_text = str(args)
    return f"{name}({args_text})"


def _current_turn_tool_call_signatures(state: AgentState) -> list[str]:
    signatures = []
    for msg in _current_turn_messages(state):
        for tool_call in getattr(msg, "tool_calls", None) or []:
            signatures.append(_safe_tool_call_signature(tool_call))
    return signatures


def _flight_request_missing_origin(text: str) -> bool:
    """Detects flight searches that cannot safely call fetch_flights yet."""
    lowered = text.lower()
    asks_for_flight = any(word in lowered for word in ("flight", "flights", "fly"))
    has_destination = bool(re.search(r"\b(to|for)\s+[A-Za-z][A-Za-z\s-]{1,40}", text, re.IGNORECASE))
    has_origin = bool(re.search(r"\bfrom\s+[A-Za-z][A-Za-z\s-]{1,40}", text, re.IGNORECASE))
    return asks_for_flight and has_destination and not has_origin


def _extract_requested_flight_destination(text: str) -> str:
    """Best-effort destination extraction for simple flight requests."""
    match = re.search(r"\b(?:to|for)\s+([A-Za-z][A-Za-z\s-]{1,40})", text, re.IGNORECASE)
    if not match:
        return ""
    destination = re.sub(r"\b(?:please|thanks|thank you)\b", "", match.group(1), flags=re.IGNORECASE)
    return destination.strip(" .?!,").title()


def _looks_like_origin_answer(text: str) -> bool:
    """Detects short follow-up answers like TLV, Tel Aviv, London."""
    clean = text.strip()
    if not clean or len(clean) > 40:
        return False
    if re.search(r"\b(flight|hotel|trip|plan|travel|visit|book|visa|currency)\b", clean, re.IGNORECASE):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z\s-]{1,39}", clean))


def _is_quota_error(error: Exception) -> bool:
    err_str = str(error)
    return (
        "RESOURCE_EXHAUSTED" in err_str
        or "429" in err_str
        or "quota" in err_str.lower()
    )


# ============================================================================
# 4. Nodes
# ============================================================================

def context_extractor_node(state: AgentState):
    """Extracts budget from the user message (per-turn)."""
    text = state["messages"][-1].content
    updates = {}
    # Matches: $1200, 1200$, 1200 dollars, 1200 Dollars, 1200 dollar
    # budget_match = re.search(r"\$\s?([\d,]+)|([\d,]+)\s?\$", text)
    budget_match = re.search(r"\$\s?([\d,]+)|([\d,]+)\s?(?:\$|dollars?)", text, re.IGNORECASE)
    
    if budget_match:
        # Take the first non-None group, remove commas, and convert to float
        budget_val = budget_match.group(1) or budget_match.group(2)
        updates["total_budget"] = float(budget_val.replace(",", ""))
    return updates


def intent_classifier_node(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    keywords = {
        "visa":      ["visa", "passport", "entry requirement"],
        "currency":  ["exchange rate", "currency", "convert"],
        "full_trip": ["flight", "hotel", "trip", "plan", "travel", "visit", "book"],
    }
    for intent, words in keywords.items():
        if any(w in last_msg for w in words):
            return {"intent": intent}
    return {"intent": "general"}


def preference_persist_node(state: AgentState):
    """Reads save_preference tool results from the current turn and merges into state."""
    existing = state.get("user_preferences") or {}
    updated = dict(existing)

    for msg in _current_turn_tool_messages(state):
        if getattr(msg, "name", None) == "save_preference" and msg.content.startswith("saved:"):
            k, v = msg.content[6:].split("=", 1)
            updated[k] = v
            print(f"\n[Memory] Saved preference: {k}={v}")

    if updated == existing:
        return {}
    return {"user_preferences": updated}


def alternatives_injector_node(state: AgentState):
    """
    Injects a structured system message when no results were found — either because
    the destination doesn't exist (lookup NO_MATCH) or because it exists but has no
    inventory for this specific route/city (fetch failure).
    Routes back to call_model so all normal agent logic (step limits, errors) still applies.
    """
    no_matches = _extract_no_matches(state)

    prefs = state.get("user_preferences") or {}
    pref_block = (
        "User's saved preferences:\n" +
        "\n".join(f"  - {k.replace('_', ' ').title()}: {v}" for k, v in prefs.items())
        if prefs else
        "No saved preferences on file — use geographic and contextual reasoning only."
    )

    items_block = ""
    for nm in no_matches:
        avail = nm.get("available_locations") or []
        avail_str = f"Available options in DB: {avail}" if avail else ""
        items_block += f"\n- {nm['search_term']}. {avail_str}"

    if not items_block:
        items_block = "\n- (see tool results for details)"

    instruction = f"""[SYSTEM — NO RESULTS FOUND, SUGGEST ALTERNATIVES]
The user's request returned no results for one or more items:
{items_block}

{pref_block}

Instructions:
1. INTENT CHECK: If the user is simply introducing themselves or making small talk, DO NOT suggest alternatives.
   Just politely greet them back, acknowledge their location/name, and ask how you can help. Ignore the rest of these instructions.
1. Clearly tell the user that the exact request could not be fulfilled (no matching flights/hotels/activities).
2. Proactively suggest 2–4 relevant alternatives WITHOUT waiting to be asked.
   For each suggestion explain in one sentence WHY it was chosen:
   - Same geographic region or continent
   - Similar climate or travel season
   - Similar luxury/budget level (infer from the request or preferences)
   - Similar travel style (beach, culture, adventure, city, nature…)
   - Alignment with saved user preferences
   - For flights: other destinations served from the same origin
3. If multiple items have no results, address each one.
4. End by asking the user if they'd like to plan a trip to any of the suggested alternatives.
5. If user preferences exist (like a preferred airline), call fetch_flights with the origin city to see 
which destinations match that preference before suggesting alternatives.
"""

    return {
        "messages": [SystemMessage(content=instruction)],
    }


def personalization_node(state: AgentState):
    """Formats stored user_preferences into a string injected into the system prompt."""
    prefs = state.get("user_preferences") or {}
    if not prefs:
        return {"personalization_context": ""}
    lines = [f"  - {k.replace('_', ' ').title()}: {v}" for k, v in prefs.items()]
    context = "User's Saved Preferences (apply to all recommendations):\n" + "\n".join(lines)
    return {"personalization_context": context}


def planner_node(state: AgentState):
    """Creates a high-level plan before any execution starts."""
    user_request = _last_human_content(state)
    intent = state.get("intent", "general")
    prefs = state.get("user_preferences") or {}
    pref_lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items()) or "- None"
    pending_destination = state.get("awaiting_origin_for_destination") or ""

    if pending_destination and _looks_like_origin_answer(user_request):
        origin = user_request.strip().upper() if len(user_request.strip()) <= 4 else user_request.strip().title()
        steps = [
            f"Verify {origin} and {pending_destination} as valid flight locations.",
            f"Find available flights from {origin} to {pending_destination}.",
            "Present the flight options clearly to the user.",
        ]
        print("\n[Planner] Created Plan:")
        print(_format_steps(steps))
        return {
            "plan": steps,
            "completed_steps": [],
            "current_plan_step": steps[0],
            "response": "",
            "replan_count": 0,
            "awaiting_origin_for_destination": "",
            "current_city": pending_destination,
        }

    if _flight_request_missing_origin(user_request):
        destination = _extract_requested_flight_destination(user_request)
        response = "Sure — what city or airport would you like to fly from?"
        print("\n[Planner] Created Plan:")
        print("  1. Ask the user for the missing origin city or airport.")
        return {
            "messages": [AIMessage(content=response)],
            "plan": [],
            "completed_steps": [],
            "current_plan_step": "",
            "response": response,
            "replan_count": 0,
            "awaiting_origin_for_destination": destination,
            "current_city": "",
        }

    prompt = f"""
You are the Planner in a Plan-and-Execute travel agent.

Create a short, ordered plan for the Executor. The Executor has access to travel
database tools and must execute the steps one at a time.

User request:
{user_request}

Classified intent: {intent}
Saved user preferences:
{pref_lines}

Planning rules:
- Keep the plan focused: 1-6 steps.
- Do not repeat steps.
- If travel data is needed, include a location lookup/verification step before fetch steps.
- If the user asks for flights but does not provide an origin city/airport, create exactly
  one step: ask the user for the origin. Do not plan flight lookup/fetch steps yet.
- Group independent fetches into a single step when they can be executed together.
- Include preference saving when the user states airline, food, seat, cabin, or travel-style preferences.
- For small talk or introductions, make one conversational step and do not plan tool usage.
- The Executor will avoid calling the same tool twice with identical arguments.
"""
    try:
        plan = planner_model.invoke(prompt)
        steps = _dedupe_steps(plan.steps)
    except Exception as e:
        if _is_quota_error(e):
            print("[Planner] Model quota exceeded while planning; using fallback plan.")
        else:
            print(f"[Planner] Structured planning failed, using fallback plan: {e}")
        if intent == "general":
            steps = ["Respond conversationally without calling tools."]
        else:
            steps = [
                "Verify relevant locations using lookup_location_options.",
                "Fetch the requested travel data with the available tools.",
                "Summarize the results clearly and mention missing data or alternatives.",
            ]

    if not steps:
        steps = ["Answer the user directly based on the request."]

    print("\n[Planner] Created Plan:")
    print(_format_steps(steps))

    return {
        "plan": steps,
        "completed_steps": [],
        "current_plan_step": steps[0],
        "response": "",
        "replan_count": 0,
        "awaiting_origin_for_destination": "",
    }


def executor_context_node(state: AgentState):
    """Prepares the current plan step for the executor without bloating history."""
    plan = state.get("plan") or []
    current_step = plan[0] if plan else "Finish the response."
    completed = state.get("completed_steps") or []
    prior_tool_calls = _current_turn_tool_call_signatures(state)

    instruction = f"""[SYSTEM - PLAN EXECUTOR]
Original user request:
{_last_human_content(state)}

Completed plan steps:
{_format_steps(completed)}

Remaining plan:
{_format_steps(plan)}

Current step to execute now:
{current_step}

Execution rules:
1. Focus only on the current step unless a tool result makes the next step obvious.
2. Do not call the same tool twice with the exact same arguments in this turn.
3. Tool calls already made this turn:
{chr(10).join(f"  - {sig}" for sig in prior_tool_calls) if prior_tool_calls else "  - None"}
4. When a step needs several independent lookups or fetches, call those tools in the same response.
5. Never call fetch_flights without an origin. If the origin is missing, ask the user for it.
6. If a tool result is empty or unavailable, stop and let the replanner adapt the remaining plan.
7. If the current step can be completed without tools, answer clearly and concisely.
"""
    print(f"\n[Executor] Current Step: {current_step}")
    return {
        "executor_instruction": instruction,
        "current_plan_step": current_step,
    }


def replanner_node(state: AgentState):
    """Observes the latest execution result and updates the remaining plan or finishes."""
    current_step = state.get("current_plan_step") or ""
    completed = list(state.get("completed_steps") or [])
    if current_step and current_step not in completed:
        completed.append(current_step)

    remaining_plan = list(state.get("plan") or [])
    if remaining_plan and remaining_plan[0] == current_step:
        remaining_plan = remaining_plan[1:]

    latest_message = state["messages"][-1]
    latest_content = getattr(latest_message, "content", "")
    tool_observations = []
    for msg in _current_turn_tool_messages(state):
        tool_observations.append(f"{getattr(msg, 'name', 'tool')}: {msg.content}")

    prompt = f"""
You are the Replanner in a Plan-and-Execute travel agent.

Original user request:
{_last_human_content(state)}

Completed steps:
{_format_steps(completed)}

Remaining plan before review:
{_format_steps(remaining_plan)}

Latest executor message:
{latest_content}

Tool observations from this turn:
{chr(10).join(tool_observations) if tool_observations else "No tool observations yet."}

Decide what to do next:
- If the user's request is fully answered, set response to the final answer and leave remaining_steps empty.
- If more work is needed, leave response empty and return only the updated remaining_steps.
- Remove completed steps and avoid duplicate steps.
- If only part of the current step was completed, add a precise unfinished follow-up step.
- If a tool returned no data, adapt the remaining plan to suggest alternatives or proceed with available information.
- Do not re-add a step whose required tool call was already made with the same arguments.
"""

    try:
        decision = replanner_model.invoke(prompt)
        response = (decision.response or "").strip()
        next_steps = _dedupe_steps(decision.remaining_steps or remaining_plan)
        rationale = (decision.rationale or "").strip()
    except Exception as e:
        if _is_quota_error(e):
            print("[Replanner] Model quota exceeded while replanning; using remaining plan.")
        else:
            print(f"[Replanner] Structured replanning failed, using fallback decision: {e}")
        response = ""
        next_steps = _dedupe_steps(remaining_plan)
        rationale = "Fallback: continue with the remaining plan."

    replan_count = state.get("replan_count", 0) + 1

    if response:
        print("\n[Replanner] Final response is ready.")
        if rationale:
            print(f"[Replanner] Rationale: {rationale}")
        return {
            "messages": [AIMessage(content=response)],
            "response": response,
            "plan": [],
            "completed_steps": completed,
            "current_plan_step": "",
            "replan_count": replan_count,
        }

    if not next_steps or replan_count >= MAX_REPLANS:
        fallback_response = str(latest_content).strip() or "I completed the available steps for this request."
        if replan_count >= MAX_REPLANS:
            fallback_response += "\n\n[Note] Replanning limit reached, so this is the best available summary."
        print("\n[Replanner] No useful remaining steps; finishing with the latest result.")
        return {
            "messages": [AIMessage(content=fallback_response)],
            "response": fallback_response,
            "plan": [],
            "completed_steps": completed,
            "current_plan_step": "",
            "replan_count": replan_count,
        }

    print("\n[Replanner] Updated Remaining Plan:")
    print(_format_steps(next_steps))
    if rationale:
        print(f"[Replanner] Rationale: {rationale}")

    return {
        "plan": next_steps,
        "completed_steps": completed,
        "current_plan_step": next_steps[0],
        "response": "",
        "replan_count": replan_count,
    }


def call_model(state: AgentState):
    system_content = SYSTEM_PROMPT
    ctx = state.get("personalization_context") or ""
    if ctx:
        system_content += f"\n\n{ctx}"
    executor_instruction = state.get("executor_instruction") or ""
    try:
        messages = [SystemMessage(content=system_content)]
        if executor_instruction:
            messages.append(SystemMessage(content=executor_instruction))
        messages.extend(state["messages"])
        response = model.invoke(messages)
    except Exception as e:
        if _is_quota_error(e):
            err_text = (
                "[ERROR] Google Gemini API quota is exhausted for the configured key. "
                "Further model calls will keep failing until the quota resets or you switch API keys/providers."
            )
        else:
            err_text = f"[ERROR] Something went wrong: {e}"
        return {
            "messages": [AIMessage(content=err_text)],
            "step_count": state.get("step_count", 0) + 1,
        }
    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}

def _extract_no_matches(state: AgentState) -> list[dict]:
    """
    Scans current-turn ToolMessages for two kinds of "no result" signals:

    1. lookup_location_options returned a no_direct_match dict AND the LLM
       decided Case B (genuine mismatch) — detectable because the LLM then
       writes the string "NO_MATCH:<term>" as its *next* AIMessage content
       before calling another tool, OR because the tool result itself is a
       no_direct_match dict with no subsequent successful fetch.

    2. A fetch tool (fetch_flights, fetch_hotels, fetch_activities) returned
       a plain "No X found" error string — meaning the location existed in
       the lookup table but has no actual inventory for this route/city.

    Returns a list of dicts: [{search_term, source, available_locations}, ...]
    """
    no_matches = []
    tool_msgs = _current_turn_tool_messages(state)

    # --- Case B signal: LLM wrote "NO_MATCH:<term>" after lookup returned no_direct_match ---
    for msg in tool_msgs:
        if getattr(msg, "name", None) != "lookup_location_options":
            continue
        content = msg.content
        if isinstance(content, str) and content.startswith("NO_MATCH:"):
            search_term = content[len("NO_MATCH:"):].strip()
            no_matches.append({"search_term": search_term, "source": "lookup", "available_locations": []})
        else:
            try:
                data = json.loads(content) if isinstance(content, str) else content
                if isinstance(data, dict) and data.get("no_direct_match"):
                    # Tool returned no_direct_match; include its available list
                    # so alternatives_injector can reason over it even if the
                    # LLM didn't explicitly write NO_MATCH:<term>
                    no_matches.append({
                        "search_term": data.get("search_term", "requested location"),
                        "source": "lookup",
                        "available_locations": data.get("available_locations", []),
                    })
            except (json.JSONDecodeError, TypeError):
                pass

    # --- Fetch failure: tool returned a "No X found" plain-text error ---
    FETCH_TOOLS = {"fetch_flights", "fetch_hotels", "fetch_activities"}
    NO_RESULT_PHRASES = ("no flights found", "no hotels found", "no activities found")
    for msg in tool_msgs:
        if getattr(msg, "name", None) not in FETCH_TOOLS:
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        if any(phrase in content.lower() for phrase in NO_RESULT_PHRASES):
            no_matches.append({
                "search_term": content,   # e.g. "No flights found from TLV to London."
                "source": "fetch",
                "available_locations": [],
            })

    return no_matches


def step_limit_node(_state: AgentState):
    return {
        "messages": [
            AIMessage(
                content=f"[STEP_LIMIT_REACHED] The agent exceeded the maximum number of "
                        f"steps ({MAX_STEPS}). Generating a partial summary."
            )
        ]
    }


def validator_node(state: AgentState):
    # Extract all tool names called in the current turn
    tool_names_called = {
        m.name for m in _current_turn_tool_messages(state)
    }
    
    intent = state.get("intent", "general")
    missing = []
    
    # Check if the graph is currently in the alternatives injection phase
    is_suggesting_alternatives = any(
        m.__class__.__name__ == "SystemMessage" and "SUGGEST ALTERNATIVES" in str(m.content) 
        for m in state["messages"][-3:]
    )
    
    # Context-Aware Validation Logic
    if intent == "full_trip" and not is_suggesting_alternatives:
        # Check if we have a locked destination city in the state
        current_city = state.get("current_city") or ""
        has_destination = (current_city and current_city != "YOUR DESTINATION")
            
        # Only require flight data if the user has actually provided a destination
        if has_destination and "fetch_flights" not in tool_names_called:
            missing.append("flight data")
        # if "fetch_hotels" not in tool_names_called:
        #     missing.append("hotel data")
            
    return {"missing_data": missing}


def retry_injector_node(state: AgentState):
    missing = state.get("missing_data", [])
    return {
        "messages": [
            SystemMessage(
                content=f"Missing data for this request: {', '.join(missing)}. "
                        f"You must call the required tools now before answering."
            )
        ]
    }


def cost_calculator_node(state: AgentState):
    tool_msgs = _current_turn_tool_messages(state)
    
    # First check if calculate_trip_cost was called at all this turn. If not, we should skip the cost calculation and not overwrite calculated_total with 0.0.
    was_tool_called = any(getattr(msg, "name", None) == "calculate_trip_cost" for msg in tool_msgs)
    
    if not was_tool_called:
        return {} 

    total = 0.0
    for msg in tool_msgs:
        # If the agent called calculate_trip_cost, extract the total_estimate from its result and add to the running total.
        if getattr(msg, "name", None) == "calculate_trip_cost":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    total += float(data.get("total_estimate") or 0)
            except Exception:
                pass
    return {"calculated_total": total}


def budget_check_node(state: AgentState):
    budget = state.get("total_budget", 0)
    total = state.get("calculated_total", 0)
    return {"over_budget": budget > 0 and total > budget}


def tool_error_node(state: AgentState):
    errors = []
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            break
        if msg.__class__.__name__ == "ToolMessage":
            content = msg.content
            if isinstance(content, str) and not content.startswith("["):
                errors.append(content)
    error_summary = "\n".join(f"- {e}" for e in errors)
    return {"messages": [AIMessage(content=f"I wasn't able to find the requested data:\n{error_summary}")]}


def formatter_node(state: AgentState):
    last_msg = state["messages"][-1]
    raw_content = last_msg.content

    if isinstance(raw_content, str) and raw_content.startswith("[ERROR]"):
        clean = raw_content.replace("[ERROR]", "").strip()
        return {"messages": [AIMessage(content=f"\n⚠️  {clean}\n")]}

    step_limit_hit = isinstance(raw_content, str) and raw_content.startswith("[STEP_LIMIT_REACHED]")

    if isinstance(raw_content, list):
        raw_content = " ".join([b.get("text", "") for b in raw_content if b.get("type") == "text"])

    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_content)
    clean_text = clean_text.replace("\\n", "\n").strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    current_turn_core_tools = {
        getattr(m, "name", None)
        for m in _current_turn_tool_messages(state)
    } & {"fetch_flights", "find_connecting_flights", "fetch_hotels", "fetch_activities", "calculate_trip_cost"}

    # Clarifying questions and lightweight lookup responses should not be wrapped
    # as a trip report just because an older turn had a destination in state.
    if state.get("response") and not current_turn_core_tools:
        return {"messages": [AIMessage(content=clean_text)]}

    # =========================================================================
    #  Stable logic for determining trip context (replacing the fluctuating Intent and cost)
    # =========================================================================
    
    # 1. Extract destination: First check if there is already a locked city in the State from previous turns
    city = state.get("current_city") or ""
    
    # 2. If no city is saved in the State, extract it from the tool call history
    if not city or city == "YOUR DESTINATION":
        city = "YOUR DESTINATION"
        for message in reversed(state["messages"]):
        # We only care about AI messages that triggered tool calls

            if not hasattr(message, "tool_calls") or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                if tool_name == "fetch_flights" and tool_args.get("destination"):
                    city = tool_args.get("destination")
                    break
                if tool_name == "fetch_hotels" and tool_args.get("city"):
                    city = tool_args.get("city")
                    break
            if city != "YOUR DESTINATION":
                break

    # 3. Define stable anchors (Invariants) to determine if this is a trip context:
    
    # Anchor A: We have a verified destination city in the conversation (different from the default)
    has_verified_city = (city and city != "YOUR DESTINATION")
    
    # Anchor B: The model activated one of the core travel tools at least once throughout the conversation history
    TRAVEL_TOOLS = {"fetch_flights", "find_connecting_flights", "fetch_hotels", "fetch_activities", "calculate_trip_cost"}
    has_called_travel_tools = any(
        getattr(m, "name", None) in TRAVEL_TOOLS for m in state["messages"]
    )
    
    # Anchor C (Recommendation): The user set an active budget in the State (saved throughout the session)
    has_active_budget = state.get("total_budget", 0) > 0

    # If none of the stable anchors are met -> this is pure chit-chat, return clean text only
    if not has_verified_city and not has_called_travel_tools and not has_active_budget:
        return {"messages": [AIMessage(content=clean_text)]}

    # =========================================================================
    #  Build the report (If we reached here, one of the anchors is met and it's a trip)
    # =========================================================================
    report = f"  TRIP SUMMARY FOR: {city.upper()}\n"
    report += "=" * 40 + "\n\n"

    if step_limit_hit:
        report += f"⚠️  WARNING: Agent reached the {MAX_STEPS}-step limit.\n\n"
        clean_text = clean_text.replace("[STEP_LIMIT_REACHED]", "").strip()

    report += clean_text + "\n\n"
    report += "=" * 40 + "\n"
    
    total = state.get("calculated_total", 0)
    if total > 0:
        report += f" ESTIMATED TOTAL COST: ${total:.2f}\n"
        if state.get("over_budget"):
            report += " BUDGET ALERT: This plan exceeds your set limit!\n"

    prefs = state.get("user_preferences") or {}
    if prefs:
        pref_line = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in prefs.items())
        report += f" Preferences on file: {pref_line}\n"

    report += "=" * 40

    return {"messages": [AIMessage(content=report)], "current_city": city}
# ============================================================================
# 5. Routing / Conditional Edges
# ============================================================================

def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    # 1. Step limit reached
    if state.get("step_count", 0) >= MAX_STEPS:
        return "step_limit"
    # 2. API error detected in the last message -> formater for error display
    if isinstance(last_msg.content, str) and "[ERROR]" in last_msg.content:
        return "formatter"
    # 3. If the last message includes tool calls -> tools 
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    # Otherwise, go to validator
    return "validator"


def check_tool_errors(state: AgentState):
    """
    Runs after preference_persist. Decides the next hop:
    - 'alternatives' if the LLM signalled NO_MATCH (Case B) for any lookup this turn
    - 'tool_error'   if a required tool returned a hard error string
    - 'replanner'   otherwise (normal Plan-and-Execute observation)
    """
    if _extract_no_matches(state):
        # print(f"\n[Router] NO_MATCH signal detected → alternatives_injector")
        return "alternatives"

    # required_tools = {"fetch_flights", "fetch_hotels"}
    required_tools = {}
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            break
        if msg.__class__.__name__ == "ToolMessage" and msg.name in required_tools:
            content = msg.content
            if isinstance(content, str) and not (
                content.startswith("[") or content.startswith("{")
            ):
                return "tool_error"
    return "replanner"


def should_retry(state: AgentState):
    missing = state.get("missing_data", [])
    tools_called = {
        msg.name for msg in state["messages"]
        if msg.__class__.__name__ == "ToolMessage"
    }
    if missing and tools_called and state.get("step_count", 0) < MAX_STEPS:
        return "retry_injector"
    return "cost_calc"


def should_personalize(state: AgentState):
    """Routes through personalization_node only when stored preferences exist."""
    prefs = state.get("user_preferences") or {}
    return "personalization" if prefs else "planner"


def should_continue_plan(state: AgentState):
    """Routes after replanning: continue executing or finish the turn."""
    if state.get("response"):
        return "finish"
    if not state.get("plan"):
        return "finish"
    if state.get("replan_count", 0) >= MAX_REPLANS:
        return "finish"
    return "execute"


def should_execute_plan(state: AgentState):
    """Routes after planning: execute normal plans or finish direct planner responses."""
    if state.get("response"):
        return "finish"
    if not state.get("plan"):
        return "finish"
    return "execute"

# ============================================================================
# 6. Build the Graph
# ============================================================================
builder = StateGraph(AgentState)

builder.add_node("intent_classifier", intent_classifier_node)
builder.add_node("context_extractor", context_extractor_node)
builder.add_node("personalization", personalization_node)
builder.add_node("planner", planner_node)
builder.add_node("executor_context", executor_context_node)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("preference_persist", preference_persist_node)
builder.add_node("alternatives_injector", alternatives_injector_node)
builder.add_node("tool_error", tool_error_node)
builder.add_node("step_limit", step_limit_node)
builder.add_node("validator", validator_node)
builder.add_node("retry_injector", retry_injector_node)
builder.add_node("replanner", replanner_node)
builder.add_node("cost_calc", cost_calculator_node)
builder.add_node("budget_check", budget_check_node)
builder.add_node("formatter", formatter_node)

builder.add_edge(START, "intent_classifier")
builder.add_edge("intent_classifier", "context_extractor")
builder.add_conditional_edges(
    "context_extractor", should_personalize,
    {"personalization": "personalization", "planner": "planner"},
)
builder.add_edge("personalization", "planner")
builder.add_conditional_edges(
    "planner", should_execute_plan,
    {"execute": "executor_context", "finish": "formatter"},
)
builder.add_edge("executor_context", "agent")
builder.add_conditional_edges(
    "agent", should_continue,
    {"tools": "tools", "validator": "validator",
     "step_limit": "step_limit", "formatter": "formatter"},
)
builder.add_edge("tools", "preference_persist")
builder.add_conditional_edges(
    "preference_persist", check_tool_errors,
    {"alternatives": "alternatives_injector", "tool_error": "tool_error", "replanner": "replanner"},
)
# alternatives_injector injects a SystemMessage then hands back to call_model.
builder.add_edge("alternatives_injector", "agent")
builder.add_edge("tool_error", "formatter")
builder.add_edge("step_limit", "formatter")
builder.add_conditional_edges(
    "validator", should_retry,
    {"retry_injector": "retry_injector", "cost_calc": "replanner"},
)
builder.add_edge("retry_injector", "agent")
builder.add_conditional_edges(
    "replanner", should_continue_plan,
    {"execute": "executor_context", "finish": "cost_calc"},
)
builder.add_edge("cost_calc", "budget_check")
builder.add_edge("budget_check", "formatter")
builder.add_edge("formatter", END)

# ============================================================================
# 7. Persistence — SqliteSaver for cross-session memory
# ============================================================================
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# ============================================================================
# 8. Interactive Execution
# ============================================================================
def run_agent():
    print(BANNER)
    print("Your preferences and conversation history persist across sessions.\n")

    thread_id = input("Enter Session ID (e.g., student_01): ").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    # Greet returning users and show their stored preferences
    try:
        existing = graph.get_state(config)
        
        if not existing or not existing.values:
            # print(f"[System] Session '{thread_id}' not found. Initializing default state values...")
            graph.update_state(config, {
                "current_city": "",
                "total_budget": 0.0,
                "calculated_total": 0.0,
                "over_budget": False,
                "intent": "general",
                "missing_data": [],
                "user_preferences": {},
                "plan": [],
                "completed_steps": [],
                "current_plan_step": "",
                "executor_instruction": "",
                "response": "",
                "replan_count": 0,
                "awaiting_origin_for_destination": "",
            })
            
        # if existing and existing.values:
        else:
            prefs = existing.values.get("user_preferences") or {}
            if prefs:
                print(f"\n[Memory] Welcome back! Loaded preferences for '{thread_id}':")
                for k, v in prefs.items():
                    print(f"  - {k.replace('_', ' ').title()}: {v}")
                print()
            else:
                print(f"\n[Memory] Welcome back! No stored preferences found for '{thread_id}'.\n")
    except Exception as e:
        print(f"Note during session initialization: {e}")
    
    print("Hey! Let's fly high. How can I help? (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye — safe travels!")
                break

            if user_input.lower() == "/history":
                history = list(graph.get_state_history(config))
                if not history:
                    print("No checkpoint history found.\n")
                else:
                    print(f"\nCheckpoint history for '{thread_id}' ({len(history)} snapshots):")
                    for i, snapshot in enumerate(history):
                        chk_id = snapshot.config["configurable"].get("checkpoint_id", "?")
                        prefs = snapshot.values.get("user_preferences") or {}
                        msgs = len(snapshot.values.get("messages") or [])
                        print(f"  [{i}] id={chk_id[:8]}...  msgs={msgs}  prefs={prefs}")
                    print()
                continue

            if user_input.lower().startswith("/goto"):
                parts = user_input.split()
                if len(parts) < 2 or not parts[1].isdigit():
                    print("Usage: /goto <index>  (use /history to see indices)\n")
                else:
                    history = list(graph.get_state_history(config))
                    idx = int(parts[1])
                    if idx >= len(history):
                        print(f"Index {idx} out of range — only {len(history)} checkpoints.\n")
                    else:
                        target = history[idx]
                        graph.update_state(config, target.values)
                        prefs = target.values.get("user_preferences") or {}
                        msgs = len(target.values.get("messages") or [])
                        print(f"[Goto] Restored checkpoint [{idx}] — msgs={msgs}, prefs={prefs}\n")
                continue

            if user_input.lower() == "/undo":
                history = list(graph.get_state_history(config))
                if len(history) < 2:
                    print("Nothing to undo.\n")
                else:
                    prev = history[1]
                    graph.update_state(config, prev.values)
                    prefs = prev.values.get("user_preferences") or {}
                    print(f"[Undo] Restored to previous checkpoint — prefs={prefs}\n")
                continue

            # Reset only per-turn execution fields. Long-term preferences stay
            # in the checkpointed state and are intentionally not overwritten.
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
                "plan": [],
                "completed_steps": [],
                "current_plan_step": "",
                "executor_instruction": "",
                "response": "",
                "replan_count": 0,
                # "current_city": "",
                # "total_budget": 0.0,
                # "calculated_total": 0.0,
                # "over_budget": False,
                # "intent": "general",
                # "missing_data": [],
                # "personalization_context": "",
            }


            print("Searching...\n")

            # Changing stream_mode to "updates"
            for chunk in graph.stream(initial_state, config, stream_mode="updates"):
                if not chunk:
                    continue
                for node_name, response in chunk.items():
                    if response is None:
                        if VERBOSE_DEBUG:
                            print(f"[Debug] Node '{node_name}' returned None response, skipping.")
                        continue
                    if VERBOSE_DEBUG:
                        try:
                            print("\n" + "="*50)
                            print(f"--- NODE EXECUTION CHUNK: {node_name} ---")
                            node_updates = {k: v for k, v in response.items() if k != "messages"}
                            print(f"Fields Updated in State: {node_updates}")

                            if "messages" in response and response["messages"]:
                                last_node_msg = response["messages"][-1]
                                print(f"Messages Added by Node: {len(response['messages'])}")
                                print(f"Last Added Actor: {last_node_msg.__class__.__name__}")
                                print(f"Last Added Content: {last_node_msg.content}")

                            print("="*50 + "\n")
                        except Exception as e:
                            print(f"[Debug Error] Failed to parse chunk for node '{node_name}': {e}")
                            continue
                    
                    # If there is a matching progress message for the completed node - print it
                    if node_name in PROGRESS_MAP:
                        print(PROGRESS_MAP[node_name])
                    
                    # If we reached the formatter - print the final report directly from its update!
                    if node_name == "formatter":
                        # Extract the last AIMessage it generated
                        final_report = response["messages"][-1].content
                        print("\n" + final_report + "\n")

            # Note: The print(last_msg.content) that was outside the loop has been removed!

        except KeyboardInterrupt:
            print("\nGoodbye — safe travels!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    run_agent()
