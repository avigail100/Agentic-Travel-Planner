#!/usr/bin/env python3
"""
Autonomous LangGraph-powered Agent: Manages State and Routing automatically
"""

import json
import re
import uuid
from typing import Annotated, TypedDict
from dotenv import load_dotenv

try:
    from rich.console import Console
    console = Console()
except Exception:
    console = None

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone, lookup_location_options
)

load_dotenv()

MAX_STEPS = 6

SYSTEM_PROMPT = """You are a proactive travel planning assistant. Use the available tools to fetch real data and answer the user directly.

Rules:
- Call tools immediately with reasonable assumptions — do not ask clarifying questions first.
  If the user says "from Israel", use origin=TLV. If they say "to Japan", use destination=Tokyo.
- When a trip requires both flight and hotel data, call fetch_flights AND fetch_hotels in the
  same response so they run in parallel — do not wait for one before calling the other.
- Once you have tool results, give a direct answer. Do not ask follow-up questions if you
  already have enough data to respond.
- If a city or route is not found in the database, say so clearly instead of retrying.
"""

BANNER = r"""
____   ____  _   _  ____
| __ ) / __ \| \ | ||  _ \
|  _ \| |  | |  \| || | | |
| |_) | |__| | |\  || |_| |
|____/ \____/|_| \_||____/
"""

# ============================================================================
# 1. Define the Shared State
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float
    step_count: int
    calculated_total: float
    over_budget: bool
    intent: str
    missing_data: list

# ============================================================================
# 2. Initialize Model and Tools
# ============================================================================
tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone,
    lookup_location_options
    
]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0).bind_tools(tools)

# ============================================================================
# 3. Define Nodes
# ============================================================================

def intent_classifier_node(state: AgentState):
    """
    Categorizes the user's request. Checks most-specific intents first to
    avoid keyword collisions on mixed queries.
    """
    last_msg = state["messages"][-1].content.lower()
    # Ordered from most specific to least — first match wins
    keywords = {
        "visa":      ["visa", "passport", "entry requirement"],
        "currency":  ["exchange rate", "currency", "convert"],
        "full_trip": ["flight", "hotel", "trip", "plan", "travel", "visit", "book"],
    }
    for intent, words in keywords.items():
        if any(w in last_msg for w in words):
            return {"intent": intent}
    return {"intent": "general"}


def context_extractor_node(state: AgentState):
    """
    Extracts destination city and budget from the user's message using regex.
    Populates state so the formatter can display a proper header.
    The main model handles actual city/intent reasoning via the system prompt.
    """
    text = state["messages"][-1].content
    updates = {}

    # Up to 3-word city names; \b on the tail prevents "New York next" → "York next"
    city_match = re.search(
        r'\b(?:to|in|visit(?:ing)?|trip\s+to)\s+([A-Za-z]+(?:\s[A-Za-z]+){0,2})\b',
        text, re.IGNORECASE
    )
    if city_match:
        updates["current_city"] = city_match.group(1).strip().title()

    budget_match = re.search(r'\$\s?([\d,]+)', text)
    if budget_match:
        updates["total_budget"] = float(budget_match.group(1).replace(",", ""))

    # Only return fields that were actually found — preserves previous turn's
    # values on follow-up queries like "also book me a car"
    return updates


def call_model(state: AgentState):
    """
    Core agent node: decides whether to call tools or produce a final answer.
    """
    try:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model.invoke(messages)
    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
            err_text = "[ERROR] API quota exceeded. Please wait a moment and try again."
        else:
            err_text = f"[ERROR] Something went wrong: {e}"
        return {"messages": [AIMessage(content=err_text)], "step_count": state.get("step_count", 0) + 1}

    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}


def step_limit_node(_state: AgentState):
    """
    Injects the step-limit warning message. Separated from the router so the
    mutation actually persists in LangGraph state.
    """
    warning = AIMessage(
        content=f"[STEP_LIMIT_REACHED] The agent exceeded the maximum number of "
                f"steps ({MAX_STEPS}). Generating a partial summary based on collected data."
    )
    return {"messages": [warning]}


def _current_turn_tool_messages(state: AgentState):
    """Return only ToolMessages from the current user turn.
    Slices from the last HumanMessage forward so prior turns don't pollute
    the validator or cost calculator when MemorySaver accumulates history.
    """
    messages = state["messages"]
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].__class__.__name__ == "HumanMessage":
            cut = i
            break
    return [m for m in messages[cut:] if m.__class__.__name__ == "ToolMessage"]


def validator_node(state: AgentState):
    """
    Checks that the expected tools were called for the detected intent.
    Only looks at the current turn to avoid false positives from history.
    """
    tool_names_called = {m.name for m in _current_turn_tool_messages(state)}
    intent = state.get("intent", "general")
    missing = []
    if intent == "full_trip":
        if "fetch_flights" not in tool_names_called:
            missing.append("flight data")
        if "fetch_hotels" not in tool_names_called:
            missing.append("hotel data")
    return {"missing_data": missing}


def retry_injector_node(state: AgentState):
    """
    Injects a retry cue as an AIMessage so it reads as the model's own thought
    rather than a fake user message — less confusing in persisted history.
    """
    missing = state.get("missing_data", [])
    return {"messages": [SystemMessage(content=f"Missing data for this request: {', '.join(missing)}. You must call the required tools now before answering.")]}

def cost_calculator_node(state: AgentState):
    """
    Independently calculates trip cost from raw tool outputs.
    Only scans the current turn to avoid adding up prices across turns.
    """
    total = 0.0
    for msg in _current_turn_tool_messages(state):
        try:
            data = json.loads(msg.content)
            if isinstance(data, list):
                for item in data:
                    total += float(item.get("price") or item.get("price_per_night") or 0)
            elif isinstance(data, dict):
                total += float(data.get("total_estimate") or 0)
        except Exception:
            continue
    return {"calculated_total": total}


def budget_check_node(state: AgentState):
    """
    Compares the calculated total against the user's budget.
    """
    budget = state.get("total_budget", 0)
    total = state.get("calculated_total", 0)
    return {"over_budget": budget > 0 and total > budget}


def formatter_node(state: AgentState):
    """
    Cleans and structures the final response for display.
    """
    last_msg = state["messages"][-1]
    raw_content = last_msg.content

    # Surface errors directly — no trip-summary wrapper
    if isinstance(raw_content, str) and raw_content.startswith("[ERROR]"):
        clean = raw_content.replace("[ERROR]", "").strip()
        return {"messages": [AIMessage(content=f"\n⚠️  {clean}\n")]}

    step_limit_hit = isinstance(raw_content, str) and raw_content.startswith("[STEP_LIMIT_REACHED]")

    if isinstance(raw_content, list):
        raw_content = " ".join([b.get("text", "") for b in raw_content if b.get("type") == "text"])

    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_content)
    clean_text = clean_text.replace("\\n", "\n").strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    report = f"  TRIP SUMMARY FOR: {state.get('current_city', 'Your Destination').upper()}\n"
    report += "=" * 40 + "\n\n"

    if step_limit_hit:
        report += f"⚠️  WARNING: Agent reached the {MAX_STEPS}-step limit.\n"
        report += "   The summary below is based on partially collected data.\n\n"
        clean_text = clean_text.replace("[STEP_LIMIT_REACHED]", "").strip()

    report += clean_text + "\n\n"
    report += "=" * 40 + "\n"

    total = state.get("calculated_total", 0)
    if total > 0:
        report += f" ESTIMATED TOTAL COST: ${total:.2f}\n"
        if state.get("over_budget"):
            report += " BUDGET ALERT: This plan exceeds your set limit!\n"

    report += "=" * 40

    return {"messages": [AIMessage(content=report)]}


# ============================================================================
# 4. Define Routing Functions (pure — no state mutations)
# ============================================================================
def should_continue(state: AgentState):
    if state.get("step_count", 0) >= MAX_STEPS:
        return "step_limit"
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "validator"


def tool_error_node(state: AgentState):
    """
    Surfaces tool errors directly to the user instead of looping back to the agent.
    Triggered when a tool returns a 'not found' or error string rather than data.
    """
    errors = []
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            break  # only look at the most recent tool round
        if msg.__class__.__name__ == "ToolMessage":
            content = msg.content
            if isinstance(content, str) and not content.startswith("["):
                errors.append(content)

    error_summary = "\n".join(f"- {e}" for e in errors)
    message = f"I wasn't able to find the requested data:\n{error_summary}\n\nPlease check the destination name and try again."
    return {"messages": [AIMessage(content=message)]}


def check_tool_errors(state: AgentState):
    """
    After tools run: if any tool returned an error string (not JSON data),
    route to the error handler instead of back to the agent.
    """
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            break  # only check the current tool round
        if msg.__class__.__name__ == "ToolMessage":
            content = msg.content
            # Tool errors are plain strings; successful results are JSON
            if isinstance(content, str) and not content.startswith("[") and not content.startswith("{"):
                return "tool_error"
    return "agent"


def should_retry(state: AgentState):
    missing = state.get("missing_data", [])
    tools_called = {
        msg.name for msg in state["messages"]
        if msg.__class__.__name__ == "ToolMessage"
    }
    # Only retry if tools were already called — skip if agent is still clarifying
    if missing and tools_called and state.get("step_count", 0) < MAX_STEPS:
        return "retry_injector"
    return "cost_calc"


# ============================================================================
# 5. Build the Graph
# ============================================================================
builder = StateGraph(AgentState)

builder.add_node("intent_classifier", intent_classifier_node)
builder.add_node("context_extractor", context_extractor_node)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("tool_error", tool_error_node)
builder.add_node("step_limit", step_limit_node)
builder.add_node("validator", validator_node)
builder.add_node("retry_injector", retry_injector_node)
builder.add_node("cost_calc", cost_calculator_node)
builder.add_node("budget_check", budget_check_node)
builder.add_node("formatter", formatter_node)

builder.add_edge(START, "intent_classifier")
builder.add_edge("intent_classifier", "context_extractor")
builder.add_edge("context_extractor", "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools":      "tools",
    "validator":  "validator",
    "step_limit": "step_limit",
})
builder.add_conditional_edges("tools", check_tool_errors, {
    "tool_error": "tool_error",
    "agent":      "agent",
})
builder.add_edge("tool_error", "formatter")          # stop and explain — no retry
builder.add_edge("step_limit", "formatter")          # bypass validator on step limit
builder.add_conditional_edges("validator", should_retry, {
    "retry_injector": "retry_injector",
    "cost_calc":      "cost_calc",
})
builder.add_edge("retry_injector", "agent")
builder.add_edge("cost_calc", "budget_check")
builder.add_edge("budget_check", "formatter")
builder.add_edge("formatter", END)

graph = builder.compile(checkpointer=MemorySaver())  # enables thread_id persistence

# ============================================================================
# 6. Interactive Execution
# ============================================================================
def run_agent():
    # Each run gets a fresh session — no cross-session bleed
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(BANNER)
    print("Hey! let's fly high, how can I help? (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye — safe travels.")
                break

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
                "total_budget": 0.0,
                "current_city": "",
                "calculated_total": 0.0,
                "over_budget": False,
                "intent": "general",
                "missing_data": [],
            }

            print("Searching...\n")

            # Stream but only print the final formatted answer
            for event in graph.stream(initial_state, config, stream_mode="values"):
                last = event["messages"][-1]
                content = last.content
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
                if isinstance(content, str) and ("TRIP SUMMARY" in content or content.strip().startswith("⚠️")):
                    print(content)

        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    run_agent()
