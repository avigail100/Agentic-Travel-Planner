#!/usr/bin/env python3
"""
Travel Agent — Plan-and-Execute Architecture (Session 5)

Architecture:
  planner  →  execute  ⇄  tools  →  replan  → (loop or END)

- Planner:  produces a deduplicated, ordered list of steps (Pydantic-validated)
- Executor: runs one step at a time using all travel tools (ReAct style)
- Replanner: decides whether to continue, revise the plan, or emit a final answer
- step_count in replan guards against infinite re-planning loops (max MAX_REPLAN_CYCLES)

Preserved from Session 4:
- All travel tools (lookup_location_options, fetch_flights, etc.)
- Alternatives injection when destinations are not found
- Budget extraction and over-budget warning
- User preferences (SqliteSaver cross-session memory)
- Formatter producing a clean TRIP SUMMARY report
"""

import json
import re
import sqlite3
from typing import Annotated, List, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict

from tools import (
    calculate_trip_cost,
    convert_cost_to_origin_currency,
    convert_time_to_destination_timezone,
    fetch_activities,
    fetch_car_rental_agencies,
    fetch_currency_exchange_rate,
    fetch_flights,
    fetch_hotels,
    fetch_seasonal_recommendations,
    fetch_time_difference,
    fetch_visa_requirements,
    find_connecting_flights,
    lookup_location_options,
    save_preference,
    suggest_alternatives
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXECUTOR_STEPS = 8   # tool calls the executor may make per step
MAX_REPLAN_CYCLES  = 6   # how many times the replanner may issue a new plan

BANNER = r"""
____   ____  _   _  ____
| __ ) / __ \| \ | ||  _ \
|  _ \| |  | |  \| || | | |
| |_) | |__| | |\  || |_| |
|____/ \____/|_| \_||____/
"""

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    """A high-level, ordered, deduplicated plan."""
    steps: List[str] = Field(
        description=(
            "Sequential steps to complete the travel request. "
            "Each step must be distinct, no repetitions, no duplicate tool calls. "
            "Keep each step atomic and focused on a single information need."
        )
    )


class FinalResponse(BaseModel):
    """The final, formatted answer to the user."""
    response: str


class ReplanAction(BaseModel):
    """The replanner's decision: either a revised plan or a final response."""
    action: Union[Plan, FinalResponse]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]     # conversation history (executor messages)
    input: str     # original user request
    plan: List[str]     # remaining steps to execute
    past_steps: List[str]     # (step_text, tool_results_summary) pairs — tracks what was done
    response: str     # set by replanner when done
    executor_steps: int     # safety counters
    replan_count: int     # budget / cost tracking (carried over from Session 4)
    total_budget: float
    calculated_total: float
    over_budget: bool     # user preferences
    user_preferences: dict

tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone,
    lookup_location_options, find_connecting_flights,
    save_preference, suggest_alternatives
]

_base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_retries=2)

planner_model   = _base_model.with_structured_output(Plan)
replanner_model = _base_model.with_structured_output(ReplanAction)
executor_model  = _base_model.bind_tools(tools)

def _extract_budget(text: str) -> float:
    m = re.search(r"\$\s?([\d,]+)|([\d,]+)\s?(?:\$|dollars?)", text, re.IGNORECASE)
    if m:
        return float((m.group(1) or m.group(2)).replace(",", ""))
    return 0.0

def _summarise_tool_messages(messages: list) -> str:
    """Extract a concise summary of tool results from a message list."""
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            # Truncate very long DB dumps to keep context manageable
            if isinstance(content, str) and len(content) > 600:
                content = content[:600] + "…[truncated]"
            parts.append(f"[{m.name}] → {content}")
    return "\n".join(parts) if parts else "(no tool output)"


def _no_match_detected(messages: list) -> bool:
    """
    True only on CASE B — a genuine mismatch where the executor explicitly
    returned "NO_MATCH:<term>" as its response.
    CASE A (no_direct_match dict) is intentionally excluded: the executor
    should handle those by silently mapping to the closest available location.
    """
    for m in messages:
        # Check executor AIMessage content for the NO_MATCH signal
        if isinstance(m, AIMessage):
            if isinstance(m.content, str) and m.content.strip().startswith("NO_MATCH:"):
                return True
        # Also check if a tool message itself returned NO_MATCH (edge case)
        if isinstance(m, ToolMessage):
            if isinstance(m.content, str) and m.content.strip().startswith("NO_MATCH:"):
                return True
    return False


def _extract_cost(messages: list) -> float:
    total = 0.0
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "calculate_trip_cost":
            try:
                data = json.loads(m.content)
                if isinstance(data, dict):
                    total += float(data.get("total_estimate") or 0)
            except Exception:
                pass
    return total


# ---------------------------------------------------------------------------
# Node: planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a senior travel planning strategist.
Given a user travel request, produce an ordered list of UNIQUE steps.

Rules:
- Each step must call a DIFFERENT piece of information (no duplicates).
- Start with resolving the location via lookup_location_options.
- If the request asks for both flights and hotels, list them as separate steps.
- Include a cost calculation step at the end if pricing is needed.
- Prefer parallel-friendly ordering (lookup first, then fetches, then cost).
- Maximum 6 steps. Keep each step one sentence.
"""


def plan_node(state: PlanExecuteState):
    budget = _extract_budget(state["input"])
    
    # Extract and format past messages to give the Planner context/memory across turns
    history_context = ""
    if state.get("messages"):
        history_context = "Past conversation context:\n"
        for msg in state["messages"]:
            actor = "User" if msg.__class__.__name__ == "HumanMessage" else "Agent"
            # Handle both string and list content types safely
            content = msg.content
            if isinstance(content, list):
                content = " ".join([b.get("text", "") for b in content if b.get("type") == "text"])
            history_context += f"  {actor}: {content}\n"
            
    # Inject both the historical context and the fresh user input into the prompt
    prompt = (
        f"{PLANNER_SYSTEM}\n\n"
        f"{history_context}\n"
        f"New User Request: {state['input']}"
    )
    
    plan = planner_model.invoke(prompt)

    # Deduplicate while preserving order
    seen: set = set()
    unique_steps = []
    for step in plan.steps:
        key = step.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_steps.append(step)

    print(f"\n{'='*50}")
    print(f"[Planner] Generated {len(unique_steps)}-step plan:")
    for i, s in enumerate(unique_steps, 1):
        print(f"  {i}. {s}")
    print(f"{'='*50}\n")

    return {
        "plan": unique_steps,
        "past_steps": [],
        "response": "",
        "executor_steps": 0,
        "replan_count": 0,
        "total_budget": budget or state.get("total_budget", 0.0),
        "calculated_total": state.get("calculated_total", 0.0),
        "over_budget": state.get("over_budget", False),
        "messages": state.get("messages", []), # Maintain message flow
    }
    #     return {
    #     "plan": unique_steps,
    #     "past_steps": [],
    #     "response": "",
    #     "executor_steps": 0,
    #     "replan_count": 0,
    #     "total_budget": budget or state.get("total_budget", 0.0),
    #     "calculated_total": 0.0,
    #     "over_budget": False,
    #     "messages": [],
    # }

# ---------------------------------------------------------------------------
# Node: executor
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM = """You are a travel data retrieval agent.
Execute ONLY the current step listed below using the available tools.
Do not skip ahead or repeat tool calls you already made.

Location resolution rules (apply BEFORE calling any fetch tool):
1. Call lookup_location_options to resolve any city/country name.
2. If it returns a list of available_locations with no exact match:
   CASE A — Semantic equivalence (country→airport, region→city):
     Examples: "Israel"→"TLV", "Japan"→"Tokyo", "Britain"→"London"
     Action: silently pick the best match from available_locations and
             immediately call the target fetch tool with that value.
             Do NOT inform the user or ask for confirmation.
   CASE B — Genuine mismatch (no logical mapping exists):
     Action: stop and return the string "NO_MATCH:<search_term>" as your
             response. Do not call any further tools.
3. If lookup returns a direct match, use that value for the fetch tool.

After executing the step, provide a brief text summary of what you found.
"""


def _is_api_error(err: Exception) -> str | None:
    """Returns a user-friendly error string for known API errors, else None."""
    s = str(err)
    if "RESOURCE_EXHAUSTED" in s or "429" in s or "quota" in s.lower():
        return "[ERROR] API quota exceeded. Please wait a moment and try again."
    if "503" in s or "UNAVAILABLE" in s:
        return "[ERROR] The AI service is temporarily unavailable. Please try again shortly."
    return None


def execute_node(state: PlanExecuteState):
    if not state["plan"]:
        return {}

    current_step = state["plan"][0]
    
    # -------------------------------------------------------------------------
    #  GATHER PAST TOOL CALLS FOR EXACT MATCHING
    # -------------------------------------------------------------------------
    # Build a set of tuples: (tool_name, stringified_sorted_args)
    past_tool_calls = set()
    for msg in state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # json.dumps with sort_keys ensures {a:1, b:2} == {b:2, a:1}
                args_str = json.dumps(tc.get("args", {}), sort_keys=True)
                past_tool_calls.add((tc.get("name"), args_str))
    # -------------------------------------------------------------------------

    prefs = state.get("user_preferences") or {}
    pref_block = (
        "User preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items())
        if prefs else ""
    )

    history_block = ""
    if state.get("past_steps"):
        history_block = "Already completed:\n" + "\n".join(
            f"  - {s}" for s in state["past_steps"]
        )

    input_text = (
        f"{EXECUTOR_SYSTEM}\n\n"
        f"Original goal: {state['input']}\n\n"
        f"{history_block}\n\n"
        f"{'[Preferences] ' + pref_block if pref_block else ''}\n\n"
        f"Current step: {current_step}\n"
    )

    print(f"\n[Executor] Running step: {current_step}")

    try:
        response = executor_model.invoke([HumanMessage(content=input_text)])
        
        # -------------------------------------------------------------------------
        #  INFINITE LOOP GUARD: Check if the model generated an identical tool call
        # -------------------------------------------------------------------------
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                args_str = json.dumps(tc.get("args", {}), sort_keys=True)
                current_call = (tc.get("name"), args_str)
                
                # If the exact same tool with the exact same arguments was called before:
                if current_call in past_tool_calls:
                    loop_error = f"[ERROR] Infinite loop detected. The tool '{tc.get('name')}' was called again with the exact same arguments: {args_str}."
                    print(f"\n[Loop Guard] Aborting execution to prevent API quota drain.")
                    
                    return {
                        "messages": [AIMessage(content=loop_error)],
                        "plan": [],   # Clear the plan to force Replanner to handle the failure
                        "response": "I encountered an issue searching for this specific combination repeatedly. Please try adjusting your destination or origin specifications.",
                    }
        # -------------------------------------------------------------------------
        
    except Exception as e:
        err_text = _is_api_error(e) or f"[ERROR] Executor failed: {e}"
        print(f"\n[Executor] {err_text}")
        return {
            "messages": [AIMessage(content=err_text)],
            "plan": [],
            "response": err_text,
        }

    return {
        "messages": [HumanMessage(content=f"[Step] {current_step}"), response],
        "executor_steps": state.get("executor_steps", 0) + 1,
    }
# ---------------------------------------------------------------------------
# Routing after executor
# ---------------------------------------------------------------------------

def check_executor_tools(state: PlanExecuteState):
    """Route to tools if the executor requested tool calls, else to replan."""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and getattr(last_msg, "tool_calls", None):
        return "tools"
    return "replan"


def after_tools(state: PlanExecuteState):
    """
    Runs after ToolNode. Records the completed step's tool results into
    past_steps, advances the plan, and accumulates cost/preferences.
    The executor will be re-invoked (via the routing lambda) if plan is non-empty,
    or the replanner will run if the plan is exhausted.
    """
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    summary = _summarise_tool_messages(tool_msgs[-10:])

    current_step = state["plan"][0] if state["plan"] else "?"
    step_record = f"Step '{current_step}': {summary}"

    new_cost = _extract_cost(tool_msgs)
    total_cost = state.get("calculated_total", 0.0) + new_cost

    existing_prefs = dict(state.get("user_preferences") or {})
    for m in tool_msgs:
        if getattr(m, "name", None) == "save_preference" and m.content.startswith("saved:"):
            k, v = m.content[6:].split("=", 1)
            existing_prefs[k] = v
            print(f"[Memory] Saved preference: {k}={v}")

    return {
        "past_steps": state.get("past_steps", []) + [step_record],
        "plan": state["plan"][1:],   # advance to the next step
        "calculated_total": total_cost,
        "user_preferences": existing_prefs,
    }

# ---------------------------------------------------------------------------
# Routing after executor
# ---------------------------------------------------------------------------

def route_after_tools(state: PlanExecuteState) -> str:
    if _no_match_detected(state["messages"]):
        return "no_match"
    return "execute" if state.get("plan") else "replan"


# ---------------------------------------------------------------------------
# Node: no_match_injector
# ---------------------------------------------------------------------------

def no_match_injector_node(state: PlanExecuteState) -> dict:
    """
    Injects a hint into past_steps with the REAL available locations from the
    lookup tool result, so the replanner cannot hallucinate destinations.
    Clears the remaining plan so the replanner exits with a FinalResponse.
    """
    # Extract the actual available_locations list from the lookup tool result
    available = []
    for m in state["messages"]:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "lookup_location_options":
            continue
        try:
            data = json.loads(m.content) if isinstance(m.content, str) else m.content
            if isinstance(data, dict) and data.get("no_direct_match"):
                available = data.get("available_locations", [])
                break
        except (json.JSONDecodeError, TypeError):
            pass

    avail_str = ", ".join(available) if available else "none found"
    hint = (
        f"SYSTEM NOTE: The requested destination was NOT found in the database. "
        f"The ONLY real destinations available are: [{avail_str}]. "
        f"Do NOT suggest any destination not in this list. "
        f"Produce a FinalResponse telling the user their destination is unavailable "
        f"and suggest alternatives ONLY from the list above."
    )
    return {
        "past_steps": state.get("past_steps", []) + [hint],
        "plan": [],
    }


# ---------------------------------------------------------------------------
# Node: replan
# ---------------------------------------------------------------------------

# REPLANNER_SYSTEM = """You are a travel planning supervisor reviewing execution progress.

# Decide between two actions:
# A) FinalResponse: ONLY use this if ALL information goals (flights, hotels, etc.) were successfully 
#    retrieved and you have data to build a complete plan, OR if the database definitively confirms 
#    that no options exist and no further alternatives can be checked. Never dump raw rows — narrate.
# B) Plan: Use this if steps remain OR if a previous step failed/returned no results (e.g., 'No flights found'). 
#    If a flight lookup fails, YOUR REVISED PLAN MUST PROACTIVELY INCLUDE A STEP TO FIND ALTERNATIVES 
#    (like checking connecting flights via find_connecting_flights or suggesting alternative nearby destinations).

# Rules for revising plans:
# - Do NOT re-add a step that already appears in the completed history with the exact same target parameters.
# - If a step failed, adapt the plan dynamically. Do not emit an empty plan or repeat the failing step blindly.
# - CRITICAL: If the history contains a SYSTEM NOTE listing available destinations, you MUST only suggest 
#   destinations from that exact list. Never invent or hallucinate unlisted destinations.
# - If budget was provided and the total cost exceeds it, mention this clearly.
# """


REPLANNER_SYSTEM = """You are a travel planning supervisor reviewing execution progress.

Decide between two actions:
A) FinalResponse: Use this if all goals are met, OR if the database definitively confirms no options exist and no valid alternatives can be offered.
B) Plan: Use this to revise the plan if steps remain, or if a previous step failed.

Rules for revising plans:
1. NO FORCED FLIGHTS: Do not add flight searches for domestic/local trips.
2. FLIGHT FAILURES: If a requested flight search fails, add a step to use 'find_connecting_flights'.
3. SMART ALTERNATIVES: If connecting flights also fail, or if a destination is totally unreachable, 
add a step to use the 'suggest_alternatives' tool with the user's origin city to find real, valid destinations.
4. DO NOT repeat a failed step with the exact same parameters.
- Do NOT re-add a step that already appears in the completed history.
5. MEMORY: Consider the user's past conversation history (provided below) to tailor your alternative suggestions to their preferences (explicit/not).
6. If budget was provided and the total cost exceeds it, mention this clearly.
"""

def replan_node(state: PlanExecuteState):
    replan_count = state.get("replan_count", 0) + 1
    budget = state.get("total_budget", 0.0)
    total_cost = state.get("calculated_total", 0.0)

    # -------------------------------------------------------------------------
    # Build conversation memory context for the Replanner
    # -------------------------------------------------------------------------
    history_context = ""
    if state.get("messages"):
        history_context = "Past conversation context:\n"
        for msg in state["messages"]:
            actor = "User" if msg.__class__.__name__ == "HumanMessage" else "Agent"
            content = msg.content
            if isinstance(content, list):
                content = " ".join([b.get("text", "") for b in content if b.get("type") == "text"])
            # Keep context concise to save tokens
            history_context += f"  {actor}: {content[:300]}\n"

    # Safety: force a final answer if we've re-planned too many times
    if replan_count > MAX_REPLAN_CYCLES or not state["plan"]:
        force_final = replan_count > MAX_REPLAN_CYCLES
        if force_final:
            print(f"\n[Replanner] ⚠️  Replan limit ({MAX_REPLAN_CYCLES}) reached — forcing final answer.")
        else:
            print("\n[Replanner] ✅ All planned steps completed successfully. Generating final response.")
        past_steps_text = "\n".join(state.get("past_steps", [])) or "(none)"
        budget_note = (
            f"\nBudget provided: ${budget:.2f}. Estimated total: ${total_cost:.2f}."
            + (" NOTE: OVER BUDGET." if budget > 0 and total_cost > budget else "")
            if budget > 0 else ""
        )

        final_prompt = (
            f"The user asked: {state['input']}\n\n"
            f"{history_context}\n\n"
            f"Completed steps:\n{past_steps_text}\n"
            f"{budget_note}\n\n"
            "Produce a friendly, well-formatted final travel summary. "
            "No raw DB output — narrate the findings clearly."
            "If suggesting alternatives, explicitly mention the options found via tools."
        )
        final_msg = _base_model.invoke(final_prompt)
        return {
            "response": final_msg.content,
            "plan": [],
            "replan_count": replan_count,
        }

    past_steps_text = "\n".join(state.get("past_steps", [])) or "(none)"
    remaining_text = "\n".join(f"  - {s}" for s in state["plan"])
    budget_note = (
        f"Budget: ${budget:.2f}. Running total so far: ${total_cost:.2f}."
        if budget > 0 else ""
    )

    prompt = (
        f"{REPLANNER_SYSTEM}\n\n"
        f"{history_context}\n\n"
        f"Original goal: {state['input']}\n\n"
        f"Completed steps:\n{past_steps_text}\n\n"
        f"Remaining planned steps:\n{remaining_text}\n\n"
        f"{budget_note}\n\n"
        "What should happen next?"
    )

    try:
        result = replanner_model.invoke(prompt)
    except Exception as e:
        err_text = _is_api_error(e) or f"[ERROR] Replanner failed: {e}"
        print(f"\n[Replanner] {err_text}")
        return {"response": err_text, "plan": [], "replan_count": replan_count}

    print(f"\n[Replanner] Decision: {type(result.action)}")

    if isinstance(result.action, FinalResponse):
        budget_warn = ""
        if budget > 0 and total_cost > budget:
            budget_warn = f"\n\n⚠️  BUDGET ALERT: Estimated ${total_cost:.2f} exceeds your ${budget:.2f} limit."
        return {
            "response": result.action.response + budget_warn,
            "plan": [],
            "replan_count": replan_count,
            "over_budget": budget > 0 and total_cost > budget,
        }

    # Revised plan — strip any steps already done
    done_keys = {s.split(":")[0].strip().lower() for s in state.get("past_steps", [])}
    new_steps = [
        s for s in result.action.steps
        if s.strip().lower() not in done_keys
    ]

    print(f"[Replanner] Revised plan ({len(new_steps)} steps remaining):")
    for i, s in enumerate(new_steps, 1):
        print(f"  {i}. {s}")

    return {
        "plan": new_steps,
        "replan_count": replan_count,
    }


# ---------------------------------------------------------------------------
# Routing after replan
# ---------------------------------------------------------------------------

def should_end(state: PlanExecuteState) -> str:
    if state.get("response"):
        return "formatter"
    if not state.get("plan"):
        # Plan is empty but no response yet — force replan to generate one
        return "replan"
    return "execute"


# ---------------------------------------------------------------------------
# Node: formatter  (Session 4 report style)
# ---------------------------------------------------------------------------

def formatter_node(state: PlanExecuteState):
    raw = state.get("response", "")

    # Derive destination from past steps / input
    city = "YOUR DESTINATION"
    for step in state.get("past_steps", []):
        m = re.search(r"fetch_flights.*?destination['\"]?\s*[:=]\s*['\"]?([A-Za-z ]+)", step, re.IGNORECASE)
        if m:
            city = m.group(1).strip().title()
            break
        m2 = re.search(r"fetch_hotels.*?city['\"]?\s*[:=]\s*['\"]?([A-Za-z ]+)", step, re.IGNORECASE)
        if m2:
            city = m2.group(1).strip().title()
            break

    total = state.get("calculated_total", 0.0)
    budget = state.get("total_budget", 0.0)

    report = f"  TRIP SUMMARY FOR: {city.upper()}\n"
    report += "=" * 40 + "\n\n"
    report += raw.strip() + "\n\n"
    report += "=" * 40 + "\n"

    if total > 0:
        report += f" ESTIMATED TOTAL COST: ${total:.2f}\n"
        if state.get("over_budget"):
            report += " BUDGET ALERT: This plan exceeds your set limit!\n"
    if budget > 0 and total == 0:
        report += f" Budget on file: ${budget:.2f}\n"

    prefs = state.get("user_preferences") or {}
    if prefs:
        pref_line = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in prefs.items())
        report += f" Preferences on file: {pref_line}\n"

    report += "=" * 40

    print("\n" + "=" * 40)
    print(report)
    print("=" * 40 + "\n")

    return {"response": report}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(PlanExecuteState)

builder.add_node("planner",           plan_node)
builder.add_node("execute",           execute_node)
builder.add_node("tools",             ToolNode(tools))
builder.add_node("after_tools_node",  after_tools)
builder.add_node("no_match_injector", no_match_injector_node)
builder.add_node("replan",            replan_node)
builder.add_node("formatter",         formatter_node)

builder.add_edge(START,     "planner")
builder.add_edge("planner", "execute")

builder.add_conditional_edges(
    "execute", check_executor_tools,
    {"tools": "tools", "replan": "replan"},
)

builder.add_edge("tools",             "after_tools_node")
builder.add_edge("no_match_injector", "replan")

builder.add_conditional_edges(
    "after_tools_node", route_after_tools,
    {"execute": "execute", "replan": "replan", "no_match": "no_match_injector"},
)

builder.add_conditional_edges(
    "replan", should_end,
    {"execute": "execute", "formatter": "formatter", "replan": "replan"},
)

builder.add_edge("formatter", END)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
conn   = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph  = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

PROGRESS_MAP = {
    "planner":           "📋  Building travel plan...",
    "execute":           "⚙️   Executing step...",
    "tools":             "🧳  Querying travel database...",
    "after_tools_node":  "📊  Processing tool results...",
    "no_match_injector": "💡  Destination not found — searching for alternatives...",
    "replan":            "🔄  Reviewing progress and re-evaluating plan...",
    # "formatter":         "✨  Formatting final report...",
}


def run_agent():
    print(BANNER)
    print("Plan-and-Execute travel agent — session 5.\n")

    thread_id = input("Enter Session ID (e.g., student_01): ").strip() or "default"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 4,
    }

    # Load / initialise state
    try:
        existing = graph.get_state(config)
        if not existing or not existing.values:
            graph.update_state(config, {
                "input": "",
                "plan": [],
                "past_steps": [],
                "response": "",
                "executor_steps": 0,
                "replan_count": 0,
                "total_budget": 0.0,
                "calculated_total": 0.0,
                "over_budget": False,
                "user_preferences": {},
                "messages": [],
            })
        else:
            prefs = existing.values.get("user_preferences") or {}
            if prefs:
                print(f"[Memory] Welcome back! Loaded preferences for '{thread_id}':")
                for k, v in prefs.items():
                    print(f"  - {k.replace('_', ' ').title()}: {v}")
                print()
            else:
                print(f"[Memory] Welcome back! No stored preferences for '{thread_id}'.\n")
    except Exception as e:
        print(f"Note during session init: {e}")

    print("Let's plan your trip! (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye — safe travels!")
                break

            # Carry over user preferences from previous turn
            try:
                prev_state = graph.get_state(config)
                saved_prefs = (prev_state.values or {}).get("user_preferences") or {}
            except Exception:
                saved_prefs = {}

            initial_state: PlanExecuteState = {
                "input":          user_input,
                "plan":           [],
                "past_steps":     [],
                "response":       "",
                "executor_steps": 0,
                "replan_count":   0,
                "total_budget":   _extract_budget(user_input),
                "calculated_total": 0.0,
                "over_budget":    False,
                "user_preferences": saved_prefs,
                "messages":       [],
            }

            print("\nSearching...\n")

            for chunk in graph.stream(initial_state, config, stream_mode="updates"):
                if not chunk:
                    continue
                for node_name, _ in chunk.items():
                    if node_name in PROGRESS_MAP:
                        print(PROGRESS_MAP[node_name])

        except KeyboardInterrupt:
            print("\nGoodbye — safe travels!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    run_agent()
