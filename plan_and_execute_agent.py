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
import os
import re
import sqlite3
from typing import Annotated, List, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
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
    search_web,
    suggest_alternatives,
    find_hotels_by_amenity,
    find_destinations_by_preference,
    fetch_restaurants,
    fetch_beaches,
    fetch_city_transport_info,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXECUTOR_STEPS = 4   # tool calls the executor may make per step
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
    chat_history: str

tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone,
    lookup_location_options, find_connecting_flights,
    save_preference, suggest_alternatives, search_web,
    find_hotels_by_amenity, find_destinations_by_preference, fetch_restaurants,
    fetch_beaches, fetch_city_transport_info,
]

_base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_retries=2)
#_base_model = ChatGroq(
#    api_key=os.getenv("GROQ_API_KEY"),
#    model="llama-3.3-70b-versatile", 
#    temperature=0.4,
#    max_retries=1
#)

planner_model   = _base_model.with_structured_output(Plan)
replanner_model = _base_model.with_structured_output(ReplanAction)
executor_model  = _base_model.bind_tools(tools)



# # Keep these exactly as they are! They will now automatically use Groq:
# planner_model   = _base_model.with_structured_output(Plan)
# replanner_model = _base_model.with_structured_output(ReplanAction)
# executor_model  = _base_model.bind_tools(tools)

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

def _digest_tool_result(content) -> str:
    """One short, human-readable line summarising a single tool result for the
    console trace (not the full raw dump)."""
    # Parse JSON content where possible so we can count rows / pick key fields.
    data = content
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except Exception:
            data = content

    if data is None:
        return "no result"
    if isinstance(data, list):
        return f"{len(data)} result(s)" if data else "no results"
    if isinstance(data, dict):
        if data.get("no_direct_match"):
            opts = data.get("available_locations", [])
            return f"no direct match — options: {', '.join(map(str, opts))}"
        if "answer" in data:   # (not expected here, but be safe)
            return str(data["answer"])[:120]
        return ", ".join(f"{k}={v}" for k, v in list(data.items())[:3])[:120]

    text = str(data).strip()
    # search_web returns a multi-line block; the ANSWER line is the useful part.
    for line in text.splitlines():
        if line.startswith("ANSWER:"):
            text = line[len("ANSWER:"):].strip()
            break
    else:
        text = text.replace("\n", " ")
    return (text[:120] + "…") if len(text) > 120 else text

def _format_call(tc: dict) -> str:
    """Render a tool call as 'tool(key=val, …)' with compact args."""
    name = tc.get("name", "?")
    args = tc.get("args", {}) or {}
    shown = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
    return f"{name}({shown})" if shown else f"{name}()"

def _no_match_detected(messages: list) -> bool:
    """
    True only on CASE B — a genuine mismatch where the executor explicitly
    returned "NO_MATCH:<term>" as its response.
    CASE A (no_direct_match dict) is intentionally excluded: the executor
    should handle those by silently mapping to the semantic-equivalent available location.
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

PLANNER_SYSTEM = """You are an expert travel planning strategist.
Your goal is to break down the user's travel request into a logical, efficient sequence of UNIQUE execution steps.

AVAILABLE TOOLS:
{tool_descriptions}

SOURCE-OF-TRUTH BOUNDARY (read carefully):
- The DATABASE is the single source of truth for flights, hotels,
  activities, car rentals, and time differences. ALWAYS use the
  dedicated DB tools for these — NEVER the web.
- The INTERNET (search_web) is for live, real-world context NOT in our database:
  current exchange rates, weather/forecasts, local news & safety advisories,
  strikes/closures, holidays, festivals, and special events.
- If the user asks about weather, current events, news, or live exchange rates,
  plan a search_web step for it.

CRITICAL RULES FOR PLANNING:
1. Each step must call a DIFFERENT piece of information (no duplicates).
2. Tool Awareness: You MUST ONLY plan steps that can be resolved using the exact tools listed above. Never invent tools or services. Respect the source-of-truth boundary above: bookable items → DB tools, live external info → search_web.
3. Parallel Execution (Batching): The downstream Executor can use multiple tools simultaneously. Group independent data-gathering tasks into a SINGLE step, but DO NOT call more then 2 tools per step. 
   - Good: "Fetch flights, and hotels for Tokyo."
   - Bad (Too slow): Step 1: "Fetch flights to Tokyo", Step 2: "Fetch hotels in Tokyo".
4. Dependency Ordering: If a step inherently depends on the result of another, place it in a SUBSEQUENT step. 
   - Example: You must gather all trip components (flights, hotels) in early steps BEFORE adding a final step to "Calculate total trip cost".
5. Do Not use all the available tools just for the sake of it. Only include tools that are relevant to the user's request. Irrelevant steps waste time and risk hitting token limits.
6. Simplicity: Maximum 6 steps. Keep step descriptions concise and focused on the data needed. Do not plan formatting or summarization steps (the system handles the final output automatically).

Do NOT create separate lookup_location_options steps.
The Executor will do lookup automatically.
"""


def plan_node(state: PlanExecuteState):
    budget = _extract_budget(state["input"])
    
    # -------------------------------------------------------------------------
    # MEMORY INJECTION: Read from chat_history instead of messages
    # -------------------------------------------------------------------------
    history_context = ""
    if state.get("chat_history"):
        history_context = f"Past conversation context:\n{state['chat_history']}"
    tool_descriptions = "\n".join([f"- {t.name}: {t.description.splitlines()[0]}" for t in tools])
    # Inject the historical context, the available tools and the fresh user input into the prompt    
    prompt = (
    PLANNER_SYSTEM.format(tool_descriptions=tool_descriptions) + "\n\n"
    f"{history_context}\n"
        f"New User Request: {state['input']}"
    )
    
    # # Extract and format past messages to give the Planner context/memory across turns
    # history_context = ""
    # if state.get("messages"):
    #     history_context = "Past conversation context:\n"
    #     for msg in state["messages"]:
    #         actor = "User" if msg.__class__.__name__ == "HumanMessage" else "Agent"
    #         # Handle both string and list content types safely
    #         content = msg.content
    #         if isinstance(content, list):
    #             content = " ".join([b.get("text", "") for b in content if b.get("type") == "text"])
    #         history_context += f"  {actor}: {content}\n"
            
    # # Inject both the historical context and the fresh user input into the prompt
    # prompt = (
    #     f"{PLANNER_SYSTEM}\n\n"
    #     f"{history_context}\n"
    #     f"New User Request: {state['input']}"
    # )
     
    try:
        plan = planner_model.invoke(prompt)
    except Exception as e:
        err_text = _is_api_error(e) or f"[ERROR] Planner failed: {e}"
        print(f"\n[Planner] {err_text}")
        return {
            "messages": [AIMessage(content=err_text)],
            "plan": [],
            "response": err_text,
        }


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

Location resolution rules (apply before calling any fetch tool):
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
4. CRITICAL: Never invent prices or tool results.

Web search rules (search_web — for live info NOT in the database: exchange rates,
weather, news/advisories, special events):
- You MUST pass a `category`, one of: "exchange", "weather", "news", "events".
  Pick the one that fits the user's intent (e.g. a forecast → "weather",
  a currency rate → "exchange", a safety advisory or strike → "news",
  a festival or holiday → "events").
- The result may include an "ANSWER:" line with the actual value (temperature,
  rate, etc.). Report that value and cite the trusted source. If there is no
  ANSWER line, summarise the snippets — do NOT fabricate a number.

After executing the step, provide a brief text summary of what you found.
When the current step asks to fetch flights/hotels/activities, location lookup is only a preparation step.
After lookup_location_options returns a valid match, you MUST call the requested fetch tool in the same step.
Do not stop after lookup unless there is NO_MATCH.
Before calling lookup_location_options, check the "Already completed" section.
If the same location was already resolved successfully earlier in this run, do NOT call lookup_location_options again.
Reuse the resolved location directly in the target fetch tool.
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
        # if hasattr(response, "tool_calls") and response.tool_calls:
        #     for tc in response.tool_calls:
        #         print(f"    → calling {_format_call(tc)}")
        #     for tc in response.tool_calls:
        #         args_str = json.dumps(tc.get("args", {}), sort_keys=True)
        #         current_call = (tc.get("name"), args_str)
                
        #         # If the exact same tool with the exact same arguments was called before:
        #         if current_call in past_tool_calls:
        #             loop_error = f"[ERROR] Infinite loop detected. The tool '{tc.get('name')}' was called again with the exact same arguments: {args_str}."
        #             print(f"\n[Loop Guard] Aborting execution to prevent API quota drain.\n {loop_error}")
                    
        #             return {
        #                 "messages": [AIMessage(content=loop_error)],
        #                 "plan": [],   # Clear the plan to force Replanner to handle the failure
        #                 "response": "I encountered an issue searching for this specific combination repeatedly. Please try adjusting your destination or origin specifications.",
        #             }
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

    # Isolate just THIS step's tool messages for the console trace (everything
    # after the most recent "[Step] ..." marker), so we don't reprint history.
    step_tool_msgs = []
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and isinstance(m.content, str) \
                and m.content.startswith("[Step]"):
            break
        if isinstance(m, ToolMessage):
            step_tool_msgs.append(m)
    step_tool_msgs.reverse()

    called_tools = {m.name for m in step_tool_msgs}

    did_lookup = "lookup_location_options" in called_tools
    only_lookup_done = called_tools == {"lookup_location_options"}

    advance_plan = not only_lookup_done

    new_cost = _extract_cost(tool_msgs)
    total_cost = state.get("calculated_total", 0.0) + new_cost

    existing_prefs = dict(state.get("user_preferences") or {})
    for m in tool_msgs:
        if getattr(m, "name", None) == "save_preference" and m.content.startswith("saved:"):
            k, v = m.content[6:].split("=", 1)
            existing_prefs[k] = v
            print(f"[Memory] Saved preference: {k}={v}")

    for m in step_tool_msgs:
        print(f"      ✓ {m.name}: {_digest_tool_result(m.content)}")

    return {
        "past_steps": state.get("past_steps", []) + [step_record],
        "plan": state["plan"][1:] if advance_plan else state["plan"],
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



REPLANNER_SYSTEM = """You are a highly analytical Travel Planning Supervisor. 
Your core responsibility is to evaluate the executed steps against the user's ORIGINAL GOAL.

AVAILABLE TOOLS YOU CAN USE IN NEW PLANS:
{tool_descriptions}

EVALUATION CRITERIA:
1. Are all EXPLICIT user requests (flights, specific hotel stars, specific airlines, budget) fulfilled?
2. Is the logical flow complete?

CONSTRAINT RELAXATION LOGIC (When exact matches fail):
If a search fails because of strict constraints (e.g., budget too low, required airline unavailable, 5-star hotels fully booked):
- DO NOT give up immediately and DO NOT return an empty response.
- GENERATE A NEW PLAN (Action: Plan) to search again by intentionally relaxing the constraints.
- Examples of relaxing constraints: drop the airline requirement, ignore the budget limit, lower the star rating...

STRATEGIC GUIDELINES FOR FAILURES & ALTERNATIVES:
Instead of giving up when a search fails, use your reasoning to decide the best recovery strategy based on the USER'S INTENT.

1. Destination-First Priority: 
   If the user requests a specific destination, your absolute top priority is getting them there. 
   - If direct flights fail, IMMEDIATELY generate a NEW Plan (Action: Plan) using 'find_connecting_flights'. 
   - If connecting flights also fail, you must generate a NEW Plan using 'suggest_alternatives' to find alternative reachable destinations.

2. The "Direct Flight" Dilemma (Ambiguity Rule): 
   If the user explicitly demanded a "direct" flight to a specific destination and none exist, their true priority is ambiguous (Do they care more about the destination, or about flying direct?). 
   - DO NOT GUESS. Instead, gather data for both scenarios: Generate a NEW Plan to simultaneously use 'find_connecting_flights' (for their requested destination) AND 'suggest_alternatives' (to find places with direct flights). 
   - Once you have the results, issue a FinalResponse presenting the dilemma with the actual options. (Example: "I couldn't find a direct flight to Tokyo. I can offer you a connecting flight to Tokyo for $900, or direct flights to Paris or London. What do you prefer?").

3. The "Never Empty-Handed" Rule (Constraint Relaxation):
   If any search fails due to strict user constraints (e.g., specific airline, budget limit, hotel stars, specific dates), you MUST NOT return a failure message with zero options. 
   - You must generate a NEW Plan (Action: Plan) that intentionally relaxes the failing constraint to find the closest possible match. 
   - When issuing the FinalResponse, clearly present this as a fallback. (Example: "I couldn't find an El Al flight, but Air France has a flight for $400.").

USING USER CONTEXT FOR ALTERNATIVES:
When relaxing constraints or suggesting new destinations, you MUST base your strategic choices on:
- The user's stored preferences and past chat history (provided below).
- Logical similarities (e.g., same continent, similar climate, similar luxury level to their original request).

DECISION ACTIONS:
A) Action: FinalResponse
   - Use if the original goal is fully met.
   - Use if you successfully found alternatives after relaxing constraints. CRITICAL: You MUST explicitly and clearly state to the user which original constraint was broken (e.g., "I couldn't find a 5-star hotel under $500, but I found this highly-rated option for $800" or "Direct flights were unavailable, so I found an alternative with a connection").
   - Use if you have exhausted all relaxed searches and absolutely nothing is available.

B) Action: Plan
   - Use to generate new steps to relax constraints or suggest alternatives when initial searches fail.
   - Be specific: write "Call fetch_hotels in Paris without budget constraint" rather than just "Find alternatives".
"""

def replan_node(state: PlanExecuteState):
    replan_count = state.get("replan_count", 0) + 1
    budget = state.get("total_budget", 0.0)
    total_cost = state.get("calculated_total", 0.0)

    # 1. FAST-TRACK ERROR HANDLING:
    # If the Executor already caught an API error, bypass the Replanner entirely.
    if state.get("response", "").startswith("[ERROR]"):
        print("\n[Replanner] ⚠️  Detected upstream execution error. Bypassing LLM summary.")
        return {
            "plan": [],
        }

    # Extract historical context for the prompt
    history_context = ""
    if state.get("chat_history"):
        history_context = f"Past conversation context:\n{state['chat_history']}"

    # 2. SAFETY LOOP BREAKER (MAX_REPLAN_CYCLES)
    if replan_count > MAX_REPLAN_CYCLES:
        print(f"\n[Replanner] ⚠️  Replan limit ({MAX_REPLAN_CYCLES}) reached — forcing final answer.")
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
        
        # Wrap the final LLM call in try/except. If the API fails now, use plain Python
        # to stitch together a raw fallback summary without relying on the model.
        try:
            final_msg = _base_model.invoke(final_prompt)
            final_text = final_msg.content
        except Exception as e:
            err_text = _is_api_error(e) or f"[ERROR] API failed during final summary: {e}"
            print(f"\n[Replanner] {err_text}")
            
            # Python-only fallback summary
            final_text = (
                "⚠️ I apologize, but I reached my API limits before I could write a nice summary for you. "
                "However, here is the raw data I managed to collect so far:\n\n"
                f"{past_steps_text}\n"
            )

        return {
            "response": final_text,
            "plan": [],
            "replan_count": replan_count,
        }
            
    # 3. PREPARE PROMPT FOR STRATEGIC EVALUATION
    past_steps_text = "\n".join(state.get("past_steps", [])) or "(none)"
    
    # If the execution queue is empty, dynamically instruct the LLM to evaluate completeness
    if state["plan"]:
        remaining_text = "\n".join(f"  - {s}" for s in state["plan"])
    else:
        remaining_text = (
            "(The execution queue is currently empty. Please thoroughly evaluate if the user's Original Goal "
            "and all constraints are fully satisfied. If yes, issue a FinalResponse. If gaps or failures exist, "
            "generate a NEW Plan using Constraint Relaxation or althernative destinations to resolve them.)"
        )

    budget_note = (
        f"Budget: ${budget:.2f}. Running total so far: ${total_cost:.2f}."
        if budget > 0 else "No specific budget limit provided."
    )
    tool_descriptions = "\n".join([f"- {t.name}: {t.description.splitlines()[0]}" for t in tools])

    prompt = (
        REPLANNER_SYSTEM.format(tool_descriptions=tool_descriptions) + "\n\n"        f"{history_context}\n\n"
        f"Original goal: {state['input']}\n\n"
        f"Completed steps:\n{past_steps_text}\n\n"
        f"Remaining planned steps:\n{remaining_text}\n\n"
        f"{budget_note}\n\n"
        "What should happen next? Evaluate carefully and choose the correct action schema."
    )

    # 4. INVOKE THE REPLANNER MODEL
    try:
        result = replanner_model.invoke(prompt)
    except Exception as e:
        err_text = _is_api_error(e) or f"[ERROR] Replanner failed: {e}"
        print(f"\n[Replanner] {err_text}")
        return {"response": err_text, "plan": [], "replan_count": replan_count}

    print(f"\n[Replanner] Decision: {type(result.action).__name__}")

    # 5. EXECUTE THE DECISION SCHEMA
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

    # Action is a revised Plan — strip any steps already done to prevent loops
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
        "response": ""
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
    
    # -------------------------------------------------------------------------
    # MEMORY UPDATE: Append the current interaction to the chat history
    # -------------------------------------------------------------------------
    current_history = state.get("chat_history", "")
    new_history = current_history + f"User: {state['input']}\nAgent: {raw.strip()}\n\n"

    return {
        "response": report, 
        "chat_history": new_history  # Save to DB via SqliteSaver
    }



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

builder.add_conditional_edges(
    "after_tools_node", route_after_tools,
    {"execute": "execute", "replan": "replan", "no_match": "no_match_injector"},
)
builder.add_edge("no_match_injector", "replan")

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
