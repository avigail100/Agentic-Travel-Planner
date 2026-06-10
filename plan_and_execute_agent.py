#!/usr/bin/env python3
"""
Travel Agent — Plan-and-Execute Architecture (Session 6)

Architecture:
  planner  →  execute  ⇄  tools  ⇄   replan  ⇄  critic → formatter → END

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
from langgraph.types import Command, interrupt
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
    ask_user,
)
from tools import KNOWN_HOSTS, WEB_CATEGORIES, hosts_for_category

from sif import (
    get_sif,
    route_after_planner,
    sif_plan_gate_node,
    route_after_plan_gate,
    route_after_tools_sif,
    sif_alternatives_gate_node,
    route_after_alternatives_gate,
    maybe_sif2_budget_interrupt,
    SIF_INTERRUPT_HANDLERS,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXECUTOR_STEPS = 4   # tool calls the executor may make per step
MAX_REPLAN_CYCLES  = 6   # how many times the replanner may issue a new plan
MAX_CRITIC_CYCLES  = 2   # how many times the critic may send the agent back for fixes

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


class CriticResult(BaseModel):
    """The critic's decision for Reflection / Reflexion-style review."""
    passed: bool = Field(
        description="True only if the draft answer satisfies the original user request."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Concrete problems found in the draft answer."
    )
    fix_steps: List[str] = Field(
        default_factory=list,
        description="Concrete execution steps needed to fix the answer. Keep them tool-friendly."
    )
    reflection: str = Field(
        default="",
        description="One short lesson from this failed attempt, used as memory for the next attempt."
    )


class SupervisorDecision(BaseModel):
    """Supervisor-Router decision: which specialist agent handles the current step."""
    agent: str = Field(
        description=(
            "Which specialist agent should execute this step. "
            "Must be one of: 'transport', 'wellbeing', 'tech'."
        )
    )
    reason: str = Field(
        description="One-line justification for the routing decision."
    )


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
    reflection_memory: List[str]
    critic_passed: bool
    critic_count: int
    approved_hosts: List[str]   # web hosts the user approved for this session (HITL)

tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_currency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone,
    lookup_location_options, find_connecting_flights,
    save_preference, suggest_alternatives, search_web,
    find_hotels_by_amenity, find_destinations_by_preference, fetch_restaurants,
    fetch_beaches, fetch_city_transport_info, ask_user,
]

_base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_retries=2)
#_base_model = ChatGroq( api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.4,max_retries=1)

planner_model   = _base_model.with_structured_output(Plan)
replanner_model = _base_model.with_structured_output(ReplanAction)
critic_model    = _base_model.with_structured_output(CriticResult)
executor_model  = _base_model.bind_tools(tools)  # kept as fallback

# ---------------------------------------------------------------------------
# Multi-Agent: per-domain tool subsets
# ---------------------------------------------------------------------------

TRANSPORT_TOOLS = [
    fetch_flights,
    fetch_hotels,
    fetch_visa_requirements,
    find_connecting_flights,
    fetch_car_rental_agencies,
    lookup_location_options,
    fetch_city_transport_info,
    calculate_trip_cost,
    suggest_alternatives,
    ask_user,
]

WELLBEING_TOOLS = [
    fetch_activities,
    fetch_seasonal_recommendations,
    fetch_restaurants,
    fetch_beaches,
    find_hotels_by_amenity,
    find_destinations_by_preference,
    lookup_location_options,
    ask_user,
]

TECH_TOOLS = [
    fetch_currency_exchange_rate,
    convert_cost_to_origin_currency,
    fetch_time_difference,
    convert_time_to_destination_timezone,
    search_web,
    save_preference,
    lookup_location_options,
    ask_user,
]

transport_model  = _base_model.bind_tools(TRANSPORT_TOOLS)
wellbeing_model  = _base_model.bind_tools(WELLBEING_TOOLS)
tech_model       = _base_model.bind_tools(TECH_TOOLS)
supervisor_model = _base_model.with_structured_output(SupervisorDecision)



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

Only include a travel warnings / safety news step when the user is planning a trip, flight, hotel etc.,
or explicitly asks about safety, warnings, risks, news, strikes, closures, or current events.

Do NOT add travel warnings for simple factual questions such as:
- current weather
- time difference
- exchange rate
- visa-only question

If the user only asks for weather, answer only the weather request.

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
        "reflection_memory": [],
        "critic_passed": False,
        "critic_count": 0,
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
# Node: supervisor_router  (Multi-Agent routing)
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM = """You are a routing supervisor for a multi-agent travel assistant.
Your ONLY job is to examine the current execution step and decide which specialist
agent is best equipped to handle it.

SPECIALIST AGENTS AND THEIR DOMAINS:

• transport — Handles logistics, transportation, and accommodation:
  fetch_flights, fetch_hotels, fetch_visa_requirements, find_connecting_flights,
  fetch_car_rental_agencies, fetch_city_transport_info, calculate_trip_cost,
  suggest_alternatives

• wellbeing — Handles experiences, activities, dining, and lifestyle:
  fetch_activities, fetch_seasonal_recommendations, fetch_restaurants,
  fetch_beaches, find_hotels_by_amenity, find_destinations_by_preference

• tech — Handles financial calculations, time conversions, web search, and preferences:
  fetch_currency_exchange_rate, convert_cost_to_origin_currency,
  fetch_time_difference, convert_time_to_destination_timezone,
  search_web, save_preference

ROUTING RULES:
1. Read the current step and pick the single most relevant agent.
2. If the step mentions flights, hotels, car rentals, or visas → transport
3. If the step mentions activities, restaurants, beaches, or experiences → wellbeing
4. If the step mentions currency, time zones, web search, or exchange rates → tech
5. Mixed steps (e.g. "Fetch flights and activities"): pick the agent whose tools are
   listed FIRST in the step, or default to 'transport' for ambiguous logistics.
6. Never invent agents. Always return exactly one of: transport, wellbeing, tech.
"""


def supervisor_router_node(state: PlanExecuteState) -> Command:
    """
    Reads the current step from the plan and routes to the appropriate
    specialist sub-agent node via a Command(goto=...).
    Falls back to 'execute_fallback' if routing fails.
    """
    if not state.get("plan"):
        # Nothing to route — let the normal flow handle it
        return Command(goto="replan")

    current_step = state["plan"][0]

    prompt = (
        f"{SUPERVISOR_SYSTEM}\n\n"
        f"Current step to route: {current_step}\n\n"
        "Return your routing decision."
    )

    try:
        decision = supervisor_model.invoke(prompt)
        agent = decision.agent.strip().lower()
        if agent not in ("transport", "wellbeing", "tech"):
            raise ValueError(f"Unknown agent: {agent!r}")
        print(f"\n[Router] Routing step '{current_step[:60]}…' → {agent.upper()} agent  ({decision.reason})")
        goto_map = {
            "transport": "transport_executor",
            "wellbeing": "wellbeing_executor",
            "tech":      "tech_executor",
        }
        return Command(goto=goto_map[agent])
    except Exception as e:
        print(f"\n[Router] Routing failed ({e}) — falling back to generic executor.")
        return Command(goto="execute_fallback")


# ---------------------------------------------------------------------------
# Helper: shared executor logic (used by all three specialist nodes + fallback)
# ---------------------------------------------------------------------------

def _run_executor(state: PlanExecuteState, model, agent_system: str, agent_label: str) -> dict:
    """
    Shared execution logic for all specialist sub-agents.
    Identical to the original execute_node body but uses the passed model
    and prepends agent_system to the prompt so each specialist stays focused.
    """
    if not state["plan"]:
        return {}

    current_step = state["plan"][0]

    past_tool_calls: set = set()
    for msg in state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc.get("args", {}), sort_keys=True)
                past_tool_calls.add((tc.get("name"), args_str))

    prefs = state.get("user_preferences") or {}
    pref_block = (
        "User preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items())
        if prefs else ""
    )

    history_block = ""
    if state.get("past_steps"):
        history_block = "Already completed / tool results so far:\n" + "\n".join(
            f"  - {s}" for s in state["past_steps"]
        )

    reflection_block = ""
    if state.get("reflection_memory"):
        reflection_block = (
            "Reflection memory from previous failed attempts in this same task:\n"
            + "\n".join(f"- {m}" for m in state["reflection_memory"] if m)
        )

    input_text = (
        f"{EXECUTOR_SYSTEM}\n\n"
        f"--- {agent_label} SPECIALIST CONTEXT ---\n"
        f"{agent_system}\n"
        f"--- END SPECIALIST CONTEXT ---\n\n"
        f"Original goal: {state['input']}\n\n"
        f"{history_block}\n\n"
        f"{'[Preferences] ' + pref_block if pref_block else ''}\n\n"
        f"{reflection_block}\n\n"
        f"Current step: {current_step}\n"
    )

    print(f"\n[{agent_label}] Running step: {current_step}")

    try:
        response = model.invoke([HumanMessage(content=input_text)])
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    → calling {_format_call(tc)}")
    except Exception as e:
        err_text = _is_api_error(e) or f"[ERROR] {agent_label} failed: {e}"
        print(f"\n[{agent_label}] {err_text}")
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
# Specialist sub-agent system prompt addenda
# ---------------------------------------------------------------------------

TRANSPORT_AGENT_SYSTEM = """You are the TRANSPORT specialist.
Your focus: flights, hotels, car rentals, visas, city transport, and trip cost calculation.
Only use tools from your specialist set. If the step requires a tool outside your set,
complete what you can and note what is missing so the supervisor can re-route.
"""

WELLBEING_AGENT_SYSTEM = """You are the WELLBEING specialist.
Your focus: activities, restaurants, beaches, seasonal tips, and destination discovery.
Only use tools from your specialist set. If the step requires a tool outside your set,
complete what you can and note what is missing so the supervisor can re-route.
"""

TECH_AGENT_SYSTEM = """You are the TECH specialist.
Your focus: currency exchange, time zone conversions, web search, and saving preferences.
Only use tools from your specialist set. If the step requires a tool outside your set,
complete what you can and note what is missing so the supervisor can re-route.
"""


# ---------------------------------------------------------------------------
# Specialist sub-agent nodes
# ---------------------------------------------------------------------------

def transport_executor_node(state: PlanExecuteState) -> dict:
    """TransportExecutor — handles logistics, flights, hotels, car rentals, visas."""
    return _run_executor(state, transport_model, TRANSPORT_AGENT_SYSTEM, "TransportExecutor")


def wellbeing_executor_node(state: PlanExecuteState) -> dict:
    """WellbeingExecutor — handles activities, restaurants, beaches, experiences."""
    return _run_executor(state, wellbeing_model, WELLBEING_AGENT_SYSTEM, "WellbeingExecutor")


def tech_executor_node(state: PlanExecuteState) -> dict:
    """TechExecutor — handles currency, time zones, web search, and preferences."""
    return _run_executor(state, tech_model, TECH_AGENT_SYSTEM, "TechExecutor")


# ---------------------------------------------------------------------------
# Node: executor  (original — kept as fallback for unroutable steps)
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
        history_block = "Already completed / tool results so far:\n" + "\n".join(
            f"  - {s}" for s in state["past_steps"]
        )

    reflection_block = ""
    if state.get("reflection_memory"):
        reflection_block = (
            "Reflection memory from previous failed attempts in this same task:\n"
            + "\n".join(f"- {m}" for m in state["reflection_memory"] if m)
        )

    input_text = (
        f"{EXECUTOR_SYSTEM}\n\n"
        f"Original goal: {state['input']}\n\n"
        f"{history_block}\n\n"
        f"{'[Preferences] ' + pref_block if pref_block else ''}\n\n"
        f"{reflection_block}\n\n"
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
    """Route to tools if the executor requested tool calls, else to web_gate."""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and getattr(last_msg, "tool_calls", None):
        return "web_gate"
    return "replan"


# ---------------------------------------------------------------------------
# Human-in-the-Loop gate: approve off-allowlist web hosts before searching
# ---------------------------------------------------------------------------

def _allowed_hosts(state: PlanExecuteState) -> set:
    """Hosts the agent may search without asking: the baseline KNOWN_HOSTS plus
    anything the user already approved this session (persisted in state)."""
    return set(KNOWN_HOSTS) | set(state.get("approved_hosts") or [])


def web_gate_node(state: PlanExecuteState) -> dict:
    """
    Sits between the executor and the ToolNode. Inspects pending `search_web`
    calls and, if a search would touch a host that is NOT in the known/approved
    set, pauses the graph (interrupt) and waits for a human decision:

      - approve : run the search and remember the host(s) for this session
      - edit    : switch the search to a different (allowed) category
      - cancel  : skip the search; the executor is told it was declined

    Any non-search tool call (DB lookups, cost math, etc.) passes straight
    through — those are local and safe, so they never interrupt.
    """
    last = state["messages"][-1] if state["messages"] else None
    tool_calls = list(getattr(last, "tool_calls", None) or [])
    if not tool_calls:
        return {}

    allowed = _allowed_hosts(state)

    # Flag EVERY search_web call that would touch an off-allowlist host. The
    # executor may emit up to two tool calls per step, so we must consider all of
    # them — a single interrupt covers the whole batch (LangGraph re-runs this
    # node top-to-bottom on resume, so one interrupt() per execution is correct).
    flagged = []          # list of (tool_call, unknown_hosts)
    all_unknown = []      # de-duplicated union of unknown hosts, for the prompt
    category = None       # the (shared) category to offer alternatives for
    for tc in tool_calls:
        if tc.get("name") != "search_web":
            continue
        cat = (tc.get("args") or {}).get("category", "")
        unknown = [h for h in hosts_for_category(cat) if h not in allowed]
        if unknown:
            flagged.append((tc, unknown))
            category = category or cat
            for h in unknown:
                if h not in all_unknown:
                    all_unknown.append(h)

    if not flagged:
        return {}   # nothing to approve — proceed to tools

    flagged_ids = {tc.get("id") for tc, _ in flagged}
    # Show the hosts for the first flagged category as the "sources" context.
    hosts = hosts_for_category(category)

    # PAUSE. Everything the terminal UI needs to render the prompt goes in the
    # payload; the graph state is checkpointed (SqliteSaver) while we wait.
    decision = interrupt({
        "type": "web_host_approval",
        "query": (flagged[0][0].get("args") or {}).get("query", ""),
        "category": category,
        "hosts": hosts,
        "unknown_hosts": all_unknown,
        "alternatives": [c for c in WEB_CATEGORIES if c != category],
    })

    action = (decision or {}).get("action", "cancel")

    if action == "approve":
        # Remember the approved host(s) for the rest of the session so we don't
        # ask again, then fall through to the tools node unchanged.
        already = list(state.get("approved_hosts") or [])
        return {"approved_hosts": already + [h for h in all_unknown if h not in already]}

    if action == "edit":
        # User chose a different category. Rewrite every flagged call's category
        # by re-emitting the AIMessage (add_messages dedupes on id, replacing it).
        new_category = decision.get("category", category)
        patched_calls = []
        for c in tool_calls:
            if c.get("id") in flagged_ids:
                c = {**c, "args": {**(c.get("args") or {}), "category": new_category}}
            patched_calls.append(c)
        new_ai = AIMessage(content=last.content, tool_calls=patched_calls, id=last.id)
        return {"messages": [new_ai]}

    # action == "cancel": drop every flagged search_web call and answer each with
    # a synthetic ToolMessage so the conversation stays valid (every tool_call
    # needs a reply). The executor sees the denial and proceeds without web data.
    kept_calls = [c for c in tool_calls if c.get("id") not in flagged_ids]
    new_ai = AIMessage(content=last.content, tool_calls=kept_calls, id=last.id)
    denials = [
        ToolMessage(
            content=(
                f"[DECLINED BY USER] The web search to "
                f"{', '.join(hosts_for_category((tc.get('args') or {}).get('category', '')))} "
                f"was not approved. Proceed using only database tools and "
                f"already-gathered information; do not retry this search."
            ),
            tool_call_id=tc.get("id"),
            name="search_web",
        )
        for tc, _ in flagged
    ]
    return {"messages": [new_ai] + denials}


def route_after_gate(state: PlanExecuteState) -> str:
    """After the gate: go to tools only if approved tool calls remain, else
    skip straight to replan (e.g. the user cancelled the only pending search)."""
    last = state["messages"][-1] if state["messages"] else None
    # If the last message is our denial ToolMessage, look back at the AIMessage.
    if isinstance(last, ToolMessage):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                last = m
                break
    if last and getattr(last, "tool_calls", None):
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

    new_cost = _extract_cost(step_tool_msgs)
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
    # NOTE: route_after_tools_sif (imported from sif.py) wraps the logic
    # above and adds the SIF-2 alternatives gate.  The graph wiring below
    # uses it in place of route_after_tools for the after_tools_node edge.
 
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
   - Use if reflection memory says the previous draft answer was incomplete, but the required data already exists in completed steps. In that case, do NOT create a new Plan. Regenerate a better FinalResponse using the completed tool results and reflection memory.
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

    reflection_block = ""
    if state.get("reflection_memory"):
        reflection_block = (
            "Reflection memory from previous failed attempts in this same task:\n"
            + "\n".join(f"- {m}" for m in state["reflection_memory"] if m)
        )

    prompt = (
        REPLANNER_SYSTEM.format(tool_descriptions=tool_descriptions) + "\n\n"
        f"{history_context}\n\n"
        f"Original goal: {state['input']}\n\n"
        f"Completed steps:\n{past_steps_text}\n\n"
        f"{reflection_block}\n\n"
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
            # SIF-2: pause and ask user before delivering over-budget result
            sif_delta = maybe_sif2_budget_interrupt(state, budget, total_cost)
            if sif_delta is not None:
                # User chose approve / new_budget / cancel — let the delta
                # propagate back; replan_node returns early so the graph
                # re-routes on the updated state.
                return {**sif_delta, "replan_count": replan_count}


        final_text = result.action.response + budget_warn

        # TEST ONLY: create an incomplete draft once, so the Critic can catch it.
        # This simulates a bad final draft while keeping the collected tool data valid.
        if (
            "critic_demo" in state["input"].lower()
            and state.get("critic_count", 0) == 0
        ):
            print("[TEST] Creating incomplete final draft for Critic demo.")
            final_text = (
                "I found flights for Paris, but I did not include the requested hotel "
                "details or the safety information that was already collected."
            )

        return {
            "response": final_text,
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
    # Check SIF level. If a new plan exists and SIF is 1, go to approval gate.
    if get_sif(state) == 1:
        return "sif_plan_gate"
    return "execute"

# ---------------------------------------------------------------------------
# Node: critic 
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a Travel Plan Critic.

Your job is to review the DRAFT final response before the user sees it.
You are NOT an executor and you must NOT call tools directly.
You must critique only using:
1. The user's original request.
2. The completed tool results in past_steps.
3. The draft final response.

Do NOT invent new requirements that are unrelated to the travel request.

ASSIGNMENT FOCUS

1. Destination suitability:
- Verify that the selected destination matches the user's request.
- Use the completed tool results as evidence, not only the wording of the final answer.
- Examples of user constraints: warm destination, beach destination, family-friendly destination, specific destination, safe destination, travel warnings.
- If the user requested a specific type of destination and there is no evidence that the selected destination matches it, fail.
- If the selected destination contradicts the user's request, fail.
- For narrow factual requests, only check that the answer refers to the correct destination or subject.

2. Destination safety / travel warning:
- This criterion is mandatory ONLY for TRAVEL DECISION REQUESTS or when the user explicitly asks about safety, travel warnings, risks, current events, protests, strikes, closures, advisories, or security concerns.
- For NON-TRAVEL-INFORMATION REQUESTS, skip this criterion completely and never fail solely because safety information is missing.
- For requests where safety validation is required, review the completed tool results for travel warnings, travel advisories, security alerts, elevated risk, unsafe destination, or similar wording.
- If safety validation is required and no safety search was performed at all for the selected/requested destination, fail and return this single data-gathering fix_step:
  Search for travel warnings for the selected destination using search_web with category=news.
- If the results explicitly mention a travel warning or elevated risk for the destination, fail unless the draft clearly warns the user about it.
- If safety information exists and shows no warning, this criterion passes.

3. Quality and style:
- Verify that the final answer is professional, clear, and service-oriented.
- Verify that the requested concrete details appear in the answer.
- If the user requested flights, hotels, activities, restaurants, prices, weather, exchange rate, visa details, or times, check that those details appear or that missing data is clearly explained.
- If the user asked for "now", "current", or "today", do not accept purely typical/seasonal information as a complete answer unless the answer clearly explains that current data was unavailable.
- Do not require unrelated trip-planning details that the user did not ask for.

FIX STEP POLICY

Choose the fix_step based on the type of problem:

A. Missing data problem:
- If the required information does NOT exist in completed tool results, return exactly ONE executable data-gathering step.
- Examples:
  fetch_hotels in Paris
  fetch_restaurants in Paris
  fetch_activities in Paris
  fetch_flights from TLV to Paris
  search_web category=weather for current weather in Paris
  search_web category=news for Paris travel warnings

B. Bad draft / missing-from-answer problem:
- If the required information already EXISTS in completed tool results but is missing from the draft final response, do NOT request another fetch/search tool.
- Return exactly this fix_step:
  Regenerate the final answer using the completed tool results and reflection memory.

OUTPUT RULES

If the answer is acceptable:
- passed = True
- issues = []
- fix_steps = []
- reflection = ""

If the answer is not acceptable:
- passed = False
- issues = specific problems
- fix_steps = exactly ONE concrete step according to the FIX STEP POLICY
- reflection = one short lesson for the next attempt

Never return an empty fix_steps list when passed=False.
"""

def critic_node(state: PlanExecuteState):
    critic_count = state.get("critic_count", 0) + 1

    if critic_count > MAX_CRITIC_CYCLES:
        print(f"\n[Critic] ⚠️  Critic limit reached — allowing final answer.")
        return {
            "critic_passed": True,
            "critic_count": critic_count,
        }

    past_steps_text = "\n".join(state.get("past_steps", [])) or "(none)"
    memory_text = "\n".join(
        f"- {m}" for m in state.get("reflection_memory", []) if m
    ) or "(none)"

    prompt = (
        f"{CRITIC_SYSTEM}\n\n"
        f"Original user request:\n{state['input']}\n\n"
        f"Completed tool/execution steps:\n{past_steps_text}\n\n"
        f"Reflection memory so far:\n{memory_text}\n\n"
        f"Draft final response:\n{state.get('response', '')}\n\n"
        "Evaluate the draft answer now."
    )

    try:
        result = critic_model.invoke(prompt)
    except Exception as e:
        if _is_api_error(e):
            print("\n[Critic] ⚠️  Critic unavailable because of API quota — skipping review.")
        else:
            print(f"\n[Critic] [ERROR] Critic failed: {e}")
        return {
            "critic_passed": True,
            "critic_count": critic_count,
        }

    print("\n[Critic] Decision:", "PASS ✅" if result.passed else "FAIL")

    if result.issues:
        print("[Critic] Issues found:")
        for issue in result.issues:
            print(f"  - {issue}")

    if result.passed:
        return {
            "critic_passed": True,
            "critic_count": critic_count,
        }

    new_memory = list(state.get("reflection_memory", []))
    if result.reflection:
        new_memory.append(result.reflection)
    elif result.issues:
        new_memory.append("Previous attempt failed: " + "; ".join(result.issues))

    clean_fix_steps = [s.strip() for s in result.fix_steps if s and s.strip()]

    # Defensive fallback: the prompt says never empty, but keep the graph robust.
    if not clean_fix_steps:
        clean_fix_steps = [
            "Collect the missing destination safety/suitability information using the appropriate available tool."
        ]

    # Keep only one step to avoid burning quota.
    clean_fix_steps = clean_fix_steps[:1]

    # If the model still returns a writing/editing instruction, convert it into
    # a Replanner-friendly regeneration step instead of sending it to Executor.
    writing_markers = ["add ", "update ", "include ", "mention ", "write ", "rewrite "]
    if any(clean_fix_steps[0].lower().startswith(m) for m in writing_markers):
        clean_fix_steps = [
            "Regenerate the final answer using the completed tool results and reflection memory."
        ]

    print("[Critic] Reflection memory updated:")
    for m in new_memory:
        print(f"  - {m}")

    print("[Critic] Sending agent back with fix step:")
    print(f"  1. {clean_fix_steps[0]}")

    return {
        "critic_passed": False,
        "critic_count": critic_count,
        "reflection_memory": new_memory,
        "plan": clean_fix_steps,
        "response": "",
    }


def route_after_critic(state: PlanExecuteState) -> str:
    if state.get("critic_passed"):
        return "formatter"

    plan = state.get("plan") or []
    if not plan:
        return "replan"

    step = plan[0].lower()

    # If the Critic says the data already exists and only the final answer
    # needs regeneration, send back to the Replanner, not to the Executor.
    regenerate_markers = [
        "regenerate the final answer",
        "completed tool results",
        "reflection memory",
    ]
    if any(marker in step for marker in regenerate_markers):
        return "replan"

    # Data-gathering fix steps should go to the Executor.
    executable_markers = [
        "fetch_",
        "fetch ",
        "search_web",
        "search for",
        "check travel warnings",
        "find_",
        "find ",
        "calculate",
        "lookup",
    ]
    if any(marker in step for marker in executable_markers):
        return "execute"

    # Unknown or writing-like step: let the Replanner turn it into a final answer
    # or a concrete executable plan, rather than burning quota in the Executor.
    return "replan"


# ---------------------------------------------------------------------------
# Node: formatter  (Session 4 report style)
# ---------------------------------------------------------------------------



def _is_trip_request(user_input: str) -> bool:
    req = (user_input or "").lower()

    trip_words = [
        "trip", "travel plan", "full trip", "vacation", "holiday",
        "itinerary", "plan me", "plan a trip",
        "טיול", "חופשה", "מסלול", "תכנון טיול",
    ]

    return any(w in req for w in trip_words)

def formatter_node(state: PlanExecuteState):
    raw = state.get("response", "")

    if not _is_trip_request(state.get("input", "")):
        clean_response = raw.strip()

        print("\n" + "=" * 40)
        print(clean_response)
        print("=" * 40 + "\n")

        current_history = state.get("chat_history", "")
        new_history = current_history + f"User: {state['input']}\nAgent: {clean_response}\n\n"

        return {
            "response": clean_response,
            "chat_history": new_history,
        }


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
        "chat_history": new_history,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(PlanExecuteState)

builder.add_node("planner",           plan_node)
# --- Multi-Agent Execution Layer ---
builder.add_node("execute",           supervisor_router_node)       # NEW: Supervisor-Router
builder.add_node("transport_executor", transport_executor_node)     # NEW: Transport sub-agent
builder.add_node("wellbeing_executor", wellbeing_executor_node)     # NEW: Wellbeing sub-agent
builder.add_node("tech_executor",      tech_executor_node)          # NEW: Tech sub-agent
builder.add_node("execute_fallback",   execute_node)                # original executor (fallback)
# ToolNode covers tools for ALL sub-agents (union of all specialist tool sets = original tools list)
builder.add_node("web_gate",          web_gate_node)
builder.add_node("tools",             ToolNode(tools))
builder.add_node("after_tools_node",  after_tools)
builder.add_node("no_match_injector", no_match_injector_node)
builder.add_node("replan",            replan_node)
builder.add_node("critic",            critic_node)
builder.add_node("formatter",         formatter_node)
builder.add_node("sif_plan_gate",         sif_plan_gate_node)
builder.add_node("sif_alternatives_gate", sif_alternatives_gate_node)

# Each specialist sub-agent feeds into the same check_executor_tools gate
# (unchanged routing: if tool calls → web_gate, else → replan)
for _specialist in ("transport_executor", "wellbeing_executor", "tech_executor", "execute_fallback"):
    builder.add_conditional_edges(
        _specialist, check_executor_tools,
        {"web_gate": "web_gate", "replan": "replan"},
    )


builder.add_edge(START,     "planner")
# builder.add_edge("planner", "execute")
# SIF-1: conditionally gate the plan before execution
builder.add_conditional_edges(
    "planner", route_after_planner,
    {"sif_plan_gate": "sif_plan_gate", "execute": "execute"},
)
builder.add_conditional_edges(
    "sif_plan_gate", route_after_plan_gate,
    {"execute": "execute", END: END},
)

# NOTE: "execute" is now supervisor_router_node, which uses Command(goto=...) to route
# directly to the correct specialist. No conditional edge needed here — the Command
# overrides any static wiring. The specialists then feed into check_executor_tools above.

# Human-in-the-loop gate: may interrupt for host approval before any web search.
builder.add_conditional_edges(
    "web_gate", route_after_gate,
    {"tools": "tools", "replan": "replan"},
)

builder.add_edge("tools", "after_tools_node")

builder.add_conditional_edges(
    "after_tools_node", route_after_tools_sif,
    {"execute": "execute", "replan": "replan",
      "no_match": "no_match_injector", "sif_alternatives_gate": "sif_alternatives_gate"},
)

builder.add_edge("no_match_injector", "replan")

# SIF-2 Gate A: after the alternatives prompt, always go to replan
builder.add_conditional_edges(
    "sif_alternatives_gate", route_after_alternatives_gate,
    {"replan": "replan"},
)

builder.add_conditional_edges(
    "replan", should_end,
    {"execute": "execute", "formatter": "critic",
     "replan": "replan", "sif_plan_gate": "sif_plan_gate"},
)
builder.add_conditional_edges(
    "critic", route_after_critic,
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
    "planner":              "📋  Building travel plan...",
    "execute":              "🎯  Supervisor routing step to specialist...",
    "transport_executor":   "✈️   Transport agent executing step...",
    "wellbeing_executor":   "🌴  Wellbeing agent executing step...",
    "tech_executor":        "🔧  Tech agent executing step...",
    "execute_fallback":     "⚙️   Executing step (fallback)...",
    "tools":                "🧳  Querying travel database...",
    "after_tools_node":     "📊  Processing tool results...",
    "no_match_injector":    "💡  Destination not found — searching for alternatives...",
    "replan":               "🔄  Reviewing progress and re-evaluating plan...",
    "critic":               "🧪  Critic is checking the draft answer and reflection memory...",
    # "formatter":          "✨  Formatting final report...",
}


def _prompt_user_question(payload: dict) -> str:
    """Render an interactive clarifying-question prompt (from the ask_user tool)
    and return the user's answer string. Supports numbered options plus
    free-text; an empty answer is allowed and passed back to the agent."""
    question = (payload.get("question") or "").strip()
    options  = payload.get("options") or []

    print("\n" + "═" * 62)
    print("💬  THE AGENT NEEDS YOUR INPUT")
    print("═" * 62)
    print(f"  {question}")
    if options:
        print("-" * 62)
        for i, opt in enumerate(options, 1):
            print(f"    {i}. {opt}")
        print("-" * 62)
        print("  Pick a number, or just type your own answer.")
    print("═" * 62)

    prompt = f"  Your answer [1-{len(options)} or text]: " if options else "  Your answer: "
    raw = input(prompt).strip()

    # A bare number selects the matching option; anything else is free-text.
    if options and raw.isdigit() and 1 <= int(raw) <= len(options):
        chosen = options[int(raw) - 1]
        print(f"  ✅  You chose: {chosen}\n")
        return chosen

    if raw:
        print(f"  ✅  Noted: {raw}\n")
    else:
        print("  (No answer given — the agent will proceed with its best guess.)\n")
    return raw


def _prompt_host_approval(payload: dict) -> dict:
    """Render the interactive approve/edit/cancel prompt for a web-host
    approval interrupt and return the user's decision dict (the resume value)."""
    query   = payload.get("query", "")
    category = payload.get("category", "")
    unknown = payload.get("unknown_hosts", [])
    hosts   = payload.get("hosts", [])
    alts    = payload.get("alternatives", [])

    print("\n" + "═" * 62)
    print("⏸️   HUMAN APPROVAL NEEDED — web search to an unapproved host")
    print("═" * 62)
    print(f"  The agent wants to run a web search:")
    print(f"    • query    : {query}")
    print(f"    • category : {category}")
    print(f"    • sources  : {', '.join(hosts) or '(none)'}")
    print(f"  ⚠️  Not on the known-hosts allowlist: {', '.join(unknown)}")
    print("-" * 62)
    print("  [a] Approve  — search these source(s) (remembered for this session)")
    print("  [e] Edit     — switch to a different, trusted category")
    print("  [c] Cancel   — skip this search; let the agent continue without it")
    print("═" * 62)

    while True:
        choice = input("  Your choice [a/e/c]: ").strip().lower()
        if choice in ("a", "approve"):
            print(f"  ✅  Approved — searching {', '.join(unknown)}.\n")
            return {"action": "approve"}
        if choice in ("c", "cancel", ""):
            print("  🚫  Cancelled — the agent will proceed without this search.\n")
            return {"action": "cancel"}
        if choice in ("e", "edit"):
            if not alts:
                print("  (No alternative categories available — pick a or c.)")
                continue
            print("  Choose a replacement category:")
            for i, c in enumerate(alts, 1):
                print(f"    {i}. {c}  ({', '.join(hosts_for_category(c))})")
            sel = input(f"  Category number [1-{len(alts)}]: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(alts):
                new_cat = alts[int(sel) - 1]
                print(f"  ✏️   Switched to '{new_cat}'.\n")
                return {"action": "edit", "category": new_cat}
            print("  Invalid selection — try again.")
            continue
        print("  Please enter 'a', 'e', or 'c'.")


def _resume_value_for(payload: dict):
    """Map an interrupt payload to the right interactive prompt and return the
    value the graph should be resumed with."""
    kind = (payload or {}).get("type")
    if kind == "user_question":
        return _prompt_user_question(payload)      # -> str (the user's answer)
    if kind == "web_host_approval":
        return _prompt_host_approval(payload)      # -> dict ({"action": ...})
    # SIF gates — dispatch to handlers defined in sif.py
    if kind in SIF_INTERRUPT_HANDLERS:
        return SIF_INTERRUPT_HANDLERS[kind](payload)
    # Unknown interrupt type — fail safe by cancelling/skipping.
    print(f"  (Unrecognised approval request: {kind!r} — skipping.)")
    return {"action": "cancel"}


def _run_with_hitl(initial_state, config):
    """Stream the graph, transparently handling human-in-the-loop interrupts:
    when the graph pauses (host approval or a clarifying question), prompt the
    user and resume with Command(resume=...), looping until the run completes
    with no interrupt."""
    stream_input = initial_state
    while True:
        interrupted = False
        for chunk in graph.stream(stream_input, config, stream_mode="updates"):
            if not chunk:
                continue
            # LangGraph surfaces a pause under the "__interrupt__" key.
            if "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                payload = intr[0].value if isinstance(intr, (list, tuple)) else intr.value
                stream_input = Command(resume=_resume_value_for(payload))
                interrupted = True
                break
            for node_name in chunk:
                if node_name in PROGRESS_MAP:
                    print(PROGRESS_MAP[node_name])
        if not interrupted:
            break

SIF_DESCRIPTIONS = {
    "1": "LOW   — approve every plan before execution",
    "2": "MEDIUM — offer search narrowing + confirm budget breaches",
    "3": "HIGH  — autonomous (only web-host approval interrupts)",
}

def _handle_sif_menu(graph, config: dict) -> None:
    """Interactive SIF settings menu, triggered by '\' in the main loop.

    Reads the current SIF from user_preferences, shows the menu, and
    writes the updated value back via graph.update_state so SqliteSaver
    persists it across sessions.
    """
    try:
        current_state = graph.get_state(config)
        prefs = dict((current_state.values or {}).get("user_preferences") or {})
    except Exception:
        prefs = {}

    current_sif = prefs.get("sif", "3")

    print("\n" + "═" * 62)
    print("⚙️   AGENT SETTINGS — Self-Independence Factor (SIF)")
    print("═" * 62)
    print(f"  Current SIF: {current_sif}  ({SIF_DESCRIPTIONS.get(current_sif, '?')})\n")
    for key, desc in SIF_DESCRIPTIONS.items():
        marker = "◀" if key == current_sif else " "
        print(f"  {key}. {desc}  {marker}")
    print("-" * 62)
    print("  Press Enter to keep current setting.")
    print("═" * 62)

    choice = input("  New SIF level [1/2/3]: ").strip()
    if choice in ("1", "2", "3") and choice != current_sif:
        prefs["sif"] = choice
        graph.update_state(config, {"user_preferences": prefs})
        print(f"  ✅  SIF updated to {choice} ({SIF_DESCRIPTIONS[choice]}).\n")
    elif choice == current_sif:
        print(f"  (SIF unchanged — still {current_sif}.)\n")
    else:
        print("  (No change.)\n")

def process_request(
    user_input: str,
    thread_id: str = "default",
    progress_callback=None,
    interrupt_callback=None,
) -> dict:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 4,
    }

    logs = []

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
                "chat_history": "",
                "reflection_memory": [],
                "critic_passed": False,
                "critic_count": 0,
                "messages": [],
                "approved_hosts": [],
            })

            saved_prefs = {}
            saved_chat_history = ""
        else:
            saved_prefs = (existing.values or {}).get("user_preferences") or {}
            saved_chat_history = (existing.values or {}).get("chat_history") or ""

    except Exception:
        saved_prefs = {}
        saved_chat_history = ""

    initial_state: PlanExecuteState = {
        "input": user_input,
        "plan": [],
        "past_steps": [],
        "response": "",
        "executor_steps": 0,
        "replan_count": 0,
        "total_budget": _extract_budget(user_input),
        "calculated_total": 0.0,
        "over_budget": False,
        "user_preferences": saved_prefs,
        "chat_history": saved_chat_history,
        "reflection_memory": [],
        "critic_passed": False,
        "critic_count": 0,
        "messages": [],
        "approved_hosts": [],
    }

    final_response = ""
    existing_message_count = 0
    try:
        before_state = graph.get_state(config)
        existing_message_count = len((before_state.values or {}).get("messages") or [])
    except Exception:
        existing_message_count = 0

    stream_input = initial_state

    while True:
        interrupted = False

        for chunk in graph.stream(stream_input, config, stream_mode="updates"):
            if not chunk:
                continue

            if "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                payload = intr[0].value if isinstance(intr, (list, tuple)) else intr.value

                if interrupt_callback:
                    decision = interrupt_callback(payload)
                else:
                    decision = {"action": "cancel"}

                stream_input = Command(resume=decision)
                interrupted = True
                break

            for node_name, node_update in chunk.items():
                if node_name in PROGRESS_MAP:
                    log = PROGRESS_MAP[node_name]
                    logs.append(log)

                    if progress_callback:
                        progress_callback(log)

                if isinstance(node_update, dict) and node_update.get("response"):
                    final_response = node_update["response"]

        if not interrupted:
            break

    if not final_response:
        try:
            state = graph.get_state(config)
            final_response = (state.values or {}).get("response", "")
        except Exception:
            final_response = ""

    try:
        final_state = graph.get_state(config)
        final_values = final_state.values or {}
    except Exception:
        final_values = {}
    return {
        "response": final_response or "I finished processing, but no final response was generated.",
        "structured_data": build_structured_data_from_messages(
            (final_values.get("messages", []) or [])[existing_message_count:],
            final_response,
            user_input,
        ),
        "logs": logs,
        }


def filter_structured_data_by_request(data: dict, user_request: str) -> dict:
    """
    Full trip request -> keep all relevant categories.
    Specific/non-trip request -> keep only requested categories.

    This prevents a follow-up like "EUR to ILS" from showing old flight/hotel cards.
    """
    req = (user_request or "").lower()

    def has_any(words):
        return any(w in req for w in words)

    trip_words = [
        "trip", "travel plan", "full trip", "vacation", "holiday",
        "itinerary", "plan me", "plan a trip", "travel to",
        "weekend in", "weekend trip", "city break",
        "טיול", "חופשה", "תכנון טיול", "מסלול",
    ]

    flight_words = ["flight", "flights", "fly", "טיסה", "טיסות"]
    hotel_words = ["hotel", "hotels", "stay", "accommodation", "מלון", "מלונות"]
    activity_words = ["activity", "activities", "things to do", "attraction", "attractions", "פעילות", "אטרקציות"]
    restaurant_words = ["restaurant", "restaurants", "food", "eat", "מסעדה", "מסעדות"]
    car_words = ["car rental", "rent a car", "rental car", "car", "רכב", "השכרת רכב"]

    bookable_count = sum([
        has_any(flight_words),
        has_any(hotel_words),
        has_any(activity_words),
        has_any(restaurant_words),
        has_any(car_words),
    ])

    is_full_trip = has_any(trip_words) or bookable_count >= 2

    if is_full_trip:
        return data

    keep = {"destination"}

    if has_any(flight_words):
        keep.add("flights")

    if has_any(hotel_words):
        keep.add("hotels")

    if has_any(activity_words):
        keep.add("activities")

    if has_any(restaurant_words):
        keep.add("restaurants")

    if has_any(car_words):
        keep.add("car_rentals")

    if has_any(["visa", "ויזה"]):
        keep.add("visa")

    if has_any(["time difference", "time zone", "שעה", "הפרש שעות"]):
        keep.add("time_difference")

    if has_any(["currency", "exchange", "exchange rate", "rate", "convert", "ils", "usd", "eur", "מטבע", "שער", "המרה"]):
        keep.add("currency_exchange")

    if has_any(["transport", "metro", "bus", "public transport", "תחבורה", "מטרו", "אוטובוס"]):
        keep.add("transport_info")

    if has_any(["season", "best time", "עונה"]):
        keep.add("seasonal_recommendations")

    if has_any(["warning", "safety", "danger", "news", "advisory", "אזהרה", "בטיחות", "מסוכן"]):
        keep.add("warning")

    if has_any(["cost", "price", "budget", "total", "עלות", "מחיר", "תקציב"]):
        keep.add("estimated_cost")

    if keep == {"destination"}:
        return {}

    return {
        key: value
        for key, value in data.items()
        if key in keep and value
    }

def get_session_preferences(thread_id: str = "default") -> dict:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 4,
    }

    try:
        state = graph.get_state(config)
        prefs = dict((state.values or {}).get("user_preferences") or {})
    except Exception:
        prefs = {}

    prefs.setdefault("sif", "3")
    return prefs

def set_session_sif(thread_id: str, sif: str) -> dict:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 4,
    }

    state = graph.get_state(config)
    prefs = dict((state.values or {}).get("user_preferences") or {})

    if sif not in ("1", "2", "3"):
        raise ValueError("SIF must be 1, 2, or 3")

    prefs["sif"] = sif
    graph.update_state(config, {"user_preferences": prefs})
    return prefs

def build_structured_data_from_messages(
    messages: list,
    final_text: str = "",
    user_request: str = "",
) -> dict:
    """
    Hybrid + stable structured-data builder.

    Stable cards:
    - Flights / hotels / activities / car rentals / restaurants come from ToolMessage results.

    Natural text:
    - Visa / time difference / seasonal recommendations / warning / estimated cost
      are taken from the final LLM answer when available, because those sections
      read better as natural language.

    This avoids parsing flights/hotels from free text, but still keeps the
    explanatory sections friendly.
    """
    import json
    import re
    from langchain_core.messages import ToolMessage

    data = {
        "destination": "",
        "flights": [],
        "hotels": [],
        "activities": [],
        "car_rentals": [],
        "restaurants": [],
        "visa": "",
        "time_difference": "",
        "currency_exchange": "",
        "seasonal_recommendations": "",
        "transport_info": "",
        "warning": "",
        "estimated_cost": "",
        "notes": "",
    }

    def parse_content(content):
        if isinstance(content, (list, dict)):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return content
        return content

    def as_list(value):
        return value if isinstance(value, list) else []

    def clean_text(value):
        value = str(value or "")
        value = re.sub(r"\*\*", "", value)
        value = re.sub(r"=+", "", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip(" .;:\n\t-*")

    def tool_display_name(tool_name: str) -> str:
        """Convert a tool name to a user-friendly generic title."""
        name = str(tool_name or "").strip()

        prefixes = (
            "fetch_", "find_", "calculate_", "convert_", "lookup_", "save_"
        )
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        return name.replace("_", " ").strip().title() or "Tool Result"

    def append_note(text: str):
        """Append a note once. Notes are rendered by the GUI as a Notes card."""
        text = clean_text(text)
        if not text:
            return

        existing = data.get("notes", "") or ""
        existing_lines = [line.strip() for line in existing.splitlines() if line.strip()]
        if text not in existing_lines:
            existing_lines.append(text)
        data["notes"] = "\n".join(existing_lines)

    def is_missing_tool_result(tool_name: str, result) -> bool:
        """Generic detector for empty / unavailable / failed tool results."""
        # lookup_location_options is an internal preparation tool. A no-match there
        # is often resolved by the executor, so do not show it as a user-facing note.
        if tool_name == "lookup_location_options":
            return False

        if result is None:
            return True

        if isinstance(result, list):
            return len(result) == 0

        if isinstance(result, dict):
            return bool(
                result.get("error")
                or result.get("no_results")
                or result.get("not_found")
                or result.get("no_direct_match")
            )

        text = str(result or "").strip().lower()
        if not text:
            return True

        missing_markers = [
            "no available",
            "no availability",
            "no results",
            "no result",
            "not found",
            "no direct match",
            "unable to find",
            "could not find",
            "cannot find",
            "error invoking tool",
            "failed",
        ]
        return any(marker in text for marker in missing_markers)

    def missing_note_for_tool(tool_name: str, result) -> str:
        """Create a generic note for any tool that returned no usable data."""
        title = tool_display_name(tool_name)
        text = clean_text(result)

        # Keep the note generic, but include a short original reason when useful.
        if "error invoking tool" in text.lower():
            return f"{title}: the tool failed, so this information could not be displayed."

        return f"{title}: no available information was found."


    def split_amenities(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return [x.strip() for x in str(value).split(",") if x.strip()]

    def unique_by(records, key):
        out = []
        seen = set()
        for r in records:
            value = str(r.get(key, "")).strip().lower()
            if value and value not in seen:
                seen.add(value)
                out.append(r)
        return out

    def extract_web_answer(text: str) -> str:
        """
        search_web returns a raw block with ANSWER + sources.
        For GUI cards we keep only ANSWER when possible.
        """
        text = str(text or "").strip()
        m = re.search(r"ANSWER:\s*(.*?)(?=\n\d+\.|\Z)", text, re.I | re.S)
        if m:
            answer = clean_text(m.group(1))
        else:
            answer = text.split("Source:")[0]
            answer = re.sub(r"Web search results for .*?:", "", answer, flags=re.I)
            answer = clean_text(answer)

        # Remove URLs/sources and keep warning readable.
        answer = re.sub(r"https?://\S+", "", answer)
        answer = re.split(r"\bSource:\b|\s+\d+\.\s+", answer, maxsplit=1, flags=re.I)[0]
        answer = clean_text(answer)
        return answer[:650]

    def extract_final_text_sections(text: str) -> dict:
        """
        Robust narrative-section extractor.

        Goal:
        - Keep cards from tools: flights/hotels/activities/restaurants/cars.
        - Keep natural narrative from the LLM: visa/time/currency/transport/season/warning/cost.
        - Prevent leakage, for example:
            Time Difference -> Currency
            City Transport -> Recommended Restaurants
            Seasonal Recommendations -> Restaurants
            Warning -> Sources/raw results
        """
        raw = str(text or "")
        raw = re.sub(r"\r\n?", "\n", raw)

        def normalize_title(title: str) -> str:
            title = clean_text(title).lower()
            title = re.sub(r"\s+", " ", title)
            return title.strip()

        def classify_title(title: str):
            """
            Return:
            - canonical key for sections we want to display as narrative cards
            - "__border__" for sections that should only stop previous text
            - None for non-section titles like "Air France"
            """
            t = normalize_title(title)

            # Sections saved as narrative text
            if "visa" in t:
                return "visa"

            if "time difference" in t or "time zone" in t:
                return "time_difference"

            if (
                "currency" in t
                or "exchange rate" in t
                or t in {"exchange", "rate"}
            ):
                return "currency_exchange"

            if "season" in t or "best time" in t:
                return "seasonal_recommendations"

            if (
                "warning" in t
                or "advisory" in t
                or "safety" in t
                or "risk" in t
            ):
                return "warning"

            if (
                "transport" in t
                or "transportation" in t
                or "metro" in t
                or "public transit" in t
                or "bus network" in t
            ):
                return "transport_info"

            if (
                "estimated trip cost" in t
                or "estimated cost" in t
                or "trip cost" in t
                or "total cost" in t
            ):
                return "estimated_cost"

            # Sections rendered from tool cards only.
            # They are borders to stop leakage but are not saved here.
            border_keywords = [
                "flight", "hotel", "accommodation", "activity", "activities",
                "attraction", "restaurant", "food", "car rental", "car rentals",
                "rental car", "beach", "shopping", "local tips", "notes",
                "additional information", "recommended restaurants",
                "recommended hotels", "recommended flights",
                "recommended activities",
            ]
            if any(k in t for k in border_keywords):
                return "__border__"

            return None

        # Candidate headings:
        #   **Title:**
        #   Title:
        #   Title (details):
        #   . Title:
        # Avoid very long captures so normal sentences are not treated as headings.
        heading_pattern = re.compile(
            r"(?i)(?:^|\n|\.\s+|\*\*\s*)\s*"
            r"(?:[-*]\s*)?"
            r"([A-Z][A-Za-z /&-]{2,60})"
            r"(?:\s*\([^)]{0,80}\))?"
            r"\s*:\s*(?:\*\*)?",
            re.S,
        )

        matches = []
        for m in heading_pattern.finditer(raw):
            title = m.group(1)
            key = classify_title(title)
            if key:
                matches.append((m, key))

        sections = {}

        for i, (m, key) in enumerate(matches):
            start = m.end()
            end = matches[i + 1][0].start() if i + 1 < len(matches) else len(raw)

            content = raw[start:end]
            content = re.split(
                r"\n\s*=+\s*\n|\n\s*ESTIMATED TOTAL COST\s*:",
                content,
                maxsplit=1,
                flags=re.I,
            )[0]

            content = clean_text(content)

            # Remove common assistant closing text if it leaks into a card.
            content = re.sub(
                r"\b(?:Do you|Would you|Let me know|If you need).*?$",
                "",
                content,
                flags=re.I,
            ).strip()

            if not content:
                continue

            # Border-only sections stop the previous section but are not saved.
            if key == "__border__":
                continue

            # Defensive guard:
            # Avoid treating car-rental "Budget: 55/day, Compact..." as Estimated Trip Cost.
            if key == "estimated_cost":
                if not re.search(r"[$€₪]|\btotal\b|\bcost\b|\bestimated\b", content, re.I):
                    continue

            # Warning/search cleanup: remove raw sources and URLs.
            if key == "warning":
                content = re.sub(r"https?://\S+", "", content)
                content = re.split(
                    r"\bSource:\b|\s+\d+\.\s+",
                    content,
                    maxsplit=1,
                    flags=re.I,
                )[0]
                content = clean_text(content)
                content = content[:650]

            sections[key] = content

        return sections


    # -------------------------
    # 1. Collect real tool data
    # -------------------------
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "")
        result = parse_content(msg.content)

        if is_missing_tool_result(tool_name, result):
            append_note(missing_note_for_tool(tool_name, result))
            continue

        if tool_name == "fetch_flights":
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                item = {
                    "airline": clean_text(r.get("airline")),
                    "price": r.get("price"),
                    "flight": clean_text(r.get("flight_number")),
                    "destination": clean_text(r.get("destination")),
                    "duration": clean_text(r.get("duration_hours")),
                    "departure": clean_text(r.get("departure_time")),
                    "arrival": clean_text(r.get("arrival_time")),
                }
                data["flights"].append(item)
                if not data["destination"] and item["destination"]:
                    data["destination"] = item["destination"]

        elif tool_name == "find_connecting_flights":
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                data["flights"].append({
                    "airline": f'{r.get("airline_1", "")} + {r.get("airline_2", "")}'.strip(" +"),
                    "price": r.get("total_price"),
                    "flight": f'{r.get("flight_1", "")} → {r.get("flight_2", "")}'.strip(" →"),
                    "destination": clean_text(r.get("final_destination")),
                    "duration": "",
                    "departure": "",
                    "arrival": "",
                    "layover": clean_text(r.get("layover")),
                })
                if not data["destination"] and r.get("final_destination"):
                    data["destination"] = clean_text(r.get("final_destination"))

        elif tool_name in ("fetch_hotels", "find_hotels_by_amenity"):
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                amenities = split_amenities(r.get("amenities"))
                room_type = clean_text(r.get("room_type"))
                if room_type and room_type not in amenities:
                    amenities.append(room_type)

                data["hotels"].append({
                    "name": clean_text(r.get("name")),
                    "price": r.get("price_per_night"),
                    "stars": r.get("stars") or 0,
                    "amenities": amenities,
                    "rating": r.get("rating", ""),
                    "room_type": room_type,
                    "breakfast_included": r.get("breakfast_included", ""),
                })

        elif tool_name == "fetch_activities":
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                data["activities"].append({
                    "name": clean_text(r.get("name")),
                    "price": r.get("price"),
                    "category": clean_text(r.get("category")),
                    "duration": clean_text(r.get("duration")),
                    "suitability": clean_text(r.get("suitable_for")),
                })

        elif tool_name == "fetch_car_rental_agencies":
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                data["car_rentals"].append({
                    "company": clean_text(r.get("company")),
                    "location": clean_text(r.get("airport")),
                    "price": r.get("price_per_day"),
                    "type": clean_text(r.get("car_type")),
                    "transmission": clean_text(r.get("transmission")),
                    "seats": r.get("seats", ""),
                })

        elif tool_name == "fetch_restaurants":
            for r in as_list(result):
                if not isinstance(r, dict):
                    continue
                data["restaurants"].append({
                    "name": clean_text(r.get("name")),
                    "cuisine": clean_text(r.get("cuisine")),
                    "price_level": clean_text(r.get("price_level")),
                    "rating": r.get("rating", ""),
                    "special_features": clean_text(r.get("special_features")),
                })

        elif tool_name == "fetch_visa_requirements":
            rows = as_list(result)
            if rows and isinstance(rows[0], dict):
                r = rows[0]
                policy = clean_text(r.get("policy"))
                days = r.get("days_allowed_without_visa", "")
                visa_type = clean_text(r.get("visa_type"))
                parts = []
                if policy:
                    parts.append(policy)
                if days not in ("", None):
                    parts.append(f"Days allowed without visa: {days}")
                if visa_type:
                    parts.append(f"Visa type: {visa_type}")
                data["visa"] = ". ".join(parts)

        elif tool_name == "fetch_time_difference":
            rows = as_list(result)
            if rows and isinstance(rows[0], dict):
                hours = rows[0].get("hours_difference")
                if hours is not None:
                    data["time_difference"] = f"{hours} hours"

        elif tool_name == "fetch_currency_exchange_rate":
            rows = as_list(result)
            if rows and isinstance(rows[0], dict):
                rate = rows[0].get("exchange_rate")
                if rate is not None:
                    data["currency_exchange"] = f"Exchange rate: {rate}"

        elif tool_name == "fetch_seasonal_recommendations":
            rows = as_list(result)
            if rows and isinstance(rows[0], dict):
                r = rows[0]
                data["seasonal_recommendations"] = (
                    f"{clean_text(r.get('best_season'))} "
                    f"({clean_text(r.get('ideal_months'))}): "
                    f"{clean_text(r.get('reason'))}"
                ).strip()

        elif tool_name == "fetch_city_transport_info":
            rows = as_list(result)
            if rows and isinstance(rows[0], dict):
                r = rows[0]
                transport_type = clean_text(r.get("transport_type"))
                ticket = r.get("average_ticket_price", "")
                car_needed = r.get("car_needed", "")
                notes = clean_text(r.get("notes"))
                parts = []
                if transport_type:
                    parts.append(f"Transport: {transport_type}")
                if ticket not in ("", None):
                    parts.append(f"Average ticket: {ticket}")
                if car_needed not in ("", None):
                    parts.append(f"Car needed: {car_needed}")
                if notes:
                    parts.append(notes)
                data["transport_info"] = ". ".join(parts)

        elif tool_name == "calculate_trip_cost":
            if isinstance(result, dict):
                total = result.get("total_estimate")
                currency = clean_text(result.get("currency"))
                if total is not None:
                    data["estimated_cost"] = f"{total} {currency}".strip()

        elif tool_name == "search_web":
            text = clean_text(result)
            low = text.lower()
            if any(w in low for w in ["warning", "safety", "unrest", "violence", "strike", "arrest", "police", "advisory"]):
                data["warning"] = extract_web_answer(text)

    data["flights"] = unique_by(data["flights"], "flight")
    data["hotels"] = unique_by(data["hotels"], "name")
    data["activities"] = unique_by(data["activities"], "name")
    data["car_rentals"] = unique_by(data["car_rentals"], "company")
    data["restaurants"] = unique_by(data["restaurants"], "name")

    # Cards selection stays deterministic.
    data = select_relevant_items_for_gui(data, user_request, final_text)

    # -------------------------
    # 2. Override narrative sections from LLM final text
    # -------------------------
    narrative = extract_final_text_sections(final_text)
    if final_text:
        text = re.sub(r"\s+", " ", final_text)

        if not narrative.get("time_difference"):
            m = re.search(
                r"([^.]*?(?:ahead of|behind)[^.]*\.)",
                text,
                re.I,
            )
            if m:
                narrative["time_difference"] = clean_text(m.group(1))

        if not narrative.get("seasonal_recommendations"):
            m = re.search(
                r"(The best season to visit .*?\.)",
                text,
                re.I,
            )
            if m:
                narrative["seasonal_recommendations"] = clean_text(m.group(1))

        if not narrative.get("warning"):
            m = re.search(
                r"((?:Recent reports|Recent news reports|Following|As of).*?(?:caution|conditions|safety|trip)\.)",
                text,
                re.I,
            )
            if m:
                narrative["warning"] = clean_text(m.group(1))

        if not narrative.get("visa"):
            m = re.search(
                r"(No visa .*?\.)",
                text,
                re.I,
            )
            if m:
                narrative["visa"] = clean_text(m.group(1))
    for key in [
        "visa",
        "time_difference",
        "currency_exchange",
        "seasonal_recommendations",
        "transport_info",
        "warning",
        "estimated_cost",
    ]:
        if narrative.get(key):
            data[key] = narrative[key]
    # Do not hide cards that were actually found by tools.
    # If the agent found flight/hotel/etc. — always show them as cards.
    return {
        key: value
        for key, value in data.items()
        if value
    }

def select_relevant_items_for_gui(data: dict, user_request: str = "", final_text: str = "") -> dict:
    """
    Deterministic selector.

    This keeps the GUI stable while still respecting requests like:
    - cheap / budget / זול
    - luxury / 5-star / spa
    - family
    - all options
    """
    import re

    request = (user_request or "").lower()
    final_lower = (final_text or "").lower()

    cheap_words = [
        "cheap", "cheapest", "budget", "low cost", "low-cost", "affordable",
        "זול", "הכי זול", "תקציב", "חסכוני"
    ]
    luxury_words = [
        "luxury", "5 star", "5-star", "five star", "spa", "deluxe",
        "יוקרתי", "5 כוכבים", "חמישה כוכבים", "ספא"
    ]
    family_words = [
        "family", "kids", "children", "families",
        "משפחה", "ילדים"
    ]
    all_words = [
        "all options", "all the options", "show all", "all flights", "all hotels",
        "כל האפשרויות", "הכל", "כל הטיסות", "כל המלונות"
    ]

    wants_cheap = any(w in request for w in cheap_words)
    wants_luxury = any(w in request for w in luxury_words)
    wants_family = any(w in request for w in family_words)
    wants_all = any(w in request for w in all_words)

    def numeric_price(item):
        value = item.get("price")
        try:
            return float(value)
        except Exception:
            return float("inf")

    def mentioned_filter(items, name_keys):
        """
        If the final answer explicitly mentions item names, keep only those.
        This makes cards match what the agent actually recommended.
        """
        if not final_lower:
            return items

        mentioned = []
        for item in items:
            names = []
            for key in name_keys:
                value = str(item.get(key, "")).strip()
                if value:
                    names.append(value)

            if any(name.lower() in final_lower for name in names):
                mentioned.append(item)

        # Only use this filter if it found something.
        return mentioned if mentioned else items

    def limit_default(items, limit=3):
        return items if wants_all else items[:limit]

    # First align with final answer names where possible
    data["flights"] = mentioned_filter(data.get("flights", []), ["airline", "flight"])
    data["hotels"] = mentioned_filter(data.get("hotels", []), ["name"])
    data["activities"] = mentioned_filter(data.get("activities", []), ["name"])
    data["car_rentals"] = mentioned_filter(data.get("car_rentals", []), ["company"])
    data["restaurants"] = mentioned_filter(data.get("restaurants", []), ["name"])

    # Then apply explicit user constraints
    if wants_cheap:
        if data["flights"]:
            data["flights"] = sorted(data["flights"], key=numeric_price)[:1]
        if data["hotels"]:
            data["hotels"] = sorted(data["hotels"], key=numeric_price)[:1]
        if data["activities"]:
            data["activities"] = sorted(data["activities"], key=numeric_price)[:3]
        if data["car_rentals"]:
            data["car_rentals"] = sorted(data["car_rentals"], key=numeric_price)[:1]
        if data["restaurants"]:
            # price_level is text, so prefer cheap/moderate if present.
            preferred = [
                r for r in data["restaurants"]
                if str(r.get("price_level", "")).lower() in ("cheap", "moderate", "low", "budget")
            ]
            data["restaurants"] = preferred or data["restaurants"][:2]

    elif wants_luxury:
        if data["hotels"]:
            luxury_hotels = [
                h for h in data["hotels"]
                if int(h.get("stars") or 0) >= 5
                or any("spa" in str(a).lower() or "deluxe" in str(a).lower() for a in h.get("amenities", []))
                or "deluxe" in str(h.get("room_type", "")).lower()
            ]
            data["hotels"] = luxury_hotels or sorted(
                data["hotels"],
                key=lambda h: (int(h.get("stars") or 0), float(h.get("rating") or 0)),
                reverse=True,
            )[:1]

        if data["restaurants"]:
            luxury_restaurants = [
                r for r in data["restaurants"]
                if str(r.get("price_level", "")).lower() in ("expensive", "luxury", "high")
                or "fine" in str(r.get("special_features", "")).lower()
            ]
            data["restaurants"] = luxury_restaurants or data["restaurants"][:2]

    elif wants_family:
        if data["activities"]:
            family_activities = [
                a for a in data["activities"]
                if "family" in str(a.get("suitability", "")).lower()
                or "children" in str(a.get("suitability", "")).lower()
                or "kids" in str(a.get("suitability", "")).lower()
            ]
            data["activities"] = family_activities or data["activities"][:3]

        if data["car_rentals"]:
            family_cars = []
            for c in data["car_rentals"]:
                try:
                    seats = int(c.get("seats") or 0)
                except Exception:
                    seats = 0
                if seats >= 5 or "suv" in str(c.get("type", "")).lower():
                    family_cars.append(c)
            data["car_rentals"] = family_cars or data["car_rentals"][:2]

    else:
        # Default: do not overwhelm the GUI.
        data["flights"] = limit_default(data.get("flights", []), 3)
        data["hotels"] = limit_default(data.get("hotels", []), 3)
        data["activities"] = limit_default(data.get("activities", []), 3)
        data["car_rentals"] = limit_default(data.get("car_rentals", []), 3)
        data["restaurants"] = limit_default(data.get("restaurants", []), 3)

    return data


def run_agent():
    print(BANNER)
    print("Plan-and-Execute travel agent — session 7 (Multi-Agent).\n")

    thread_id = input("Enter Session ID (e.g., student_01): ").strip() or "default"
    config = {
        "configurable": {"thread_id": thread_id},
        #   "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 4,
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
                "chat_history": "",
                "reflection_memory": [],
                "critic_passed": False,
                "critic_count": 0,
                "messages": [],
                "approved_hosts": [],
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
    sif_level = (graph.get_state(config).values or {}).get("user_preferences", {}).get("sif", "3")
    print(f"[SIF] Autonomy level: {sif_level} — {SIF_DESCRIPTIONS.get(sif_level, '')}  (type '/' to change)\n")


    print("Let's plan your trip! (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye — safe travels!")
                break
            # '/' opens the SIF settings menu
            if user_input == "/":
                _handle_sif_menu(graph, config)
                continue

            # Carry over user preferences and session host approvals from the
            # previous turn so we don't re-ask for the same host.
            try:
                prev_state = graph.get_state(config)
                saved_prefs = (prev_state.values or {}).get("user_preferences") or {}
                saved_chat_history = (prev_state.values or {}).get("chat_history") or ""
                saved_hosts = (prev_state.values or {}).get("approved_hosts") or []
            except Exception:
                saved_prefs, saved_hosts = {}, []
                saved_chat_history = ""

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
                "chat_history":   saved_chat_history,
                "reflection_memory": [],
                "critic_passed":  False,
                "critic_count":   0,
                "messages":       [],
                "approved_hosts": list(saved_hosts),
            }

            print("\nSearching...\n")

            _run_with_hitl(initial_state, config)

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