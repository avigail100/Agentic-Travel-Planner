#!/usr/bin/env python3
"""
Autonomous LangGraph-powered Agent: Manages State and Routing automatically
"""

import os
import json
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import all tools
from tools import (
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_curency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone
)

load_dotenv()

# ============================================================================
# 1. Define the Shared State
# ============================================================================
class AgentState(TypedDict):
    """
    The central data structure shared by all nodes.
    'messages': Appends new messages to the history using the add_messages reducer.
    'current_city': Tracks the target destination throughout the flow.
    'total_budget': Tracks the remaining funds for the trip.
    """
    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float
    step_count: int          # Internal counter to prevent infinite loops
    calculated_total: float  # Final arithmetic sum of the trip
    over_budget: bool        # Flag indicating if budget was exceeded

# ============================================================================
# 2. Initialize Model and Tools
# ============================================================================
tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_curency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone
]

# Note: max_retries=0 is kept so it fails fast on quota limits instead of hanging indefinitely
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0).bind_tools(tools)

# ============================================================================
# 3. Define the Agent Logic (Agent Node)
# ============================================================================
def call_model(state: AgentState):
    """
    Examines the current state and decides whether to trigger a tool call
    or provide a final answer to the user.
    """
    response = model.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1  # Increment step count to track iterations
        }
# Define a separate node to calculate the total cost. 
def cost_calculator_node(state: AgentState):
    """
    Hard-coded logic to extract prices from tool outputs and calculate a reliable total.
    """
    total = 0.0
    for msg in state["messages"]:
        if msg.__class__.__name__ == "ToolMessage":
            try:
                data = json.loads(msg.content)
                if isinstance(data, list):
                    for item in data:
                        # Adding flight or hotel prices found in list results
                        total += float(item.get("price", 0))
                        total += float(item.get("price_per_night", 0))
                elif isinstance(data, dict):
                    # Adding results from calculation tools
                    total += float(data.get("total_estimate", 0))
            except:
                continue
    return {"calculated_total": total}

def budget_check_node(state: AgentState):
    """
    Compares the calculated total against the user's budget.
    """
    budget = state.get("total_budget", 0)
    total = state.get("calculated_total", 0)
    
    is_over = False
    if budget > 0 and total > budget:
        is_over = True
    return {"over_budget": is_over}

def formatter_node(state: AgentState):
    """
    A strict formatter that cleans Markdown noise and restructures the final text.
    """
    last_msg = state["messages"][-1]
    raw_content = last_msg.content

    # Extract text from complex multi-block content if necessary
    if isinstance(raw_content, list):
        raw_content = " ".join([b.get("text", "") for b in raw_content if b.get("type") == "text"])

    # 1. Remove Markdown bolding (**text** -> text)
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_content)
    
    # 2. Standardize newlines and remove excessive whitespace
    clean_text = clean_text.replace("\\n", "\n").strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    # 3. Build a structured report header
    report = f"  TRIP SUMMARY FOR: {state.get('current_city', 'Your Destination').upper()}\n"
    report += "=" * 40 + "\n\n"
    report += clean_text + "\n\n"
    report += "=" * 40 + "\n"

    # 4. Inject the calculated total and budget warnings
    total = state.get("calculated_total", 0)
    if total > 0:
        report += f" ESTIMATED TOTAL COST: ${total:.2f}\n"
        if state.get("over_budget"):
            report += " BUDGET ALERT: This plan exceeds your set limit!\n"
    
    report += "=" * 40

    return {"messages": [AIMessage(content=report)]}

# ============================================================================
# 4. Define Conditional Logic (Edges)
# ============================================================================
def should_continue(state: AgentState):
    """
    Determines the next path in the graph based on the model's output.
    Returns 'tools' if the model wants to call a function, otherwise 'END'.
    """
    """
    Routing logic: Tools -> Step Limit -> Final Report.
    """
    # 1. Check for infinite loops (Safety Guardrail)
    # if state.get("step_count", 0) >= 20:
        # return "human_review"
    
    last_message = state["messages"][-1]
    # 2. Check if the model requested a tool
    if last_message.tool_calls:
        return "tools"
    # 3. Otherwise, proceed to final cost calculation and formatting
    # return "cost_calc"
    return "formatter"
    

# def formatter_node(state: AgentState):
#     last_msg = state["messages"][-1]
#     content = last_msg.content
    
#     # Basic formatting to clean up the model's response for user display
#     if isinstance(content, list):
#         # If the content is a list of blocks, extract text
#         content = " ".join([block.get("text", "") for block in content if block.get("type") == "text"])
    
#     formatted_text = content.strip().replace("\\n", "\n")
    
#     # Append total cost if available and not already included in the response
#     total = state.get("calculated_total", 0)
#     if total > 0 and str(total) not in formatted_text:
#         formatted_text += f"\n\n--- Estimated Total Cost: ${total} ---"
#         if state.get("over_budget"):
#             formatted_text += "\n Note: This trip exceeds your specified budget!"

#     # Return the formatted message as an AIMessage to be sent back to the user
#     from langchain_core.messages import AIMessage
#     return {"messages": [AIMessage(content=formatted_text)]}

# ============================================================================
# 5. Build the Graph Workflow
# ============================================================================
builder = StateGraph(AgentState)

# Nodes
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("formatter", formatter_node)
# builder.add_node("cost_calc", cost_calculator_node)
# builder.add_node("budget_check", budget_check_node)

# Edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "formatter": "formatter",
    # "cost_calc": "cost_calc"
})
builder.add_edge("tools", "agent")  # Loop back to the brain after tool execution
# builder.add_edge("cost_calc", "budget_check")
# builder.add_edge("budget_check", "formatter")
builder.add_edge("formatter", END)

# Compile the Graph
graph = builder.compile()

# ============================================================================
# 6. Interactive Execution
# ============================================================================
def run_agent():
    # thread_id identifies a specific conversation session
    config = {"configurable": {"thread_id": "student_session_01"}}
    
    print("--- Autonomous Travel Agent Started ---")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: 
            break
            
        # Run the graph and stream the state updates
        try:
            for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
                last_msg = event["messages"][-1]
                
                # Print the type of message and a snippet of its content
                msg_type = last_msg.__class__.__name__
                # content = str(last_msg.content)[:200] + "..." if len(str(last_msg.content)) > 200 else str(last_msg.content)
                content = str(last_msg.content)
                print(f"[{msg_type}] Content: {content}")
                
                # Print explicit state variables for tracking
                print(f"City: {event.get('current_city')} | Budget: {event.get('total_budget')}")
                print("-" * 20)
        except Exception as e:
            print(f"\n Error during graph execution: {e}")

if __name__ == "__main__":
    run_agent()
