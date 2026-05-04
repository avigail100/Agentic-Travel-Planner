#!/usr/bin/env python3
"""
Autonomous LangGraph-powered Agent: Manages State and Routing automatically
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
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
    return {"messages": [response]}

# ============================================================================
# 4. Define Conditional Logic (Edges)
# ============================================================================
def should_continue(state: AgentState):
    """
    Determines the next path in the graph based on the model's output.
    Returns 'tools' if the model wants to call a function, otherwise 'END'.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ============================================================================
# 5. Build the Graph Workflow
# ============================================================================
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

# Define Connections
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # Loop back to the brain after tool execution

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
                content = str(last_msg.content)[:100] + "..." if len(str(last_msg.content)) > 100 else str(last_msg.content)
                print(f"[{msg_type}] Content: {content}")
                
                # Print explicit state variables for tracking
                print(f"City: {event.get('current_city')} | Budget: {event.get('total_budget')}")
                print("-" * 20)
        except Exception as e:
            print(f"\n❌ Error during graph execution: {e}")

if __name__ == "__main__":
    run_agent()
