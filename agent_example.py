#!/usr/bin/env python3
"""
LLM-powered Agent: Takes ONE user prompt, LLM decides which tools to call
"""

import sys
sys.path.insert(0, '/Users/omerburshan/Documents/Agentic-Travel-Planner')

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

from tools import (
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost, fetch_curency_exchange_rate,
    convert_cost_to_origin_currency, fetch_car_rental_agencies,
    fetch_seasonal_recommendations, convert_time_to_destination_timezone
)

load_dotenv()

# ============================================================================
# ONE USER PROMPT (LLM WILL PARSE IT)
# ============================================================================
user_prompt = "I want to plan a 3-day trip from Tel Aviv to Paris. I have a budget of $150/night for hotels. Please find the best flight and hotel options, and calculate the total cost. Also check visa requirements and time difference."

print("\n" + "=" * 70)
print("USER PROMPT:")
print("=" * 70)
print(user_prompt)
print("=" * 70)

# ============================================================================
# INITIALIZE LLM WITH TOOLS
# ============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0)

# Relevant tools to bind
relevant_tools = [
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost
]

llm_with_tools = llm.bind_tools(relevant_tools)

# Tool mapping (all available tools)
tool_map = {
    "fetch_flights": fetch_flights,
    "fetch_hotels": fetch_hotels,
    "fetch_activities": fetch_activities,
    "fetch_visa_requirements": fetch_visa_requirements,
    "fetch_time_difference": fetch_time_difference,
    "calculate_trip_cost": calculate_trip_cost,
    "fetch_curency_exchange_rate": fetch_curency_exchange_rate,
    "convert_cost_to_origin_currency": convert_cost_to_origin_currency,
    "fetch_car_rental_agencies": fetch_car_rental_agencies,
    "fetch_seasonal_recommendations": fetch_seasonal_recommendations,
    "convert_time_to_destination_timezone": convert_time_to_destination_timezone,
}

# ============================================================================
# AGENT LOOP: Keep calling tools until LLM is done
# ============================================================================
print("\n🤖 LLM ANALYZING PROMPT AND CALLING TOOLS...\n")

try:
    messages = [HumanMessage(content=user_prompt)]
    ai_msg = llm_with_tools.invoke(messages)

    max_iterations = 10
    iteration = 0

    while ai_msg.tool_calls and iteration < max_iterations:
        iteration += 1
        tool_responses = []
        
        # Execute each tool call
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"→ Calling {tool_name}({tool_args})")
            
            selected_tool = tool_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            print(f"  ✓ Done\n")
            
            tool_responses.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
        
        # Add to conversation history
        messages.append(ai_msg)
        messages.extend(tool_responses)
        
        # Get next response from LLM
        ai_msg = llm_with_tools.invoke(messages)

    # ============================================================================
    # LLM GENERATES FINAL RESPONSE
    # ============================================================================
    print("=" * 70)
    print("AGENT FINAL ANSWER:")
    print("=" * 70)
    if hasattr(ai_msg, 'content') and ai_msg.content:
        print(ai_msg.content)
    else:
        print("(Empty response from LLM)")
    print("=" * 70 + "\n")

except Exception as e:
    print(f"\n❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
