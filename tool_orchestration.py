import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import fetch_flights, fetch_hotels, calculate_trip_cost, fetch_activities, fetch_visa_requirements, fetch_curency_exchange_rate, convert_cost_to_origin_currency, fetch_car_rental_agencies, fetch_seasonal_recommendations

load_dotenv()
# 1. Initialize the Model (Gemini 2.5 Flash - 2026 Standard)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Tool Binding
tools = [fetch_flights, fetch_hotels, calculate_trip_cost]
llm_with_tools = llm.bind_tools(tools)
try:
    # 3. Step A: Intent Detection
    query = "I want to find a flight from London to tokyo" 
    ai_msg = llm_with_tools.invoke(query)

    # 4. Step B: Manual Execution (The Manual Loop)
    if ai_msg.tool_calls:
        tool_responses = []
        
        # Mapping tool names to actual functions
        tool_map = {
        "fetch_flights": fetch_flights,
        "fetch_hotels": fetch_hotels,
        "calculate_trip_cost": calculate_trip_cost,
        "fetch_activities": fetch_activities,
        "fetch_visa_requirements": fetch_visa_requirements,
        "fetch_currency_exchange_rate": fetch_curency_exchange_rate,
        "convert_cost_to_origin_currency": convert_cost_to_origin_currency,
        "fetch_car_rental_agencies": fetch_car_rental_agencies,
        "fetch_seasonal_recommendations": fetch_seasonal_recommendations
        }
        
        # Iterate over each tool call identified by the model and execute them
        for tool_call in ai_msg.tool_calls:
            print(f"Model identified intent to call: {tool_call['name']}")
            selected_tool = tool_map[tool_call["name"]]
            
            # Execute the Python function with AI-generated arguments
            tool_output = selected_tool.invoke(tool_call["args"])
            print(f"Tool Output: {tool_output}")
            
            # Append the tool output back to the conversation history for the final response
            tool_responses.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))

        # 5. Step C: Closing the Loop
        # Send history + tool output back to the LLM for a final human response
        final_response = llm_with_tools.invoke([
        HumanMessage(content=query),
        ai_msg,
        *tool_responses # Unpacking the list of tool responses into a single list of messages
        ])
        print(f"\nFinal Agent Response: {final_response.content}")

except Exception as e:
    print(f"An error occurred: {e}")