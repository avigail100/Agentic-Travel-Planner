#!/usr/bin/env python3
"""
Example: Plan a trip to Paris from Tel Aviv
Run this script to see tools in action
"""

import sys
sys.path.insert(0, '/Users/omerburshan/Documents/Agentic-Travel-Planner')

from tools import (
    fetch_flights, fetch_hotels, fetch_activities,
    fetch_visa_requirements, fetch_time_difference,
    calculate_trip_cost
)

print("\n" + "=" * 70)
print("TRIP PLANNER: TEL AVIV → PARIS (3 days)")
print("=" * 70 + "\n")

# Step 1: Search for flights
print("🛫 STEP 1: Finding flights from TLV to Paris...")
flights = fetch_flights.invoke({'origin': 'TLV', 'destination': 'Paris'})
print(f"   Found {len(flights)} flight options:")
for i, flight in enumerate(flights, 1):
    print(f"   {i}. {flight['airline']} (Flight: {flight['flight_number']}) - ${flight['price']}")
selected_flight = flights[0]
flight_price = selected_flight['price']
print(f"   ✓ Selected: {selected_flight['airline']} - ${flight_price}\n")

# Step 2: Search for hotels
print("🏨 STEP 2: Finding hotels in Paris (max $150/night)...")
hotels = fetch_hotels.invoke({'city': 'Paris', 'max_price': 150})
print(f"   Found {len(hotels)} hotel options:")
for i, hotel in enumerate(hotels, 1):
    print(f"   {i}. {hotel['name']} ({hotel['stars']}⭐) - ${hotel['price_per_night']}/night")
selected_hotel = hotels[0]
hotel_price = selected_hotel['price_per_night']
print(f"   ✓ Selected: {selected_hotel['name']} - ${hotel_price}/night\n")

# Step 3: Find activities
print("🎭 STEP 3: Finding activities in Paris (max $40)...")
activities = fetch_activities.invoke({'city': 'Paris', 'max_price': 40})
print(f"   Found {len(activities)} activity options:")
for activity in activities:
    print(f"   • {activity['name']} ({activity['category']}) - ${activity['price']}")
print()

# Step 4: Calculate trip cost
print("💰 STEP 4: Calculating total trip cost...")
duration = 3
trip_cost = calculate_trip_cost.invoke({
    'flight_price': flight_price,
    'hotel_price_per_night': hotel_price,
    'duration_days': duration
})
print(f"   Flight:       ${trip_cost['breakdown']['flight']}")
print(f"   Hotel (3 nights): ${trip_cost['breakdown']['hotel_total']} (${hotel_price}/night)")
print(f"   TOTAL:        ${trip_cost['total_estimate']}\n")

# Step 5: Check visa requirements
print("📋 STEP 5: Checking visa requirements...")
visa_info = fetch_visa_requirements.invoke({'origin': 'Israel', 'destination': 'France'})
if visa_info and not isinstance(visa_info, str):
    print(f"   Policy: {visa_info[0]['policy']}")
    print(f"   Days allowed without visa: {visa_info[0]['days_allowed_without_visa']}")
else:
    print(f"   {visa_info}")
print()

# Step 6: Check time difference
print("⏰ STEP 6: Checking time difference...")
time_diff = fetch_time_difference.invoke({'origin': 'Tel Aviv', 'destination': 'Paris'})
if time_diff and not isinstance(time_diff, str):
    hours_diff = time_diff[0]['hours_difference']
    print(f"   When it's 12:00 PM in Tel Aviv, it's {12 + hours_diff}:00 in Paris")
else:
    print(f"   {time_diff}")
print()

# Summary
print("=" * 70)
print("TRIP SUMMARY")
print("=" * 70)
print(f"Destination:      Paris")
print(f"Duration:         {duration} days")
print(f"Flight:           {selected_flight['airline']} (${flight_price})")
print(f"Hotel:            {selected_hotel['name']} (${hotel_price}/night)")
print(f"Total Cost:       ${trip_cost['total_estimate']}")
print(f"Activities:       {len(activities)} options available")
print("=" * 70 + "\n")
