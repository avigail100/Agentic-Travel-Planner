from langchain_core.tools import tool
import sqlite3
import os

DB_PATH = "travel_agency.db"

def _run_query(query: str, params: tuple = ()):
    """Helper function to execute a SQL query and return results."""
    if not os.path.exists(DB_PATH):
        return "Error: Database file not found. Please run init_db.py first."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        # Get column names to return a list of dictionaries
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in results]
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()

@tool
def lookup_location_options(search_term: str, service_type: str):
    """
    Use this tool FIRST to resolve ANY location mention based on the service_type.
    
    IMPORTANT: If this tool returns a list of "Available locations", 
    DO NOT apologize or ask the user for clarification. Map the requested location 
    to the correct item in the returned list and IMMEDIATELY call the target tool.
    
    Input: 
    - search_term (example: 'Israel', 'France', 'Lod', 'Paris').
    - service_type: 'flight', 'hotel', 'activity', "best_season", "car_rental", "visa_requirements", "time_difference"
    """
    search_term = search_term.strip().lower()
    svc = service_type.strip().lower()
    like_param = f"%{search_term}%"
    
    # Map the service type to the relevant table and columns for searching
    SERVICE_MAP = {
        "flight": {"table": "flights", "cols": ["origin", "destination"]},
        "hotel": {"table": "hotels", "cols": ["city"]},
        "activity": {"table": "activities", "cols": ["city"]},
        "best_season": {"table": "best_seasons", "cols": ["city"]},
        "car_rental": {"table": "car_rentals", "cols": ["city"]},
        "visa_requirements": {"table": "visa_requirements", "cols": ["origin", "destination"]},
        "time_difference": {"table": "time_differences", "cols": ["origin", "destination"]}
    }

    # If the service type is invalid, return an error message with valid options
    if svc not in SERVICE_MAP:
        valid_services = list(SERVICE_MAP.keys())
        return f"ERROR: '{svc}' is invalid, it must be one of: {valid_services}"

    # Dynamically build the query based on the service type's relevant table and columns
    config = SERVICE_MAP[svc]
    table = config["table"]
    cols = config["cols"]

    # Build a query that searches for the search term in all relevant columns for that service type using UNION to combine results
    query_parts = [f"SELECT DISTINCT {col} AS available_location FROM {table} WHERE LOWER({col}) LIKE ?" for col in cols]
    query = " UNION ".join(query_parts)
    params = tuple([like_param] * len(cols))
    
    matches = _run_query(query, params)

    # If no matches found, fetch all available locations for that service type and return an instruction to map the search term to the relevant location
    if not matches or isinstance(matches, str):
         all_loc_parts = [f"SELECT DISTINCT {col} AS loc FROM {table}" for col in cols]
         all_loc_query = " UNION ".join(all_loc_parts)
         
         all_locs = _run_query(all_loc_query)
         
         if isinstance(all_locs, list):
             available_locations = [loc['loc'] for loc in all_locs if 'loc' in loc]
             # Return a structured NO_MATCH signal — the graph will route to alternatives_node
             return {
                 "status": "NO_MATCH",
                 "search_term": search_term,
                 "service_type": svc,
                 "available_locations": available_locations,
             }
            #  # Instruct the AI to map the search term to the relevant location and execute the target tool immediately
            #  return (
            #      f"Instruction: No exact match for '{search_term}' in the {svc} database. "
            #      f"Available options are: {available_locations}. "
            #      f"ACTION: Do not tell the user there is no match. "
            #      f"Map '{search_term}' to the relevant item from this specific list "
            #      f"and execute the target tool immediately."
            #  )
         return f"No {svc} items found matching '{search_term}'."

    return matches

@tool
def fetch_flights(origin: str, destination: str = None):
    """
    Search for available flights from origin to destination.
    If you dont get destintion and only origin, Return all the flights from this origin to all destinations.
    Input: EXACT location names retrieved from the lookup_location_options tool.
            Destination can be None to fetch all flights from the origin to any destination.
    Returns: List of matching flights with origin, airline, price, and flight number, destination. If no matches, return a message indicating no flights found.
    """
    
    # Pre-processing inputs to match the database format
    origin_param = origin.strip().lower()

    if destination is not None:
        dest_param = destination.strip().lower()
        # Using LOWER() to ensure case-insensitive matching in the database
        query = "SELECT airline, price, flight_number, destination FROM flights WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
        matches = _run_query(query, (origin_param, dest_param))
    else:
        query = "SELECT airline, price, flight_number, destination FROM flights WHERE LOWER(origin) = ?"
        matches = _run_query(query, (origin_param,))
    
    if not matches or isinstance(matches, str):
        if destination:
                return f"No flights found from {origin} to {destination}."
        return f"No flights found departing from {origin}."

    return matches

@tool
def fetch_hotels(city: str, max_price: int = None):
    """
    Find hotels in a specific city from the database.
    Input: city name (string), max_price (optional integer).
    """
    query = "SELECT name, price_per_night, stars FROM hotels WHERE LOWER(city) = ?"
    params = [city.strip().lower()]
    
    if max_price is not None:
        query += " AND price_per_night <= ?"
        params.append(max_price)
    
    matches = _run_query(query, tuple(params))
    
    if not matches or isinstance(matches, str):
        return f"No hotels found in {city} meeting those criteria."
    return matches

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, duration_days: int):
    """
    Calculates the total cost for a trip including flight and hotel stay.
    Input: flight_price (float), hotel_price_per_night (float), duration_days (int).
    """
    try:
        total_hotel = float(hotel_price_per_night) * int(duration_days)
        total_grand = float(flight_price) + total_hotel
        
        return {
            "breakdown": {
                "flight": flight_price,
                "hotel_total": total_hotel,
                "days": duration_days
            },
            "total_estimate": total_grand,
            "currency": "USD"
        }
    except (ValueError, TypeError):
        return "Error: Please provide valid numbers for prices and duration."

@tool
def fetch_activities(city: str, max_price: int = None):
    """
    Find activities in a specific city from the database.
    Input: city name (string), max_price (optional integer).
    """
    query = "SELECT name, price, category FROM activities WHERE LOWER(city) = ?"
    params = [city.strip().lower()]
    
    if max_price is not None:
        query += " AND price <= ?"
        params.append(max_price)
    
    matches = _run_query(query, tuple(params))
    
    if not matches or isinstance(matches, str):
        return f"No activities found in {city} meeting those criteria."
    return matches

@tool
def fetch_visa_requirements(origin: str, destination: str):
    """
    Fetch visa requirements for travelers from the origin country to the destination country.
    IMPORTANT: This tool uses country names so if you get from the user a city name, change it to the relevant country name first before calling this tool.
    for example if the user says "I want to travel from Tel Aviv to Paris", you should resolve "Tel Aviv" to origin="israel", destination="france" before calling this tool.
    you do this by first mapping the city to the country and then calling this tool with the resolved country names.
    Returns visa requirement policy and the amount of days you can stay without a visa.
    """
    query = "SELECT days_allowed_without_visa, policy FROM visa_requirements WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
    origin_param = origin.strip().lower()
    dest_param = destination.strip().lower()
    
    matches = _run_query(query, (origin_param, dest_param))
    
    if not matches or isinstance(matches, str):
        return f"No visa requirement information found for travelers from {origin} to {destination}."
    return matches

@tool
def fetch_currency_exchange_rate(origin_currency: str, destination_currency: str):
    """
    Fetch the current exchange rate between the origin currency and the destination currency.
    IMPORTANT: This tool uses currency codes so if you get from the user a currency name, change it to the relevant currency code first before calling this tool.
    for example if the user says "I want to convert from US dollars to Israeli shekels", 
    you should resolve "US dollars" to origin_currency="USD", "Israeli shekels" to destination_currency="ILS" before calling this tool.
    you do this by first mapping the currency name to the currency code and then calling this tool with the resolved currency codes.
    Returns the exchange rate as of today.
    """
    query = "SELECT exchange_rate FROM exchange_rates WHERE LOWER(origin_currency) = ? AND LOWER(destination_currency) = ?"
    origin_param = origin_currency.strip().lower()
    dest_param = destination_currency.strip().lower()
    
    matches = _run_query(query, (origin_param, dest_param))
    
    if not matches or isinstance(matches, str):
        return f"No exchange rate information found for {origin_currency} to {destination_currency}."
    return matches

@tool
def convert_cost_to_origin_currency(cost_in_destination_currency: float, exchange_rate: float):
    """
    Convert a cost from the destination currency to the origin currency using the provided exchange rate.
    Returns the converted cost in the origin currency.
    """
    try:
        converted_cost = cost_in_destination_currency / exchange_rate
        return converted_cost
    except Exception as e:
        return f"Error converting cost: {e}"

@tool
def fetch_car_rental_agencies(city: str):
    """
    Fetch available car rental agencies in a specific city from the database.
    Returns a list of car rental agencies with price per day and car types.
    """
    query = "SELECT company, airport, price_per_day, car_type FROM car_rentals WHERE LOWER(city) = ?"
    city_param = city.strip().lower()
    
    matches = _run_query(query, (city_param,))
    
    if not matches or isinstance(matches, str):
        return f"No available car rental agencies found in {city}."
    return matches

@tool
def fetch_seasonal_recommendations(city: str):
    """
    Fetch seasonal travel recommendations for a specific city from the database.
    Returns the best season to visit and the months when it's ideal.
    """
    query = "SELECT season as best_season, months as ideal_months FROM best_seasons WHERE LOWER(city) = ?"
    city_param = city.strip().lower()
    
    matches = _run_query(query, (city_param,))
    
    if not matches or isinstance(matches, str):
        return f"No seasonal recommendations found for {city}."
    return matches

@tool
def fetch_time_difference(origin: str, destination: str):
    """
    Fetch the time difference in hours between the origin city and the destination city.
    IMPORTANT: This tool uses city names so if you get from the user a country name, change it to the relevant city name first before calling this tool.
    for example if the user says "I want to know the time difference between Israel and France", 
    you should resolve "Israel" to origin="Tel Aviv", "France" to destination="Paris" before calling this tool.
    you do this by first mapping the country name to the main city name and then calling this tool with the resolved city names.
    Returns the time difference in hours.
    """
    query = "SELECT hours_difference FROM time_differences WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
    origin_param = origin.strip().lower()
    dest_param = destination.strip().lower()
    
    matches = _run_query(query, (origin_param, dest_param))
    
    if not matches or isinstance(matches, str):
        return f"No time difference information found for {origin} to {destination}."
    return matches

@tool
def convert_time_to_destination_timezone(time_in_origin_timezone: str, time_difference_hours: int):
    """
    Convert a time from the origin timezone to the destination timezone using the provided time difference.
    Input: time_in_origin_timezone (string in format "YYYY-MM-DD HH:MM"), time_difference_hours (integer).
    Returns the converted time in the destination timezone.
    """
    try:
        from datetime import datetime, timedelta

        origin_time = datetime.strptime(time_in_origin_timezone, "%Y-%m-%d %H:%M")
        destination_time = origin_time + timedelta(hours=time_difference_hours)
        return destination_time.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        return f"Error converting time: {e}"


    
# if __name__ == "__main__":
    # origin = "London"
    # destination = "paris"
    # flights = fetch_flights.invoke({'origin': origin, 'destination': destination})
    # print(f"Available flights from {origin} to {destination}:")
    # print(flights)
    # city = "Paris"
    # hotels = fetch_hotels.invoke({'city': city})
    # print(f"Available hotels in {city} within the budget:")
    # print(hotels)
    # flight_price = 150.0
    # hotel_price_per_night = 100.0
    # nights = 3
    # total_cost = calculate_trip_cost.invoke({'flight_price': flight_price, 'hotel_price_per_night': hotel_price_per_night, 'nights': nights})
    # print(f"Total estimated cost of the trip: ${total_cost:.2f}")
    # city = "Paris"
    # activities = fetch_activities.invoke({'city': city, 'budget': 30})
    # print(f"Available activities in {city} within the budget:")
    # print(activities)
    # origin = "Israel"
    # destination = "France"
    # visa_info = fetch_visa_requirements.invoke({'origin': origin, 'destination': destination})
    # print(f"Visa requirements for travelers from {origin} to {destination}:")
    # print(visa_info)
    # origin_currency = "USD"
    # destination_currency = "ILS"
    # exchange_rate = fetch_currency_exchange_rate.invoke({'origin_currency': origin_currency, 'destination_currency': destination_currency})
    # print(f"Current exchange rate from {origin_currency} to {destination_currency}:")
    # print(exchange_rate)
    # cost_in_destination_currency = 100.0
    # exchange_rate = 3.65
    # converted_cost = convert_cost_to_origin_currency.invoke({'cost_in_destination_currency': cost_in_destination_currency, 'exchange_rate': exchange_rate})
    # print(f"Cost in origin currency: {converted_cost:.2f}")
    # city = "Paris"
    # car_rental_agencies = fetch_car_rental_agencies.invoke({'city': city})
    # print(f"Available car rental agencies in {city}:")
    # print(car_rental_agencies)
    # city = "Paris"
    # seasonal_recommendations = fetch_seasonal_recommendations.invoke({'city': city})
    # print(f"Seasonal travel recommendations for {city}:")
    # print(seasonal_recommendations)
    # city = "Paris"
    # time_difference = fetch_time_difference.invoke({'origin': "Tel Aviv", 'destination': city})
    # print(f"Time difference between Tel Aviv and {city}:")
    # print(time_difference)
    # time_in_origin_timezone = "2024-07-01 12:00"
    # converted_time = convert_time_to_destination_timezone.invoke({'time_in_origin_timezone': time_in_origin_timezone, 'time_difference_hours': time_difference[0]['hours_difference']})
    # print(f"Time in {city} when it's {time_in_origin_timezone} in Tel Aviv: {converted_time}")

    #סוכנויות רכב בשדה
    # עונה מומלצת
    # המרת שעות