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

    When this tool returns a list of available locations and NO exact match was found,
    YOU must decide — before doing anything else — whether the situation is:

    CASE A — Semantic equivalence (country→hub, city→airport, region→main city):
        Examples: "Israel" → "TLV", "Rechovot" → "TLV", "Britain" → "London",
                  "Japan" → "Tokyo", "Ben Gurion" → "TLV"
        Action: Silently map to the correct item from the list and immediately call
                the target tool. Do NOT inform the user. Do NOT ask for confirmation.

    CASE B — Genuine destination mismatch (the user asked for somewhere we don't serve):
        Examples: User asked for "Buenos Aires" and we only have European cities.
        Action: Return the string "NO_MATCH:<search_term>" so the graph can
                route to the alternatives flow.
                Format exactly: NO_MATCH:<original search term>

    Input:
    - search_term: e.g. 'Israel', 'France', 'Lod', 'Paris'.
    - service_type: 'flight', 'hotel', 'activity', 'best_season', 'car_rental',
                    'visa_requirements', 'time_difference'
    """
    raw = search_term.strip().lower()
    svc = service_type.strip().lower()
    like_param = f"%{raw}%"

    SERVICE_MAP = {
        "flight":            {"table": "flights",           "cols": ["origin", "destination"]},
        "hotel":             {"table": "hotels",            "cols": ["city"]},
        "activity":          {"table": "activities",        "cols": ["city"]},
        "best_season":       {"table": "best_seasons",      "cols": ["city"]},
        "car_rental":        {"table": "car_rentals",       "cols": ["city"]},
        "visa_requirements": {"table": "visa_requirements", "cols": ["origin", "destination"]},
        "time_difference":   {"table": "time_differences",  "cols": ["origin", "destination"]},
    }

    if svc not in SERVICE_MAP:
        return f"ERROR: '{svc}' is invalid, must be one of: {list(SERVICE_MAP.keys())}"

    table = SERVICE_MAP[svc]["table"]
    cols  = SERVICE_MAP[svc]["cols"]

    # Direct LIKE search first
    query_parts = [
        f"SELECT DISTINCT {col} AS available_location FROM {table} WHERE LOWER({col}) LIKE ?"
        for col in cols
    ]
    query = " UNION ".join(query_parts)
    matches = _run_query(query, tuple([like_param] * len(cols)))

    if matches and not isinstance(matches, str):
        return matches

    # No direct match — return full list so the LLM can apply CASE A or CASE B logic
    all_loc_parts = [f"SELECT DISTINCT {col} AS loc FROM {table}" for col in cols]
    all_locs = _run_query(" UNION ".join(all_loc_parts))
    available_locations = (
        [loc["loc"] for loc in all_locs if "loc" in loc]
        if isinstance(all_locs, list) else []
    )

    return {
        "no_direct_match": True,
        "search_term": search_term.strip(),
        "service_type": svc,
        "available_locations": available_locations,
        "instruction": (
            "Decide: is this semantic equivalence (CASE A) or a genuine mismatch (CASE B)? "
            "CASE A → silently pick the correct item from available_locations and call the target tool immediately. "
            "CASE B → do not retry, return the exact string:'NO_MATCH:{search_term}'."
        ),
    }

@tool
def fetch_flights(origin: str, destination: str = None):
    """
    Search for available flights from origin to destination.
    If you dont get destintion and only origin, Return all the flights from this origin to all destinations.
    Input: EXACT location names retrieved from the lookup_location_options tool.
            Destination can be None to fetch all flights from the origin to any destination.
    Returns: List of matching flights with destination, airline, price, flight number, availability, duration, departure time and arrival time. 
    If no matches, return a message indicating no flights found.
    """
    
    # Pre-processing inputs to match the database format
    origin_param = origin.strip().lower()

    if destination is not None:
        dest_param = destination.strip().lower()
        # Using LOWER() to ensure case-insensitive matching in the database
        query = "SELECT airline, price, flight_number, destination, duration_hours, departure_time, arrival_time FROM flights WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
        matches = _run_query(query, (origin_param, dest_param))
    else:
        query = "SELECT airline, price, flight_number, destination, duration_hours, departure_time, arrival_time FROM flights WHERE LOWER(origin) = ?"
        matches = _run_query(query, (origin_param,))
    
    if not matches or isinstance(matches, str):
        if destination:
                return f"No flights found from {origin} to {destination}."
        return f"No flights found departing from {origin}."

    return matches

@tool
def find_connecting_flights(origin: str, destination: str):
    """
    Search for connecting flights with exactly one stop (layover) from origin to destination.
    Use this if direct flights (fetch_flights) are not available.
    Returns a list of flight combinations including the layover city, flight numbers, and total price.
    """
    # the qury routes with exactly one layover by joining the flights table twice to find pairs of flights where the destination of the first flight matches the origin of the second flight, 
    # and the first flight departs from the specified origin while the second flight arrives at the specified destination. 
    # It returns details about both flights and calculates the total price for the connecting trip.
    query = """
    SELECT 
        f1.origin AS origin, 
        f1.destination AS layover, 
        f2.destination AS final_destination,
        f1.airline AS airline_1, 
        f1.flight_number AS flight_1, 
        f1.price AS price_1,
        f2.airline AS airline_2, 
        f2.flight_number AS flight_2, 
        f2.price AS price_2,
        (f1.price + f2.price) AS total_price
    FROM flights f1
    JOIN flights f2 ON LOWER(f1.destination) = LOWER(f2.origin)
    WHERE LOWER(f1.origin) = ? AND LOWER(f2.destination) = ?
    """
    
    origin_param = origin.strip().lower()
    dest_param = destination.strip().lower()
    
    matches = _run_query(query, (origin_param, dest_param))
    
    if not matches or isinstance(matches, str):
        return f"No connecting flights found with exactly one stop from {origin} to {destination}."
    
    return matches

@tool
def fetch_hotels(city: str, max_price: int = None):
    """
    Find hotels in a specific city from the database.
    Input: city name (string), max_price (optional integer).
    When showing hotel results to the user, include ALL returned fields (name, price_per_night, stars, amenities, rating, room_type, breakfast_included) to give them a complete picture of the options available.
    """
    query = "SELECT name, price_per_night, stars, amenities, rating, room_type, breakfast_included FROM hotels WHERE LOWER(city) = ?"
    params = [city.strip().lower()]
    
    if max_price is not None:
        query += " AND price_per_night <= ?"
        params.append(max_price)
    
    matches = _run_query(query, tuple(params))
    
    if not matches or isinstance(matches, str):
        return f"No hotels found in {city} meeting those criteria."
    return matches

@tool
def find_hotels_by_amenity(amenity: str, max_price: int = None):
    """
    Find hotels across all cities that include a specific amenity.
    Use this when the user asks for hotels with spa, pool, WiFi, breakfast, etc.,
    and does not specify a city.
    """

    query = """
    SELECT city, name, price_per_night, stars, amenities,
           rating, room_type, breakfast_included
    FROM hotels
    WHERE LOWER(amenities) LIKE ?
    """

    params = [f"%{amenity.strip().lower()}%"]
    if max_price is not None:
        query += " AND price_per_night <= ?"
        params.append(max_price)

    matches = _run_query(query, tuple(params))
    print("matches:", matches)

    if not matches or isinstance(matches, str):
        return f"No hotels found with amenity: {amenity}."

    return matches

@tool
def calculate_trip_cost(items: list[dict]):
    """
    Calculates the total grand cost of the trip from a list of dynamic expense items.
    Use this tool to sum up ALL expenses found during the planning (flights, hotels, activities, car rentals, food, etc.).
    
    Input 'items' is a list of dictionaries, where each dictionary MUST contain:
    - 'name': str (e.g., 'Flight', 'Hotel Stay', 'Louvre Museum')
    - 'price': float (the cost per unit/night/ticket)
    - 'quantity': int (optional, defaults to 1. Use for number of nights, tickets, days of car rental, etc.)
    """
    try:
        breakdown = {}
        total_grand = 0.0
        
        for item in items:
            name = item.get("name", "Unknown Expense")
            price = float(item.get("price", 0.0))
            quantity = int(item.get("quantity", 1))
            
            item_total = price * quantity
            total_grand += item_total
            
            breakdown[name] = {
                "unit_price": price,
                "quantity": quantity,
                "item_total": item_total
            }
            
        return {
            "breakdown": breakdown,
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
    When presenting activity results, include all returned fields (name, price, category, duration, suitable_for) to give the user a comprehensive view of their options.
    """
    query = "SELECT name, price, category, duration, suitable_for FROM activities WHERE LOWER(city) = ?"
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
    Returns visa requirement policy, the amount of days you can stay without a visa and the visa type.
    """
    query = "SELECT days_allowed_without_visa, policy, visa_type FROM visa_requirements WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
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
    This tool supports: direct exchange rates, reverse exchange rates and indirect conversions through an intermediate currency if needed
    Returns the exchange rate as of today.
    """
    query = "SELECT exchange_rate FROM exchange_rates WHERE LOWER(origin_currency) = ? AND LOWER(destination_currency) = ?"

    origin_param = origin_currency.strip().lower()
    dest_param = destination_currency.strip().lower()
    
    # First try to find the direct exchange rate
    res = _run_query(query, (origin_param, dest_param))
    
    if res and not isinstance(res, str):
        return res

    # if no direct exchange rate is found, try to find the reverse exchange rate and invert it
    res_reverse = _run_query(query, (dest_param, origin_param))
    if res_reverse and not isinstance(res_reverse, str):
        original_rate = res_reverse[0]['exchange_rate']
        return [{"exchange_rate": 1 / original_rate}]
    
    # Try to find a bridge currency
    bridge_query = """
    SELECT
        r1.destination_currency AS bridge_currency,
        r1.exchange_rate * r2.exchange_rate AS exchange_rate
    FROM exchange_rates r1
    JOIN exchange_rates r2
        ON LOWER(r1.destination_currency) = LOWER(r2.origin_currency)
    WHERE LOWER(r1.origin_currency) = ?
    AND LOWER(r2.destination_currency) = ?
    LIMIT 1
    """

    bridge_res = _run_query(bridge_query, (origin_param, dest_param))

    if bridge_res and not isinstance(bridge_res, str):
        return bridge_res

    return f"No exchange rate found between {origin_currency} and {destination_currency}."

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
    Returns a list of car rental agencies with price per day, car types, transmission, and seats.
    Use this tool for requests about: automatic/manual cars, SUV/economy/luxury cars, family cars, number of seats, rental prices
    """
    query = "SELECT company, airport, price_per_day, car_type, transmission, seats FROM car_rentals WHERE LOWER(city) = ?"
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
    also include the reason for the recommendation if available (e.g. weather, events, tourist crowds) to help the user understand the context of the recommendation.
    """
    query = "SELECT season as best_season, months as ideal_months, reason FROM best_seasons WHERE LOWER(city) = ?"
    city_param = city.strip().lower()
    
    matches = _run_query(query, (city_param,))
    
    if not matches or isinstance(matches, str):
        return f"No seasonal recommendations found for {city}."
    return matches

# TODO: לחדד את עניין הפילטור לפי ים ומזג אוויר וכו
@tool
def find_destinations_by_preference(preference: str):
    """
    Find destinations across all cities based on a weather or season preference.
    Use this when the user asks where to go based on weather, season,
    climate, or travel timing without specifying a city.
    Examples: warm weather, beach weather, not humid, cherry blossoms, outdoor events
    """

    month_mapping = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    pref = preference.strip().lower()

    # Month-based search
    if pref in month_mapping:
        month = month_mapping[pref]

        query = """
        SELECT city, season, months, reason
        FROM best_seasons
        WHERE
            (start_month <= end_month AND ? BETWEEN start_month AND end_month)
            OR
            (start_month > end_month AND (? >= start_month OR ? <= end_month))
        """

        matches = _run_query(query, (month, month, month))

    # General preference search
    else:
        query = """
        SELECT city, season, months, reason
        FROM best_seasons
        WHERE LOWER(season) LIKE ?
           OR LOWER(reason) LIKE ?
        """

        param = f"%{pref}%"
        matches = _run_query(query, (param, param, param))

    if not matches or isinstance(matches, str):
        return f"No destinations found for preference: {preference}."

    return matches

@tool
def fetch_time_difference(origin: str, destination: str):
    """
    Fetch the time difference in hours between the origin city and the destination city.
    IMPORTANT: This tool uses city names so if you get from the user a country name, change it to the relevant city name first before calling this tool.
    for example if the user says "I want to know the time difference between Israel and France", 
    you should resolve "Israel" to origin="Tel Aviv", "France" to destination="Paris" before calling this tool.
    you do this by first mapping the country name to the main city name and then calling this tool with the resolved city names.
    Supports: direct lookup, reverse lookup and indirect lookup through an intermediate location.
    Returns the time difference in hours.
    """
    query = "SELECT hours_difference FROM time_differences WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
    origin_param = origin.strip().lower()
    dest_param = destination.strip().lower()
    
      # Direct lookup
    matches = _run_query(query, (origin_param, dest_param))

    if matches and not isinstance(matches, str):
        return matches

    # Reverse lookup
    reverse_matches = _run_query(query, (dest_param, origin_param))

    if reverse_matches and not isinstance(reverse_matches, str):
        reversed_diff = reverse_matches[0]["hours_difference"]
        return [{"hours_difference": -reversed_diff}]

    # Bridge lookup through intermediate location
    bridge_query = """
    SELECT
        t1.destination AS bridge_location,
        t1.hours_difference + t2.hours_difference AS hours_difference
    FROM time_differences t1
    JOIN time_differences t2
        ON LOWER(t1.destination) = LOWER(t2.origin)
    WHERE LOWER(t1.origin) = ?
      AND LOWER(t2.destination) = ?
    LIMIT 1
    """

    bridge_matches = _run_query(bridge_query, (origin_param, dest_param))

    if bridge_matches and not isinstance(bridge_matches, str):
        return bridge_matches

    return f"No time difference information found for {origin} to {destination}."

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

@tool
def save_preference(key: str, value: str) -> str:
    """Save a user travel preference for future sessions.
    key: one of preferred_airline, food_preference, travel_style, seat_preference, class_preference
    value: the preference value (e.g. 'El Al', 'kosher', 'luxury')
    """
    return f"saved:{key}={value}"