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
def lookup_location_options(search_terms: list[str], service_type: str):
    """
    Use this tool FIRST to resolve ALL location mentions based on the service_type in a SINGLE call.
    You can pass one or multiple locations at once (e.g., origin and destination) to save time.

    When this tool returns its dictionary of results, check each term. 
    If a term has "no_direct_match": True, decide — before doing anything else:

    CASE A — Semantic equivalence (country→hub, city→airport, region→main city):
        Examples: "Israel" → "TLV", "Rehovot" → "TLV", "Britain" → "London",
                  "Japan" → "Tokyo", "Ben Gurion" → "TLV"
        Action: Silently map to the correct item from the list and immediately call
                the target tool. Do NOT inform the user. Do NOT ask for confirmation.

    CASE B — Genuine destination mismatch (the user asked for somewhere we don't serve):
        Examples: User asked for "Buenos Aires" and we only have European cities.
        Action: Return the string "NO_MATCH:<search_term>" so the graph can
                route to the alternatives flow.
                Format exactly: NO_MATCH:<original search term>

    Input:
    - search_terms: A list of strings, e.g. ['Israel', 'France'] or ['Lod'].
    - service_type: 'flight', 'hotel', 'activity', 'best_season', 'car_rental',
                    'visa_requirements', 'time_difference'
    Returns:
    A dictionary mapping each requested search term to its specific result, UNLESS an invalid service_type is provided (which returns a error string). 
    For each search term in the returned dictionary, the value will be one of two structures:

    1. Direct Match (List of Dicts):
       If the requested location matches an entry in the database, the value is the raw results directly from the SQL query.
       Example: "paris": [{"available_location": "Paris"}]

    2. No Direct Match (Dictionary):
       If the location is NOT found directly, the value is a JSON-like dictionary containing all valid locations for that service type.
       This dictionary acts as a signal for you to execute CASE A or CASE B logic.
       Example:
       "lod": {
           "no_direct_match": True,
           "search_term": "lod",
           "service_type": "flight",
           "available_locations": ["Tel Aviv", "London", ...],
           "instruction": "<Specific instructions on how to handle CASE A vs CASE B>"
       }

    3. Error Message (String):
       If an unsupported service_type is requested, the tool bypasses the dictionary and returns a clear string indicating the error.
       Format: "ERROR: '<service_type>' is invalid, must be one of: [...]"
    """
    # check if no duplicates and lowercase and strip the tearms
    cleaned_terms = [term.strip().lower() for term in search_terms]
    if len(cleaned_terms) > 1 and len(set(cleaned_terms)) != len(cleaned_terms):
        return "ERROR: Duplicate locations detected in search_terms. Origin and destination cannot be the same city."
    
    svc = service_type.strip().lower()

    SERVICE_MAP = {
        "flight":            {"table": "flights",           "cols": ["origin", "destination"]},
        "hotel":             {"table": "hotels",            "cols": ["city"]},
        "activity":          {"table": "activities",        "cols": ["city"]},
        "best_season":       {"table": "best_seasons",      "cols": ["city"]},
        "car_rental":        {"table": "car_rentals",       "cols": ["city"]},
        "visa_requirements": {"table": "visa_requirements", "cols": ["origin", "destination"]},
        "time_difference":   {"table": "time_differences",  "cols": ["origin", "destination"]},
    }
    
    all_locs_cache = None
    results = {}
    for term, raw in zip(search_terms, cleaned_terms):
        like_param = f"%{raw}%"

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
            results[term] = matches
        else:
            # No direct match — lazy load full list for CASE A/B logic
            if all_locs_cache is None:
                all_loc_parts = [f"SELECT DISTINCT {col} AS loc FROM {table}" for col in cols]
                all_locs = _run_query(" UNION ".join(all_loc_parts))
                all_locs_cache = (
                    [loc["loc"] for loc in all_locs if "loc" in loc]
                    if isinstance(all_locs, list) else []
                )

            results[term] = {
                "no_direct_match": True,
                "search_term": term,
                "service_type": svc,
                "available_locations": all_locs_cache,
                "instruction": (
                    "Decide: is this semantic equivalence (CASE A) or a genuine mismatch (CASE B)? "
                    "CASE A → silently pick the correct item from available_locations and call the target tool immediately. "
                    "CASE B → do not retry, return the exact string:'NO_MATCH:{search_term}'."
                )
            }
    return results

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
def suggest_alternatives(origin: str):
    """
    Find available alternative flight destinations from a specific origin city.
    Use this ONLY when the originally requested destination is unreachable (direct or connection flights) or flights are unavailable.
    Input: The origin city (e.g., 'Tel Aviv', 'London').
    """
    query = "SELECT DISTINCT destination FROM flights WHERE LOWER(origin) = ?"
    origin_param = origin.strip().lower()
    
    matches = _run_query(query, (origin_param,))
    
    if not matches or isinstance(matches, str):
        return f"No alternative flight destinations found departing from {origin}."
    
    # Extract just the destination names into a clean list
    destinations = [row["destination"] for row in matches if "destination" in row]
    return f"Available destinations from {origin}: {', '.join(destinations)}"

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

@tool
def find_destinations_by_preference(preference: str):
    """
    Find destinations across all cities based on a season preference.
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

# Trusted source allowlists per category. The LLM chooses the category (it
# understands the user's intent); the tool then biases Tavily toward these hosts
# via include_domains AND hard-enforces them in code, so results only come from
# reputable sources. "events" is open web — there's no single canonical source.
_WEB_SOURCES = {
    "news":     ["bbc.com", "reuters.com", "apnews.com"],
    "exchange": ["xe.com", "x-rates.com"],
    "weather":  ["weather.com", "accuweather.com"],
    "events":   ["reuters.com"],
}

# Human-in-the-loop policy: hosts the agent is pre-authorised to search without
# asking the user. Any host a web search would touch that is NOT in this set
# triggers an interrupt so the user can approve/edit/cancel before we hit the
# network (see the web_gate node in plan_and_execute_agent.py).
#
# Note: "accuweather.com" is intentionally left OUT of the baseline so a normal
# weather search exercises the approval flow — useful for the demo. The user can
# approve it for the session, after which it won't ask again.
KNOWN_HOSTS = {
    "bbc.com", "reuters.com", "apnews.com",
    "xe.com", "x-rates.com",
    "weather.com",
}


def hosts_for_category(category: str) -> list:
    """Hosts that search_web would query for a given category (its allowlist).
    Returns [] for an unknown category."""
    return list(_WEB_SOURCES.get((category or "").strip().lower(), []))


# The valid search_web categories (used by the HITL gate to offer alternatives).
WEB_CATEGORIES = list(_WEB_SOURCES.keys())


def _host_on_allowlist(result: dict, hosts: list) -> bool:
    """True only if a search result's URL is https AND its hostname is on the
    allowlist. Matches the parsed hostname (or a subdomain of it), never a loose
    substring, so spoofs like https://evil.com/bbc.com are rejected."""
    from urllib.parse import urlparse

    url = (result.get("href") or result.get("url") or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    netloc = parsed.netloc.lower().split(":")[0]  # drop any :port
    return any(netloc == h or netloc.endswith("." + h) for h in hosts)

@tool
def search_web(query: str, category: str) -> str:
    """
    Search the live internet for real-world, time-sensitive travel information that is
    NOT stored in our database.

    USE THIS TOOL ONLY FOR: current currency/exchange rates, weather and forecasts,
    local news, public safety advisories, strikes/closures, holidays, festivals, and
    special events at a destination.

    DO NOT use this tool for flights, hotels, activities, car rentals, or
    time differences — those come exclusively from the database tools (the single
    source of truth). Never use this tool to invent prices for bookable items.

    YOU must choose the correct `category` based on what the user is asking about.
    Each category restricts results to its trusted sources:
      - "exchange" → currency / exchange-rate questions (xe.com, x-rates.com)
      - "weather"  → weather & forecasts (weather.com, accuweather.com)
      - "news"     → news, safety advisories, strikes, closures (bbc.com, reuters.com, apnews.com)
      - "events"   → festivals, holidays, concerts, special events (general trusted web)

    Input:
    - query: a focused natural-language query, e.g. "current USD to EUR exchange rate"
             or "weather in Tokyo late May 2026" or "special events in Paris June 2026".
    - category: one of "exchange", "weather", "news", "events".
    Returns: a synthesized direct answer (when available) plus the supporting
             trusted-source snippets, or a message indicating no results were found.
    """
    cleaned = query.strip()
    if not cleaned:
        return "No search query provided."

    cat = category.strip().lower()
    if cat not in _WEB_SOURCES:
        return (
            f"Invalid category '{category}'. "
            f"Choose one of: {list(_WEB_SOURCES.keys())}."
        )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: web search is unavailable — TAVILY_API_KEY is not set. "
            "Add it to your .env file."
        )

    import requests

    hosts = _WEB_SOURCES[cat]
    # Tavily's news topic returns recent, dated articles — ideal for advisories,
    # strikes, and closures. Everything else uses the general topic.
    topic = "news" if cat == "news" else "general"

    payload = {
        "query": cleaned,
        "search_depth": "basic",
        "topic": topic,
        "include_answer": True,   # Tavily synthesizes a direct answer (e.g. the actual rate/temp)
        "max_results": 8,
    }
    if hosts:
        # Bias Tavily toward the trusted hosts; we still hard-enforce below.
        payload["include_domains"] = hosts
    if topic == "news":
        payload["days"] = 14   # only recent news

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError:
        code = resp.status_code
        if code in (401, 403):
            return "Web search failed: Tavily rejected the API key (check TAVILY_API_KEY)."
        if code == 429:
            return "Web search failed: Tavily rate limit / quota exceeded. Try again later."
        return f"Web search failed: Tavily returned HTTP {code}."
    except Exception as e:
        return f"Web search failed: {e}"

    results = data.get("results") or []

    # HARD-enforce the allowlist so no off-list host can leak through. Match on
    # the parsed hostname (not a loose substring) and require https, so a spoofed
    # URL like https://evil.com/bbc.com or any http:// result can't slip past.
    if hosts:
        results = [r for r in results if _host_on_allowlist(r, hosts)]

    answer = (data.get("answer") or "").strip()
    results = results[:5]

    if not answer and not results:
        if hosts:
            return (
                f"No results from trusted {cat} sources ({', '.join(hosts)}) "
                f"for: {cleaned}."
            )
        return f"No web results found for: {cleaned} (category: {cat})."

    lines = [f"Web search results for '{cleaned}' [category: {cat}]:"]
    if answer:
        lines.append(f"ANSWER: {answer}")
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        if len(content) > 400:
            content = content[:400] + "…"
        url = r.get("url") or r.get("href") or ""
        date = (r.get("published_date") or "").strip()
        date_str = f" ({date[:10]})" if date else ""
        lines.append(f"{i}. {title}{date_str}\n   {content}\n   Source: {url}")
    return "\n".join(lines)

@tool
def save_preference(key: str, value: str) -> str:
    """Save a user travel preference for future sessions.
    key: one of preferred_airline, food_preference, travel_style, seat_preference, class_preference
    value: the preference value (e.g. 'El Al', 'kosher', 'luxury')
    """
    return f"saved:{key}={value}"

@tool
def fetch_restaurants(city: str):
    """
    Fetch recommended restaurants in a city.
    Use this tool for restaurant, food, cuisine, romantic dinner, cheap food, local food, or fine dining requests.
    returns a list of restaurants with their cuisine type, price level, rating, and any special features (e.g. vegan options, outdoor seating).
    """

    query = """
    SELECT name, cuisine, price_level, rating, special_features
    FROM restaurants
    WHERE LOWER(city) = ?
    """

    params = [city.strip().lower()]
    matches = _run_query(query, tuple(params))

    if not matches or isinstance(matches, str):
        return f"No restaurants found in {city}."

    return matches

@tool
def fetch_beaches(city: str = None):
    """
    Fetch beach recommendations.
    Use this tool for beach, swimming, sunbathing, nightlife near beach, family beach, or water sports requests.
    If city is not provided, search beaches across all cities.
    returns a list of beaches with their type (sandy, rocky, pebbly), suitability (family-friendly, good for parties, water sports), and any special notes (e.g. lifeguards on duty, nearby amenities).
    """

    query = """
    SELECT city, beach_name, beach_type, suitable_for, notes
    FROM beaches
    WHERE 1=1
    """

    params = []

    if city is not None:
        query += " AND LOWER(city) = ?"
        params.append(city.strip().lower())


    matches = _run_query(query, tuple(params))

    if not matches or isinstance(matches, str):
        return "No beach recommendations found."

    return matches

@tool
def fetch_city_transport_info(city: str):
    """
    Get basic city transportation info.
    Use only for public transport, metro, subway, buses, or getting around a city.
    Do not use for rental cars.
    """

    query = """
    SELECT transport_type, average_ticket_price, car_needed, notes
    FROM public_transport
    WHERE LOWER(city) = ?
    """

    city_param = city.strip().lower()

    matches = _run_query(query, (city_param,))

    if not matches or isinstance(matches, str):
        return f"No public transport information found for {city}."

    return matches

@tool
def ask_user(question: str, options: list[str] = None) -> str:
    """
    Ask the human user a clarifying question and WAIT for their answer before
    continuing. Use this when the request is genuinely ambiguous or missing a
    preference you cannot reasonably infer (e.g. budget not stated, "a warm
    place" without a vibe, unclear dates) — NOT for things you can decide
    yourself or already know from memory.

    This pauses the agent (a human-in-the-loop checkpoint) and surfaces the
    question in the terminal. The user can pick one of your suggested options or
    type a free-text answer.

    Input:
    - question: a single, specific question, e.g.
        "What's your approximate budget for this trip?"
    - options: OPTIONAL list of 2-5 short suggested answers the user can pick by
        number, e.g. ["Beaches & relaxation", "City & culture", "Adventure"].
        Omit it for purely open questions. The user may always answer freely.

    Returns: the user's answer as a string. Treat it as authoritative and
    incorporate it into the rest of the plan.
    """
    # Imported lazily so tools.py stays importable without langgraph at hand.
    from langgraph.types import interrupt

    q = (question or "").strip()
    if not q:
        return "No question was provided to ask the user."

    opts = [str(o).strip() for o in (options or []) if str(o).strip()]

    # PAUSE the graph. The interactive terminal UI (in plan_and_execute_agent.py)
    # renders this payload, collects the answer, and resumes the graph with it.
    answer = interrupt({
        "type": "user_question",
        "question": q,
        "options": opts,
    })

    answer = (str(answer) if answer is not None else "").strip()
    if not answer:
        return f"(The user was asked: '{q}' but did not provide an answer.)"
    return f"The user answered: {answer}"