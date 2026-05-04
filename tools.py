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
def fetch_flights(origin: str, destination: str):
    """
    Search for available flights between two locations in the database.
    Input:
    - origin: 3-letter airport code (e.g., 'TLV', 'JFK', 'LHR').
    - destination: Full city name (e.g., 'Paris', 'London', 'Tokyo').
    Returns: List of matching flights with airline, price, and flight number.
    """
    # Using LOWER() to ensure case-insensitive matching in the database
    query = "SELECT airline, price, flight_number FROM flights WHERE LOWER(origin) = ? AND LOWER(destination) = ?"
    
    # Pre-processing inputs to match the database format
    origin_param = origin.strip().lower()
    dest_param = destination.strip().lower()
    
    matches = _run_query(query, (origin_param, dest_param))
    
    if not matches or isinstance(matches, str):
        return f"No flights found from {origin} to {destination}. Make sure to use airport codes for origin."
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
def fetch_curency_exchange_rate(origin_currency: str, destination_currency: str):
    """
    Fetch the current exchange rate between the origin currency and the destination currency.
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
    # exchange_rate = fetch_curency_exchange_rate.invoke({'origin_currency': origin_currency, 'destination_currency': destination_currency})
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