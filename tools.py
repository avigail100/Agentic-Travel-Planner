from langchain_core.tools import tool
import json

@tool
def fetch_flights(origin: str, destination: str):
    """
    Fetch available flights between two cities from the local database.
    Returns a list of flights with prices, availability status and airline.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        flights = travel_db.get("flights", [])
        matching_flights = [
            flight for flight in flights
            if flight["origin"].lower() == origin.lower() and flight["destination"].lower() == destination.lower() 
            and flight["availability"].lower() != "unavailable"
        ]

        if not matching_flights:
            return f"No available flights found from {origin} to {destination}."

        results = []

        for flight in matching_flights:
            filtered_flight = {
                "airline": flight["airline"],
                "flight_number": flight["flight_number"],
                "price": flight["price"],
                "availability": flight["availability"]
            }
            results.append(filtered_flight)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:  
        return f"error fetching flights: {e}"

@tool
def fetch_hotels(city: str, budget: float = None):
    """
    Fetch available hotels in a specific city from the local database.
    Optional budget parameter to filter hotels by price.
    Returns a list of hotels with prices, star ratings, and amenities.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        hotels = travel_db.get("hotels", [])
        matching_hotels = [
            hotel for hotel in hotels
            if hotel["city"].lower() == city.lower() and (budget is None or hotel["price_per_night"] <= budget)
        ]

        if not matching_hotels:
            return f"No available hotels found in {city}."

        results = []

        for hotel in matching_hotels:
            filtered_hotel = {
                "name": hotel["name"],
                "price_per_night": hotel["price_per_night"],
                "stars": hotel["stars"],
                "amenities": hotel["amenities"]
            }
            results.append(filtered_hotel)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching hotels: {e}"

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, nights: int):
    """
    Calculate the total cost of a trip based on flight price, hotel price per night, and number of nights.
    Returns the total estimated cost of the trip with service charges.
    """
    try:
        total_cost = flight_price + (hotel_price_per_night * nights)
        return total_cost*1.1
    except Exception as e:
        return f"Error calculating trip cost: {e}"

@tool
def fetch_activities(city: str, budget: float = None):
    """
    Fetch available activities in a specific city from the local database.
    Optional budget parameter to filter activities by price.
    Returns a list of activities with prices, ratings, and descriptions.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        activities = travel_db.get("activities", [])
        matching_activities = [
            activity for activity in activities
            if activity["city"].lower() == city.lower() and (budget is None or activity["price"] <= budget)
            
        ]

        if not matching_activities:
            return f"No available activities found in {city}."

        results = []

        for activity in matching_activities:
            filtered_activity = {
                "name": activity["name"],
                "price": activity["price"],
                "category": activity["category"]
            }
            results.append(filtered_activity)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching activities: {e}"

@tool
def fetch_visa_requirements(origin: str, destination: str):
    """
    Fetch visa requirements for travelers from the origin country to the destination country.
    Returns visa requirement policy and the amount of days you can stay without a visa.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        visas = travel_db.get("visa_requirements", [])
        matching_visa = [
            visa for visa in visas
             if visa["origin"].lower() == origin.lower() and visa["destination"].lower() == destination.lower()
        ]

        if not matching_visa:
            return f"No visa requirement information found for travelers from {origin} to {destination}."

        results = []

        for visa in matching_visa:
            filtered_visa = {
                "days_allowed_without_visa": visa["days_allowed_without_visa"],
                "policy": visa["policy"]
            }
            results.append(filtered_visa)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching visa requirements: {e}"

@tool
def fetch_curency_exchange_rate(origin_currency: str, destination_currency: str):
    """
    Fetch the current exchange rate between the origin currency and the destination currency.
    Returns the exchange rate as of today.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        exchange_rates = travel_db.get("exchange_rates", [])
        matching_rate = [
            rate for rate in exchange_rates
            if rate["origin_currency"].lower() == origin_currency.lower() and rate["destination_currency"].lower() == destination_currency.lower()
        ]

        if not matching_rate:
            return f"No exchange rate information found for {origin_currency} to {destination_currency}."

        results = []

        for rate in matching_rate:
            filtered_rate = {
                "exchange_rate": rate["exchange_rate"]
            }
            results.append(filtered_rate)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching currency exchange rates: {e}"

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
    Fetch available car rental agencies in a specific city from the local database.
    Returns a list of car rental agencies with price per day and car types.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        car_rentals = travel_db.get("car_rentals", [])
        matching_agencies = [
            agency for agency in car_rentals
            if agency["city"].lower() == city.lower()
        ]

        if not matching_agencies:
            return f"No available car rental agencies found in {city}."

        results = []

        for agency in matching_agencies:
            filtered_agency = {
                "company": agency["company"],
                "airport": agency["airport"],
                "price_per_day": agency["price_per_day"],
                "car_type": agency["car_type"]
            }
            results.append(filtered_agency)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching car rental agencies: {e}"

@tool
def fetch_seasonal_recommendations(city: str):
    """
    Fetch seasonal travel recommendations for a specific city from the local database.
    Returns the best season to visit and the months when it's ideal.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        seasonal_recommendations = travel_db.get("best_seasons", [])
        matching_recommendation = [
            recommendation for recommendation in seasonal_recommendations
            if recommendation["city"].lower() == city.lower()
        ]

        if not matching_recommendation:
            return f"No seasonal recommendations found for {city}."

        results = []

        for recommendation in matching_recommendation:
            filtered_recommendation = {
                "best_season": recommendation["season"],
                "ideal_months": recommendation["months"]
            }
            results.append(filtered_recommendation)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching seasonal recommendations: {e}"

@tool
def fetch_time_difference(origin: str, destination: str):
    """
    Fetch the time difference in hours between the origin city and the destination city.
    Returns the time difference in hours.
    """
    try:
        travel_db = load_json_db("travel_db.json")
        time_differences = travel_db.get("time_differences", [])
        matching_time_difference = [
            time_diff for time_diff in time_differences
            if time_diff["origin"].lower() == origin.lower() and time_diff["destination"].lower() == destination.lower()
        ]

        if not matching_time_difference:
            return f"No time difference information found for {origin} to {destination}."

        results = []

        for time_diff in matching_time_difference:
            filtered_time_diff = {
                "hours_difference": time_diff["hours_difference"]
            }
            results.append(filtered_time_diff)
        return results
    except FileNotFoundError as e:
        return f"Error finding file: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error fetching time difference: {e}"

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

def load_json_db(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
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