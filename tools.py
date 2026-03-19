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