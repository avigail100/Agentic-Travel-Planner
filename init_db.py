import sqlite3

def create_travel_db():
    conn = sqlite3.connect("travel_agency.db")
    cursor = conn.cursor()

    # 1. Drop existing tables if they exist
    tables = [
        "hotels", "flights", "activities", "visa_requirements", 
        "exchange_rates", "car_rentals", "best_seasons", "time_differences"
    ]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # 2. Create Tables
    cursor.execute("""
    CREATE TABLE hotels (
        id INTEGER PRIMARY KEY,
        city TEXT,
        name TEXT,
        price_per_night INTEGER,
        stars INTEGER,
        amenities TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE flights (
        id INTEGER PRIMARY KEY,
        origin TEXT,
        destination TEXT,
        airline TEXT,
        price INTEGER,
        flight_number TEXT,
        availability TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE activities (
        id INTEGER PRIMARY KEY,
        city TEXT,
        name TEXT,
        category TEXT,
        price INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE visa_requirements (
        id INTEGER PRIMARY KEY,
        origin TEXT,
        destination TEXT,
        policy TEXT,
        days_allowed_without_visa INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE exchange_rates (
        id INTEGER PRIMARY KEY,
        origin_currency TEXT,
        destination_currency TEXT,
        exchange_rate REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE car_rentals (
        id INTEGER PRIMARY KEY,
        airport TEXT,
        city TEXT,
        company TEXT,
        price_per_day INTEGER,
        car_type TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE best_seasons (
        id INTEGER PRIMARY KEY,
        city TEXT,
        season TEXT,
        months TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE time_differences (
        id INTEGER PRIMARY KEY,
        origin TEXT,
        destination TEXT,
        hours_difference INTEGER
    )
    """)

    # 3. Insert Data

    # --- Hotels ---
    hotels = [
        ('Paris', 'Hotel Ritz', 450, 5, 'Spa, Fine Dining'),
        ('Paris', 'EcoStay Paris', 85, 2, 'Free WiFi'),
        ('London', 'The Savoy', 600, 5, 'Butler Service, River View'),
        ('Paris', 'Hotel de Ville', 150, 3, 'None'),
        ('Tokyo', 'Shibuya Capsule', 50, 2, 'None'),
        ('New York', 'The Plaza', 850, 5, 'Luxury')
    ]
    cursor.executemany("INSERT INTO hotels (city, name, price_per_night, stars, amenities) VALUES (?, ?, ?, ?, ?)", hotels)

    # --- Flights ---
    flights = [
        ('TLV', 'Paris', 'AirFrance', 120, 'AF123', 'Available'),
        ('New York', 'London', 'Virgin Atlantic', 550, 'VS001', 'Limited'),
        ('London', 'Tokyo', 'JAL', 950, 'JL402', 'Available'),
        ('TLV', 'Paris', 'El Al', 350, 'LY321', 'Available'),
        ('TLV', 'Tokyo', 'EL AL', 950, 'LY091', 'Available')
    ]
    cursor.executemany("INSERT INTO flights (origin, destination, airline, price, flight_number, availability) VALUES (?, ?, ?, ?, ?, ?)", flights)

    # --- Activities ---
    activities = [
        ('Paris', 'Louvre Museum', 'Culture', 20),
        ('Paris', 'Eiffel Tower Tour', 'Sightseeing', 35),
        ('London', 'London Eye', 'Sightseeing', 30),
        ('Tokyo', 'Akihabara Gaming Tour', 'Entertainment', 50),
        ('Paris', 'Disneyland', 'Family', 95)
    ]
    cursor.executemany("INSERT INTO activities (city, name, category, price) VALUES (?, ?, ?, ?)", activities)

    # --- Visa Requirements ---
    visa = [
        ('Israel', 'France', 'No visa required for tourism up to 90 days', 90),
        ('Israel', 'Japan', 'No visa required for tourism up to 90 days', 90),
        ('India', 'France', 'Schengen Visa required', 0)
    ]
    cursor.executemany("INSERT INTO visa_requirements (origin, destination, policy, days_allowed_without_visa) VALUES (?, ?, ?, ?)", visa)

    # --- Exchange Rates ---
    rates = [
        ('USD', 'ILS', 3.65),
        ('EUR', 'USD', 1.08),
        ('GBP', 'USD', 1.27)
    ]
    cursor.executemany("INSERT INTO exchange_rates (origin_currency, destination_currency, exchange_rate) VALUES (?, ?, ?)", rates)

    # --- Car Rentals ---
    cars = [
        ('CDG', 'Paris', 'Hertz', 45, 'Economy'),
        ('CDG', 'Paris', 'Avis', 60, 'SUV'),
        ('LHR', 'London', 'Europcar', 55, 'Compact')
    ]
    cursor.executemany("INSERT INTO car_rentals (airport, city, company, price_per_day, car_type) VALUES (?, ?, ?, ?, ?)", cars)

    # --- Best Seasons ---
    seasons = [
        ('Paris', 'Spring', 'March, April, May'),
        ('London', 'Summer', 'June, July, August'),
        ('Tokyo', 'Autumn', 'September, October, November')
    ]
    cursor.executemany("INSERT INTO best_seasons (city, season, months) VALUES (?, ?, ?)", seasons)

    # --- Time Differences ---
    times = [
        ('Tel Aviv', 'Paris', -1),
        ('Tel Aviv', 'London', -2),
        ('Tel Aviv', 'Tokyo', 7),
        ('Paris', 'Tokyo', 8)
    ]
    cursor.executemany("INSERT INTO time_differences (origin, destination, hours_difference) VALUES (?, ?, ?)", times)

    conn.commit()
    conn.close()
    print("Database 'travel_agency.db' created with all expanded fields and new tables!")

if __name__ == "__main__":
    create_travel_db()