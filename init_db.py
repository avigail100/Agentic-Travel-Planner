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
        amenities TEXT,
        rating REAL,
        room_type TEXT,
        breakfast_included TEXT
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
        availability TEXT,
        duration_hours TEXT,
        departure_time TEXT,
        arrival_time TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE activities (
        id INTEGER PRIMARY KEY,
        city TEXT,
        name TEXT,
        category TEXT,
        price INTEGER,
        duration TEXT,
        suitable_for TEXT
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
    # Paris
    ('Paris', 'Hotel Ritz', 450, 5, 'Spa, Fine Dining', 9.4, 'Deluxe', 'Yes'),
    ('Paris', 'EcoStay Paris', 85, 2, 'WiFi', 7.8, 'Standard', 'No'),
    ('Paris', 'Hotel de Ville', 150, 3, 'Breakfast, WiFi', 8.3, 'Standard', 'Yes'),
    # London
    ('London', 'The Savoy', 600, 5, 'Butler Service, River View', 9.6, 'Suite', 'Yes'),
    ('London', 'Central Inn', 140, 3, 'WiFi', 8.0, 'Standard', 'No'),
    # Tokyo
    ('Tokyo', 'Shibuya Capsule', 50, 2, 'Shared Lounge', 7.5, 'Capsule', 'No'),
    ('Tokyo', 'Tokyo Grand Hotel', 320, 4, 'Gym, Spa, Breakfast', 8.9, 'Deluxe', 'Yes'),
    # New York
    ('New York', 'The Plaza', 850, 5, 'Luxury Spa, City View', 9.7, 'Suite', 'Yes'),
    ('New York', 'Brooklyn Budget Stay', 110, 2, 'WiFi', 7.9, 'Standard', 'No'),
    # Rome
    ('Rome', 'Colosseum Hotel', 210, 4, 'Breakfast, Rooftop', 8.7, 'Deluxe', 'Yes'),
    # Barcelona
    ('Barcelona', 'Beachside Resort', 260, 4, 'Pool, Sea View', 8.8, 'Suite', 'Yes'),
    # Bangkok
    ('Bangkok', 'Bangkok Palace', 180, 4, 'Pool, Gym', 8.4, 'Deluxe', 'Yes')
    ]
    cursor.executemany("""INSERT INTO hotels 
                       (city, name, price_per_night, stars, amenities, rating, room_type, breakfast_included) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       """, hotels)

    # --- Flights ---
    flights = [
    # Israel -> Europe
    ('TLV', 'Paris', 'Air France', 420, 'AF123', 'Available', '4h 30m', '08:30', '12:00'),
    ('TLV', 'Rome', 'ITA Airways', 280, 'AZ809', 'Available', '3h 45m', '10:10', '12:55'),
    ('TLV', 'London', 'British Airways', 510, 'BA405', 'Limited', '5h 20m', '07:15', '11:35'),
    ('TLV', 'Barcelona', 'Vueling', 320, 'VY7845', 'Available', '4h 40m', '06:45', '10:25'),
    ('TLV', 'Amsterdam', 'KLM', 390, 'Available', 'KL462', '5h 05m', '05:20', '09:25'),
    # Israel -> USA
    ('TLV', 'New York', 'El Al', 1150, 'LY007', 'Available', '11h 50m', '01:00', '07:50'),
    ('TLV', 'New York', 'United Airlines', 980, 'UA091', 'Limited', '12h 05m', '23:45', '06:50'),
    ('TLV', 'Los Angeles', 'El Al', 1450, 'LY005', 'Available', '15h 20m', '00:30', '06:50'),
    ('TLV', 'Miami', 'American Airlines', 1050, 'AA053', 'Available', '13h 10m', '22:10', '05:20'),
    # Europe internal
    ('Paris', 'London', 'easyJet', 120, 'U22438', 'Available', '1h 15m', '09:00', '09:15'),
    ('Rome', 'Barcelona', 'Ryanair', 95, 'FR6972', 'Available', '1h 50m', '14:20', '16:10'),
    ('Amsterdam', 'Berlin', 'KLM', 140, 'KL1771', 'Available', '1h 20m', '11:00', '12:20'),
    # Europe -> Asia
    ('London', 'Tokyo', 'Japan Airlines', 940, 'JL402', 'Available', '13h 40m', '09:30', '07:10'),
    ('Paris', 'Tokyo', 'Air France', 890, 'AF276', 'Available', '13h 30m', '10:00', '06:30'),
    # USA routes
    ('New York', 'London', 'Virgin Atlantic', 650, 'VS001', 'Available', '7h 05m', '19:10', '07:15'),
    ('New York', 'Paris', 'Delta', 720, 'DL220', 'Limited', '7h 20m', '18:40', '08:00'),
    # Asia
    ('Tokyo', 'Bangkok', 'Thai Airways', 310, 'TG677', 'Available', '6h 10m', '13:20', '18:30'),
    ('Dubai', 'Tokyo', 'Emirates', 780, 'EK312', 'Available', '9h 25m', '02:40', '17:05'),
    # Cheap options
    ('TLV', 'Larnaca', 'Arkia', 95, 'IZ901', 'Available', '1h 05m', '08:00', '09:05'),
    ('Rome', 'Athens', 'Aegean', 110, 'A3651', 'Available', '2h 00m', '16:10', '18:10'),
    # Edge cases
    ('TLV', 'Paris', 'Transavia', 180, 'TO3451', 'Unavailable', '4h 40m', '05:50', '09:30'),
    ('Bangkok', 'London', 'British Airways', 1100, 'BA010', 'Limited', '12h 15m', '23:45', '05:00')
    ]
    cursor.executemany("""
        INSERT INTO flights 
        (origin, destination, airline, price, flight_number, availability, duration_hours, departure_time, arrival_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, flights)

    # --- Activities ---
    activities = [
    # Paris
    ('Paris', 'Louvre Museum', 'Culture', 20, '3h', 'Couples'),
    ('Paris', 'Eiffel Tower Tour', 'Sightseeing', 35, '2h', 'Everyone'),
    ('Paris', 'Disneyland Paris', 'Family', 95, 'Full Day', 'Families'),
    # London
    ('London', 'London Eye', 'Sightseeing', 30, '1h', 'Everyone'),
    ('London', 'British Museum', 'Culture', 0, '2h', 'Everyone'),
    ('London', 'Thames River Cruise', 'Relaxation', 45, '2h', 'Couples'),
    # Rome
    ('Rome', 'Colosseum Tour', 'History', 40, '2h', 'Everyone'),
    ('Rome', 'Vatican Museum', 'Culture', 35, '3h', 'Adults'),
    # Barcelona
    ('Barcelona', 'Sagrada Familia Tour', 'Culture', 28, '2h', 'Everyone'),
    ('Barcelona', 'Beach Day', 'Relaxation', 0, 'Full Day', 'Families'),
    # Amsterdam
    ('Amsterdam', 'Canal Cruise', 'Relaxation', 25, '1h', 'Couples'),
    ('Amsterdam', 'Van Gogh Museum', 'Culture', 22, '2h', 'Adults'),
    # Berlin
    ('Berlin', 'Berlin Wall Memorial', 'History', 0, '2h', 'Everyone'),
    # Tokyo
    ('Tokyo', 'Akihabara Gaming Tour', 'Entertainment', 50, '4h', 'Gamers'),
    ('Tokyo', 'Shibuya Food Tour', 'Food', 70, '3h', 'Adults'),
    ('Tokyo', 'Tokyo Disneyland', 'Family', 90, 'Full Day', 'Families'),
    # Bangkok
    ('Bangkok', 'Floating Market Tour', 'Culture', 45, '5h', 'Everyone'),
    ('Bangkok', 'Thai Cooking Workshop', 'Food', 55, '3h', 'Adults'),
    # New York
    ('New York', 'Statue of Liberty Cruise', 'Sightseeing', 30, '3h', 'Everyone'),
    ('New York', 'Broadway Show', 'Entertainment', 120, '3h', 'Couples'),
    # Los Angeles
    ('Los Angeles', 'Hollywood Tour', 'Sightseeing', 60, '4h', 'Everyone'),
    # Miami
    ('Miami', 'South Beach Day', 'Relaxation', 0, 'Full Day', 'Everyone'),
    # Athens
    ('Athens', 'Acropolis Tour', 'History', 25, '2h', 'Everyone'),
    # Larnaca
    ('Larnaca', 'Finikoudes Beach', 'Relaxation', 0, 'Full Day', 'Families'),
    # Dubai
    ('Dubai', 'Burj Khalifa Visit', 'Sightseeing', 55, '2h', 'Everyone'),
    ]
    cursor.executemany("INSERT INTO activities (city, name, category, price, duration, suitable_for) VALUES (?, ?, ?, ?, ?, ?)", activities)

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