import sqlite3

def create_travel_db():
    conn = sqlite3.connect("travel_agency.db")
    cursor = conn.cursor()

    # 1. Drop existing tables if they exist
    tables = [
        "hotels", "flights", "activities", "visa_requirements", 
        "exchange_rates", "car_rentals", "best_seasons", "time_differences",
        "restaurants", "beaches", "public_transport"
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
        days_allowed_without_visa INTEGER,
        visa_type TEXT
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
        car_type TEXT,
        transmission TEXT,
        seats INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE best_seasons (
        id INTEGER PRIMARY KEY,
        city TEXT,
        season TEXT,
        months TEXT,
        reason TEXT,
        start_month INTEGER,
        end_month INTEGER
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

    cursor.execute("""
    CREATE TABLE restaurants (
        id INTEGER PRIMARY KEY,
        city TEXT,
        name TEXT,
        cuisine TEXT,
        price_level TEXT,
        rating REAL,
        special_features TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE beaches (
        id INTEGER PRIMARY KEY,
        city TEXT,
        beach_name TEXT,
        beach_type TEXT,
        suitable_for TEXT,
        notes TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE public_transport (
        id INTEGER PRIMARY KEY,
        city TEXT,
        transport_type TEXT,
        average_ticket_price REAL,
        car_needed TEXT,
        notes TEXT
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
    ('Rome', 'Budget Rome Inn', 90, 2, 'WiFi', 7.6, 'Standard', 'No'),
    # Barcelona
    ('Barcelona', 'Beachside Resort', 260, 4, 'Pool, Sea View', 8.8, 'Suite', 'Yes'),
    # Bangkok
    ('Bangkok', 'Bangkok Palace', 180, 4, 'Pool, Gym', 8.4, 'Deluxe', 'Yes'),
    # Dubai
    ('Dubai', 'Palm Luxury Resort', 520, 5, 'Private Beach, Spa, Pool', 9.5, 'Suite', 'Yes'),
    # Athens
    ('Athens', 'Acropolis View Hotel', 170, 3, 'Breakfast, Rooftop', 8.4, 'Standard', 'Yes'),
    # Larnaca
    ('Larnaca', 'Sunny Coast Hotel', 140, 3, 'Beach Access, Pool', 8.1, 'Standard', 'Yes'),
    # Amsterdam
    ('Amsterdam', 'Canal Boutique Hotel', 240, 4, 'Canal View, Breakfast', 8.8, 'Deluxe', 'Yes'),
    # Berlin
    ('Berlin', 'Berlin Central Stay', 130, 3, 'WiFi, Breakfast', 8.0, 'Standard', 'Yes'),
    # Los Angeles
    ('Los Angeles', 'Sunset Boulevard Hotel', 390, 4, 'Pool, Gym', 8.7, 'Suite', 'Yes'),
    # Miami
    ('Miami', 'Ocean Drive Resort', 410, 5, 'Beachfront, Pool, Spa', 9.2, 'Suite', 'Yes'),
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
    ('Israel', 'France', 'No visa required for tourism up to 90 days', 90, 'Tourist'),
    ('Israel', 'Italy', 'No visa required for tourism up to 90 days', 90, 'Tourist'),
    ('Israel', 'United Kingdom', 'No visa required for tourism up to 180 days', 180, 'Tourist'),
    ('Israel', 'United States', 'ESTA approval required before travel', 90, 'ESTA'),
    ('Israel', 'Japan', 'No visa required for tourism up to 90 days', 90, 'Tourist'),
    ('Israel', 'Thailand', 'No visa required for tourism up to 30 days', 30, 'Tourist'),

    ('United States', 'France', 'No visa required for tourism up to 90 days', 90, 'Tourist'),
    ('United States', 'Japan', 'No visa required for tourism up to 90 days', 90, 'Tourist'),

    ('United Kingdom', 'United States', 'ESTA approval required before travel', 90, 'ESTA'),

    ('France', 'Japan', 'No visa required for tourism up to 90 days', 90, 'Tourist'),

    ('Japan', 'Thailand', 'No visa required for tourism up to 30 days', 30, 'Tourist'),
    ]
    cursor.executemany("INSERT INTO visa_requirements (origin, destination, policy, days_allowed_without_visa, visa_type) VALUES (?, ?, ?, ?, ?)", visa)

    # --- Exchange Rates ---
    rates = [
        ('USD', 'EUR', 0.92),
    ('EUR', 'USD', 1.09),

    ('USD', 'ILS', 3.65),
    ('ILS', 'USD', 0.27),

    ('EUR', 'ILS', 3.98),
    ('ILS', 'EUR', 0.25),

    ('USD', 'JPY', 155.2),
    ('JPY', 'USD', 0.0064),

    ('USD', 'GBP', 0.79),
    ('GBP', 'USD', 1.27),

    ('USD', 'THB', 36.1),
    ('THB', 'USD', 0.028),

    ('USD', 'AED', 3.67),
    ('AED', 'USD', 0.27),
    ]
    cursor.executemany("INSERT INTO exchange_rates (origin_currency, destination_currency, exchange_rate) VALUES (?, ?, ?)", rates)

    # --- Car Rentals ---
    cars = [
    # Paris
    ('CDG', 'Paris', 'Hertz', 70, 'Economy', 'Automatic', 5),
    ('CDG', 'Paris', 'Avis', 120, 'SUV', 'Automatic', 7),
    ('ORY', 'Paris', 'Budget', 55, 'Compact', 'Manual', 4),
    # Rome
    ('FCO', 'Rome', 'Europcar', 65, 'Compact', 'Manual', 5),
    # London
    ('LHR', 'London', 'Sixt', 110, 'Luxury', 'Automatic', 5),
    ('LGW', 'London', 'Europcar', 75, 'Compact', 'Manual', 5),
    # Tokyo
    ('HND', 'Tokyo', 'Toyota Rent', 80, 'Compact', 'Automatic', 5),
    ('NRT', 'Tokyo', 'Nippon Rent', 140, 'Luxury', 'Automatic', 5),
    # Bangkok
    ('BKK', 'Bangkok', 'Budget', 45, 'Economy', 'Automatic', 5),
    # New York
    ('JFK', 'New York', 'Enterprise', 95, 'SUV', 'Automatic', 7),
    ('LGA', 'New York', 'Budget', 60, 'Economy', 'Automatic', 5),
    # Los Angeles
    ('LAX', 'Los Angeles', 'Alamo', 130, 'Convertible', 'Automatic', 4),
    # Dubai
    ('DXB', 'Dubai', 'Luxury Cars Dubai', 220, 'Luxury SUV', 'Automatic', 7),
    ('DXB', 'Dubai', 'Budget', 65, 'Economy', 'Automatic', 5),
    # Athens
    ('ATH', 'Athens', 'Avis', 55, 'Economy', 'Manual', 5),
    # Larnaca
    ('LCA', 'Larnaca', 'Hertz', 50, 'Compact', 'Automatic', 5),
    ]
    cursor.executemany("INSERT INTO car_rentals (airport, city, company, price_per_day, car_type, transmission, seats) VALUES (?, ?, ?, ?, ?, ?, ?)", cars)

    # --- Best Seasons ---
    seasons = [
    ('Paris', 'Spring', 'April-June', 'Mild weather, flowers, and comfortable walking conditions', 4, 6),
    ('London', 'Summer', 'June-August', 'Warmest season with outdoor events and long daylight', 6, 8),
    ('Rome', 'Spring', 'April-June', 'Warm but not too hot, ideal for sightseeing', 4, 6),
    ('Barcelona', 'Summer', 'June-September', 'Warm beach weather and sunny days', 6, 9),
    ('Amsterdam', 'Spring', 'March-May', 'Mild weather and tulip season', 3, 5),
    ('Berlin', 'Summer', 'May-August', 'Warm weather, festivals, and outdoor nightlife', 5, 8),
    ('Tokyo', 'Spring', 'March-April', 'Cherry blossom season and pleasant weather', 3, 4),
    ('Bangkok', 'Winter', 'November-February', 'Warm weather with less humidity than summer', 11, 2),
    ('Dubai', 'Winter', 'November-March', 'Warm sunny weather without extreme summer heat', 11, 3),
    ('New York', 'Fall', 'September-November', 'Comfortable weather and beautiful autumn scenery', 9, 11),
    ('Los Angeles', 'Spring', 'March-May', 'Sunny and warm weather with fewer crowds', 3, 5),
    ('Athens', 'Spring', 'April-June', 'Warm weather, good for historical tours', 4, 6),
    ('Larnaca', 'Summer', 'May-September', 'Warm sunny beach season', 5, 9)
    ]
    cursor.executemany("INSERT INTO best_seasons (city, season, months, reason, start_month, end_month) VALUES (?, ?, ?, ?, ?, ?)", seasons)

    # --- Time Differences ---
    times = [
        # Israel
    ('Tel Aviv', 'Paris', -1),
    ('Tel Aviv', 'Rome', -1),
    ('Tel Aviv', 'London', -2),
    ('Tel Aviv', 'New York', -7),
    ('Tel Aviv', 'Tokyo', 6),
    ('Tel Aviv', 'Bangkok', 4),
    ('Tel Aviv', 'Dubai', 1),
    ('Tel Aviv', 'Athens', 0),
    # USA
    ('New York', 'Tokyo', 13),
    ('New York', 'Paris', 6),
    ('New York', 'Bangkok', 12),
    ('New York', 'London', 5),
    # UK
    ('London', 'Tokyo', 9),
    ('London', 'Bangkok', 7),
    ('London', 'Paris', 1),
    # France
    ('Paris', 'Tokyo', 8),
    ('Paris', 'Bangkok', 6),
    ('Paris', 'Dubai', 3),
    # Japan
    ('Tokyo', 'Bangkok', -2),
    ('Tokyo', 'Dubai', -5),
    # UAE
    ('Dubai', 'Paris', -3),
    ('Dubai', 'London', -4),
    # Greece
    ('Athens', 'Tokyo', 7),
    ]
    cursor.executemany("INSERT INTO time_differences (origin, destination, hours_difference) VALUES (?, ?, ?)", times)

    # --- Restaurants ---
    restaurants = [
        ('Paris', 'Le Gourmet', 'French', 'Expensive', 9.3, 'Fine Dining, Romantic'),
        ('Paris', 'Cafe Montmartre', 'French', 'Moderate', 8.6, 'Local Atmosphere, Breakfast'),
        ('London', 'The Riverside Grill', 'British', 'Expensive', 9.0, 'River View, Romantic'),
        ('London', 'Soho Bites', 'International', 'Moderate', 8.4, 'Trendy, Good Nightlife Area'),
        ('Tokyo', 'Sakura Sushi', 'Japanese', 'Moderate', 9.1, 'Fresh Sushi, Local Experience'),
        ('Tokyo', 'Shibuya Ramen House', 'Japanese', 'Cheap', 8.7, 'Ramen, Casual'),
        ('Rome', 'Mama Roma', 'Italian', 'Moderate', 8.8, 'Authentic Pizza, Pasta'),
        ('Barcelona', 'Tapas Corner', 'Spanish', 'Moderate', 8.6, 'Tapas, Local Food'),
        ('Bangkok', 'Street Thai', 'Thai', 'Cheap', 8.5, 'Street Food, Local Experience'),
        ('Dubai', 'Sky Lounge', 'International', 'Luxury', 9.4, 'Rooftop View, Romantic'),
        ('New York', 'Manhattan Steakhouse', 'American', 'Expensive', 9.0, 'Steakhouse, City View'),
        ('Miami', 'Ocean Grill', 'Seafood', 'Moderate', 8.7, 'Sea View, Beach Area')
    ]
    cursor.executemany("INSERT INTO restaurants (city, name, cuisine, price_level, rating, special_features)VALUES (?, ?, ?, ?, ?, ?)", restaurants)

    # --- Beaches ---
    beaches = [
        ('Barcelona', 'Barceloneta Beach', 'Urban Beach', 'Swimming, Nightlife', 'Popular beach close to the city center'),
        ('Barcelona', 'Bogatell Beach', 'Relaxed Urban Beach', 'Couples, Swimming', 'Quieter than Barceloneta and good for a relaxed beach day'),
        ('Miami', 'South Beach', 'Party Beach', 'Nightlife, Sunbathing', 'Famous beach with restaurants and nightlife'),
        ('Miami', 'Crandon Park Beach', 'Family Beach', 'Families, Relaxation', 'Calmer beach option suitable for families'),
        ('Larnaca', 'Finikoudes Beach', 'Family Beach', 'Families, Swimming', 'Easy beach with promenade and calm atmosphere'),
        ('Larnaca', 'Mackenzie Beach', 'Lively Beach', 'Swimming, Restaurants', 'Popular beach with cafes and restaurants nearby'),
        ('Dubai', 'JBR Beach', 'Luxury Beach', 'Families, Water Sports', 'Modern beach area with restaurants nearby'),
        ('Dubai', 'Kite Beach', 'Sport Beach', 'Water Sports, Families', 'Good for active travelers and beach activities'),
        ('Athens', 'Vouliagmeni Beach', 'Relaxed Beach', 'Couples, Swimming', 'Good option for a beach escape near Athens'),
        ('Bangkok', 'Pattaya Beach Day Trip', 'Day Trip Beach', 'Families, Relaxation', 'Beach option outside Bangkok for a day trip'),
        ('Los Angeles', 'Santa Monica Beach', 'Urban Beach', 'Families, Walking, Sunsets', 'Iconic beach with pier and restaurants'),
        ('Los Angeles', 'Venice Beach', 'Lively Beach', 'Walking, Street Culture', 'Good for people-watching and a lively atmosphere')
    ]
    cursor.executemany("INSERT INTO beaches (city, beach_name, beach_type, suitable_for, notes) VALUES (?, ?, ?, ?, ?)", beaches)

    public_transport = [
        ('Paris', 'Metro', 2.5, 'No', 'Fast and extensive network'),
        ('Paris', 'Bus and Metro', 2.2, 'No','Easy to explore tourist areas without a car'),
        ('London', 'Underground', 3.0, 'No','Public transport is usually easier than driving'),
        ('London', 'Bus', 1.8, 'No','Good for sightseeing around the city'),
        ('Tokyo', 'Train', 1.8, 'No','Very punctual and efficient'),
        ('Tokyo', 'Metro', 1.7, 'No','Best option for most tourist attractions'),
        ('Bangkok', 'Skytrain', 1.5, 'No','Avoids heavy traffic'),
        ('Bangkok', 'Taxi', 4.0, 'Sometimes','Cheap but traffic can be very heavy'),
        ('Dubai', 'Metro', 2.0, 'Sometimes','Metro is modern but taxis are common'),
        ('Dubai', 'Taxi', 6.0, 'Sometimes','Very common and convenient for tourists'),
        ('New York', 'Subway', 2.9, 'No','Runs 24/7'),
        ('New York', 'Taxi', 12.0, 'No','Useful late at night but expensive in traffic'),
        ('Rome', 'Bus and Metro', 1.7, 'No','Historic center is walkable'),
        ('Barcelona', 'Metro', 2.4, 'No','Good coverage for tourists'),
        ('Barcelona', 'Bus', 2.1, 'No','Convenient for beach areas'),
        ('Los Angeles', 'Car', 0, 'Yes','Renting a car is strongly recommended'),
        ('Los Angeles', 'Metro', 1.9, 'Sometimes','Limited compared to other major cities'),
        ('Larnaca', 'Bus and Taxi', 1.5, 'Sometimes','Useful to have a car for beaches outside the center'),
    ]
    cursor.executemany("INSERT INTO public_transport (city, transport_type, average_ticket_price, car_needed, notes) VALUES (?, ?, ?, ?, ?)", public_transport)

    conn.commit()
    conn.close()
    print("Database 'travel_agency.db' created with all expanded fields and new tables!")

if __name__ == "__main__":
    create_travel_db()