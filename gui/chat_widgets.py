import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy

class ChatMessageWidget(QFrame):
    """
    Robust chat message widget.

    Best mode:
        Uses structured_data from the agent:
        {
            "destination": "...",
            "flights": [...],
            "hotels": [...],
            "activities": [...],
            "visa": "...",
            "time_difference": "...",
            "warning": "..."
        }

    Fallback mode:
        Displays clean plain text if structured_data is missing.
    """

    def __init__(
        self,
        role: str,
        speaker: str,
        text: str,
        theme: dict,
        robot_path: Path | None = None,
        structured_data: dict | None = None,
    ):
        super().__init__()
        self.role = role
        self.speaker = speaker
        self.raw_text = str(text or "")
        self.theme = theme
        self.robot_path = robot_path
        self.structured_data = structured_data or {}

        self.setObjectName("chatBubble")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.build_ui()

    def build_ui(self):
        t = self.theme
        is_agent = self.role == "agent"

        bg = t.get("bubble_agent", t["card2"]) if is_agent else t.get("bubble_user", t["card2"])
        border = t["agent_border"] if is_agent else t["user_border"]

        self.setStyleSheet(f"""
            QFrame#chatBubble {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 16px;
            }}
            QLabel {{
                background: transparent;
                color: {t['text']};
            }}
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(14, 12, 14, 12)
        self.main_layout.setSpacing(10)

        self.add_header(is_agent)

        if is_agent and self.has_structured_data():
            self.render_structured_trip()
        else:
            self.add_plain_text(self.raw_text)

    def add_header(self, is_agent: bool):
        t = self.theme

        header = QHBoxLayout()
        header.setSpacing(8)

        icon = QLabel()
        icon.setFixedSize(26, 26)
        icon.setAlignment(Qt.AlignCenter)

        if is_agent and self.robot_path and self.robot_path.exists():
            pixmap = QPixmap(str(self.robot_path))
            icon.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon.setText("👤" if not is_agent else "🤖")
            icon.setStyleSheet("font-size: 18px;")

        name = QLabel(self.speaker)
        name.setStyleSheet(f"""
            color: {t['accent']};
            font-size: 15px;
            font-weight: 800;
            background: transparent;
        """)

        header.addWidget(icon)
        header.addWidget(name)
        header.addStretch()
        self.main_layout.addLayout(header)

    def add_plain_text(self, text: str):
        label = QLabel(str(text))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet(f"""
            color: {self.theme['text']};
            font-size: 15px;
            line-height: 1.55;
            background: transparent;
        """)
        self.main_layout.addWidget(label)

    # ---------------- structured rendering ----------------

    def has_structured_data(self) -> bool:
        d = self.structured_data
        return bool(
            d.get("destination")
            or d.get("flights")
            or d.get("hotels")
            or d.get("activities")
            or d.get("restaurants")
            or d.get("car_rentals")
            or d.get("transport_info")
            or d.get("visa")
            or d.get("time_difference")
            or d.get("currency_exchange")
            or d.get("seasonal_recommendations")
            or d.get("warning")
            or d.get("estimated_cost")
            or d.get("notes")
        )

    def render_structured_trip(self):
        d = self.structured_data

        if d.get("destination"):
            self.add_section_title("✨", "Trip Summary")
            dest = QLabel(str(d["destination"]).upper())
            dest.setStyleSheet(f"""
                color: {self.theme['accent']};
                font-size: 17px;
                font-weight: 800;
                letter-spacing: 1px;
                background: transparent;
            """)
            self.main_layout.addWidget(dest)

        if d.get("flights"):
            self.add_section_title("✈️", "Flights")
            for flight in d["flights"]:
                self.add_flight_card(flight)

        if d.get("hotels"):
            self.add_section_title("🏨", "Hotels")
            for hotel in d["hotels"]:
                self.add_hotel_card(hotel)

        if d.get("activities"):
            self.add_section_title("🎯", "Activities")
            for activity in d["activities"]:
                self.add_activity_card(activity)

        if d.get("restaurants"):
            self.add_section_title("🍽️", "Restaurants")
            for restaurant in d["restaurants"]:
                self.add_restaurant_card(restaurant)

        if d.get("car_rentals"):
            self.add_section_title("🚗", "Car Rentals")
            for rental in d["car_rentals"]:
                self.add_car_rental_card(rental)
        
        if d.get("transport_info"):
            self.add_section_title("🚇", "City Transport")
            self.add_info_card("🚇", str(d["transport_info"]), "#0EA5E9")

        if d.get("visa"):
            self.add_section_title("📋", "Visa Information")
            self.add_info_card("📋", str(d["visa"]), "#3B82F6")

        if d.get("time_difference"):
            self.add_section_title("🕒", "Time Difference")
            self.add_info_card("🕒", str(d["time_difference"]), "#6366F1")

        if d.get("currency_exchange"):
            self.add_section_title("💱", "Currency Exchange")
            self.add_info_card("💱", str(d["currency_exchange"]), "#8B5CF6")

        if d.get("seasonal_recommendations"):
            self.add_section_title("🌤️", "Seasonal Recommendations")
            self.add_info_card("🌤️", str(d["seasonal_recommendations"]), "#10B981")

        if d.get("estimated_cost"):
            self.add_section_title("💰", "Estimated Trip Cost")
            self.add_info_card("💰", str(d["estimated_cost"]), "#16A34A")

        if d.get("warning"):
            self.add_section_title("⚠️", "Travel Warning")
            self.add_info_card("⚠️", str(d["warning"]), "#EF4444")

        if d.get("notes"):
            self.add_section_title("📝", "Notes")
            self.add_info_card("•", str(d["notes"]), "#94A3B8")

    def add_section_title(self, icon: str, title: str):
        label = QLabel(f"{icon} {title}")
        label.setStyleSheet(f"""
            color: {self.theme['text']};
            font-size: 17px;
            font-weight: 800;
            margin-top: 10px;
            background: transparent;
        """)
        self.main_layout.addWidget(label)

    def make_card(self, border_color: str):
        card = QFrame()
        card.setObjectName("innerCard")
        card.setStyleSheet(f"""
            QFrame#innerCard {{
                background-color: {self.theme['card2']};
                border: 1px solid {self.theme['border']};
                border-left: 4px solid {border_color};
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)
        return card, layout

    def add_badges(self, layout, badges: list[tuple[str, str]]):
        badges = [(text, color) for text, color in badges if text not in (None, "", [])]
        if not badges:
            return

        row = QHBoxLayout()
        row.setSpacing(6)

        for text, color in badges:
            badge = QLabel(str(text))
            badge.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 9px;
                    padding: 3px 8px;
                    font-size: 12px;
                    font-weight: 700;
                }}
            """)
            row.addWidget(badge)

        row.addStretch()
        layout.addLayout(row)

    def add_flight_card(self, flight: dict):
        card, layout = self.make_card("#3B82F6")

        title = QLabel(f"✈️ {flight.get('airline', 'Flight')}")
        title.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {self.theme['text']};")
        layout.addWidget(title)

        badges = [
            (flight.get("flight"), "#3B82F6"),
            (self.money(flight.get("price")), "#16A34A"),
            (self.time_range(flight), "#6366F1"),
            (flight.get("duration"), "#0EA5E9"),
        ]
        self.add_badges(layout, badges)
        self.main_layout.addWidget(card)

    def add_hotel_card(self, hotel: dict):
        card, layout = self.make_card("#EAB308")

        stars = hotel.get("stars")
        stars_text = ""
        if isinstance(stars, int) and stars > 0:
            stars_text = " " + ("★" * stars + "☆" * (5 - stars))

        title = QLabel(f"🏨 {hotel.get('name', 'Hotel')}{stars_text}")
        title.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 800;
            color: {self.theme['text']};
        """)
        layout.addWidget(title)

        badges = [
            (self.money(hotel.get("price"), suffix="/night"), "#CA8A04"),
            (f"Rating {hotel.get('rating')}" if hotel.get("rating") else "", "#6366F1"),
        ]

        for amenity in hotel.get("amenities", []) or []:
            badges.append((amenity, "#64748B"))

        self.add_badges(layout, badges)
        self.main_layout.addWidget(card)

    def add_activity_card(self, activity: dict):
        card, layout = self.make_card("#06B6D4")

        title = QLabel(f"🎯 {activity.get('name', 'Activity')}")
        title.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {self.theme['text']};")
        layout.addWidget(title)

        badges = [
            (self.money(activity.get("price")), "#0891B2"),
            (activity.get("duration"), "#6366F1"),
            (activity.get("category"), "#64748B"),
            (activity.get("suitability"), "#8B5CF6"),
        ]
        self.add_badges(layout, badges)
        self.main_layout.addWidget(card)


    def add_car_rental_card(self, rental: dict):
        card, layout = self.make_card("#F97316")

        title = QLabel(f"🚗 {rental.get('company', 'Car Rental')}")
        title.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {self.theme['text']};")
        layout.addWidget(title)

        badges = [
            (rental.get("type"), "#F97316"),
            (rental.get("transmission"), "#6366F1"),
            (rental.get("seats"), "#64748B"),
            (self.money(rental.get("price"), suffix="/day"), "#16A34A"),
        ]
        self.add_badges(layout, badges)
        self.main_layout.addWidget(card)

    def add_restaurant_card(self, restaurant: dict):
        card, layout = self.make_card("#F59E0B")

        title = QLabel(f"🍽️ {restaurant.get('name', 'Restaurant')}")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 800; color: {self.theme['text']};"
        )
        layout.addWidget(title)

        badges = [
            (restaurant.get("price_level"), "#F59E0B"),
            (restaurant.get("cuisine"), "#6366F1"),
            (f"Rating {restaurant.get('rating')}" if restaurant.get("rating") else "", "#64748B"),
            (restaurant.get("special_features"), "#10B981"),
        ]

        self.add_badges(layout, badges)
        self.main_layout.addWidget(card)

    def add_info_card(self, icon: str, text: str, border_color: str):
        card, layout = self.make_card(border_color)

        clean_text = self.clean_display_text(text)
        label = QLabel(f"{icon} {clean_text}")
        label.setWordWrap(True)
        label.setMinimumHeight(42)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet(f"""
            font-size: 14px;
            line-height: 1.6;
            padding: 6px;
            color: {self.theme['text']};
            background: transparent;
        """)
        layout.addWidget(label)
        self.main_layout.addWidget(card)


    def clean_display_text(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(
            r"\b(?:Do you need|Would you like|Let me know|If you need).*?$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip(" .;:\n\t")

    def money(self, value, currency="$", suffix="") -> str:
        if value in (None, ""):
            return ""
        value = str(value)
        if value.startswith("$") or value.startswith("€") or value.startswith("₪"):
            return f"{value}{suffix}"
        return f"{currency}{value}{suffix}"

    def time_range(self, flight: dict) -> str:
        dep = flight.get("departure")
        arr = flight.get("arrival")
        if dep and arr:
            return f"{dep} → {arr}"
        if dep:
            return f"Departure {dep}"
        return ""
