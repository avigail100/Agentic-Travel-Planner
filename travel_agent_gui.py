
import sys
import html
import io
import threading
import re
from contextlib import redirect_stdout
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QTextEdit, QLineEdit, QPushButton, QLabel, QFrame,
    QMessageBox, QInputDialog, QComboBox,
    QScrollArea
)

import plan_and_execute_agent as agent_backend

process_request = agent_backend.process_request
get_session_preferences = agent_backend.get_session_preferences

from gui.gui_config import (
    APP_DIR, ROBOT_IMAGE, SIF_DESCRIPTIONS, LIGHT_THEME, DARK_THEME,
    set_session_sif,
)

class StreamToSignal(io.StringIO):
    def __init__(self, signal, original_stdout):
        super().__init__()
        self.signal = signal
        self.original_stdout = original_stdout

    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        if text.strip():
            self.signal.emit(text)
        return len(text)

    def flush(self):
        self.original_stdout.flush()

class AgentWorker(QThread):
    progress = Signal(str)
    terminal_log = Signal(str)
    interrupt_requested = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, user_text, thread_id):
        super().__init__()
        self.user_text = user_text
        self.thread_id = thread_id
        self._interrupt_event = threading.Event()
        self._interrupt_decision = {"action": "cancel"}

    def ask_interrupt(self, payload):
        self._interrupt_decision = {"action": "cancel"}
        self._interrupt_event.clear()
        self.interrupt_requested.emit(payload)
        self._interrupt_event.wait()
        return self._interrupt_decision

    def provide_interrupt_decision(self, decision):
        self._interrupt_decision = decision
        self._interrupt_event.set()

    def run(self):
        try:
            stream = StreamToSignal(self.terminal_log, sys.__stdout__)
            with redirect_stdout(stream):
                result = process_request(
                    self.user_text,
                    self.thread_id,
                    progress_callback=self.progress.emit,
                    interrupt_callback=self.ask_interrupt,
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

from gui.gui_dialogs import DetailsDialog, PlanApprovalDialog
from gui.chat_widgets import ChatMessageWidget

class TravelAgentGUI(QWidget):
    def __init__(self):
        super().__init__()

        if ROBOT_IMAGE.exists():
            self.setWindowIcon(QIcon(str(ROBOT_IMAGE)))

        self.thread_id = "default"
        self.details_log_text = ""
        self.details_dialog = None
        self.dark_mode = False
        self.theme = LIGHT_THEME
        self.chat_messages = []  # list of (role, text, structured_data)

        self.setWindowTitle("Bond Travel Planner")
        self.setGeometry(140, 70, 1180, 730)

        self.stack = QStackedWidget(self)
        self.login_page = self.build_login_page()
        self.chat_page = self.build_chat_page()

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.chat_page)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.stack)

        self.apply_theme()

    def robot_label(self, size=96):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        if ROBOT_IMAGE.exists():
            pixmap = QPixmap(str(ROBOT_IMAGE))
            label.setPixmap(
                pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            label.setText("🤖")
            label.setStyleSheet(f"font-size:{int(size * 0.65)}px;")
        return label

    def apply_theme(self):
        t = DARK_THEME if self.dark_mode else LIGHT_THEME
        self.theme = t

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {t['app_bg']};
                font-family: Arial;
                color: {t['text']};
            }}

            QFrame#mainCard, QFrame#sideCard, QFrame#loginCard {{
                background-color: {t['card']};
                border: 1px solid {t['border']};
                border-radius: 24px;
            }}

            QLabel#appTitle {{
                font-size: 25px;
                font-weight: 800;
                color: {t['text']};
            }}

            QLabel#loginTitle {{
                font-size: 32px;
                font-weight: 800;
                color: {t['text']};
                background: transparent;
            }}

            QLabel#sectionTitle {{
                font-size: 18px;
                font-weight: 800;
                color: {t['text']};
            }}

            QLabel#subtitle {{
                font-size: 14px;
                color: {t['muted']};
                background: transparent;
            }}

            QTextEdit {{
                background-color: {t['input_bg']};
                border: 1px solid {t['border']};
                border-radius: 20px;
                padding: 18px;
                font-size: 15px;
                color: {t['text']};
            }}

            QLineEdit {{
                background-color: {t['input_bg']};
                border: 1px solid {t['border']};
                border-radius: 18px;
                padding: 14px;
                font-size: 15px;
                color: {t['text']};
            }}

            QComboBox {{
                background-color: {t['input_bg']};
                border: 1px solid {t['border']};
                border-radius: 14px;
                padding: 10px;
                font-size: 14px;
                color: {t['text']};
            }}

            QPushButton {{
                background-color: {t['accent']};
                color: white;
                border: none;
                border-radius: 18px;
                padding: 12px 18px;
                font-size: 16px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: {t['accent_hover']};
            }}

            QPushButton:disabled {{
                background-color: {t['border']};
                color: {t['muted']};
            }}

            QPushButton#secondaryButton {{
                background-color: {t['card2']};
                color: {t['text']};
                border: 1px solid {t['border']};
            }}

            QPushButton#dangerButton {{
                background-color: #E74C3C;
                color: white;
            }}

            QCheckBox {{
                background-color: {t['card']};
                border: 1px solid {t['border']};
                border-radius: 14px;
                padding: 12px;
                font-size: 14px;
                color: {t['text']};
            }}

            QDialog {{
                background-color: {t['app_bg']};
                font-family: Arial;
                color: {t['text']};
            }}

            QLabel#dialogTitle {{
                font-size: 22px;
                font-weight: 800;
                color: {t['text']};
            }}

            QTextEdit#logArea {{
                background-color: {t['input_bg']};
                border: 1px solid {t['border']};
                border-radius: 18px;
                padding: 14px;
                font-family: Menlo, Monaco, monospace;
                font-size: 12px;
                color: {t['text']};
            }}
        """)

        if hasattr(self, "theme_button"):
            self.theme_button.setText("☀️ Light Mode" if self.dark_mode else "🌙 Dark Mode")
        if hasattr(self, "login_theme_button"):
            self.login_theme_button.setText("☀️ Light Mode" if self.dark_mode else "🌙 Dark Mode")
        if hasattr(self, "sif_description_label"):
            self.sif_description_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {t['card2']};
                    border-radius: 14px;
                    padding: 12px;
                    color: {t['muted']};
                    font-size: 13px;
                }}
            """)
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(f"""
                color: {t['muted']};
                font-size: 14px;
                padding-left: 8px;
                font-style: italic;
            """)
        if hasattr(self, "chat_scroll"):
            self.chat_scroll.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {t['card']};
                    border: none;
                    border-radius: 16px;
                }}
                QScrollBar:vertical {{
                    background: transparent;
                    width: 10px;
                }}
                QScrollBar::handle:vertical {{
                    background: {t['border']};
                    border-radius: 5px;
                }}
            """)
            self.chat_container.setStyleSheet(f"background-color: {t['card']};")

        if hasattr(self, "memory_area"):
            self.memory_area.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {t['input_bg']};
                    border: 1px solid {t['border']};
                    border-radius: 16px;
                    padding: 10px;
                    font-size: 13px;
                    color: {t['text']};
                }}
            """)
        if hasattr(self, "open_details_hint"):
            self.open_details_hint.setStyleSheet(f"color: {t['muted']}; font-size: 13px;")
        if hasattr(self, "footer"):
            self.footer.setStyleSheet(f"color: {t['muted']}; font-size: 13px;")

    def build_login_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(345, 90, 345, 90)

        card = QFrame()
        card.setObjectName("loginCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(38, 34, 38, 34)
        card_layout.setSpacing(16)

        logo = self.robot_label(130)

        title = QLabel("Bond Travel Planner")
        title.setObjectName("loginTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Continue your saved travel session")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("Session ID, e.g. student_01")
        self.session_input.returnPressed.connect(self.start_chat)

        self.login_theme_button = QPushButton("🌙 Dark Mode")
        self.login_theme_button.setObjectName("secondaryButton")
        self.login_theme_button.clicked.connect(self.toggle_theme)

        start_button = QPushButton("Start Mission")
        start_button.clicked.connect(self.start_chat)

        card_layout.addWidget(logo)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.session_input)
        card_layout.addWidget(start_button)
        card_layout.addWidget(self.login_theme_button)

        layout.addWidget(card)
        return page

    def build_chat_page(self):
        page = QWidget()

        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(28, 22, 28, 18)
        outer_layout.setSpacing(14)

        header_layout = QHBoxLayout()

        logo = self.robot_label(48)

        title_layout = QVBoxLayout()
        title = QLabel("Bond Travel Planner")
        title.setObjectName("appTitle")

        self.session_label = QLabel("Logged in as: default")
        self.session_label.setObjectName("subtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(self.session_label)

        self.theme_button = QPushButton("🌙 Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)

        self.details_button = QPushButton("Details")
        self.details_button.clicked.connect(self.open_details_dialog)

        self.change_user_button = QPushButton("Change User")
        self.change_user_button.clicked.connect(self.back_to_login)

        self.new_chat_button = QPushButton("Clear Screen")
        self.new_chat_button.clicked.connect(self.clear_chat)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.theme_button)
        header_layout.addWidget(self.details_button)
        header_layout.addWidget(self.change_user_button)
        header_layout.addWidget(self.new_chat_button)

        outer_layout.addLayout(header_layout)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        chat_card = QFrame()
        chat_card.setObjectName("mainCard")

        chat_layout = QVBoxLayout(chat_card)
        chat_layout.setContentsMargins(22, 22, 22, 18)
        chat_layout.setSpacing(14)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)

        self.chat_container = QWidget()
        self.chat_messages_layout = QVBoxLayout(self.chat_container)
        self.chat_messages_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_messages_layout.setSpacing(12)
        self.chat_messages_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)

        self.status_label = QLabel("")
        self.status_label.setObjectName("subtitle")

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message...")
        self.user_input.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("➤")
        self.send_button.setFixedWidth(65)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)

        chat_layout.addWidget(self.chat_scroll)
        chat_layout.addWidget(self.status_label)
        chat_layout.addLayout(input_layout)

        side_card = QFrame()
        side_card.setObjectName("sideCard")
        side_card.setFixedWidth(315)

        side_layout = QVBoxLayout(side_card)
        side_layout.setContentsMargins(20, 22, 20, 20)
        side_layout.setSpacing(12)

        autonomy_title = QLabel("Agent Autonomy")
        autonomy_title.setObjectName("sectionTitle")

        autonomy_subtitle = QLabel("Change SIF for this session")
        autonomy_subtitle.setObjectName("subtitle")

        self.sif_combo = QComboBox()
        self.sif_combo.addItem("SIF 1 - Low approval", "1")
        self.sif_combo.addItem("SIF 2 - Guided", "2")
        self.sif_combo.addItem("SIF 3 - Autonomous", "3")
        self.sif_combo.currentIndexChanged.connect(self.on_sif_changed)

        self.sif_description_label = QLabel("")
        self.sif_description_label.setWordWrap(True)

        memory_title = QLabel("Bond Memory")
        memory_title.setObjectName("sectionTitle")

        memory_subtitle = QLabel("Saved preferences for this session")
        memory_subtitle.setObjectName("subtitle")

        self.memory_area = QTextEdit()
        self.memory_area.setReadOnly(True)

        self.open_details_hint = QLabel("Use Details to open terminal logs in a separate window.")
        self.open_details_hint.setWordWrap(True)

        side_layout.addWidget(autonomy_title)
        side_layout.addWidget(autonomy_subtitle)
        side_layout.addWidget(self.sif_combo)
        side_layout.addWidget(self.sif_description_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(memory_title)
        side_layout.addWidget(memory_subtitle)
        side_layout.addWidget(self.memory_area, stretch=1)
        side_layout.addSpacing(8)
        side_layout.addWidget(self.open_details_hint)

        body_layout.addStretch(1)
        body_layout.addWidget(chat_card, 7)
        body_layout.addWidget(side_card, 2)
        body_layout.addStretch(1)

        outer_layout.addLayout(body_layout)

        self.footer = QLabel("Travel smart. Travel happy. 🌍")
        self.footer.setAlignment(Qt.AlignCenter)

        outer_layout.addWidget(self.footer)

        return page

    def toggle_theme(self):
        was_near_bottom = True
        if hasattr(self, "chat_scroll"):
            bar = self.chat_scroll.verticalScrollBar()
            was_near_bottom = bar.value() >= bar.maximum() - 20

        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.rebuild_chat_widgets(scroll_to_bottom=was_near_bottom)

    def rebuild_chat_widgets(self, scroll_to_bottom=True):
        if not hasattr(self, "chat_messages_layout"):
            return

        while self.chat_messages_layout.count():
            item = self.chat_messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # Important: keep the stretch in the layout BEFORE inserting messages.
        # _add_message_widget inserts before the final stretch, so without this
        # the order becomes reversed after theme changes.
        self.chat_messages_layout.addStretch()

        for item in self.chat_messages:
            if len(item) == 3:
                role, text, structured_data = item
            else:
                role, text = item
                structured_data = None
            self._add_message_widget(role, text, store=False, structured_data=structured_data)

        if scroll_to_bottom:
            self.scroll_chat_to_bottom()

    def start_chat(self):
        session = self.session_input.text().strip()
        self.thread_id = session if session else "default"
        self.session_label.setText(f"Logged in as: {self.thread_id}")

        self.clear_chat()
        self.add_agent_message(
            f"Welcome! Your session ID is {self.thread_id}. "
            "How can I help you plan your trip?"
        )

        self.refresh_memory_panel()
        self.stack.setCurrentWidget(self.chat_page)

    def back_to_login(self):
        self.stack.setCurrentWidget(self.login_page)

    def format_agent_response(self, text):
        raw = str(text).strip()

        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and set(stripped) <= {"="}:
                continue
            lines.append(line.rstrip())

        cleaned = "\n".join(lines).strip()
        cleaned = cleaned.replace("TRIP SUMMARY FOR:", "Trip Summary For:")

        safe_text = html.escape(cleaned)

        replacements = {
            "Trip Summary For:": "<span style='font-size:20px; font-weight:800;'>✨ Trip Summary For:</span>",
            "Flights:": "✈️ <b>Flights:</b>",
            "Flight:": "✈️ <b>Flight:</b>",
            "Hotels:": "🏨 <b>Hotels:</b>",
            "Hotel:": "🏨 <b>Hotel:</b>",
            "Activities:": "🎟️ <b>Activities:</b>",
            "Activity:": "🎟️ <b>Activity:</b>",
            "Restaurants:": "🍽️ <b>Restaurants:</b>",
            "Restaurant:": "🍽️ <b>Restaurant:</b>",
            "Estimated total:": "💰 <b>Estimated total:</b>",
            "Total:": "💰 <b>Total:</b>",
            "Travel warning:": "⚠️ <b>Travel warning:</b>",
            "Warning:": "⚠️ <b>Warning:</b>",
        }

        for old, new in replacements.items():
            safe_text = safe_text.replace(html.escape(old), new)

        safe_text = safe_text.replace("\n", "<br>")
        return safe_text

    def _robot_html(self, size=28):
        if ROBOT_IMAGE.exists():
            return f'<img src="{ROBOT_IMAGE.as_posix()}" width="{size}" height="{size}" style="vertical-align:middle;">'
        return "🤖"

    def _add_message_widget(self, role: str, text: str, store: bool = True, structured_data: dict | None = None):
        if store:
            self.chat_messages.append((role, str(text), structured_data))

        speaker = "Bond" if role == "agent" else self.thread_id
        widget = ChatMessageWidget(
            role=role,
            speaker=speaker,
            text=str(text),
            theme=self.theme,
            robot_path=ROBOT_IMAGE,
            structured_data=structured_data,
        )

        # Insert before the final stretch. If there is no stretch yet, add directly.
        if self.chat_messages_layout.count() == 0:
            self.chat_messages_layout.addWidget(widget)
        else:
            index = max(0, self.chat_messages_layout.count() - 1)
            self.chat_messages_layout.insertWidget(index, widget)

        self.scroll_chat_to_bottom()

    def scroll_chat_to_bottom(self):
        QApplication.processEvents()
        if hasattr(self, "chat_scroll"):
            bar = self.chat_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

    def add_agent_message(self, text, store=True, structured_data: dict | None = None):
        self._add_message_widget("agent", text, store=store, structured_data=structured_data)

    def add_user_message(self, text, store=True):
        self._add_message_widget("user", text, store=store, structured_data=None)

    def add_agent_progress_message(self, text):

        self.status_label.setText(f"🤖 {html.escape(str(text))}")

    def open_details_dialog(self):
        if self.details_dialog is None:
            self.details_dialog = DetailsDialog(self)
        self.details_dialog.set_log_text(self.details_log_text)
        self.details_dialog.show()
        self.details_dialog.raise_()
        self.details_dialog.activateWindow()

    def add_detail_log(self, text):
        self.details_log_text += str(text)
        if self.details_dialog is not None:
            self.details_dialog.set_log_text(self.details_log_text)

    def send_message(self):
        text = self.user_input.text().strip()
        if not text:
            return

        self.add_user_message(text)
        self.user_input.clear()

        self.add_agent_progress_message("Bond is working...")
        self.add_detail_log(f"\n=== Session: {self.thread_id} ===\n")

        self.send_button.setEnabled(False)
        self.user_input.setEnabled(False)
        self.sif_combo.setEnabled(False)

        self.worker = AgentWorker(text, self.thread_id)
        self.worker.progress.connect(self.on_agent_progress)
        self.worker.terminal_log.connect(self.add_detail_log)
        self.worker.interrupt_requested.connect(self.on_interrupt_requested)
        self.worker.finished.connect(self.on_agent_finished)
        self.worker.error.connect(self.on_agent_error)
        self.worker.start()

    def on_agent_progress(self, log):
        self.add_agent_progress_message(log)

    def on_interrupt_requested(self, payload):
        kind = payload.get("type", "")

        if kind == "web_host_approval":
            query = payload.get("query", "")
            category = payload.get("category", "")
            hosts = payload.get("hosts", [])
            unknown_hosts = payload.get("unknown_hosts", [])
            alternatives = payload.get("alternatives", [])

            text = (
                f"Bond wants to run a web search:\n\n"
                f"Query: {query}\n"
                f"Category: {category}\n"
                f"Sources: {', '.join(hosts)}\n"
                f"Unapproved hosts: {', '.join(unknown_hosts)}\n\n"
                f"What do you want to do?"
            )

            msg = QMessageBox(self)
            msg.setWindowTitle("Human Approval Needed")
            msg.setText(text)

            approve_btn = msg.addButton("Approve", QMessageBox.AcceptRole)
            edit_btn = msg.addButton("Edit Category", QMessageBox.ActionRole)
            msg.addButton("Cancel", QMessageBox.RejectRole)

            msg.exec()
            clicked = msg.clickedButton()

            if clicked == approve_btn:
                decision = {"action": "approve"}
            elif clicked == edit_btn:
                if alternatives:
                    chosen, ok = QInputDialog.getItem(
                        self,
                        "Edit Search Category",
                        "Choose replacement category:",
                        alternatives,
                        0,
                        False,
                    )
                    decision = {"action": "edit", "category": chosen} if ok and chosen else {"action": "cancel"}
                else:
                    decision = {"action": "cancel"}
            else:
                decision = {"action": "cancel"}

            self.add_detail_log(f"\n[GUI INTERRUPT] {kind} -> {decision}\n")
            self.worker.provide_interrupt_decision(decision)
            return

        if kind == "user_question":
            question = payload.get("question", "Bond needs your input.")
            options = payload.get("options", [])

            if options:
                answer, ok = QInputDialog.getItem(self, "Bond Question", question, options, 0, False)
            else:
                answer, ok = QInputDialog.getText(self, "Bond Question", question)

            decision = answer if ok else ""
            self.add_detail_log(f"\n[GUI INTERRUPT] {kind} -> {decision}\n")
            self.worker.provide_interrupt_decision(decision)
            return

        if kind == "sif_plan_approval":
            plan = payload.get("plan", [])
            dialog = PlanApprovalDialog(plan, self)
            dialog.exec()
            decision = dialog.decision
            self.add_detail_log(f"\n[GUI INTERRUPT] {kind} -> {decision}\n")
            self.worker.provide_interrupt_decision(decision)
            return

        if kind == "sif_alternatives_offer":
            msg = QMessageBox(self)
            msg.setWindowTitle("SIF-2: Narrow Search?")
            msg.setText(
                "Bond finished the first search round.\n\n"
                "Do you want to guide/narrow the next search, or continue automatically?"
            )
            continue_btn = msg.addButton("Continue", QMessageBox.AcceptRole)
            narrow_btn = msg.addButton("Narrow Search", QMessageBox.ActionRole)
            msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == narrow_btn:
                pref, ok = QInputDialog.getText(
                    self,
                    "Narrow Search",
                    "Enter preference, e.g. Europe only, under $800, beach resort:",
                )
                decision = {"action": "narrow", "preference": pref.strip()} if ok and pref.strip() else {"action": "continue"}
            else:
                decision = {"action": "continue"}

            self.add_detail_log(f"\n[GUI INTERRUPT] {kind} -> {decision}\n")
            self.worker.provide_interrupt_decision(decision)
            return

        if kind == "sif_budget_breach":
            budget = float(payload.get("budget", 0) or 0)
            total_cost = float(payload.get("total_cost", 0) or 0)
            overage = float(payload.get("overage", 0) or 0)

            msg = QMessageBox(self)
            msg.setWindowTitle("SIF-2: Budget Breach")
            msg.setText(
                f"The current plan is over budget.\n\n"
                f"Budget: ${budget:.2f}\n"
                f"Found total: ${total_cost:.2f}\n"
                f"Overage: ${overage:.2f}\n\n"
                "What do you want to do?"
            )
            approve_btn = msg.addButton("Approve Over Budget", QMessageBox.AcceptRole)
            new_budget_btn = msg.addButton("Set New Budget", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel Request", QMessageBox.RejectRole)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == approve_btn:
                decision = {"action": "approve"}
            elif clicked == new_budget_btn:
                amount, ok = QInputDialog.getDouble(
                    self, "New Budget", "Enter new budget:",
                    max(total_cost, budget), 0, 1000000, 2,
                )
                decision = {"action": "new_budget", "amount": amount} if ok else {"action": "cancel"}
            elif clicked == cancel_btn:
                decision = {"action": "cancel"}
            else:
                decision = {"action": "cancel"}

            self.add_detail_log(f"\n[GUI INTERRUPT] {kind} -> {decision}\n")
            self.worker.provide_interrupt_decision(decision)
            return

        self.add_detail_log(f"\n[GUI INTERRUPT] Unknown interrupt: {payload}\n")
        self.worker.provide_interrupt_decision({"action": "cancel"})

    def on_agent_finished(self, result):
        self.status_label.clear()

        response = result.get("response", "")
        structured_data = result.get("structured_data") or result.get("travel_data") or {}
        self.add_agent_message(response, structured_data=structured_data)
        self.refresh_memory_panel()

        self.send_button.setEnabled(True)
        self.user_input.setEnabled(True)
        self.sif_combo.setEnabled(True)
        self.user_input.setFocus()

    def on_agent_error(self, error):
        self.status_label.clear()

        self.add_agent_message(f"Error: {error}")
        self.add_detail_log(f"\nERROR: {error}\n")

        self.send_button.setEnabled(True)
        self.user_input.setEnabled(True)
        self.sif_combo.setEnabled(True)
        self.user_input.setFocus()

    def clear_chat(self):
        self.chat_messages = []
        if hasattr(self, "chat_messages_layout"):
            while self.chat_messages_layout.count():
                item = self.chat_messages_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.chat_messages_layout.addStretch()
        self.details_log_text = ""
        if self.details_dialog is not None:
            self.details_dialog.set_log_text("")
        if hasattr(self, "status_label"):
            self.status_label.clear()

    def refresh_memory_panel(self):
        prefs = get_session_preferences(self.thread_id)

        sif = str(prefs.get("sif", "3"))
        if sif not in ("1", "2", "3"):
            sif = "3"

        self.sif_combo.blockSignals(True)
        self.sif_combo.setCurrentIndex(int(sif) - 1)
        self.sif_combo.blockSignals(False)
        self.sif_description_label.setText(SIF_DESCRIPTIONS.get(sif, SIF_DESCRIPTIONS["3"]))

        if not prefs:
            self.memory_area.setPlainText(f"SIF: {sif}\nNo saved preferences yet.")
            return

        lines = [f"SIF: {sif} - {SIF_DESCRIPTIONS.get(sif, '')}"]
        for key, value in prefs.items():
            if key == "sif":
                continue
            lines.append(f"{key}: {value}")

        self.memory_area.setPlainText("\n".join(lines))

    def on_sif_changed(self):
        sif = self.sif_combo.currentData()
        if not sif:
            return

        try:
            set_session_sif(self.thread_id, str(sif))
            self.sif_description_label.setText(SIF_DESCRIPTIONS[str(sif)])
            self.refresh_memory_panel()
            self.add_detail_log(f"\n[GUI SETTINGS] SIF updated to {sif}\n")
        except Exception as e:
            QMessageBox.warning(self, "SIF Update Failed", str(e))
            self.refresh_memory_panel()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if ROBOT_IMAGE.exists():
        app.setWindowIcon(QIcon(str(ROBOT_IMAGE)))
    window = TravelAgentGUI()
    window.show()
    sys.exit(app.exec())
