from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QCheckBox
)

class DetailsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.setWindowTitle("Bond Details")
        self.resize(760, 560)
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        title = QLabel("⚙️ Bond Details")
        title.setObjectName("dialogTitle")

        subtitle = QLabel("Terminal logs and agent workflow")
        subtitle.setObjectName("subtitle")

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("logArea")

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.log_area)
        layout.addWidget(close_button, alignment=Qt.AlignRight)

    def set_log_text(self, text: str):
        self.log_area.setPlainText(text)
        self.log_area.moveCursor(QTextCursor.End)


class PlanApprovalDialog(QDialog):
    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.plan = list(plan or [])
        self.decision = {"action": "cancel"}

        self.setWindowTitle("SIF-1 Plan Approval")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        title = QLabel("📋 SIF-1: Plan Approval Required")
        title.setObjectName("dialogTitle")

        subtitle = QLabel("Select which steps Bond should execute, or edit the plan manually below.")
        subtitle.setObjectName("subtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.checkboxes = []
        for i, step in enumerate(self.plan, 1):
            cb = QCheckBox(f"{i}. {step}")
            cb.setChecked(True)
            self.checkboxes.append(cb)
            layout.addWidget(cb)

        manual_label = QLabel("Optional manual revised plan — one step per line:")
        manual_label.setObjectName("subtitle")
        self.manual_plan = QTextEdit()
        self.manual_plan.setPlaceholderText(
            "Leave empty to use the selected checkboxes.\n"
            "Example:\nFetch flights from TLV to Paris\nFetch hotels in Paris"
        )
        self.manual_plan.setFixedHeight(110)

        layout.addWidget(manual_label)
        layout.addWidget(self.manual_plan)

        buttons = QHBoxLayout()
        approve_selected = QPushButton("Approve Selected")
        approve_all = QPushButton("Approve All")
        cancel = QPushButton("Cancel")
        cancel.setObjectName("dangerButton")
        approve_all.setObjectName("secondaryButton")

        approve_selected.clicked.connect(self._approve_selected)
        approve_all.clicked.connect(self._approve_all)
        cancel.clicked.connect(self._cancel)

        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(approve_all)
        buttons.addWidget(approve_selected)
        layout.addLayout(buttons)

    def _manual_steps(self):
        text = self.manual_plan.toPlainText().strip()
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _approve_selected(self):
        manual_steps = self._manual_steps()
        if manual_steps:
            self.decision = {"action": "edit", "plan": manual_steps}
            self.accept()
            return

        selected = [step for step, cb in zip(self.plan, self.checkboxes) if cb.isChecked()]
        if not selected:
            self.decision = {"action": "cancel"}
        elif len(selected) == len(self.plan):
            self.decision = {"action": "approve"}
        else:
            self.decision = {"action": "edit", "plan": selected}
        self.accept()

    def _approve_all(self):
        self.decision = {"action": "approve"}
        self.accept()

    def _cancel(self):
        self.decision = {"action": "cancel"}
        self.reject()
