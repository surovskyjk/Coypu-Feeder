"""
6-step numbered sidebar, always visible.
Clicking a completed step navigates back.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


STEPS = [
    ("1", "Find Railway"),
    ("2", "Select Section"),
    ("3", "Configure"),
    ("4", "Candidates"),
    ("5", "Refine"),
    ("6", "Export"),
]


class StepButton(QPushButton):
    def __init__(self, number: str, label: str, parent=None):
        super().__init__(parent)
        self._number = number
        self._label = label
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(54)
        self._set_style("inactive")

    def set_state(self, state: str):
        """state: 'inactive' | 'active' | 'done'"""
        self._set_style(state)

    def _set_style(self, state: str):
        base = "border-radius:6px; text-align:left; padding:8px 12px;"
        if state == "active":
            bg = "background:#2a82da; color:#fff;"
            num_col = "#fff"
        elif state == "done":
            bg = "background:#2e5e2e; color:#8bc34a;"
            num_col = "#8bc34a"
        else:
            bg = "background:#3c3c3f; color:#999;"
            num_col = "#666"
        self.setStyleSheet(f"QPushButton {{ {base} {bg} }}")
        self.setText(f"  {self._number}  {self._label}")


class StepSidebar(QWidget):
    step_clicked = Signal(int)  # 0-based step index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self._max_reached = 0
        self._buttons: list[StepButton] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(6)

        title = QLabel("Coypu-Feeder")
        title.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color:#2a82da; margin-bottom:8px;")
        layout.addWidget(title)

        sub = QLabel("OSM → LandXML")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#666; font-size:10px; margin-bottom:12px;")
        layout.addWidget(sub)

        for i, (num, label) in enumerate(STEPS):
            btn = StepButton(num, label)
            btn.clicked.connect(lambda checked=False, idx=i: self._on_clicked(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()
        self._update_styles()

    def _on_clicked(self, idx: int):
        if idx <= self._max_reached:
            self.step_clicked.emit(idx)

    def _update_styles(self):
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                btn.set_state("active")
            elif i < self._current:
                btn.set_state("done")
            else:
                btn.set_state("inactive")

    def set_step(self, idx: int):
        self._current = idx
        self._max_reached = max(self._max_reached, idx)
        self._update_styles()

    def reset(self):
        self._current = 0
        self._max_reached = 0
        self._update_styles()
