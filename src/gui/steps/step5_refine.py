"""
Step 5 — Refine Alignment.
User sees the selected candidate and can optionally insert Euler spirals
at Line–Arc transitions before proceeding to export.

Phase A/B: placeholder UI with Back and Accept buttons.
Spiral insertion logic will be added in Phase D.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont


class Step5Refine(QWidget):
    refinement_done = Signal(list)   # final list[dict] elements
    back_requested  = Signal()       # user wants to go back to candidates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidate = None
        self._xy = None
        self._chainages = None
        self._settings: dict = {}
        self._working_elements: list[dict] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title (updated by prepare())
        self._title_lbl = QLabel("Refine Alignment")
        self._title_lbl.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        layout.addWidget(self._title_lbl)

        # Algorithm sub-title
        self._algo_lbl = QLabel("")
        self._algo_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._algo_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        layout.addWidget(sep)

        # Placeholder for spiral insertion controls (Phase D)
        placeholder = QLabel("Spiral insertion controls\nwill appear here in Phase D.")
        placeholder.setStyleSheet(
            "color: #666; font-size: 11px; padding: 20px;"
            "border: 1px dashed #444; border-radius: 4px;"
        )
        placeholder.setAlignment(__import__('PySide6.QtCore', fromlist=['Qt']).Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()

        # Metrics (placeholder)
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(
            "QFrame { background: #2a2a2e; border-radius: 4px; padding: 6px; }"
        )
        mf = QVBoxLayout(metrics_frame)
        mf.setContentsMargins(8, 6, 8, 6)
        self._metrics_lbl = QLabel("No candidate selected.")
        self._metrics_lbl.setStyleSheet("color: #888; font-size: 10px;")
        mf.addWidget(self._metrics_lbl)
        layout.addWidget(metrics_frame)

        # Navigation buttons
        btn_row = QHBoxLayout()

        self._back_btn = QPushButton("← Back to Candidates")
        self._back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(self._back_btn)

        self._accept_btn = QPushButton("Accept & Export →")
        self._accept_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: #fff; border-radius: 4px; "
            "font-weight: bold; padding: 6px 12px; }"
            "QPushButton:hover { background: #2ecc71; }"
            "QPushButton:disabled { background: #444; color: #777; }"
        )
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(self._accept_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, candidate, xy, chainages, settings: dict):
        """
        Called by App when user selects a candidate from Step 4.

        Parameters
        ----------
        candidate : CandidateAlignment dataclass (or dict with 'elements' key)
        xy        : np.ndarray of projected coordinates (N, 2) — may be None
        chainages : np.ndarray of chainages — may be None
        settings  : dict from Step3Configure
        """
        self._candidate = candidate
        self._xy        = xy
        self._chainages = chainages
        self._settings  = settings

        # Build working copy of elements
        if hasattr(candidate, 'elements'):
            self._working_elements = list(candidate.elements)
            label = getattr(candidate, 'label', 'Unknown Algorithm')
            n_el  = len(candidate.elements)
            max_d = getattr(candidate, 'max_deviation', 0.0)
            rmse  = getattr(candidate, 'rmse', 0.0)
        elif isinstance(candidate, dict):
            self._working_elements = list(candidate.get('elements', []))
            label = candidate.get('label', 'Unknown Algorithm')
            n_el  = len(self._working_elements)
            max_d = candidate.get('max_deviation', 0.0)
            rmse  = candidate.get('rmse', 0.0)
        else:
            self._working_elements = []
            label = 'Unknown'
            n_el  = 0
            max_d = 0.0
            rmse  = 0.0

        self._title_lbl.setText(f"Refine: {label}")
        self._algo_lbl.setText(f"Algorithm: {label}")
        self._metrics_lbl.setText(
            f"Elements: {n_el}    Max deviation: {max_d:.2f} m    RMSE: {rmse:.2f} m"
        )
        self._accept_btn.setEnabled(len(self._working_elements) > 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_accept(self):
        self.refinement_done.emit(list(self._working_elements))
