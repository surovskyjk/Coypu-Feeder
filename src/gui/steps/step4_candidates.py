"""
Step 4 — Candidate Alignments.
Runs three algorithms (Curvature Heuristic, RANSAC Arc Fit, Greedy Split)
in a background thread and shows metrics cards for each.
The user picks one candidate to proceed to Step 5 (Refine).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QFrame, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


# ---------------------------------------------------------------------------
# CandidateCard
# ---------------------------------------------------------------------------

class CandidateCard(QGroupBox):
    """A single algorithm result card with metrics and a Select button."""
    select_clicked = Signal(str)   # algorithm_id

    def __init__(self, algo_id: str, label: str, color: str, parent=None):
        super().__init__(label, parent)
        self._algo_id = algo_id
        self._color   = color
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Coloured indicator bar
        dot = QFrame()
        dot.setFixedHeight(4)
        dot.setStyleSheet(f"background: {self._color}; border-radius: 2px;")
        layout.addWidget(dot)

        # Metrics
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(2)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Max deviation:"))
        self._dev_lbl = QLabel("—")
        self._dev_lbl.setStyleSheet("color: #aaa;")
        dev_row.addWidget(self._dev_lbl)
        dev_row.addStretch()
        metrics_layout.addLayout(dev_row)

        rmse_row = QHBoxLayout()
        rmse_row.addWidget(QLabel("RMSE:"))
        self._rmse_lbl = QLabel("—")
        self._rmse_lbl.setStyleSheet("color: #aaa;")
        rmse_row.addWidget(self._rmse_lbl)
        rmse_row.addStretch()
        metrics_layout.addLayout(rmse_row)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Elements:"))
        self._n_lbl = QLabel("—")
        self._n_lbl.setStyleSheet("color: #aaa;")
        n_row.addWidget(self._n_lbl)
        n_row.addStretch()
        metrics_layout.addLayout(n_row)

        layout.addLayout(metrics_layout)

        # Select button
        self._btn = QPushButton("Select This")
        self._btn.setEnabled(False)
        self._btn.setStyleSheet(
            f"QPushButton {{ background: {self._color}; color: #000; border-radius: 4px; "
            f"font-weight: bold; padding: 4px 8px; }}"
            f"QPushButton:hover {{ opacity: 0.9; }}"
            f"QPushButton:disabled {{ background: #444; color: #777; }}"
        )
        self._btn.clicked.connect(lambda: self.select_clicked.emit(self._algo_id))
        layout.addWidget(self._btn)

    def set_pending(self):
        """Reset card to 'computing' state."""
        self._dev_lbl.setText("computing…")
        self._rmse_lbl.setText("—")
        self._n_lbl.setText("—")
        self._btn.setEnabled(False)

    def set_result(self, candidate):
        """Update card with completed CandidateAlignment metrics."""
        self._dev_lbl.setText(f"{candidate.max_deviation:.2f} m")
        self._rmse_lbl.setText(f"{candidate.rmse:.2f} m")
        self._n_lbl.setText(str(candidate.n_elements))
        self._btn.setEnabled(len(candidate.elements) > 0)

    def set_error(self):
        """Mark card as failed."""
        self._dev_lbl.setText("error")
        self._rmse_lbl.setText("—")
        self._n_lbl.setText("—")
        self._btn.setEnabled(False)


# ---------------------------------------------------------------------------
# Step4Candidates
# ---------------------------------------------------------------------------

class Step4Candidates(QWidget):
    candidate_selected   = Signal(object)   # CandidateAlignment
    candidate_map_update = Signal(list)     # list[CandidateAlignment] for map overlay
    back_requested       = Signal()         # user clicked ← Back to Configure

    ALGO_DEFS = [
        ("tight",    "Tight Segmentation", "#ff9800"),
        ("balanced", "Balanced Merge",     "#66bb6a"),
        ("smooth",   "Smooth Merge",       "#42a5f5"),
        ("raw",      "OSM Polyline",       "#e040fb"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidates: dict = {}   # algo_id → CandidateAlignment
        self._worker = None
        self._cards: list[CandidateCard] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Candidate Alignments")
        title.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # Progress bar (hidden initially)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_lbl = QLabel("Configure settings and proceed from Step 3.")
        self._status_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._status_lbl.setWordWrap(True)
        layout.addWidget(self._status_lbl)

        # Algorithm cards
        for algo_id, label, color in self.ALGO_DEFS:
            card = CandidateCard(algo_id, label, color, self)
            card.select_clicked.connect(self._on_select)
            layout.addWidget(card)
            self._cards.append(card)

        layout.addStretch()

        # Back button
        self._back_btn = QPushButton("← Back to Configure")
        self._back_btn.clicked.connect(self.back_requested.emit)
        layout.addWidget(self._back_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, tracks, settings: dict, xy_list: list,
                chainages_list: list, work_epsg: int):
        """
        Called by App after Step 3 config confirmed and coordinates projected.

        Parameters
        ----------
        tracks        : list of Track objects (raw OSM)
        settings      : dict from Step3Configure
        xy_list       : list of np.ndarray, one per track, projected coords
        chainages_list: list of np.ndarray, one per track
        work_epsg     : EPSG of the projected coordinate system
        """
        from gui.worker import CandidateWorker

        self._tracks         = tracks
        self._settings       = settings
        self._xy_list        = xy_list
        self._chainages_list = chainages_list
        self._work_epsg      = work_epsg
        self._candidates.clear()

        # Reset all cards
        for card in self._cards:
            card.set_pending()

        if not xy_list:
            self._status_lbl.setText("No projected coordinates available.")
            self._progress.setVisible(False)
            return

        self._status_lbl.setText("Running algorithms…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)   # indeterminate

        # Stop any existing worker
        if self._worker is not None:
            try:
                self._worker.candidate_ready.disconnect()
                self._worker.all_done.disconnect()
                self._worker.failed.disconnect()
            except Exception:
                pass
            self._worker = None

        settings_copy = dict(settings)
        settings_copy["_work_epsg"] = work_epsg

        self._worker = CandidateWorker(xy_list, chainages_list, settings_copy, self)
        self._worker.candidate_ready.connect(self._on_candidate_ready)
        self._worker.candidate_error.connect(self._on_candidate_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_candidate_ready(self, algo_id: str, candidate):
        self._candidates[algo_id] = candidate
        for card in self._cards:
            if card._algo_id == algo_id:
                if candidate.elements:
                    card.set_result(candidate)
                else:
                    card.set_error()
        # Notify App to update map overlay with all received candidates so far
        self.candidate_map_update.emit(list(self._candidates.values()))

    def _on_all_done(self):
        self._progress.setVisible(False)
        n = sum(1 for c in self._candidates.values() if c.elements)
        self._status_lbl.setText(
            f"{n}/{len(self.ALGO_DEFS)} algorithms succeeded. Select one to continue."
        )

    def _on_candidate_error(self, algo_id: str, error: str):
        """One algorithm failed — mark its card, keep others running."""
        for card in self._cards:
            if card._algo_id == algo_id:
                card.set_error()
        # Show truncated error in status
        short = error.split("\n")[0][:120]
        self._status_lbl.setText(f"⚠ {algo_id}: {short}")

    def _on_failed(self, error: str):
        self._progress.setVisible(False)
        self._status_lbl.setText(f"Error: {error}")

    def _on_select(self, algo_id: str):
        c = self._candidates.get(algo_id)
        if c and c.elements:
            self.candidate_selected.emit(c)
