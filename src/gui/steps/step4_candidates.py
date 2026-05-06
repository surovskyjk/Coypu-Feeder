"""
Step 4 — Candidate Alignments.

Runs four algorithms (Segment & Fit, DP Segmentation, Progressive MC, Raw OSM)
in a background thread and shows a metrics card per algorithm.

Each card has:
  • A thin coloured bar identifying the algorithm
  • A per-card indeterminate progress bar while the algorithm runs
  • A live status label showing what the algorithm is doing
  • Quality metrics once the algorithm finishes
  • A "Select This" button to proceed to Step 5

Progressive MC also emits intermediate (preview) results every ~7 s so the map
shows a preliminary alignment before the final result is ready.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QFrame, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QColor


# ---------------------------------------------------------------------------
# CandidateCard
# ---------------------------------------------------------------------------

class CandidateCard(QGroupBox):
    """A single algorithm result card with progress indicator, metrics and Select button."""
    select_clicked = Signal(str)   # algorithm_id

    def __init__(self, algo_id: str, label: str, color: str, parent=None):
        super().__init__(label, parent)
        self._algo_id = algo_id
        self._color   = color
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(4)

        # ── Coloured accent bar ───────────────────────────────────────
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(f"background: {self._color}; border-radius: 2px;")
        layout.addWidget(bar)

        # ── Per-card progress bar (indeterminate) ─────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate mode
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ background: #333; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {self._color}; border-radius: 3px; }}"
        )
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # ── Live status label ─────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setVisible(False)
        layout.addWidget(self._status_lbl)

        # ── Quality metrics ───────────────────────────────────────────
        gform = QVBoxLayout()
        gform.setSpacing(2)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Max deviation:"))
        self._dev_lbl = QLabel("—")
        self._dev_lbl.setStyleSheet("color: #aaa;")
        dev_row.addWidget(self._dev_lbl)
        dev_row.addStretch()
        gform.addLayout(dev_row)

        rmse_row = QHBoxLayout()
        rmse_row.addWidget(QLabel("RMSE:"))
        self._rmse_lbl = QLabel("—")
        self._rmse_lbl.setStyleSheet("color: #aaa;")
        rmse_row.addWidget(self._rmse_lbl)
        rmse_row.addStretch()
        gform.addLayout(rmse_row)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Elements:"))
        self._n_lbl = QLabel("—")
        self._n_lbl.setStyleSheet("color: #aaa;")
        n_row.addWidget(self._n_lbl)
        n_row.addStretch()
        gform.addLayout(n_row)

        jump_row = QHBoxLayout()
        jump_lbl = QLabel("C1 mismatch:")
        jump_lbl.setToolTip(
            "Maximum tangent-direction mismatch between successive elements\n"
            "(0° = perfect C1 continuity)."
        )
        jump_row.addWidget(jump_lbl)
        self._jump_lbl = QLabel("—")
        self._jump_lbl.setStyleSheet("color: #aaa;")
        jump_row.addWidget(self._jump_lbl)
        jump_row.addStretch()
        gform.addLayout(jump_row)

        layout.addLayout(gform)

        # ── Select button ─────────────────────────────────────────────
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

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def set_pending(self):
        """Reset card to 'waiting to start' state."""
        self._dev_lbl.setText("—")
        self._dev_lbl.setStyleSheet("color: #aaa;")
        self._rmse_lbl.setText("—")
        self._n_lbl.setText("—")
        self._jump_lbl.setText("—")
        self._jump_lbl.setStyleSheet("color: #aaa;")
        self._btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)
        self._status_lbl.setText("")

    def set_running(self, message: str = "Starting…"):
        """Mark card as actively computing."""
        self._dev_lbl.setText("computing…")
        self._dev_lbl.setStyleSheet("color: #888;")
        self._progress_bar.setVisible(True)
        self._status_lbl.setText(message)
        self._status_lbl.setVisible(True)

    def set_progress(self, message: str):
        """Update the live status label while running."""
        self._status_lbl.setText(message)
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
        if not self._status_lbl.isVisible():
            self._status_lbl.setVisible(True)

    def set_preview(self, candidate):
        """Show preliminary metrics while the algorithm is still running."""
        if candidate.n_elements > 0:
            self._n_lbl.setText(f"{candidate.n_elements}  (preliminary)")
            self._n_lbl.setStyleSheet("color: #888;")
        # max_deviation is 0 in preview CandidateAlignment objects (not computed yet)
        # so only show it if it's non-zero
        if candidate.max_deviation > 0:
            self._dev_lbl.setText(f"{candidate.max_deviation:.2f} m  ⟳")
            self._dev_lbl.setStyleSheet("color: #888;")

    def set_result(self, candidate):
        """Update card with completed CandidateAlignment metrics."""
        self._dev_lbl.setText(f"{candidate.max_deviation:.2f} m")
        self._dev_lbl.setStyleSheet("color: #aaa;")
        self._rmse_lbl.setText(f"{candidate.rmse:.2f} m")
        self._rmse_lbl.setStyleSheet("color: #aaa;")
        self._n_lbl.setText(str(candidate.n_elements))
        self._n_lbl.setStyleSheet("color: #aaa;")
        # Heading-jump (C1 sanity): green if < 0.1°, amber if < 1°, red otherwise
        jump = float(getattr(candidate, "max_heading_jump_deg", 0.0) or 0.0)
        self._jump_lbl.setText(f"{jump:.3f}°")
        if jump < 0.1:
            self._jump_lbl.setStyleSheet("color: #66bb6a;")
        elif jump < 1.0:
            self._jump_lbl.setStyleSheet("color: #ffca28;")
        else:
            self._jump_lbl.setStyleSheet("color: #ef5350;")
        self._btn.setEnabled(len(candidate.elements) > 0)
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)

    def set_error(self):
        """Mark card as failed."""
        self._dev_lbl.setText("error")
        self._dev_lbl.setStyleSheet("color: #e57373;")
        self._rmse_lbl.setText("—")
        self._n_lbl.setText("—")
        self._jump_lbl.setText("—")
        self._jump_lbl.setStyleSheet("color: #aaa;")
        self._btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)


# ---------------------------------------------------------------------------
# Step4Candidates
# ---------------------------------------------------------------------------

class Step4Candidates(QWidget):
    candidate_selected   = Signal(object)   # CandidateAlignment
    candidate_map_update = Signal(list)     # list[CandidateAlignment] for map overlay
    back_requested       = Signal()         # user clicked ← Back to Configure

    ALGO_DEFS = [
        ("segment_fit",         "Segment & Fit",           "#ff9800"),
        ("segment_fit_spirals", "Segment & Fit (Spirals)", "#26c6da"),
        ("dp_segment",          "DP Segmentation",         "#66bb6a"),
        ("progressive_mc",      "Progressive MC",          "#42a5f5"),
        ("raw",                 "OSM Polyline",            "#e040fb"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidates: dict = {}   # algo_id → final CandidateAlignment
        self._previews:   dict = {}   # algo_id → preliminary CandidateAlignment
        self._worker = None
        self._cards: list[CandidateCard] = []
        self._n_done = 0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Candidate Alignments")
        title.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # Overall progress bar (determinate: 0 → number of algorithms)
        self._progress = QProgressBar()
        self._progress.setRange(0, len(self.ALGO_DEFS))
        self._progress.setValue(0)
        self._progress.setFixedHeight(10)
        self._progress.setTextVisible(False)
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
        self._previews.clear()
        self._n_done = 0

        # Reset all cards
        for card in self._cards:
            card.set_pending()

        if not xy_list:
            self._status_lbl.setText("No projected coordinates available.")
            self._progress.setVisible(False)
            return

        n_algos = len(self.ALGO_DEFS)
        self._status_lbl.setText(f"Running {n_algos} algorithms…")
        self._progress.setRange(0, n_algos)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        # Stop any existing worker
        if self._worker is not None:
            try:
                self._worker.candidate_ready.disconnect()
                self._worker.candidate_preview.disconnect()
                self._worker.progress_update.disconnect()
                self._worker.all_done.disconnect()
                self._worker.failed.disconnect()
            except Exception:
                pass
            self._worker = None

        settings_copy = dict(settings)
        settings_copy["_work_epsg"] = work_epsg

        self._worker = CandidateWorker(xy_list, chainages_list, settings_copy, self)
        self._worker.candidate_ready.connect(self._on_candidate_ready)
        self._worker.candidate_preview.connect(self._on_candidate_preview)
        self._worker.progress_update.connect(self._on_progress_update)
        self._worker.candidate_error.connect(self._on_candidate_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

        # Mark first card as running immediately (algorithms run sequentially)
        if self._cards:
            self._cards[0].set_running()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress_update(self, algo_id: str, message: str):
        """Route live status messages to the matching card."""
        for card in self._cards:
            if card._algo_id == algo_id:
                card.set_progress(message)

    def _on_candidate_preview(self, algo_id: str, preview):
        """Intermediate result from MC — update map but don't mark card as done."""
        self._previews[algo_id] = preview
        for card in self._cards:
            if card._algo_id == algo_id:
                card.set_preview(preview)
        self._emit_map_update()

    def _on_candidate_ready(self, algo_id: str, candidate):
        """Final result — update card metrics and map overlay."""
        self._candidates[algo_id] = candidate
        self._previews.pop(algo_id, None)   # final result supersedes any preview
        self._n_done += 1
        self._progress.setValue(self._n_done)

        for card in self._cards:
            if card._algo_id == algo_id:
                if candidate.elements:
                    card.set_result(candidate)
                else:
                    card.set_error()

        self._emit_map_update()

        # Mark the next card as running if algorithms run sequentially
        algo_ids = [a for a, _, _ in self.ALGO_DEFS]
        if algo_id in algo_ids:
            next_idx = algo_ids.index(algo_id) + 1
            if next_idx < len(self._cards) and next_idx >= self._n_done:
                self._cards[next_idx].set_running()

        remaining = len(self.ALGO_DEFS) - self._n_done
        if remaining > 0:
            self._status_lbl.setText(
                f"{self._n_done}/{len(self.ALGO_DEFS)} done — "
                f"{remaining} algorithm{'s' if remaining > 1 else ''} still running…"
            )

    def _on_all_done(self):
        self._progress.setVisible(False)
        n = sum(1 for c in self._candidates.values() if c.elements)
        self._status_lbl.setText(
            f"{n}/{len(self.ALGO_DEFS)} algorithms succeeded. "
            f"Select one to continue to Step 5."
        )

    def _on_candidate_error(self, algo_id: str, error: str):
        """One algorithm failed — mark its card, keep others running."""
        self._n_done += 1
        self._progress.setValue(self._n_done)
        for card in self._cards:
            if card._algo_id == algo_id:
                card.set_error()
        short = error.split("\n")[0][:120]
        self._status_lbl.setText(f"⚠ {algo_id}: {short}")

    def _on_failed(self, error: str):
        self._progress.setVisible(False)
        self._status_lbl.setText(f"Error: {error}")

    def _on_select(self, algo_id: str):
        c = self._candidates.get(algo_id)
        if c and c.elements:
            self.candidate_selected.emit(c)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_map_update(self):
        """Combine final results with any in-flight previews and update the map."""
        # Final results take precedence over previews for the same algo_id
        combined: dict = {**self._previews, **self._candidates}
        self.candidate_map_update.emit(list(combined.values()))
