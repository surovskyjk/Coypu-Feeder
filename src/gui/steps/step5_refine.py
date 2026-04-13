"""
Step 5 — Refine Alignment.
User sees the selected candidate and can optionally insert Euler spirals
at Line–Arc transitions before proceeding to export.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QDoubleSpinBox, QGroupBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class _TransitionRow(QWidget):
    """One row in the transition list: label + spin box + Insert/Remove button."""

    insert_clicked = Signal(int, float)   # transition_idx, L_spiral
    remove_clicked = Signal(int)          # transition_idx

    def __init__(self, idx: int, transition, parent=None):
        super().__init__(parent)
        self._idx        = idx
        self._transition = transition
        self._has_spiral = transition.has_spiral
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # Label
        lbl = QLabel(self._transition.label)
        lbl.setMinimumWidth(110)
        lbl.setStyleSheet("font-size: 10px;")
        row.addWidget(lbl)

        # Spiral length spin box
        self._spin = QDoubleSpinBox()
        self._spin.setRange(10.0, 300.0)
        self._spin.setSingleStep(5.0)
        self._spin.setSuffix(" m")
        from geometry.spiral_insertion import auto_suggest_L
        suggested = auto_suggest_L(
            self._transition.R,
            self._transition.available_L,
        )
        self._spin.setValue(suggested)
        self._spin.setFixedWidth(80)
        row.addWidget(self._spin)

        # Insert / Remove button
        self._btn = QPushButton()
        self._btn.setFixedWidth(70)
        self._btn.clicked.connect(self._on_btn_clicked)
        row.addWidget(self._btn)

        self._update_state()

    def _update_state(self):
        if self._has_spiral:
            self._btn.setText("Remove")
            self._btn.setStyleSheet(
                "QPushButton { background: #c0392b; color: #fff; border-radius: 3px; "
                "font-size: 10px; padding: 3px 6px; }"
                "QPushButton:hover { background: #e74c3c; }"
            )
            self._spin.setEnabled(False)
        else:
            self._btn.setText("Insert")
            self._btn.setStyleSheet(
                "QPushButton { background: #27ae60; color: #fff; border-radius: 3px; "
                "font-size: 10px; padding: 3px 6px; }"
                "QPushButton:hover { background: #2ecc71; }"
                "QPushButton:disabled { background: #444; color: #777; }"
            )
            avail = self._transition.available_L
            self._spin.setEnabled(avail > 11.0)
            self._btn.setEnabled(avail > 11.0)

    def set_has_spiral(self, has_spiral: bool):
        self._has_spiral = has_spiral
        self._update_state()

    def _on_btn_clicked(self):
        if self._has_spiral:
            self.remove_clicked.emit(self._idx)
        else:
            self.insert_clicked.emit(self._idx, self._spin.value())


class Step5Refine(QWidget):
    refinement_done            = Signal(list)   # final list[dict] elements
    back_requested             = Signal()       # user wants to go back to candidates
    alignment_update_requested = Signal(list)   # live map refresh during refinement

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidate = None
        self._xy = None
        self._chainages = None
        self._settings: dict = {}
        self._working_elements: list[dict] = []
        self._transitions: list = []
        self._history: list[list[dict]] = []   # undo stack
        self._rows: list[_TransitionRow] = []
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # Title
        self._title_lbl = QLabel("Refine Alignment")
        self._title_lbl.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        v.addWidget(self._title_lbl)

        self._algo_lbl = QLabel("")
        self._algo_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        v.addWidget(self._algo_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        v.addWidget(sep)

        # Spiral insertion group
        self._spiral_group = QGroupBox("Spiral Insertions")
        self._spiral_group.setStyleSheet("font-size: 10px;")
        self._spiral_layout = QVBoxLayout(self._spiral_group)
        self._spiral_layout.setContentsMargins(6, 6, 6, 6)
        self._spiral_layout.setSpacing(4)

        self._no_trans_lbl = QLabel("No Line–Arc transitions found.")
        self._no_trans_lbl.setStyleSheet("color: #666; font-size: 10px;")
        self._spiral_layout.addWidget(self._no_trans_lbl)

        v.addWidget(self._spiral_group)

        # Metrics
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(
            "QFrame { background: #2a2a2e; border-radius: 4px; }"
        )
        mf = QVBoxLayout(metrics_frame)
        mf.setContentsMargins(8, 6, 8, 6)
        self._metrics_lbl = QLabel("No candidate selected.")
        self._metrics_lbl.setStyleSheet("color: #888; font-size: 10px;")
        mf.addWidget(self._metrics_lbl)
        v.addWidget(metrics_frame)

        v.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Navigation row (outside scroll)
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(8, 4, 8, 8)
        nav_row.setSpacing(6)

        self._back_btn = QPushButton("← Back")
        self._back_btn.setFixedWidth(70)
        self._back_btn.clicked.connect(self.back_requested.emit)
        nav_row.addWidget(self._back_btn)

        self._undo_btn = QPushButton("↩ Undo")
        self._undo_btn.setFixedWidth(70)
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_undo)
        nav_row.addWidget(self._undo_btn)

        self._accept_btn = QPushButton("Accept & Export →")
        self._accept_btn.setMinimumHeight(36)
        self._accept_btn.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        self._accept_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: #fff; border-radius: 4px; "
            "font-weight: bold; padding: 6px 12px; }"
            "QPushButton:hover { background: #2ecc71; }"
            "QPushButton:disabled { background: #444; color: #777; }"
        )
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._on_accept)
        nav_row.addWidget(self._accept_btn)

        outer.addLayout(nav_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, candidate, xy, chainages, settings: dict):
        """
        Called by App when user selects a candidate from Step 4.

        Parameters
        ----------
        candidate : CandidateAlignment dataclass (or dict with 'elements' key)
        xy        : np.ndarray projected coordinates (N,2) — may be None
        chainages : np.ndarray chainages — may be None
        settings  : dict from Step3Configure
        """
        self._candidate = candidate
        self._xy        = xy
        self._chainages = chainages
        self._settings  = settings
        self._history.clear()

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
        self._accept_btn.setEnabled(len(self._working_elements) > 0)
        self._undo_btn.setEnabled(False)

        self._refresh_metrics(n_el, max_d, rmse)
        self._refresh_transitions()

    # ------------------------------------------------------------------
    # Transition list helpers
    # ------------------------------------------------------------------

    def _refresh_transitions(self):
        """Rebuild the transition rows from the current working elements."""
        from geometry.spiral_insertion import find_transitions

        # Clear old rows
        for row in self._rows:
            row.deleteLater()
        self._rows.clear()

        self._transitions = find_transitions(self._working_elements)

        if not self._transitions:
            self._no_trans_lbl.setVisible(True)
            return

        self._no_trans_lbl.setVisible(False)

        for idx, trans in enumerate(self._transitions):
            row = _TransitionRow(idx, trans, self)
            row.insert_clicked.connect(self._on_insert)
            row.remove_clicked.connect(self._on_remove)
            self._spiral_layout.addWidget(row)
            self._rows.append(row)

    # ------------------------------------------------------------------
    # Spiral insert / remove / undo
    # ------------------------------------------------------------------

    def _on_insert(self, transition_idx: int, L_spiral: float):
        from geometry.spiral_insertion import insert_spiral
        trans = self._transitions[transition_idx]
        try:
            new_els = insert_spiral(
                self._working_elements,
                trans.arc_idx,
                trans.side,
                L_spiral,
            )
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot insert spiral", str(exc))
            return

        self._history.append(list(self._working_elements))
        self._working_elements = new_els
        self._undo_btn.setEnabled(True)
        self._refresh_metrics_from_elements()
        self._refresh_transitions()
        self._emit_alignment()

    def _on_remove(self, transition_idx: int):
        from geometry.spiral_insertion import remove_spiral
        trans = self._transitions[transition_idx]
        try:
            new_els = remove_spiral(
                self._working_elements,
                trans.arc_idx,
                trans.side,
            )
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot remove spiral", str(exc))
            return

        self._history.append(list(self._working_elements))
        self._working_elements = new_els
        self._undo_btn.setEnabled(True)
        self._refresh_metrics_from_elements()
        self._refresh_transitions()
        self._emit_alignment()

    def _on_undo(self):
        if not self._history:
            return
        self._working_elements = self._history.pop()
        self._undo_btn.setEnabled(bool(self._history))
        self._refresh_metrics_from_elements()
        self._refresh_transitions()
        self._emit_alignment()

    def _refresh_metrics_from_elements(self):
        """Recompute metrics from current working elements."""
        n_el = len(self._working_elements)
        max_d = 0.0
        rmse  = 0.0
        if self._xy is not None and self._chainages is not None and n_el > 0:
            try:
                from geometry.candidates import evaluate_candidate
                m = evaluate_candidate(
                    self._working_elements,
                    self._xy,
                    self._chainages,
                )
                max_d = m.get("max_deviation", 0.0)
                rmse  = m.get("rmse", 0.0)
            except Exception:
                pass
        self._refresh_metrics(n_el, max_d, rmse)
        self._accept_btn.setEnabled(n_el > 0)

    def _refresh_metrics(self, n_el: int, max_d: float, rmse: float):
        self._metrics_lbl.setText(
            f"Elements: {n_el}    "
            f"Max deviation: {max_d:.2f} m    "
            f"RMSE: {rmse:.2f} m"
        )

    def _emit_alignment(self):
        """Ask App to refresh the map overlay with the current working elements."""
        self.alignment_update_requested.emit(list(self._working_elements))

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self):
        self.refinement_done.emit(list(self._working_elements))
