"""
Step 2 — Select Section.
Shows the fetched tracks; user picks which ones to export.

Highlight: clicking a row in the list auto-highlights it on the map.
Use the "👁 Highlight" button to highlight after a multi-selection change.
Use "📍 Fit to all tracks" to re-centre the map.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class Step2Section(QWidget):
    section_confirmed     = Signal(list)   # selected Track objects
    highlight_changed     = Signal(int)    # track index  (-1 = reset all)
    fit_to_tracks_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: list = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        lbl = QLabel("Select tracks to export:")
        lbl.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        layout.addWidget(lbl)

        hint = QLabel(
            "Click a track to highlight it on the map.\n"
            "Hold Ctrl/Shift to select multiple tracks."
        )
        hint.setStyleSheet("color:#888; font-size:10px;")
        layout.addWidget(hint)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Single-click on a row → highlight that track on map
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, stretch=1)

        # ── Map / selection buttons ────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.clicked.connect(self._list.selectAll)

        self._clear_sel_btn = QPushButton("Clear")
        self._clear_sel_btn.clicked.connect(self._list.clearSelection)

        self._highlight_btn = QPushButton("👁  Highlight")
        self._highlight_btn.setToolTip("Highlight the currently focused track on the map")
        self._highlight_btn.clicked.connect(self._on_highlight_btn)

        self._reset_hl_btn = QPushButton("Reset highlight")
        self._reset_hl_btn.setToolTip("Reset all track colours")
        self._reset_hl_btn.clicked.connect(lambda: self.highlight_changed.emit(-1))

        self._fit_btn = QPushButton("📍  Fit map")
        self._fit_btn.setToolTip("Zoom and pan the map to show all loaded tracks")
        self._fit_btn.clicked.connect(self.fit_to_tracks_requested.emit)

        for btn in (self._select_all_btn, self._clear_sel_btn,
                    self._highlight_btn, self._reset_hl_btn, self._fit_btn):
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

        # ── Next button ──────────────────────────────────────────────
        self._next_btn = QPushButton("Next →  Configure")
        self._next_btn.setMinimumHeight(38)
        self._next_btn.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        self._next_btn.setStyleSheet(
            "QPushButton { background:#2a82da; color:#fff; border-radius:5px; }"
            "QPushButton:hover { background:#3a92ea; }"
            "QPushButton:disabled { background:#444; color:#777; }"
        )
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._next_btn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def populate(self, tracks):
        self._tracks = tracks
        self._list.clear()
        for i, t in enumerate(tracks):
            node_count = len(t.nodes) if hasattr(t, "nodes") else 0
            item = QListWidgetItem(f"{t.name}  ({node_count} nodes)")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)
        self._list.selectAll()

    def get_selected_tracks(self) -> list:
        selected = self._list.selectedItems()
        indices  = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        return [self._tracks[i] for i in indices if i < len(self._tracks)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_row_changed(self, row: int):
        """Single-click on a row → highlight that track."""
        self.highlight_changed.emit(row)

    def _on_highlight_btn(self):
        """Explicit highlight button — highlight the current (focused) item."""
        row = self._list.currentRow()
        self.highlight_changed.emit(row)

    def _on_next(self):
        selected = self.get_selected_tracks()
        if not selected:
            QMessageBox.warning(self, "No tracks",
                                "Please select at least one track.")
            return
        self.section_confirmed.emit(selected)
