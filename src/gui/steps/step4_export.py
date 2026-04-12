"""
Step 4 — Export.
File chooser, progress bar with named stages, start button.
After success: shows alignment on map and offers 'Export Another Railway'.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.worker import ExportWorker

STAGE_PCT = {
    "Projecting":  10,
    "Fitting":     30,
    "Querying":    55,
    "Building":    82,
    "Writing":     92,
}


class Step4Export(QWidget):
    # Emitted after successful export
    export_finished            = Signal(str, int)  # filepath, work_epsg
    osm_track_ready            = Signal(list)      # raw OSM [[lat,lon],...] per track
    alignment_ready            = Signal(list)      # reconstructed geometric points
    fit_to_alignment_requested = Signal()
    start_over_requested       = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks  = []
        self._settings: dict = {}
        self._worker: ExportWorker | None = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        lbl = QLabel("Export LandXML")
        lbl.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # ── File path ──────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose output file…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # ── Progress ────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._stage_lbl = QLabel("Ready.")
        self._stage_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(self._stage_lbl)

        self._station_lbl = QLabel("")
        self._station_lbl.setStyleSheet("color:#888; font-size:10px;")
        layout.addWidget(self._station_lbl)

        layout.addStretch()

        # ── Post-export actions (hidden until done) ────────────────────
        self._post_frame = QFrame()
        self._post_frame.setVisible(False)
        pv = QVBoxLayout(self._post_frame)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;")
        pv.addWidget(sep)

        ok_lbl = QLabel(
            "✅  Both overlays drawn on map:\n"
            "  🔴 Red solid  — fitted LandXML alignment\n"
            "  🔵 Cyan dashed — original OSM polyline"
        )
        ok_lbl.setStyleSheet("color:#8bc34a; font-size:10px;")
        ok_lbl.setWordWrap(True)
        pv.addWidget(ok_lbl)

        post_btn_row = QHBoxLayout()
        self._fit_aln_btn = QPushButton("📍  Zoom to exported line")
        self._fit_aln_btn.clicked.connect(self.fit_to_alignment_requested.emit)

        self._restart_btn = QPushButton("🔄  Export Another Railway")
        self._restart_btn.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        self._restart_btn.clicked.connect(self.start_over_requested.emit)

        post_btn_row.addWidget(self._fit_aln_btn)
        post_btn_row.addWidget(self._restart_btn)
        pv.addLayout(post_btn_row)

        layout.addWidget(self._post_frame)

        # ── Start button ──────────────────────────────────────────────
        self._start_btn = QPushButton("▶  Start Export")
        self._start_btn.setMinimumHeight(44)
        self._start_btn.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        self._start_btn.setStyleSheet(
            "QPushButton { background:#27ae60; color:#fff; border-radius:6px; }"
            "QPushButton:hover { background:#2ecc71; }"
            "QPushButton:disabled { background:#444; color:#777; }"
        )
        self._start_btn.clicked.connect(self._start_export)
        layout.addWidget(self._start_btn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def prepare(self, tracks, settings: dict):
        self._tracks   = tracks
        self._settings = settings
        self._progress.setValue(0)
        self._stage_lbl.setText("Ready.")
        self._station_lbl.setText("")
        self._start_btn.setEnabled(True)
        self._post_frame.setVisible(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save LandXML file", "",
            "LandXML files (*.xml);;All files (*.*)",
        )
        if path:
            self._path_edit.setText(path)

    def _start_export(self):
        filepath = self._path_edit.text().strip()
        if not filepath:
            QMessageBox.warning(self, "No file",
                                "Please choose an output file first.")
            return
        if not self._tracks:
            QMessageBox.warning(self, "No data", "No tracks to export.")
            return

        self._start_btn.setEnabled(False)
        self._post_frame.setVisible(False)
        self._progress.setValue(0)
        self._stage_lbl.setText("Starting…")
        self._station_lbl.setText("")

        self._worker = ExportWorker(self._tracks, self._settings, filepath, self)
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.station_progress.connect(self._on_station_progress)
        self._worker.osm_track_ready.connect(self.osm_track_ready.emit)   # forward
        self._worker.alignment_ready.connect(self.alignment_ready.emit)   # forward
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_stage(self, stage: str):
        self._stage_lbl.setText(stage)
        pct = self._progress.value()
        for key, val in STAGE_PCT.items():
            if stage.startswith(key):
                pct = val
                break
        self._progress.setValue(pct)

    def _on_station_progress(self, current: float, total: float):
        if total > 0:
            self._station_lbl.setText(
                f"Chainage: {current:.0f} / {total:.0f} m"
            )

    def _on_finished(self, filepath: str, work_epsg: int):
        self._progress.setValue(100)
        self._stage_lbl.setText("Export complete!")
        self._start_btn.setEnabled(True)
        self._post_frame.setVisible(True)
        self.export_finished.emit(filepath, work_epsg)

    def _on_failed(self, error: str):
        self._stage_lbl.setText("Export failed.")
        self._start_btn.setEnabled(True)
        QMessageBox.critical(self, "Export failed", error)
