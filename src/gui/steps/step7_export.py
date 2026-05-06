"""
Step 7 — Export.
CRS selection, file chooser, progress bar with named stages, start button.
After success: shows alignment on map and offers 'Export Another Railway'.

The internal geometry is always fitted in auto-UTM.  Here the user picks
the output CRS; FinalExportWorker reprojects element coordinates before
writing LandXML.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox, QFrame, QComboBox, QCheckBox,
    QFormLayout, QGroupBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from geometry.projection import CRS_PRESETS
from gui.worker import FinalExportWorker

STAGE_PCT = {
    "Reconstruct": 10,
    "Querying":    40,
    "Building":    80,
    "Writing":     92,
}


class Step7Export(QWidget):
    # Emitted after successful export
    export_finished            = Signal(str, int)  # filepath, output_epsg
    osm_track_ready            = Signal(list)      # raw OSM [[lat,lon],...] per track
    alignment_ready            = Signal(list)      # reconstructed geometric points
    alignment_segments_ready   = Signal(list)      # per-element WGS84 segments
    fit_to_alignment_requested = Signal()
    start_over_requested       = Signal()
    back_requested             = Signal()          # user wants to go back to Refine

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks    = []
        self._settings: dict = {}
        self._elements_list: list[list[dict]] = []
        self._work_epsg: int = 32633   # internal auto-UTM
        self._xy_list: list  = []
        self._worker: FinalExportWorker | None = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        lbl = QLabel("Export LandXML")
        lbl.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # ── Output CRS ────────────────────────────────────────────────
        crs_group = QGroupBox("Output Coordinate System")
        cf = QFormLayout(crs_group)

        self._crs_combo = QComboBox()
        for label, _ in CRS_PRESETS:
            self._crs_combo.addItem(label)
        # Default to UTM 33N — a reasonable European starting point
        # (the internal work CRS is auto-UTM so this is a sensible match)
        default_idx = next(
            (i for i, (_, epsg) in enumerate(CRS_PRESETS) if epsg == 32633), 0
        )
        self._crs_combo.setCurrentIndex(default_idx)
        cf.addRow("Preset:", self._crs_combo)

        self._epsg_edit = QLineEdit()
        self._epsg_edit.setPlaceholderText("e.g. 5514  (overrides preset)")
        cf.addRow("Custom EPSG:", self._epsg_edit)

        self._force_pos_chk = QCheckBox("Force all coordinates positive")
        self._force_pos_chk.setToolTip(
            "Applies abs() to output coordinates — strips the minus sign\n"
            "without changing the numeric value.\n"
            "Use for S-JTSK positive convention (EPSG:5514)."
        )
        cf.addRow("", self._force_pos_chk)

        layout.addWidget(crs_group)

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

        # ── Back button ───────────────────────────────────────────────
        self._back_btn = QPushButton("← Back to Refine")
        self._back_btn.clicked.connect(self.back_requested.emit)
        layout.addWidget(self._back_btn)

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

    def prepare(self, elements_list: list, tracks, settings: dict,
                work_epsg: int, xy_list: list):
        """
        Called by App after Step 5 refinement_done.

        Parameters
        ----------
        elements_list : list of element lists — one per track
        tracks        : list of Track objects (for OSM reference + elevation)
        settings      : dict from Step3Configure (no epsg/force_positive here)
        work_epsg     : internal auto-UTM EPSG used during candidate fitting
        xy_list       : projected coordinates per track in work_epsg (for display)
        """
        self._elements_list = elements_list
        self._tracks        = tracks
        self._settings      = settings
        self._work_epsg     = work_epsg
        self._xy_list       = xy_list
        self._progress.setValue(0)
        self._stage_lbl.setText("Ready.")
        self._station_lbl.setText("")
        self._start_btn.setEnabled(True)
        self._back_btn.setEnabled(True)
        self._post_frame.setVisible(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_output_epsg(self) -> int | None:
        """Return the chosen output EPSG, or None if invalid."""
        custom = self._epsg_edit.text().strip()
        if custom:
            if not custom.isdigit():
                QMessageBox.warning(self, "Invalid EPSG",
                                    "Custom EPSG must be a number.")
                return None
            return int(custom)
        idx = self._crs_combo.currentIndex()
        epsg = CRS_PRESETS[idx][1]
        if epsg == -1:
            # "Auto UTM" preset — use the same as the internal work CRS
            return self._work_epsg
        return epsg

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

        output_epsg = self._resolve_output_epsg()
        if output_epsg is None:
            return

        force_positive = self._force_pos_chk.isChecked()

        self._start_btn.setEnabled(False)
        self._back_btn.setEnabled(False)
        self._post_frame.setVisible(False)
        self._progress.setValue(0)
        self._stage_lbl.setText("Starting…")
        self._station_lbl.setText("")

        self._worker = FinalExportWorker(
            self._elements_list, self._tracks, self._settings,
            filepath,
            work_epsg=self._work_epsg,
            output_epsg=output_epsg,
            force_positive=force_positive,
            xy_list=self._xy_list,
            parent=self,
        )
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.station_progress.connect(self._on_station_progress)
        self._worker.osm_track_ready.connect(self.osm_track_ready.emit)
        self._worker.alignment_ready.connect(self.alignment_ready.emit)
        self._worker.alignment_segments_ready.connect(self.alignment_segments_ready.emit)
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

    def _on_finished(self, filepath: str, output_epsg: int):
        self._progress.setValue(100)
        self._stage_lbl.setText("Export complete!")
        self._start_btn.setEnabled(True)
        self._back_btn.setEnabled(True)
        self._post_frame.setVisible(True)
        self.export_finished.emit(filepath, output_epsg)

    def _on_failed(self, error: str):
        self._stage_lbl.setText("Export failed.")
        self._start_btn.setEnabled(True)
        self._back_btn.setEnabled(True)
        QMessageBox.critical(self, "Export failed", error)
