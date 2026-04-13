"""
Main application window — PySide6.
3-column layout: StepSidebar | MapWidget | QStackedWidget (step panels).
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget,
    QSizePolicy, QMessageBox, QApplication,
)

from .map_widget import MapWidget
from .step_sidebar import StepSidebar
from .steps.step1_find import Step1Find
from .steps.step2_section import Step2Section
from .steps.step3_configure import Step3Configure
from .steps.step4_candidates import Step4Candidates
from .steps.step5_refine import Step5Refine
from .steps.step6_export import Step6Export

# Maximum map-view span (km) allowed for "search in view"
_MAX_VIEW_KM = 20.0


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coypu-Feeder — OSM Railway to LandXML")
        self.resize(1340, 820)
        self.setMinimumSize(1050, 680)

        self._tracks: list          = []
        self._selected_tracks: list = []
        self._settings: dict        = {}
        self._bbox_workers: list    = []
        self._selected_candidate    = None
        self._final_elements: list  = []
        self._xy_list: list         = []
        self._chainages_list: list  = []
        self._work_epsg: int        = 32633

        self._build_layout()
        self._wire_signals()
        self._connect_scheme_changes()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.sidebar = StepSidebar()
        self.sidebar.setFixedWidth(200)
        h.addWidget(self.sidebar)

        self.map_widget = MapWidget()
        self.map_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        h.addWidget(self.map_widget, stretch=1)

        self.stack = QStackedWidget()
        self.stack.setFixedWidth(340)

        self.step1              = Step1Find()
        self.step2              = Step2Section()
        self.step3              = Step3Configure()
        self.step4_candidates   = Step4Candidates()
        self.step5_refine       = Step5Refine()
        self.step6_export       = Step6Export()

        self.stack.addWidget(self.step1)            # 0
        self.stack.addWidget(self.step2)            # 1
        self.stack.addWidget(self.step3)            # 2
        self.stack.addWidget(self.step4_candidates) # 3
        self.stack.addWidget(self.step5_refine)     # 4
        self.stack.addWidget(self.step6_export)     # 5

        h.addWidget(self.stack)
        self.statusBar().showMessage(
            "Ready. Search for a railway or use 'Lines in View' after zooming in."
        )

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self):
        # Sidebar back-navigation
        self.sidebar.step_clicked.connect(self._goto_step)

        # Map JS errors → status bar
        self.map_widget.js_error.connect(
            lambda msg: self.statusBar().showMessage(f"⚠ {msg}")
        )

        # Map bounds → "Lines in View" search
        self.step1.search_in_view_requested.connect(self._on_search_in_view)
        self.map_widget.bounds_ready.connect(self._on_map_bounds_ready)

        # Step 1 → fetch railway
        self.step1.railway_fetched.connect(self._on_railway_fetched)

        # Step 2 → highlight / fit / confirm
        self.step2.highlight_changed.connect(self._on_highlight_changed)
        self.step2.fit_to_tracks_requested.connect(self._on_fit_to_tracks)
        self.step2.section_confirmed.connect(self._on_section_confirmed)

        # Step 3 → config confirmed
        self.step3.config_confirmed.connect(self._on_config_confirmed)

        # Step 4 (candidates) → map update + selection
        self.step4_candidates.candidate_map_update.connect(self._on_candidate_map_update)
        self.step4_candidates.candidate_selected.connect(self._on_candidate_selected)

        # Step 5 (refine) → done / back
        self.step5_refine.refinement_done.connect(self._on_refinement_done)
        self.step5_refine.back_requested.connect(lambda: self._goto_step(3))

        # Step 6 (export) → alignment display / fit / export / restart
        self.step6_export.osm_track_ready.connect(self._on_osm_track_ready)
        self.step6_export.alignment_ready.connect(self._on_alignment_ready)
        self.step6_export.fit_to_alignment_requested.connect(self._on_fit_to_alignment)
        self.step6_export.export_finished.connect(self._on_export_finished)
        self.step6_export.start_over_requested.connect(self._start_over)

    # ------------------------------------------------------------------
    # System colour-scheme changes (dark ↔ light)
    # ------------------------------------------------------------------

    def _connect_scheme_changes(self):
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(
                self._on_color_scheme_changed
            )
        except Exception:
            pass  # Qt < 6.5 — no signal, static theme is fine

    def _on_color_scheme_changed(self, scheme):
        from gui.theme import apply_theme
        dark = (scheme == Qt.ColorScheme.Dark)
        apply_theme(QApplication.instance(), dark)
        self.map_widget.set_theme(dark)

    # ------------------------------------------------------------------
    # Step transitions
    # ------------------------------------------------------------------

    def _goto_step(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.sidebar.set_step(idx)

    # ------------------------------------------------------------------
    # Step 2 map interactions
    # ------------------------------------------------------------------

    def _on_highlight_changed(self, idx: int):
        self.map_widget.highlight_track(idx)
        if idx < 0:
            self.statusBar().showMessage("Track highlight reset — all tracks shown.")
        elif idx < len(self._tracks):
            self.statusBar().showMessage(
                f"Highlighted track {idx + 1}: {self._tracks[idx].name}"
            )

    def _on_fit_to_tracks(self):
        if self._tracks:
            self.map_widget.fly_to_tracks()
            self.statusBar().showMessage(
                f"Zooming map to {len(self._tracks)} track(s)."
            )
        else:
            self.statusBar().showMessage("No tracks loaded yet.")

    # ------------------------------------------------------------------
    # Step 6 map interactions
    # ------------------------------------------------------------------

    def _on_osm_track_ready(self, alignments: list):
        """Show the raw OSM polyline as a dashed cyan reference while fitting runs."""
        if alignments and any(len(a) > 0 for a in alignments):
            self.map_widget.show_osm_reference(alignments)
            self.statusBar().showMessage(
                "OSM reference polyline drawn (dashed cyan). Fitting geometry…"
            )

    def _on_alignment_ready(self, alignments: list):
        if not alignments or not any(len(a) > 0 for a in alignments):
            self.statusBar().showMessage(
                "⚠ Export finished but alignment contains no points — "
                "nothing drawn on map."
            )
            return
        self.map_widget.show_alignment(alignments)
        total_pts = sum(len(a) for a in alignments)
        self.statusBar().showMessage(
            f"Both overlays ready — 🔴 red: fitted LandXML ({total_pts} pts)  "
            f"🔵 cyan dashed: OSM reference  ({len(alignments)} track(s))."
        )

    def _on_fit_to_alignment(self):
        self.map_widget.fly_to_alignment()
        self.statusBar().showMessage("Zooming map to exported alignment.")

    # ------------------------------------------------------------------
    # "Lines in View" search
    # ------------------------------------------------------------------

    def _on_search_in_view(self):
        """Step 1 requested a search — ask the map for its current bounds."""
        self.step1.set_view_search_busy(True)
        self.map_widget.request_bounds()

    def _on_map_bounds_ready(self, s: float, w: float, n: float, e: float):
        """Map returned its bounds; validate area then run Overpass query."""
        center_lat = (s + n) / 2.0
        lat_km = (n - s) * 111.0
        lon_km = (e - w) * 111.0 * math.cos(math.radians(center_lat))

        if lat_km > _MAX_VIEW_KM or lon_km > _MAX_VIEW_KM:
            self.step1.set_view_search_busy(False)
            self.step1.show_view_results(
                [],
                status=(
                    f"⚠ View too large ({lat_km:.0f} × {lon_km:.0f} km). "
                    f"Zoom in to ≤ {_MAX_VIEW_KM:.0f} km and try again."
                ),
            )
            self.statusBar().showMessage(
                "View too large for 'Lines in View' search — zoom in more."
            )
            return

        self.statusBar().showMessage(
            f"Searching railway lines in {lat_km:.1f} × {lon_km:.1f} km view…"
        )

        from gui.worker import SearchWorker
        worker = SearchWorker("bbox", (s, w, n, e), self)
        worker.results_ready.connect(self._on_view_results_ready)
        worker.failed.connect(self._on_view_search_failed)
        worker.finished.connect(lambda: self.step1.set_view_search_busy(False))
        worker.finished.connect(
            lambda: self._bbox_workers.remove(worker)
            if worker in self._bbox_workers else None
        )
        self._bbox_workers.append(worker)
        worker.start()

    def _on_view_results_ready(self, results: list):
        n = len(results)
        status = f"Found {n} railway line{'s' if n != 1 else ''} in current view."
        self.step1.show_view_results(results, status)
        self._goto_step(0)
        self.statusBar().showMessage(status + " Click a result to load it.")

    def _on_view_search_failed(self, error: str):
        self.step1.show_view_results([], status=f"Search failed: {error}")
        self.statusBar().showMessage(f"Lines-in-View search failed: {error}")

    # ------------------------------------------------------------------
    # Railway loaded
    # ------------------------------------------------------------------

    def _on_railway_fetched(self, overpass_data, relation_info: dict):
        from osm.parser import parse_tracks
        try:
            self._tracks = parse_tracks(overpass_data)
        except Exception as exc:
            QMessageBox.critical(self, "Parse error",
                                 f"Failed to parse track data:\n{exc}")
            self.statusBar().showMessage(f"⚠ Parse error: {exc}")
            return

        if not self._tracks:
            QMessageBox.warning(
                self, "No tracks found",
                "The relation was fetched but no continuous track could be "
                "extracted.\nThe relation may have no ways, or its ways are "
                "not connected."
            )
            self.statusBar().showMessage("⚠ No tracks found in relation.")
            return

        self.map_widget.clear_alignment()
        self.map_widget.show_tracks(self._tracks)
        self.step2.populate(self._tracks)
        n = len(self._tracks)
        name = relation_info.get("name", "")
        self.statusBar().showMessage(
            f"✓ Loaded '{name}' — {n} track{'s' if n != 1 else ''} drawn on map. "
            "Select tracks and click Next."
        )
        self._goto_step(1)

    # ------------------------------------------------------------------
    # Section confirmed
    # ------------------------------------------------------------------

    def _on_section_confirmed(self, selected_tracks: list):
        self._selected_tracks = selected_tracks
        self.statusBar().showMessage(
            f"{len(selected_tracks)} track(s) selected. Configure export settings."
        )
        self._goto_step(2)

    # ------------------------------------------------------------------
    # Config confirmed → project coordinates + launch candidate worker
    # ------------------------------------------------------------------

    def _on_config_confirmed(self, settings: dict):
        from geometry.projection import wgs84_to_projected, auto_utm_epsg
        from geometry.curvature import compute_chainages
        import numpy as np

        self._settings = settings
        epsg       = settings["epsg"]
        work_epsg  = epsg if epsg != -1 else auto_utm_epsg(self._selected_tracks[0].nodes)
        self._work_epsg = work_epsg

        force_positive = settings.get("force_positive", False)
        xy_list        = []
        chainages_list = []

        for track in self._selected_tracks:
            xy = np.array(wgs84_to_projected(track.nodes, work_epsg))
            if force_positive:
                xy = np.abs(xy)
            xy_list.append(xy)
            chainages_list.append(compute_chainages(xy))

        self._xy_list        = xy_list
        self._chainages_list = chainages_list

        self.map_widget.clear_candidates()

        self.step4_candidates.prepare(
            self._selected_tracks, settings, xy_list, chainages_list, work_epsg
        )
        self.statusBar().showMessage(
            "Projecting coordinates… Running candidate algorithms."
        )
        self._goto_step(3)

    # ------------------------------------------------------------------
    # Candidate map overlay update
    # ------------------------------------------------------------------

    def _on_candidate_map_update(self, candidates: list):
        """Called each time a candidate algorithm completes — update map overlays."""
        payload = [
            {
                "nodes": [[lat, lon] for lat, lon in c.geo_wgs84],
                "color": c.color_hex,
                "label": c.label,
            }
            for c in candidates
            if c.geo_wgs84
        ]
        self.map_widget.show_candidates(payload)

    # ------------------------------------------------------------------
    # Candidate selected → go to Step 5
    # ------------------------------------------------------------------

    def _on_candidate_selected(self, candidate):
        self._selected_candidate = candidate
        # Clear candidate overlays; Step 5 will show the chosen one
        self.map_widget.clear_candidates()
        xy        = self._xy_list[0]        if self._xy_list        else None
        chainages = self._chainages_list[0] if self._chainages_list else None
        self.step5_refine.prepare(candidate, xy, chainages, self._settings)
        self.statusBar().showMessage(
            f"Candidate '{getattr(candidate, 'label', '')}' selected. "
            "Optionally refine spiral transitions, then Accept."
        )
        self._goto_step(4)

    # ------------------------------------------------------------------
    # Refinement done → go to Step 6
    # ------------------------------------------------------------------

    def _on_refinement_done(self, elements: list):
        self._final_elements = elements
        self.step6_export.prepare(elements, self._selected_tracks, self._settings)
        self.statusBar().showMessage(
            "Refinement complete. Choose a file and click 'Start Export'."
        )
        self._goto_step(5)

    # ------------------------------------------------------------------
    # Export finished
    # ------------------------------------------------------------------

    def _on_export_finished(self, filepath: str, work_epsg: int):
        # alignment_ready signal from step6 already triggered map display
        self.statusBar().showMessage(
            f"✓ Export complete (EPSG:{work_epsg}) — alignment shown on map. {filepath}"
        )

    # ------------------------------------------------------------------
    # Start over
    # ------------------------------------------------------------------

    def _start_over(self):
        self._tracks             = []
        self._selected_tracks    = []
        self._settings           = {}
        self._selected_candidate = None
        self._final_elements     = []
        self._xy_list            = []
        self._chainages_list     = []
        self.map_widget.clear_all()   # clears tracks + cyan OSM ref + red alignment + candidates
        self.sidebar.reset()
        self._goto_step(0)
        self.statusBar().showMessage(
            "Ready. Search for a new railway or use 'Lines in View'."
        )
