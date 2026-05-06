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
from .steps.step5_refine        import Step5Refine
from .steps.step6_crosssection  import Step6CrossSection
from .steps.step7_export        import Step7Export

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
        self.step6_crosssec     = Step6CrossSection()
        self.step7_export       = Step7Export()

        self.stack.addWidget(self.step1)            # 0
        self.stack.addWidget(self.step2)            # 1
        self.stack.addWidget(self.step3)            # 2
        self.stack.addWidget(self.step4_candidates) # 3
        self.stack.addWidget(self.step5_refine)     # 4
        self.stack.addWidget(self.step6_crosssec)   # 5
        self.stack.addWidget(self.step7_export)     # 6

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

        # Step 2 → highlight / fit / confirm / back
        self.step2.highlight_changed.connect(self._on_highlight_changed)
        self.step2.fit_to_tracks_requested.connect(self._on_fit_to_tracks)
        self.step2.section_confirmed.connect(self._on_section_confirmed)
        self.step2.back_requested.connect(lambda: self._goto_step(0))

        # Step 3 → config confirmed / back
        self.step3.config_confirmed.connect(self._on_config_confirmed)
        self.step3.back_requested.connect(lambda: self._goto_step(1))

        # Step 4 (candidates) → map update + selection + back
        self.step4_candidates.candidate_map_update.connect(self._on_candidate_map_update)
        self.step4_candidates.candidate_selected.connect(self._on_candidate_selected)
        self.step4_candidates.back_requested.connect(self._on_candidates_back)

        # Step 5 (refine) → done / back / live map update
        self.step5_refine.refinement_done.connect(self._on_refinement_done)
        self.step5_refine.back_requested.connect(self._on_refine_back)
        self.step5_refine.alignment_update_requested.connect(
            self._on_refine_alignment_update
        )

        # Step 6 (cross-section) → back / done / map overlay
        self.step6_crosssec.back_requested.connect(lambda: self._goto_step(4))
        self.step6_crosssec.analysis_done.connect(self._on_analysis_done)
        self.step6_crosssec.cross_section_ready.connect(self._on_cross_section_ready)

        # Step 7 (export) → back
        self.step7_export.back_requested.connect(lambda: self._goto_step(5))

        # Step 7 (export) → alignment display / fit / export / restart
        self.step7_export.osm_track_ready.connect(self._on_osm_track_ready)
        self.step7_export.alignment_ready.connect(self._on_alignment_ready)
        self.step7_export.alignment_segments_ready.connect(self._on_alignment_segments_ready)
        self.step7_export.fit_to_alignment_requested.connect(self._on_fit_to_alignment)
        self.step7_export.export_finished.connect(self._on_export_finished)
        self.step7_export.start_over_requested.connect(self._start_over)

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
        # Default merged-red drawing; superseded by per-element rendering
        # below when alignment_segments_ready fires.
        self.map_widget.show_alignment(alignments)
        total_pts = sum(len(a) for a in alignments)
        self.statusBar().showMessage(
            f"Both overlays ready — 🔴 red: fitted LandXML ({total_pts} pts)  "
            f"🔵 cyan dashed: OSM reference  ({len(alignments)} track(s))."
        )

    def _on_alignment_segments_ready(self, segments: list):
        """Per-element coloured rendering after final export."""
        if not segments:
            return
        self.map_widget.show_alignment_segmented(segments)
        n_line   = sum(1 for s in segments if s.get("type") == "Line")
        n_arc    = sum(1 for s in segments if s.get("type") == "Arc")
        n_spiral = sum(1 for s in segments if s.get("type") == "Spiral")
        self.statusBar().showMessage(
            f"Per-element view drawn — 🔵 {n_line} Lines · 🔴 {n_arc} Arcs · "
            f"🟢 {n_spiral} Spirals. Hover any segment for parameters."
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
        """Map returned its bounds; run 'Lines in View' search."""
        # ── Overpass "Lines in View" search ──────────────────────────
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

        # Internal working CRS is always auto-UTM (metric, undistorted).
        # The user chooses the output CRS in Step 6 just before exporting.
        work_epsg = auto_utm_epsg(self._selected_tracks[0].nodes)
        self._work_epsg = work_epsg

        xy_list        = []
        chainages_list = []

        for track in self._selected_tracks:
            xy = np.array(wgs84_to_projected(track.nodes, work_epsg))
            xy_list.append(xy)
            chainages_list.append(compute_chainages(xy))

        self._xy_list        = xy_list
        self._chainages_list = chainages_list

        self.map_widget.clear_candidates()
        self.map_widget.clear_alignment()

        # Show dashed cyan OSM reference immediately so it's visible during
        # candidate generation (Step 4) and not just after export.
        osm_ref = [[[lat, lon] for lat, lon in t.nodes]
                   for t in self._selected_tracks]
        if any(len(r) > 0 for r in osm_ref):
            self.map_widget.show_osm_reference(osm_ref)

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
        # Clear candidate overlays; Step 5 will show the chosen one as
        # per-element coloured alignment with hover tooltips.
        self.map_widget.clear_candidates()
        xy        = self._xy_list[0]        if self._xy_list        else None
        chainages = self._chainages_list[0] if self._chainages_list else None
        self.step5_refine.prepare(candidate, xy, chainages, self._settings)

        # Always draw the merged red polyline first so the user is guaranteed
        # to see *something* even if the segmented call has any issue. Then
        # overlay the per-element coloured polylines (which replace the merged
        # one via clearAlignment() inside showAlignmentSegmented).
        merged = getattr(candidate, "geo_wgs84", None)
        if merged:
            self.map_widget.show_alignment([[list(pt) for pt in merged]])

        segments = getattr(candidate, "geo_segments_wgs84", None) or []
        n_line   = sum(1 for s in segments if s.get("type") == "Line")
        n_arc    = sum(1 for s in segments if s.get("type") == "Arc")
        n_spiral = sum(1 for s in segments if s.get("type") == "Spiral")
        print(f"[App] candidate '{getattr(candidate, 'label', '?')}' selected: "
              f"{len(segments)} segments ({n_line} Lines, {n_arc} Arcs, {n_spiral} Spirals); "
              f"merged_pts={len(merged) if merged else 0}")
        if segments:
            self.map_widget.show_alignment_segmented(segments)

        self.statusBar().showMessage(
            f"Candidate '{getattr(candidate, 'label', '')}' selected — "
            f"{n_line} Lines · {n_arc} Arcs · {n_spiral} Spirals shown. "
            "Hover any segment for parameters."
        )
        self._goto_step(4)

    def _on_candidates_back(self):
        """Go back from Candidates to Configure — clear overlays."""
        self.map_widget.clear_candidates()
        self.map_widget.clear_alignment()
        self.map_widget.clear_osm_reference()
        self._goto_step(2)

    def _on_refine_alignment_update(self, elements: list):
        """Live map update from Step 5 when a spiral is inserted/removed."""
        from geometry.alignment import (
            reconstruct_alignment_projected,
            reconstruct_alignment_per_element,
        )
        from geometry.projection import projected_to_wgs84

        if not elements or not self._xy_list:
            return
        try:
            # Build per-element segments for tooltip-rich rendering.
            segments_payload = []
            try:
                from gui.worker import _serialise_element_params
                per_el = reconstruct_alignment_per_element(elements, sample_interval=2.0)
                for el, pts in per_el:
                    if not pts:
                        continue
                    wgs = projected_to_wgs84(pts, self._work_epsg)
                    segments_payload.append({
                        "type":   el.get("type", "Line"),
                        "params": _serialise_element_params(el),
                        "points": [list(p) for p in wgs],
                    })
            except Exception:
                segments_payload = []

            if segments_payload:
                self.map_widget.show_alignment_segmented(segments_payload)
            else:
                geo_xy     = reconstruct_alignment_projected(elements, sample_interval=5.0)
                geo_latlon = projected_to_wgs84(geo_xy, self._work_epsg)
                self.map_widget.show_alignment([[[lat, lon] for lat, lon in geo_latlon]])

            n_spirals  = sum(1 for e in elements if e.get("type") == "Spiral")
            self.statusBar().showMessage(
                f"Alignment updated — {len(elements)} elements "
                f"({n_spirals} spiral{'s' if n_spirals != 1 else ''})."
            )
        except Exception as exc:
            self.statusBar().showMessage(f"⚠ Map update failed: {exc}")

    def _on_refine_back(self):
        """Go back from Refine to Candidates — restore all candidate overlays."""
        self.map_widget.clear_alignment()
        # Re-emit candidate overlays if available
        candidates = [c for c in self.step4_candidates._candidates.values()
                      if c.geo_wgs84]
        if candidates:
            payload = [
                {"nodes": [[lat, lon] for lat, lon in c.geo_wgs84],
                 "color": c.color_hex, "label": c.label}
                for c in candidates
            ]
            self.map_widget.show_candidates(payload)
        self._goto_step(3)

    # ------------------------------------------------------------------
    # Refinement done → go to Step 6
    # ------------------------------------------------------------------

    def _on_refinement_done(self, elements: list):
        self._final_elements = elements
        self.step6_crosssec.prepare(elements, self._work_epsg)
        self.statusBar().showMessage(
            "Refinement complete. Run cross-section analysis or skip to export."
        )
        self._goto_step(5)

    # ------------------------------------------------------------------
    # Cross-section analysis done → go to Step 7
    # ------------------------------------------------------------------

    def _on_analysis_done(self, results: list):
        # results may be [] if the step was skipped
        elements_list = [self._final_elements]
        self.step7_export.prepare(
            elements_list, self._selected_tracks, self._settings,
            self._work_epsg, self._xy_list,
        )
        n = len(results)
        msg = (
            f"Cross-section: {n} stations analysed. Choose a file and export."
            if n else
            "Skipped cross-section analysis. Choose a file and export."
        )
        self.statusBar().showMessage(msg)
        self._goto_step(6)

    def _on_cross_section_ready(self, left_pts: list, right_pts: list):
        """Show coloured cross-section overlays on the map."""
        self.map_widget.show_cross_section(left_pts, right_pts)

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
        self.map_widget.clear_all()   # clears tracks + osmRef + alignment + candidates + cross-section
        self.sidebar.reset()
        self._goto_step(0)
        self.statusBar().showMessage(
            "Ready. Search for a new railway or use 'Lines in View'."
        )
