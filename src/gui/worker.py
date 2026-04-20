"""
QThread workers for background tasks.
All network / CPU work runs off the main thread;
results are returned via Qt signals.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Search worker
# ---------------------------------------------------------------------------

class SearchWorker(QThread):
    """
    mode: "ref" | "name" | "number_in_name" | "bbox" | "relation_members"
    query: str for text modes; (south, west, north, east) tuple for bbox
    """
    results_ready  = Signal(list)
    failed         = Signal(str)
    status_update  = Signal(str)   # live progress string (endpoint attempts, cache hit…)

    def __init__(self, mode: str, query, parent=None):
        super().__init__(parent)
        self._mode  = mode
        self._query = query

    def run(self):
        try:
            from osm.query import (
                search_by_ref,
                search_railways_by_name,
                search_by_number_in_name,
                search_relations_in_bbox,
                fetch_relation_members,
            )
            cb = lambda msg: self.status_update.emit(msg)
            q = self._query
            if self._mode == "ref":
                results = search_by_ref(q, progress_cb=cb)
            elif self._mode == "name":
                results = search_railways_by_name(q, progress_cb=cb)
            elif self._mode == "number_in_name":
                results = search_by_number_in_name(q, progress_cb=cb)
            elif self._mode == "bbox":
                s, w, n, e = q
                results = search_relations_in_bbox(s, w, n, e, progress_cb=cb)
            elif self._mode == "relation_members":
                results = fetch_relation_members(int(q), progress_cb=cb)
            else:
                results = []
            self.results_ready.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Fetch worker
# ---------------------------------------------------------------------------

class FetchWorker(QThread):
    """Fetches full way/node data + metadata for a single OSM relation."""
    data_ready    = Signal(object, dict)   # (overpass_data, relation_info)
    failed        = Signal(str)
    status_update = Signal(str)            # live progress string

    def __init__(self, relation_id: int, parent=None):
        super().__init__(parent)
        self._rid = relation_id

    def run(self):
        try:
            from osm.query import fetch_relation_ways, fetch_relation_metadata
            cb   = lambda msg: self.status_update.emit(msg)
            data = fetch_relation_ways(self._rid, progress_cb=cb)
            info = fetch_relation_metadata(self._rid, progress_cb=None) or {
                "id": self._rid, "name": str(self._rid)
            }
            self.data_ready.emit(data, info)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Candidate worker
# ---------------------------------------------------------------------------

class CandidateWorker(QThread):
    """
    Runs all four candidate alignment algorithms sequentially in the background.

    Signals
    -------
    candidate_ready(str, object)    algorithm_id, CandidateAlignment (final)
    candidate_preview(str, object)  algorithm_id, CandidateAlignment (intermediate —
                                    emitted periodically by Progressive MC for live
                                    map updates while the algorithm is still running)
    progress_update(str, str)       algorithm_id, human-readable status message
    candidate_error(str, str)       algorithm_id, error message (per-algo failure)
    all_done()                      all algorithms finished
    failed(str)                     fatal setup error (e.g. no data)
    """
    candidate_ready   = Signal(str, object)
    candidate_preview = Signal(str, object)   # intermediate result
    progress_update   = Signal(str, str)      # (algo_id, message)
    candidate_error   = Signal(str, str)
    all_done          = Signal()
    failed            = Signal(str)

    _ALGO_COLORS = {
        "segment_fit":         "#ff9800",
        "segment_fit_spirals": "#26c6da",
        "dp_segment":          "#66bb6a",
        "progressive_mc":      "#42a5f5",
        "raw":                 "#e040fb",
    }

    def __init__(self, xy_list, chainages_list, settings: dict, parent=None):
        super().__init__(parent)
        self._xy_list        = xy_list        # list of np.ndarray
        self._chainages_list = chainages_list # list of np.ndarray
        self._settings       = settings

    def run(self):
        try:
            from geometry.candidates import CandidateGenerator, CandidateAlignment
            from geometry.projection import projected_to_wgs84
            from geometry.alignment import reconstruct_alignment_projected

            # Use first track only for candidate generation (multi-track is Phase F)
            xy        = self._xy_list[0]
            chs       = self._chainages_list[0]
            work_epsg = self._settings.get("_work_epsg", 32633)

            gen = CandidateGenerator(xy, chs, self._settings)

            for algo_id in ["segment_fit", "segment_fit_spirals", "dp_segment", "progressive_mc", "raw"]:
                color = self._ALGO_COLORS.get(algo_id, "#ffffff")

                # --- progress text callback (called from algorithm internals) ---
                def _pcb(msg, _id=algo_id):
                    self.progress_update.emit(_id, msg)

                # --- preview callback (MC only: intermediate element list → WGS84) ---
                def _prev_cb(elements, _id=algo_id, _color=color):
                    if not elements:
                        return
                    try:
                        geo_xy    = reconstruct_alignment_projected(elements, sample_interval=5.0)
                        geo_wgs84 = projected_to_wgs84(geo_xy, work_epsg)
                        preview = CandidateAlignment(
                            algorithm_id=_id,
                            label=gen.LABELS.get(_id, _id),
                            elements=elements,
                            n_elements=len(elements),
                            color_hex=_color,
                            geo_wgs84=geo_wgs84,
                        )
                        self.candidate_preview.emit(_id, preview)
                    except Exception:
                        pass

                try:
                    _pcb("Starting…")
                    c = gen._run_one(algo_id, progress_cb=_pcb, preview_cb=_prev_cb)
                    c.color_hex = color
                    # Compute dense WGS84 points for map display
                    if c.elements:
                        geo_xy      = reconstruct_alignment_projected(
                            c.elements, sample_interval=5.0)
                        c.geo_wgs84 = projected_to_wgs84(geo_xy, work_epsg)
                    else:
                        c.geo_wgs84 = []
                    self.candidate_ready.emit(algo_id, c)
                except Exception as exc:
                    import traceback
                    err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    self.candidate_error.emit(algo_id, err_msg)

            self.all_done.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Final export worker (uses pre-fitted elements — no re-fitting)
# ---------------------------------------------------------------------------

class FinalExportWorker(QThread):
    """
    Export a pre-fitted alignment to LandXML.

    Skips projection + geometry fitting (elements already chosen in Step 4/5).
    Does: element reprojection → OSM reference emit → elevation → LandXML build → write.

    The internal elements are in *work_epsg* (auto-UTM).  Before writing,
    all element coordinates are transformed to *output_epsg* via
    ``geometry.projection.transform_elements``.

    Parameters
    ----------
    elements_list   : list[list[dict]] — one fitted element list per track.
                      Coordinates are in work_epsg (auto-UTM).
    tracks          : list of Track objects (for OSM nodes + elevation sampling)
    settings        : dict from Step3Configure (geometry settings only)
    filepath        : output .xml path
    work_epsg       : internal auto-UTM EPSG (coordinate system of elements)
    output_epsg     : user-chosen output EPSG for LandXML
    force_positive  : if True, abs() applied to output coordinates
    xy_list         : list[np.ndarray] projected coords in work_epsg (for elevation)
    """
    stage_changed    = Signal(str)
    station_progress = Signal(float, float)
    osm_track_ready  = Signal(list)
    alignment_ready  = Signal(list)
    finished         = Signal(str, int)   # filepath, output_epsg
    failed           = Signal(str)

    def __init__(self, elements_list, tracks, settings: dict,
                 filepath: str, work_epsg: int, output_epsg: int,
                 force_positive: bool = False, xy_list=None, parent=None):
        super().__init__(parent)
        self._elements_list  = elements_list
        self._tracks         = tracks
        self._settings       = settings
        self._filepath       = filepath
        self._work_epsg      = work_epsg
        self._output_epsg    = output_epsg
        self._force_positive = force_positive
        self._xy_list        = xy_list or []

    def run(self):
        try:
            self._export()
            self.finished.emit(self._filepath, self._output_epsg)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")

    def _export(self):
        from geometry.alignment import reconstruct_alignment_projected
        from geometry.elevation import (
            interpolate_along_alignment, sample_elevations, fit_vertical_geometry,
        )
        from geometry.curvature import compute_chainages
        from geometry.projection import projected_to_wgs84, transform_elements
        from landxml.builder import build_landxml, write_landxml

        s               = self._settings
        sample_interval = s.get("sample_interval", 20.0)
        vc_length       = s.get("vc_length", 100.0)
        project_name    = s.get("project_name", "Railway Alignment")
        work_epsg       = self._work_epsg
        output_epsg     = self._output_epsg
        force_positive  = self._force_positive

        # ── Emit raw OSM reference overlay (always WGS84) ───────────────────
        self.osm_track_ready.emit(
            [[[lat, lon] for lat, lon in t.nodes] for t in self._tracks]
        )

        alignments: list[dict]       = []
        geo_wgs84_tracks: list[list] = []

        for idx, track in enumerate(self._tracks):
            # Pre-fitted elements in work_epsg (auto-UTM)
            h_elements_utm = (self._elements_list[idx]
                              if idx < len(self._elements_list) else [])

            self.stage_changed.emit(
                f"Reconstructing geometry ({idx + 1}/{len(self._tracks)}): {track.name}…"
            )

            # Reconstruct dense display geometry in work_epsg (undistorted UTM)
            if h_elements_utm:
                geo_xy_utm = reconstruct_alignment_projected(
                    h_elements_utm, sample_interval=5.0
                )
            else:
                geo_xy_utm = []

            # Convert to WGS84 for map display (always from UTM, no sign flip)
            geo_latlon = projected_to_wgs84(geo_xy_utm, work_epsg) if geo_xy_utm else []
            geo_wgs84_tracks.append([[lat, lon] for lat, lon in geo_latlon])

            # ── Elevation (uses work_epsg xy for chainage) ───────────────────
            self.stage_changed.emit(
                f"Querying DEM elevation ({idx + 1}/{len(self._tracks)}): {track.name}…"
            )
            xy_arr   = (self._xy_list[idx] if idx < len(self._xy_list)
                        else np.array([[0.0, 0.0]]))
            chainages = compute_chainages(xy_arr)
            total_len = float(chainages[-1]) if len(chainages) else 0.0

            sample_ch, sample_latlon = interpolate_along_alignment(
                track.nodes, chainages, sample_interval=sample_interval
            )
            self.station_progress.emit(0.0, total_len)
            elevs      = sample_elevations(sample_latlon)
            self.station_progress.emit(total_len, total_len)
            v_elements = fit_vertical_geometry(sample_ch, elevs, vc_length=vc_length)

            # ── Reproject element coordinates to output CRS ──────────────────
            self.stage_changed.emit(
                f"Reprojecting to EPSG:{output_epsg}…"
            )
            h_elements_out = transform_elements(
                h_elements_utm,
                from_epsg=work_epsg,
                to_epsg=output_epsg,
                force_positive=force_positive,
            )

            alignments.append({
                "name":      track.name,
                "elements":  h_elements_out,
                "vertical":  v_elements,
                "sta_start": 0.0,
            })

        # ── Emit reconstructed alignment for map display ─────────────────────
        self.alignment_ready.emit(geo_wgs84_tracks)

        self.stage_changed.emit("Building LandXML…")
        root = build_landxml(
            alignments,
            output_epsg=output_epsg,
            project_name=project_name,
            force_positive=force_positive,
        )

        self.stage_changed.emit("Writing file…")
        write_landxml(root, self._filepath)


# ---------------------------------------------------------------------------
# Cross-section worker
# ---------------------------------------------------------------------------

class CrossSectionWorker(QThread):
    """
    Background worker for cross-section elevation analysis.

    Samples terrain elevation at the central alignment and at left/right
    perpendicular offsets every *interval_m* metres, using OpenTopoData.

    Signals
    -------
    progress(current: int, total: int)  — stations processed so far
    finished(results: list)             — list of result dicts
    failed(error: str)
    """

    progress = Signal(int, int)
    finished = Signal(list)
    failed   = Signal(str)

    def __init__(
        self,
        elements:   list,
        work_epsg:  int,
        offset_m:   float,
        interval_m: float = 5.0,
        parent=None,
    ):
        super().__init__(parent)
        self._elements   = elements
        self._work_epsg  = work_epsg
        self._offset_m   = offset_m
        self._interval_m = interval_m

    def run(self):
        try:
            from geometry.cross_section import compute_cross_section
            results = compute_cross_section(
                self._elements,
                self._work_epsg,
                self._offset_m,
                self._interval_m,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Export worker
# ---------------------------------------------------------------------------

class ExportWorker(QThread):
    """
    Runs: projection → geometry fit → elevation → LandXML.

    Signals
    -------
    stage_changed(str)              named stage banner
    station_progress(float,float)   current / total chainage (m)
    osm_track_ready(list)           raw OSM [[lat,lon],...] per track (emitted
                                    before fitting, for map reference overlay)
    alignment_ready(list)           reconstructed geometric [[lat,lon],...] per
                                    track derived from fitted elements (WGS84)
    finished(str, int)              filepath, work_epsg
    failed(str)                     error message
    """
    stage_changed    = Signal(str)
    station_progress = Signal(float, float)
    osm_track_ready  = Signal(list)   # raw OSM nodes, emitted before fitting
    alignment_ready  = Signal(list)   # reconstructed geometry, emitted after fit
    finished         = Signal(str, int)
    failed           = Signal(str)

    def __init__(self, tracks, settings: dict, filepath: str, parent=None):
        super().__init__(parent)
        self._tracks   = tracks
        self._settings = settings
        self._filepath = filepath

    def run(self):
        try:
            work_epsg = self._export()
            self.finished.emit(self._filepath, work_epsg)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _export(self) -> int:
        from geometry.projection import wgs84_to_projected, auto_utm_epsg
        from geometry.alignment import fit_alignment
        from geometry.elevation import (
            interpolate_along_alignment, sample_elevations, fit_vertical_geometry,
        )
        from geometry.curvature import compute_chainages
        from landxml.builder import build_landxml, write_landxml

        s = self._settings
        epsg            = s["epsg"]
        force_positive  = s.get("force_positive", False)
        smooth_window   = s["smooth_window"]
        sample_interval = s["sample_interval"]
        vc_length       = s["vc_length"]
        project_name    = s.get("project_name", "Railway Alignment")
        min_line        = s.get("min_line_length", 10.0)
        min_arc         = s.get("min_arc_length",  10.0)
        min_spiral      = s.get("min_spiral_length", 10.0)
        min_radius      = s.get("min_radius", 150.0)

        from geometry.alignment import reconstruct_alignment_projected
        from geometry.projection import projected_to_wgs84

        # Resolve working EPSG
        work_epsg = epsg if epsg != -1 else auto_utm_epsg(self._tracks[0].nodes)

        # ── Emit raw OSM polyline for the reference overlay ──────────────────
        # Emitted before any fitting so the cyan dashed line appears immediately.
        self.osm_track_ready.emit(
            [[[lat, lon] for lat, lon in t.nodes] for t in self._tracks]
        )

        self.stage_changed.emit("Projecting coordinates…")
        all_projected: list[np.ndarray] = []
        # Keep the sign of the *original* projected coordinates so we can undo
        # force_positive after reconstruction.
        orig_signs: list[np.ndarray] = []
        for track in self._tracks:
            xy_orig = np.array(wgs84_to_projected(track.nodes, work_epsg))
            orig_signs.append(np.sign(xy_orig[0]))   # sign of first point (x, y)
            if force_positive:
                xy = np.abs(xy_orig)
            else:
                xy = xy_orig
            all_projected.append(xy)

        alignments      = []
        geo_wgs84_tracks: list[list[list[float]]] = []   # for alignment_ready
        n_tracks        = len(self._tracks)

        for idx, (track, xy) in enumerate(zip(self._tracks, all_projected)):
            self.stage_changed.emit(
                f"Fitting geometry ({idx + 1}/{n_tracks}): {track.name}…"
            )
            chainages  = compute_chainages(xy)
            h_elements = fit_alignment(
                xy,
                min_radius=min_radius,
                smooth_window=smooth_window,
                min_line_length=min_line,
                min_arc_length=min_arc,
                min_spiral_length=min_spiral,
                max_deviation=s.get("max_deviation",  0.50),
                check_interval=s.get("check_interval", 5.0),
            )

            # ── Reconstruct dense geometric points from fitted elements ──────
            # sample_interval defaults to 5 m for a visually smooth line
            geo_xy = reconstruct_alignment_projected(h_elements, sample_interval=5.0)

            # Undo force_positive: restore original coordinate signs
            if force_positive and geo_xy:
                sx, sy = float(orig_signs[idx][0]), float(orig_signs[idx][1])
                if sx == 0.0:
                    sx = 1.0
                if sy == 0.0:
                    sy = 1.0
                geo_xy = [(x * sx, y * sy) for x, y in geo_xy]

            # Convert projected → WGS84 (lat, lon)
            geo_latlon = projected_to_wgs84(geo_xy, work_epsg)
            geo_wgs84_tracks.append([[lat, lon] for lat, lon in geo_latlon])

            self.stage_changed.emit(
                f"Querying DEM elevation ({idx + 1}/{n_tracks}): {track.name}…"
            )
            total_len = float(chainages[-1]) if len(chainages) else 0.0
            sample_ch, sample_latlon = interpolate_along_alignment(
                track.nodes, chainages, sample_interval=sample_interval
            )
            self.station_progress.emit(0.0, total_len)
            elevs      = sample_elevations(sample_latlon)
            self.station_progress.emit(total_len, total_len)
            v_elements = fit_vertical_geometry(sample_ch, elevs, vc_length=vc_length)

            alignments.append({
                "name":     track.name,
                "elements": h_elements,
                "vertical": v_elements,
                "sta_start": 0.0,
            })

        # ── Emit reconstructed geometric alignment (WGS84) for map display ─
        self.alignment_ready.emit(geo_wgs84_tracks)

        self.stage_changed.emit("Building LandXML…")
        root = build_landxml(
            alignments,
            output_epsg=work_epsg,
            project_name=project_name,
            force_positive=force_positive,
        )

        self.stage_changed.emit("Writing file…")
        write_landxml(root, self._filepath)
        return work_epsg
