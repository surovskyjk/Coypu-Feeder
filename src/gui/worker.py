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
    mode: "ref" | "name" | "number_in_name" | "bbox"
    query: str for text modes; (south, west, north, east) tuple for bbox
    """
    results_ready = Signal(list)
    failed        = Signal(str)

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
            q = self._query
            if self._mode == "ref":
                results = search_by_ref(q)
            elif self._mode == "name":
                results = search_railways_by_name(q)
            elif self._mode == "number_in_name":
                results = search_by_number_in_name(q)
            elif self._mode == "bbox":
                s, w, n, e = q
                results = search_relations_in_bbox(s, w, n, e)
            elif self._mode == "relation_members":
                results = fetch_relation_members(int(q))
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
    data_ready = Signal(object, dict)   # (overpass_data, relation_info)
    failed     = Signal(str)

    def __init__(self, relation_id: int, parent=None):
        super().__init__(parent)
        self._rid = relation_id

    def run(self):
        try:
            from osm.query import fetch_relation_ways, fetch_relation_metadata
            data = fetch_relation_ways(self._rid)
            info = fetch_relation_metadata(self._rid) or {
                "id": self._rid, "name": str(self._rid)
            }
            self.data_ready.emit(data, info)
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
