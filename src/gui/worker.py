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
    stage_changed(str)          named stage banner
    station_progress(float,float)  current / total chainage (m)
    alignment_ready(list)       [[lat,lon],...] per track — for map display
    finished(str, int)          filepath, work_epsg
    failed(str)                 error message
    """
    stage_changed    = Signal(str)
    station_progress = Signal(float, float)
    alignment_ready  = Signal(list)   # list of [[lat,lon],...] per track
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

        # Resolve working EPSG
        work_epsg = epsg if epsg != -1 else auto_utm_epsg(self._tracks[0].nodes)

        # ── Emit WGS84 track paths for map display (before any projection/abs) ──
        # Use original OSM nodes — always correct, independent of CRS/force_positive
        self.alignment_ready.emit(
            [[[lat, lon] for lat, lon in t.nodes] for t in self._tracks]
        )

        self.stage_changed.emit("Projecting coordinates…")
        all_projected: list[np.ndarray] = []
        for track in self._tracks:
            xy = np.array(wgs84_to_projected(track.nodes, work_epsg))
            if force_positive:
                xy = np.abs(xy)
            all_projected.append(xy)

        alignments = []
        n_tracks   = len(self._tracks)

        for idx, (track, xy) in enumerate(zip(self._tracks, all_projected)):
            self.stage_changed.emit(
                f"Fitting geometry ({idx + 1}/{n_tracks}): {track.name}…"
            )
            chainages  = compute_chainages(xy)
            h_elements = fit_alignment(
                xy,
                smooth_window=smooth_window,
                min_line_length=min_line,
                min_arc_length=min_arc,
                min_spiral_length=min_spiral,
            )

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
