"""
Multi-algorithm candidate alignment generator — redesigned.

Root cause of the old approach: ``_build_elements_c1`` propagated arc deflection
angles measured from noisy OSM points, producing cumulative position drift that
grew progressively along the alignment.

The new algorithms use **position-anchored, tangent-point junctions**:
each arc is fitted independently to its OSM point cluster; each line direction
is fitted independently; junctions are solved geometrically (O(1), no drift).

Algorithm IDs
-------------
``segment_fit``     Segment & Fit   — curvature segmentation + tangent-point assembly
``dp_segment``      DP Segmentation — Imai-Iri DP, globally optimal element count
``progressive_mc``  Progressive MC  — greedy insertion + simulated annealing
``raw``             OSM Polyline    — one Line per vertex pair (unchanged)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateAlignment:
    algorithm_id:  str
    label:         str
    elements:      list
    max_deviation: float = 0.0
    rmse:          float = 0.0
    n_elements:    int   = 0
    color_hex:     str   = "#ffffff"
    geo_wgs84:     list  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Segment:
    seg_type:   str    # "Line" | "Arc"
    start_idx:  int    # index into xy (inclusive)
    end_idx:    int    # index into xy (inclusive)
    R_median:   float  # fitted radius; math.inf for Line
    rot:        str    # "ccw" | "cw"
    deflection: float  # signed total heading change (rad)


# Minimum arc deflection below which a curved segment is treated as a line
_MIN_ARC_DEFLECTION_RAD = math.radians(5.0)


# ---------------------------------------------------------------------------
# Candidate evaluation  (unchanged public API)
# ---------------------------------------------------------------------------

def evaluate_candidate(
    elements:      list[dict],
    xy:            np.ndarray,
    chainages:     np.ndarray,
    check_interval: float = 5.0,
) -> dict:
    """
    Compute quality metrics for a list of fitted elements vs the OSM polyline.

    Returns dict with keys: max_deviation (float), rmse (float), n_elements (int)
    """
    from geometry.alignment import max_deviation_element

    if not elements or len(xy) < 2:
        return {"max_deviation": 0.0, "rmse": 0.0, "n_elements": len(elements)}

    max_dev = 0.0
    sq_sum  = 0.0
    sq_cnt  = 0

    for el in elements:
        sta0   = el.get("sta_start", 0.0)
        sta1   = sta0 + el.get("length", 0.0)
        mask   = (chainages >= sta0 - 0.1) & (chainages <= sta1 + 0.1)
        xy_seg = xy[mask]
        if len(xy_seg) < 2:
            continue
        dev = max_deviation_element(el, xy_seg, check_interval)
        if dev > max_dev:
            max_dev = dev

    for i, pt in enumerate(xy):
        ch = float(chainages[i])
        min_dist = float("inf")
        for el in elements:
            sta0 = el.get("sta_start", 0.0)
            sta1 = sta0 + el.get("length", 0.0)
            if ch < sta0 - 1.0 or ch > sta1 + 1.0:
                continue
            d = _point_to_element_dist(pt, el)
            if d < min_dist:
                min_dist = d
        if math.isfinite(min_dist):
            sq_sum += min_dist * min_dist
            sq_cnt += 1

    rmse = math.sqrt(sq_sum / sq_cnt) if sq_cnt > 0 else 0.0
    return {
        "max_deviation": float(max_dev),
        "rmse":          float(rmse),
        "n_elements":    len(elements),
    }


def _point_to_element_dist(pt: np.ndarray, el: dict) -> float:
    """Perpendicular distance from point to the nearest point on a Line/Arc element."""
    etype = el.get("type", "Line")
    if etype == "Line":
        start = np.array(el["start"])
        end   = np.array(el["end"])
        seg   = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return float(np.linalg.norm(pt - start))
        t = float(np.dot(pt - start, seg)) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        return float(np.linalg.norm(pt - (start + t * seg)))
    elif etype == "Arc":
        center = np.array(el["center"])
        r      = el.get("radius", 1.0)
        return abs(float(np.linalg.norm(pt - center)) - r)
    else:
        start = np.array(el.get("start", [0.0, 0.0]))
        end   = np.array(el.get("end",   [0.0, 0.0]))
        seg   = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return float(np.linalg.norm(pt - start))
        t = float(np.dot(pt - start, seg)) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        return float(np.linalg.norm(pt - (start + t * seg)))


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------

class CandidateGenerator:
    """Run all four candidate algorithms on a projected XY polyline."""

    COLORS = ["#ff9800", "#66bb6a", "#42a5f5", "#e040fb"]
    LABELS = {
        "segment_fit":    "Segment & Fit",
        "dp_segment":     "DP Segmentation",
        "progressive_mc": "Progressive MC",
        "raw":            "OSM Polyline",
    }

    def __init__(self, xy: np.ndarray, chainages: np.ndarray, settings: dict):
        self.xy             = xy
        self.chainages      = chainages
        self.settings       = settings
        self.min_radius     = settings.get("min_radius",       150.0)
        self.smooth_window  = settings.get("smooth_window",    21)
        self.max_deviation  = settings.get("max_deviation",    0.5)
        self.check_interval = settings.get("check_interval",   5.0)
        self.merge_pct         = settings.get("merge_radius_pct",  15.0)
        self.time_budget_s     = settings.get("time_budget_s",     60.0)
        self.division_length    = settings.get("division_length",    500.0)
        self.min_tangent_length = settings.get("min_tangent_length",  30.0)
        self.min_kappa_radius   = settings.get("min_kappa_radius",     0.0)
        self.min_kappa_length   = settings.get("min_kappa_length",   200.0)

        # Precompute forced-line chainage ranges once (shared by all algorithms)
        self._forced_ch_ranges: list[tuple[float, float]] = _compute_forced_line_ranges(
            self.xy, self.chainages,
            smooth_window    = self.smooth_window,
            min_kappa_radius = self.min_kappa_radius,
            min_kappa_length = self.min_kappa_length,
        )

    def run_all(self) -> list[CandidateAlignment]:
        results = []
        algo_ids = ["segment_fit", "dp_segment", "progressive_mc", "raw"]
        for algo_id, color in zip(algo_ids, self.COLORS):
            try:
                c = self._run_one(algo_id)
                c.color_hex = color
            except Exception:
                c = CandidateAlignment(
                    algorithm_id=algo_id,
                    label=self.LABELS[algo_id],
                    elements=[],
                    color_hex=color,
                )
            results.append(c)
        return results

    def _run_one(self, algo_id: str, progress_cb=None, preview_cb=None) -> CandidateAlignment:
        if algo_id == "segment_fit":
            return self._run_segment_fit(progress_cb=progress_cb)
        elif algo_id == "dp_segment":
            return self._run_dp(progress_cb=progress_cb)
        elif algo_id == "progressive_mc":
            return self._run_progressive_mc(progress_cb=progress_cb, preview_cb=preview_cb)
        elif algo_id == "raw":
            return self._run_raw()
        else:
            raise ValueError(f"Unknown algorithm: {algo_id!r}")

    # ------------------------------------------------------------------
    # Algorithm 1 — Segment & Fit
    # ------------------------------------------------------------------

    def _run_segment_fit(self, progress_cb=None) -> CandidateAlignment:
        """
        Curvature segmentation + independent primitive fitting + tangent-point
        C0/C1 assembly.  Fast and reliable for clean OSM data.
        """
        from geometry.alignment import _fit_circle_kasa
        from geometry.curvature import compute_curvature, smooth_curvature

        def _p(msg):
            if progress_cb:
                progress_cb(msg)

        xy, chainages = self.xy, self.chainages
        N = len(xy)

        if N < 2:
            return CandidateAlignment("segment_fit", "Segment & Fit", [])
        if N == 2:
            elements = _two_point_line(xy, chainages)
            metrics  = evaluate_candidate(elements, xy, chainages, self.check_interval)
            return CandidateAlignment("segment_fit", "Segment & Fit", elements, **metrics)

        LINE_TOL = 0.001

        _p("Computing curvature profile…")
        kappa        = compute_curvature(xy)
        kappa_smooth = smooth_curvature(kappa, window=self.smooth_window)

        micro_types: list[str] = []
        micro_sign:  list[int] = []

        for i in range(1, N - 1):
            k = float(kappa_smooth[i])
            if abs(k) < LINE_TOL:
                micro_types.append("Line")
                micro_sign.append(+1)
            else:
                micro_types.append("Arc")
                micro_sign.append(+1 if k > 0.0 else -1)

        _p("Segmenting by curvature sign…")
        segments = _merge_segments_by_sign(micro_types, micro_sign, xy)
        if not segments:
            elements = _two_point_line(xy, chainages)
            metrics  = evaluate_candidate(elements, xy, chainages, self.check_interval)
            return CandidateAlignment("segment_fit", "Segment & Fit", elements, **metrics)

        _p("Fitting circle arcs (Kasa)…")
        # Kasa refinement
        for seg in segments:
            if seg.seg_type == "Arc":
                pts = xy[seg.start_idx: seg.end_idx + 1]
                if len(pts) >= 3:
                    cx, cy, r = _fit_circle_kasa(pts)
                    if cx is not None and r is not None and self.min_radius <= r < 1e6:
                        seg.R_median = float(r)

        # Sagitta filter
        changed = False
        for seg in segments:
            if seg.seg_type == "Arc":
                if _arc_sagitta(seg.R_median, seg.deflection) < 1.5:
                    seg.seg_type = "Line"
                    seg.R_median = math.inf
                    changed = True
        if changed:
            segments = _merge_consecutive_lines(segments, xy)

        # Stage-2 radius merge
        _p("Merging similar-radius arcs…")
        segments = _merge_arcs_by_radius(segments, xy, self.merge_pct)

        # Re-run Kasa after merge
        for seg in segments:
            if seg.seg_type == "Arc":
                pts = xy[seg.start_idx: seg.end_idx + 1]
                if len(pts) >= 3:
                    cx, cy, r = _fit_circle_kasa(pts)
                    if cx is not None and r is not None and self.min_radius <= r < 1e6:
                        seg.R_median = float(r)

        _p("Assembling elements with tangent junctions…")
        elements = _connect_segments_tangent(segments, xy, chainages, self.min_radius)
        _p("Post-processing (C1 enforcement)…")
        elements = _post_process_elements(
            elements, self._forced_ch_ranges, self.min_radius
        )
        _p("Evaluating quality…")
        metrics  = evaluate_candidate(elements, xy, chainages, self.check_interval)
        return CandidateAlignment("segment_fit", "Segment & Fit", elements, **metrics)

    # ------------------------------------------------------------------
    # Algorithm 2 — DP Segmentation
    # ------------------------------------------------------------------

    def _run_dp(self, progress_cb=None) -> CandidateAlignment:
        """
        Imai-Iri dynamic programming: finds the globally optimal segmentation
        (minimum SSE + regularisation × element count).
        O(N²) cost table; practical for N < 1000 OSM nodes.
        """
        def _p(msg):
            if progress_cb:
                progress_cb(msg)

        # lam scales with merge_pct: higher tolerance → fewer, longer elements
        lam = max(1.0, self.merge_pct) ** 2 * 0.5
        _p("Building cost table…")
        segments = _dp_segmentation(
            self.xy, self.chainages, lam, self.min_radius,
            progress_cb=progress_cb,
            time_budget_s=self.time_budget_s,
        )
        _p("Assembling elements with tangent junctions…")
        elements = _connect_segments_tangent(
            segments, self.xy, self.chainages, self.min_radius
        )
        _p("Post-processing (C1 enforcement)…")
        elements = _post_process_elements(
            elements, self._forced_ch_ranges, self.min_radius
        )
        _p("Evaluating quality…")
        metrics = evaluate_candidate(elements, self.xy, self.chainages, self.check_interval)
        return CandidateAlignment("dp_segment", "DP Segmentation", elements, **metrics)

    # ------------------------------------------------------------------
    # Algorithm 3 — Progressive MC
    # ------------------------------------------------------------------

    def _run_progressive_mc(self, progress_cb=None, preview_cb=None) -> CandidateAlignment:
        """
        Piecewise MC with boundary constraints + Arc-Line-Arc consolidation.
        """
        # ── Phase 1: Piecewise MC ───────────────────────────────────────────
        elements = _progressive_mc_build_piecewise(
            self.xy, self.chainages,
            max_deviation   = self.max_deviation,
            min_radius      = self.min_radius,
            merge_pct       = self.merge_pct,
            max_elements    = 80,
            time_budget_s   = self.time_budget_s,
            division_length = self.division_length,
            progress_cb     = progress_cb,
            preview_cb      = preview_cb,
        )

        # ── Phase 2: Arc-Line-Arc consolidation ────────────────────────────
        if progress_cb:
            progress_cb("Consolidating Arc\u2013Line\u2013Arc patterns\u2026")
        elements = _consolidate_arc_line_arc(
            elements, self.xy, self.chainages,
            min_tangent_length = self.min_tangent_length,
            min_radius         = self.min_radius,
        )

        # ── Phase 3: C1 post-processing ────────────────────────────────────
        if progress_cb:
            progress_cb("Post-processing (C1 enforcement)\u2026")
        elements = _post_process_elements(
            elements, self._forced_ch_ranges, self.min_radius
        )

        # ── Quality evaluation ─────────────────────────────────────────────
        if progress_cb:
            progress_cb("Evaluating quality\u2026")
        metrics = evaluate_candidate(elements, self.xy, self.chainages, self.check_interval)
        return CandidateAlignment("progressive_mc", "Progressive MC", elements, **metrics)

    # ------------------------------------------------------------------
    # Raw OSM polyline (unchanged)
    # ------------------------------------------------------------------

    def _run_raw(self) -> CandidateAlignment:
        """One Line element per consecutive OSM vertex pair. No fitting, no C1."""
        xy, chainages = self.xy, self.chainages
        elements: list[dict] = []
        sta = 0.0
        for i in range(len(xy) - 1):
            p0, p1  = xy[i], xy[i + 1]
            seg_len = float(chainages[i + 1] - chainages[i])
            if seg_len < 1e-9:
                continue
            elements.append({
                "type":          "Line",
                "sta_start":     sta,
                "length":        seg_len,
                "start":         p0.tolist(),
                "end":           p1.tolist(),
                "direction_rad": math.atan2(
                    float(p1[1] - p0[1]), float(p1[0] - p0[0])
                ),
            })
            sta += seg_len

        metrics = evaluate_candidate(elements, xy, chainages, self.check_interval)
        return CandidateAlignment("raw", "OSM Polyline", elements, **metrics)


# ---------------------------------------------------------------------------
# Shared geometry helpers (used by multiple algorithms)
# ---------------------------------------------------------------------------

def _two_point_line(xy: np.ndarray, chainages: np.ndarray) -> list[dict]:
    """Trivial single-Line element for degenerate 2-point inputs."""
    seg_len = float(chainages[-1] - chainages[0])
    heading = math.atan2(float(xy[-1, 1] - xy[0, 1]), float(xy[-1, 0] - xy[0, 0]))
    return [{
        "type": "Line", "sta_start": 0.0, "length": seg_len,
        "start": xy[0].tolist(), "end": xy[-1].tolist(),
        "direction_rad": heading,
    }]


def _segment_deflection(pts: np.ndarray) -> float:
    """Signed total heading change across a point sequence (sum of turning angles)."""
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 2):
        e1 = pts[i + 1] - pts[i]
        e2 = pts[i + 2] - pts[i + 1]
        cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        dot   = float(e1[0] * e2[0] + e1[1] * e2[1])
        total += math.atan2(cross, dot)
    return total


def _arc_sagitta(R: float, defl_rad: float) -> float:
    """Maximum lateral deviation of a circular arc from its chord."""
    if not math.isfinite(R) or R <= 0:
        return 0.0
    return R * (1.0 - math.cos(abs(defl_rad) / 2.0))


# ---------------------------------------------------------------------------
# Segment-merge helpers (used by Algorithm 1)
# ---------------------------------------------------------------------------

def _merge_segments_by_sign(
    micro_types: list[str],
    micro_sign:  list[int],
    xy:          np.ndarray,
) -> list[_Segment]:
    """Stage-1: group by type + sign; demote shallow arcs; merge adjacent Lines."""
    n_micro = len(micro_types)
    if n_micro == 0:
        return []

    segments: list[_Segment] = []
    seg_start    = 0
    current_type = micro_types[0]
    current_sign = micro_sign[0]

    def _flush(seg_end_micro: int) -> None:
        xi0  = seg_start
        xi1  = min(seg_end_micro + 1, len(xy) - 1)
        pts  = xy[xi0: xi1 + 1]
        defl = _segment_deflection(pts)
        rot  = "ccw" if current_sign >= 0 else "cw"
        if current_type == "Line" or abs(defl) < _MIN_ARC_DEFLECTION_RAD:
            typ, R_med = "Line", math.inf
        else:
            typ, R_med = "Arc", math.inf
        segments.append(_Segment(
            seg_type=typ, start_idx=xi0, end_idx=xi1,
            R_median=R_med, rot=rot, deflection=defl,
        ))

    for m in range(1, n_micro):
        if (micro_types[m] != current_type
                or (micro_types[m] == "Arc" and micro_sign[m] != current_sign)):
            _flush(m - 1)
            seg_start    = m
            current_type = micro_types[m]
            current_sign = micro_sign[m]

    _flush(n_micro - 1)

    # Boundary corrections
    if segments:
        segments[0].start_idx = 0
        segments[-1].end_idx  = len(xy) - 1

    return _merge_consecutive_lines(segments, xy)


def _merge_consecutive_lines(segments: list[_Segment], xy: np.ndarray) -> list[_Segment]:
    """Merge adjacent Line segments into one."""
    merged: list[_Segment] = []
    for seg in segments:
        if merged and seg.seg_type == "Line" and merged[-1].seg_type == "Line":
            merged[-1].end_idx  = seg.end_idx
            pts = xy[merged[-1].start_idx: merged[-1].end_idx + 1]
            merged[-1].deflection = _segment_deflection(pts)
        else:
            merged.append(seg)
    return merged


def _merge_arcs_by_radius(
    segments:         list[_Segment],
    xy:               np.ndarray,
    merge_radius_pct: float,
) -> list[_Segment]:
    """Stage-2: merge adjacent same-sign Arcs with similar Kasa radii."""
    tol     = merge_radius_pct / 100.0
    changed = True
    while changed:
        changed = False
        merged: list[_Segment] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if (i + 1 < len(segments)
                    and seg.seg_type == "Arc"
                    and segments[i + 1].seg_type == "Arc"
                    and seg.rot == segments[i + 1].rot):
                nxt = segments[i + 1]
                Ra, Rb = seg.R_median, nxt.R_median
                if (Ra > 0 and Rb > 0
                        and math.isfinite(Ra) and math.isfinite(Rb)
                        and abs(Ra - Rb) / max(Ra, Rb) <= tol):
                    La = float(np.linalg.norm(xy[seg.end_idx] - xy[seg.start_idx]))
                    Lb = float(np.linalg.norm(xy[nxt.end_idx] - xy[nxt.start_idx]))
                    den = La + Lb if (La + Lb) > 0 else 1.0
                    R_m = (Ra * La + Rb * Lb) / den
                    pts  = xy[seg.start_idx: nxt.end_idx + 1]
                    defl = _segment_deflection(pts)
                    merged.append(_Segment(
                        seg_type="Arc", start_idx=seg.start_idx, end_idx=nxt.end_idx,
                        R_median=R_m, rot=seg.rot, deflection=defl,
                    ))
                    i += 2
                    changed = True
                    continue
            merged.append(seg)
            i += 1
        segments = merged
    return segments


# ---------------------------------------------------------------------------
# Core: position-anchored tangent-point junction assembly
# ---------------------------------------------------------------------------

def _fit_line_direction(xy: np.ndarray, i0: int, i1: int) -> float:
    """
    Orthogonal regression via SVD. Returns dominant heading (rad) aligned with
    overall travel direction (xy[i0] → xy[i1]).
    """
    pts = xy[i0: i1 + 1]
    if len(pts) < 2:
        i0c = max(0, i0 - 1)
        i1c = min(len(xy) - 1, i1 + 1)
        d = xy[i1c] - xy[i0c]
        return math.atan2(float(d[1]), float(d[0]))
    mean    = pts.mean(axis=0)
    centred = pts - mean
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        d = vt[0]
    except np.linalg.LinAlgError:
        d = pts[-1] - pts[0]
    # Align sign with travel direction
    overall = pts[-1] - pts[0]
    if np.dot(d, overall) < 0:
        d = -d
    norm = math.hypot(float(d[0]), float(d[1]))
    if norm < 1e-9:
        return 0.0
    return math.atan2(float(d[1]) / norm, float(d[0]) / norm)


def _fit_arc_robust(
    xy: np.ndarray, i0: int, i1: int, min_radius: float
) -> tuple[float, float, float] | None:
    """
    Fit a circle to OSM[i0..i1] using Kasa algebraic method.
    Returns (cx, cy, R) or None if the fit is degenerate.
    """
    from geometry.alignment import _fit_circle_kasa

    pts = xy[i0: i1 + 1]
    if len(pts) >= 3:
        cx, cy, r = _fit_circle_kasa(pts)
        if cx is not None and r is not None and math.isfinite(r) and min_radius <= r < 1e6:
            return float(cx), float(cy), float(r)
    # Fallback: first / middle / last only
    if len(pts) >= 3:
        sub = np.array([pts[0], pts[len(pts) // 2], pts[-1]])
        cx, cy, r = _fit_circle_kasa(sub)
        if cx is not None and r is not None and math.isfinite(r) and min_radius <= r < 1e6:
            return float(cx), float(cy), float(r)
    return None


def _arc_line_tangent_junction(
    cx: float, cy: float, R: float, rot: str, heading_rad: float
) -> tuple[float, float]:
    """
    The unique point on circle (cx, cy, R) where the tangent direction equals
    heading_rad in the direction of arc travel.

    Derivation (CCW arc, increasing θ):
        tangent at θ = (−sin θ, cos θ).  Set equal to (cos φ, sin φ):
        ⟹  θ = φ − π/2  ⟹  J = (cx + R sin φ, cy − R cos φ)

    CW arc (decreasing θ):
        tangent at θ = (sin θ, −cos θ).  Set equal to (cos φ, sin φ):
        ⟹  θ = π/2 + φ  ⟹  J = (cx − R sin φ, cy + R cos φ)
    """
    sp = math.sin(heading_rad)
    cp = math.cos(heading_rad)
    if rot == "ccw":
        return cx + R * sp, cy - R * cp
    else:
        return cx - R * sp, cy + R * cp


def _connect_segments_tangent(
    segments:   list[_Segment],
    xy:         np.ndarray,
    chainages:  np.ndarray,
    min_radius: float,
) -> list[dict]:
    """
    Build a C0+C1 element list using tangent-point junctions.

    Each arc is independently Kasa-fitted to its OSM points; each line
    direction is independently PCA-fitted.  Junction points are solved
    analytically — no forward angle propagation, no accumulated drift.

    Anchors: first element starts at xy[0]; last element ends at xy[-1].
    """
    if not segments:
        return []

    n = len(segments)

    # ── Step 1: fit geometric primitives ─────────────────────────────────
    fitted: list[dict] = []
    for seg in segments:
        if seg.seg_type == "Arc":
            result = _fit_arc_robust(xy, seg.start_idx, seg.end_idx, min_radius)
            if result is None:
                # Degenerate — treat as line
                h = _fit_line_direction(xy, seg.start_idx, seg.end_idx)
                fitted.append({"type": "Line", "heading": h, "seg": seg})
            else:
                cx, cy, R = result
                fitted.append({
                    "type": "Arc", "cx": cx, "cy": cy, "R": R,
                    "rot": seg.rot, "seg": seg,
                })
        else:
            h = _fit_line_direction(xy, seg.start_idx, seg.end_idx)
            fitted.append({"type": "Line", "heading": h, "seg": seg})

    # ── Step 2: compute junction points ──────────────────────────────────
    junctions: list[np.ndarray | None] = [None] * (n + 1)
    junctions[0] = np.array(xy[0],  dtype=float)
    junctions[n] = np.array(xy[-1], dtype=float)

    for j in range(1, n):
        left  = fitted[j - 1]
        right = fitted[j]

        if left["type"] == "Line" and right["type"] == "Arc":
            # Junction = point on right arc where tangent = left line heading.
            # Refine heading iteratively (converges in 2-3 steps).
            phi = left["heading"]
            prev = junctions[j - 1]
            for _ in range(5):
                jx, jy = _arc_line_tangent_junction(
                    right["cx"], right["cy"], right["R"], right["rot"], phi
                )
                if prev is not None:
                    dx = jx - float(prev[0])
                    dy = jy - float(prev[1])
                    dist = math.hypot(dx, dy)
                    if dist > 1e-6:
                        new_phi = math.atan2(dy, dx)
                        if abs(new_phi - phi) < 1e-5:
                            break
                        phi = new_phi
                    else:
                        break
                else:
                    break
            junctions[j] = np.array([jx, jy])

        elif left["type"] == "Arc" and right["type"] == "Line":
            # Junction = point on left arc where tangent = right line heading.
            phi = right["heading"]
            jx, jy = _arc_line_tangent_junction(
                left["cx"], left["cy"], left["R"], left["rot"], phi
            )
            junctions[j] = np.array([jx, jy])

        else:
            # Arc-Arc or Line-Line: fall back to OSM boundary point
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)

    # ── Step 3: validate junctions (bounding-box sanity check) ───────────
    for j in range(1, n):
        if junctions[j] is None:
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)
            continue
        # Must lie within the OSM bounding box (with 500 m margin)
        seg_l = fitted[j - 1]["seg"]
        seg_r = fitted[j]["seg"]
        all_pts = xy[seg_l.start_idx: seg_r.end_idx + 1]
        if len(all_pts) == 0:
            continue
        margin = 500.0
        if not (all_pts[:, 0].min() - margin <= junctions[j][0] <= all_pts[:, 0].max() + margin
                and all_pts[:, 1].min() - margin <= junctions[j][1] <= all_pts[:, 1].max() + margin):
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)

    # ── Step 4: build elements from junctions ────────────────────────────
    elements: list[dict] = []
    sta = 0.0

    for i, f in enumerate(fitted):
        start_pt = junctions[i]
        end_pt   = junctions[i + 1]

        if f["type"] == "Line":
            seg_len = float(np.linalg.norm(end_pt - start_pt))
            if seg_len < 1e-6:
                continue
            heading = math.atan2(
                float(end_pt[1] - start_pt[1]),
                float(end_pt[0] - start_pt[0]),
            )
            elements.append({
                "type":          "Line",
                "sta_start":     sta,
                "length":        seg_len,
                "start":         start_pt.tolist(),
                "end":           end_pt.tolist(),
                "direction_rad": heading,
            })
            sta += seg_len

        else:  # Arc
            cx, cy, R = f["cx"], f["cy"], f["R"]
            rot  = f["rot"]
            sign = 1.0 if rot == "ccw" else -1.0

            a_s = math.atan2(float(start_pt[1]) - cy, float(start_pt[0]) - cx)
            a_e = math.atan2(float(end_pt[1])   - cy, float(end_pt[0])   - cx)

            # Wrap arc angle to correct direction
            delta = a_e - a_s
            if rot == "ccw":
                while delta <= 0.0:
                    delta += 2.0 * math.pi
            else:
                while delta >= 0.0:
                    delta -= 2.0 * math.pi

            # Sanity: arc must be < 180° for railway geometry
            if abs(delta) > math.pi:
                # Wrong solution — fall back to line chord
                chord = float(np.linalg.norm(end_pt - start_pt))
                if chord > 1e-6:
                    heading = math.atan2(
                        float(end_pt[1] - start_pt[1]),
                        float(end_pt[0] - start_pt[0]),
                    )
                    elements.append({
                        "type": "Line", "sta_start": sta, "length": chord,
                        "start": start_pt.tolist(), "end": end_pt.tolist(),
                        "direction_rad": heading,
                    })
                    sta += chord
                continue

            arc_len = R * abs(delta)
            chord   = float(np.linalg.norm(end_pt - start_pt))

            elements.append({
                "type":        "Arc",
                "sta_start":   sta,
                "length":      arc_len,
                "start":       start_pt.tolist(),
                "end":         end_pt.tolist(),
                "center":      [cx, cy],
                "radius":      R,
                "rot":         rot,
                "chord":       chord,
                "_deflection": sign * abs(delta),
            })
            sta += arc_len

    return elements


# ---------------------------------------------------------------------------
# Algorithm 2 helpers — Dynamic Programming Segmentation
# ---------------------------------------------------------------------------

def _line_sse(xy: np.ndarray, i0: int, i1: int) -> float:
    """Sum of squared perpendicular distances from OSM[i0..i1] to the best-fit line."""
    pts = xy[i0: i1 + 1]
    if len(pts) < 2:
        return 0.0
    centred = pts - pts.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centred, full_matrices=False)
        # Minor singular value² = sum of squared perp distances
        return float(s[1] ** 2) if len(s) > 1 else 0.0
    except np.linalg.LinAlgError:
        return float("inf")


def _arc_sse_and_fit(
    xy: np.ndarray, i0: int, i1: int, min_radius: float
) -> tuple[float, float, float, float]:
    """
    Fit a circle to OSM[i0..i1]; return (radial_SSE, cx, cy, R).
    Returns (inf, 0, 0, 0) if fewer than 3 points or fit fails.
    """
    from geometry.alignment import _fit_circle_kasa

    pts = xy[i0: i1 + 1]
    if len(pts) < 3:
        return float("inf"), 0.0, 0.0, 0.0
    cx, cy, R = _fit_circle_kasa(pts)
    if cx is None or R is None or not math.isfinite(R) or R < min_radius or R > 1e6:
        return float("inf"), 0.0, 0.0, 0.0
    dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    sse   = float(np.sum((dists - R) ** 2))
    return sse, float(cx), float(cy), float(R)


def _dp_segmentation(
    xy:           np.ndarray,
    chainages:    np.ndarray,
    lam:          float,
    min_radius:   float,
    progress_cb=None,
    time_budget_s: float = 60.0,
) -> list[_Segment]:
    """
    Imai-Iri DP: find the segmentation of OSM[0..N-1] minimising
        Σ SSE(segment) + λ × number_of_segments.

    lam controls the trade-off: larger → fewer, longer elements.
    To keep O(N²) tractable, spans longer than max_span points skip the arc fit.
    """
    t_dp_start = time.monotonic()

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    N = len(xy)
    if N < 2:
        return []

    # For very dense OSM data (< 2 m node spacing), subsample to ≤ 200 nodes
    # before building the O(N²) cost table.  The subsampled indices are later
    # mapped back when constructing _Segment objects.
    target_n = 200
    if N > target_n:
        step = max(1, N // target_n)
        sub_idx = list(range(0, N, step))
        if sub_idx[-1] != N - 1:
            sub_idx.append(N - 1)
        xy_dp  = xy[sub_idx]
        ch_dp  = chainages[sub_idx]
        # build a reverse map: subsampled index → original index
        orig_idx = sub_idx
        Ndp = len(xy_dp)
        _p(f"Subsampled {N} → {Ndp} nodes for cost table…")
    else:
        xy_dp    = xy
        ch_dp    = chainages
        orig_idx = list(range(N))
        Ndp      = N

    max_span = min(Ndp, 120)  # limit arc-fit window for performance

    # Pre-compute cost table on the (possibly subsampled) point set
    cost_val  = np.full((Ndp, Ndp), np.inf, dtype=float)
    cost_type = np.empty((Ndp, Ndp), dtype=object)
    arc_cx    = np.zeros((Ndp, Ndp), dtype=float)
    arc_cy    = np.zeros((Ndp, Ndp), dtype=float)
    arc_R_arr = np.zeros((Ndp, Ndp), dtype=float)
    arc_rot   = np.empty((Ndp, Ndp), dtype=object)

    _p(f"Building cost table (0/{Ndp} rows)…")
    last_report_i = -1
    report_every  = max(1, Ndp // 10)   # report ~10 times across the table

    for i in range(Ndp - 1):
        # Time-budget check (DP rarely exceeds it, but guard anyway)
        if time.monotonic() - t_dp_start > time_budget_s:
            _p(f"Time limit reached at row {i}/{Ndp}; using partial cost table.")
            break

        if i - last_report_i >= report_every:
            _p(f"Cost table: {i}/{Ndp} rows…")
            last_report_i = i

        for j in range(i + 1, min(i + max_span, Ndp)):
            l_sse = _line_sse(xy_dp, i, j)
            best_sse  = l_sse
            best_type = "Line"

            if j - i >= 3:
                defl = abs(_segment_deflection(xy_dp[i: j + 1]))
                if defl >= _MIN_ARC_DEFLECTION_RAD:
                    a_sse, cx, cy, R = _arc_sse_and_fit(xy_dp, i, j, min_radius)
                    sag = _arc_sagitta(R, defl) if R > 0 and math.isfinite(R) else 0.0
                    if a_sse < l_sse and sag >= 1.5:
                        best_sse  = a_sse
                        best_type = "Arc"
                        arc_cx[i, j] = cx
                        arc_cy[i, j] = cy
                        arc_R_arr[i, j] = R
                        raw_defl = _segment_deflection(xy_dp[i: j + 1])
                        arc_rot[i, j] = "ccw" if raw_defl >= 0 else "cw"

            cost_val[i, j]  = best_sse
            cost_type[i, j] = best_type

    _p("Running dynamic programming…")

    # Dynamic programming on the subsampled index space
    dp_cost = np.full(Ndp, np.inf, dtype=float)
    dp_prev = np.full(Ndp, -1,    dtype=int)
    dp_cost[0] = 0.0

    for j in range(1, Ndp):
        for i in range(max(0, j - max_span), j):
            if not math.isfinite(dp_cost[i]):
                continue
            c = dp_cost[i] + cost_val[i, j] + lam
            if c < dp_cost[j]:
                dp_cost[j] = c
                dp_prev[j] = i

    _p("Tracing optimal segmentation…")

    # Reconstruct breakpoints (in subsampled space)
    bp_sub: list[int] = []
    j = Ndp - 1
    while j > 0:
        bp_sub.append(j)
        p = dp_prev[j]
        if p < 0:
            break
        j = p
    bp_sub.reverse()

    if not bp_sub:
        bp_sub = [Ndp - 1]

    # Build a reverse map: original index → subsampled index
    orig_to_sub: dict[int, int] = {orig_idx[si]: si for si in range(Ndp)}

    # Map subsampled breakpoints back to original OSM indices, keeping pairs
    # (sub_j, orig_bp) so we can later look up cost_type / arc_R_arr.
    bp_pairs: list[tuple[int, int]] = [
        (sub_j, orig_idx[sub_j]) for sub_j in bp_sub
    ]

    # Build _Segment list using original xy / chainages
    segments: list[_Segment] = []
    prev_orig = 0
    prev_sub  = 0
    for sub_j, bp in bp_pairs:
        seg_type = cost_type[prev_sub, sub_j]
        if seg_type is None:
            seg_type = "Line"

        defl = _segment_deflection(xy[prev_orig: bp + 1])

        if seg_type == "Arc":
            R   = arc_R_arr[prev_sub, sub_j]
            rot = arc_rot[prev_sub, sub_j]
            if not isinstance(rot, str):
                rot = "ccw" if defl >= 0 else "cw"
            if _arc_sagitta(R, defl) < 1.5 or abs(defl) < _MIN_ARC_DEFLECTION_RAD:
                seg_type = "Line"
                R = math.inf
        else:
            R   = math.inf
            rot = "ccw" if defl >= 0 else "cw"

        segments.append(_Segment(
            seg_type=seg_type, start_idx=prev_orig, end_idx=bp,
            R_median=R, rot=rot, deflection=defl,
        ))
        prev_orig = bp
        prev_sub  = sub_j

    return segments


# ---------------------------------------------------------------------------
# Algorithm 3 helpers — Progressive MC
# ---------------------------------------------------------------------------

def _build_elements_from_boundaries(
    boundaries: list[int],
    types:      list[str],
    xy:         np.ndarray,
    chainages:  np.ndarray,
    min_radius: float,
) -> list[dict]:
    """
    Construct element list from boundary indices + type assignments.
    Arc elements are fitted to their OSM point range; endpoints anchored
    to xy[boundary].  Used during MC iterations (no tangent-point junctions
    here — those are applied in the final assembly pass).
    """
    elements: list[dict] = []
    sta = 0.0
    for k in range(len(boundaries) - 1):
        i0  = boundaries[k]
        i1  = boundaries[k + 1]
        typ = types[k]
        s_pt = np.array(xy[i0], dtype=float)
        e_pt = np.array(xy[i1], dtype=float)

        if typ == "Line" or i1 - i0 < 3:
            seg_len = float(chainages[i1] - chainages[i0])
            if seg_len < 1e-6:
                continue
            heading = math.atan2(
                float(e_pt[1] - s_pt[1]), float(e_pt[0] - s_pt[0])
            )
            elements.append({
                "type": "Line", "sta_start": sta, "length": seg_len,
                "start": s_pt.tolist(), "end": e_pt.tolist(),
                "direction_rad": heading,
            })
            sta += seg_len

        else:  # Arc
            result = _fit_arc_robust(xy, i0, i1, min_radius)
            if result is None:
                seg_len = float(chainages[i1] - chainages[i0])
                if seg_len < 1e-6:
                    continue
                heading = math.atan2(
                    float(e_pt[1] - s_pt[1]), float(e_pt[0] - s_pt[0])
                )
                elements.append({
                    "type": "Line", "sta_start": sta, "length": seg_len,
                    "start": s_pt.tolist(), "end": e_pt.tolist(),
                    "direction_rad": heading,
                })
                sta += seg_len
                continue

            cx, cy, R = result
            defl = _segment_deflection(xy[i0: i1 + 1])
            rot  = "ccw" if defl >= 0 else "cw"
            sign = 1.0 if rot == "ccw" else -1.0
            arc_len = R * abs(defl)
            chord   = float(np.linalg.norm(e_pt - s_pt))
            elements.append({
                "type":        "Arc",
                "sta_start":   sta,
                "length":      arc_len,
                "start":       s_pt.tolist(),
                "end":         e_pt.tolist(),
                "center":      [cx, cy],
                "radius":      R,
                "rot":         rot,
                "chord":       chord,
                "_deflection": sign * abs(defl),
            })
            sta += arc_len

    return elements


def _worst_osm_deviation(
    elements: list[dict], xy: np.ndarray, chainages: np.ndarray
) -> tuple[float, int]:
    """
    Return (max_deviation, worst_osm_idx) over all OSM points.

    Vectorised per-element: computes perpendicular distances for all points
    in an element's chainage window at once using numpy, which is significantly
    faster than the previous point-by-point scalar loop.
    """
    worst_dev = 0.0
    worst_idx = len(xy) // 2

    for el in elements:
        sta0 = el.get("sta_start", 0.0)
        sta1 = sta0 + el.get("length", 0.0)
        mask    = (chainages >= sta0 - 0.1) & (chainages <= sta1 + 0.1)
        idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            continue

        pts = xy[idx_arr]  # shape (K, 2)

        etype = el.get("type", "Line")
        if etype == "Arc":
            center = np.array(el["center"], dtype=float)
            r      = float(el.get("radius", 1.0))
            dists  = np.abs(np.linalg.norm(pts - center, axis=1) - r)
        else:
            # Line (or unknown — treat as Line)
            start  = np.array(el.get("start", [0.0, 0.0]), dtype=float)
            end    = np.array(el.get("end",   [0.0, 0.0]), dtype=float)
            seg    = end - start
            seg_sq = float(np.dot(seg, seg))
            if seg_sq < 1e-18:
                dists = np.linalg.norm(pts - start, axis=1)
            else:
                t    = np.dot(pts - start, seg) / seg_sq
                t    = np.clip(t, 0.0, 1.0)
                proj = start + t[:, np.newaxis] * seg
                dists = np.linalg.norm(pts - proj, axis=1)

        local_max = float(dists.max())
        if local_max > worst_dev:
            worst_dev = local_max
            worst_idx = int(idx_arr[int(dists.argmax())])

    return worst_dev, worst_idx


def _progressive_mc_build(
    xy:            np.ndarray,
    chainages:     np.ndarray,
    max_deviation: float,
    min_radius:    float,
    merge_pct:     float = 15.0,
    max_elements:  int   = 80,
    time_budget_s: float = 60.0,
    progress_cb=None,
    preview_cb=None,
    preview_interval_s: float = 7.0,
) -> list[dict]:
    """
    Progressive insertion with simulated annealing.

    Starts with a single Line element.  At each step, inserts a new boundary
    at the OSM point with the highest deviation; tries Line split, Arc
    conversion, and Line-to-Arc hybrid; keeps the option that most reduces
    max deviation.

    Every 10 insertions a SA perturbation randomly shifts boundary indices
    by ±3 points, accepting moves probabilistically to escape local minima.

    On completion, the final boundary/type assignment is assembled with
    tangent-point junctions (via _connect_segments_tangent).

    progress_cb(msg: str)        — called at each iteration with status text
    preview_cb(elements: list)   — called every ~preview_interval_s seconds with
                                   a preliminary element list for map visualisation
    """
    N = len(xy)
    if N < 2:
        return []

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    rng = np.random.default_rng(42)
    t_start        = time.monotonic()
    t_last_preview = t_start

    # Initial state: one Line covering everything
    boundaries: list[int] = [0, N - 1]
    types:      list[str] = ["Line"]

    T = max(max_deviation * 2.0, 0.5)   # SA initial temperature

    _p("Starting — 1 element, evaluating…")

    iteration = 0
    while True:
        # ── time / size budget ───────────────────────────────────────────
        elapsed = time.monotonic() - t_start
        if elapsed > time_budget_s:
            _p(f"Time limit reached ({time_budget_s:.0f} s). Finalising…")
            break
        if len(boundaries) >= max_elements + 1:
            _p(f"Element limit reached ({max_elements}). Finalising…")
            break

        # ── evaluate current state ───────────────────────────────────────
        elements = _build_elements_from_boundaries(
            boundaries, types, xy, chainages, min_radius
        )
        if not elements:
            break
        worst_dev, worst_idx = _worst_osm_deviation(elements, xy, chainages)

        n_el = len(boundaries) - 1
        _p(
            f"Iteration {iteration + 1}  |  {n_el} element{'s' if n_el != 1 else ''}"
            f"  |  max deviation {worst_dev:.3f} m"
            f"  |  {elapsed:.0f}/{time_budget_s:.0f} s"
        )

        # ── periodic preview for map visualisation ───────────────────────
        if preview_cb is not None:
            now = time.monotonic()
            if now - t_last_preview >= preview_interval_s:
                try:
                    # Build a quick tangent-junction version of current state
                    _preview_segs: list[_Segment] = []
                    for k in range(len(boundaries) - 1):
                        i0p, i1p = boundaries[k], boundaries[k + 1]
                        typ_p  = types[k]
                        defl_p = _segment_deflection(xy[i0p: i1p + 1])
                        rot_p  = "ccw" if defl_p >= 0 else "cw"
                        R_p    = math.inf
                        if typ_p == "Arc":
                            res = _fit_arc_robust(xy, i0p, i1p, min_radius)
                            if res:
                                R_p = res[2]
                            else:
                                typ_p = "Line"
                        _preview_segs.append(_Segment(
                            seg_type=typ_p, start_idx=i0p, end_idx=i1p,
                            R_median=R_p, rot=rot_p, deflection=defl_p,
                        ))
                    preview_elements = _connect_segments_tangent(
                        _preview_segs, xy, chainages, min_radius
                    )
                    preview_cb(preview_elements)
                except Exception:
                    pass
                t_last_preview = now

        if worst_dev <= max_deviation:
            _p(f"Converged! Max deviation {worst_dev:.3f} m within {max_deviation:.3f} m target")
            break

        # ── find which segment contains the worst point ──────────────────
        seg_k = None
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= worst_idx <= boundaries[k + 1]:
                seg_k = k
                break
        if seg_k is None:
            break

        i0, i1 = boundaries[seg_k], boundaries[seg_k + 1]

        if i1 - i0 < 2:
            # Cannot split further; try SA perturbation instead
            if T < 1e-3:
                break
            _do_sa_perturbation(boundaries, types, xy, chainages, min_radius,
                                worst_dev, rng, T)
            T *= 0.90
            iteration += 1
            continue

        # ── generate candidate moves ─────────────────────────────────────
        mid = max(i0 + 1, min(i1 - 1, worst_idx))

        best_dev   = worst_dev
        best_bdry  = boundaries[:]
        best_types = types[:]

        # Move A: split into two Lines
        bdry_a = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
        typ_a  = types[:seg_k] + ["Line", "Line"] + types[seg_k + 1:]
        dev_a, _ = _worst_osm_deviation(
            _build_elements_from_boundaries(bdry_a, typ_a, xy, chainages, min_radius),
            xy, chainages,
        )
        if dev_a < best_dev:
            best_dev, best_bdry, best_types = dev_a, bdry_a, typ_a

        # Move B: convert entire segment to Arc
        defl_full = _segment_deflection(xy[i0: i1 + 1])
        if (i1 - i0 >= 3
                and abs(defl_full) >= _MIN_ARC_DEFLECTION_RAD
                and types[seg_k] != "Arc"):
            typ_b = types[:seg_k] + ["Arc"] + types[seg_k + 1:]
            dev_b, _ = _worst_osm_deviation(
                _build_elements_from_boundaries(boundaries, typ_b, xy, chainages, min_radius),
                xy, chainages,
            )
            if dev_b < best_dev:
                best_dev, best_bdry, best_types = dev_b, boundaries[:], typ_b

        # Move C: split into Line + Arc (left half as Line, right as Arc)
        if mid - i0 >= 3:
            defl_r = _segment_deflection(xy[mid: i1 + 1])
            if abs(defl_r) >= _MIN_ARC_DEFLECTION_RAD:
                bdry_c = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
                typ_c  = types[:seg_k] + ["Line", "Arc"] + types[seg_k + 1:]
                dev_c, _ = _worst_osm_deviation(
                    _build_elements_from_boundaries(bdry_c, typ_c, xy, chainages, min_radius),
                    xy, chainages,
                )
                if dev_c < best_dev:
                    best_dev, best_bdry, best_types = dev_c, bdry_c, typ_c

        # Move D: split into Arc + Line (left half as Arc, right as Line)
        if i1 - mid >= 3:
            defl_l = _segment_deflection(xy[i0: mid + 1])
            if abs(defl_l) >= _MIN_ARC_DEFLECTION_RAD:
                bdry_d = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
                typ_d  = types[:seg_k] + ["Arc", "Line"] + types[seg_k + 1:]
                dev_d, _ = _worst_osm_deviation(
                    _build_elements_from_boundaries(bdry_d, typ_d, xy, chainages, min_radius),
                    xy, chainages,
                )
                if dev_d < best_dev:
                    best_dev, best_bdry, best_types = dev_d, bdry_d, typ_d

        boundaries = best_bdry
        types      = best_types

        # ── periodic SA perturbation ─────────────────────────────────────
        if iteration % 10 == 9 and T > 1e-3:
            _do_sa_perturbation(boundaries, types, xy, chainages, min_radius,
                                best_dev, rng, T)
            T *= 0.95

        iteration += 1

    _p("Assembling final elements with tangent junctions…")

    # ── final assembly with tangent-point junctions ───────────────────────
    segments: list[_Segment] = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        typ  = types[k]
        defl = _segment_deflection(xy[i0: i1 + 1])
        rot  = "ccw" if defl >= 0 else "cw"

        R = math.inf
        if typ == "Arc":
            result = _fit_arc_robust(xy, i0, i1, min_radius)
            if result:
                R = result[2]
            else:
                typ = "Line"
                R   = math.inf

        segments.append(_Segment(
            seg_type=typ, start_idx=i0, end_idx=i1,
            R_median=R, rot=rot, deflection=defl,
        ))

    return _connect_segments_tangent(segments, xy, chainages, min_radius)


def _do_sa_perturbation(
    boundaries: list[int],
    types:      list[str],
    xy:         np.ndarray,
    chainages:  np.ndarray,
    min_radius: float,
    current_dev: float,
    rng:        np.random.Generator,
    T:          float,
) -> None:
    """In-place SA: randomly shift one interior boundary by ±1–3 indices."""
    if len(boundaries) <= 2:
        return
    j    = int(rng.integers(1, len(boundaries) - 1))
    step = int(rng.integers(-3, 4))
    if step == 0:
        return
    new_b = int(boundaries[j]) + step
    new_b = max(int(boundaries[j - 1]) + 1, min(int(boundaries[j + 1]) - 1, new_b))
    if new_b == boundaries[j]:
        return
    old_b        = boundaries[j]
    boundaries[j] = new_b
    new_elements = _build_elements_from_boundaries(
        boundaries, types, xy, chainages, min_radius
    )
    new_dev, _ = _worst_osm_deviation(new_elements, xy, chainages)
    delta = new_dev - current_dev
    if delta > 0:
        # Worsening move: accept with SA probability
        if rng.random() >= math.exp(-delta / max(T, 1e-9)):
            boundaries[j] = old_b   # reject


# ---------------------------------------------------------------------------
# Piecewise MC helpers
# ---------------------------------------------------------------------------

def _place_anchors(chainages: np.ndarray, division_length: float) -> list[int]:
    """
    Return OSM node indices used as window boundaries.

    Always includes 0 and N-1.  Between them, selects the node closest
    to each successive multiple of division_length (< total length).
    Never adds two consecutive identical indices.
    """
    N = len(chainages)
    if N < 2 or division_length <= 0:
        return [0, N - 1]
    ch0    = float(chainages[0])
    ch_end = float(chainages[-1])
    if ch_end - ch0 <= division_length:
        return [0, N - 1]

    anchors = [0]
    k_mult  = 1
    while True:
        target = ch0 + k_mult * division_length
        if target >= ch_end:
            break
        j = int(np.searchsorted(chainages, target))
        j = max(1, min(N - 2, j))
        # Pick j-1 if it's closer to the target
        if j > 0 and abs(float(chainages[j - 1]) - target) < abs(float(chainages[j]) - target):
            j -= 1
        if j != anchors[-1]:
            anchors.append(j)
        k_mult += 1

    if anchors[-1] != N - 1:
        anchors.append(N - 1)
    return anchors


def _anchor_tangent(xy: np.ndarray, anchor_idx: int, half_window: int = 5) -> float:
    """
    Estimate OSM travel-direction heading at anchor_idx via local SVD fit
    over the ±half_window neighbouring nodes.
    """
    i0 = max(0, anchor_idx - half_window)
    i1 = min(len(xy) - 1, anchor_idx + half_window)
    return _fit_line_direction(xy, i0, i1)


def _connect_segments_tangent_constrained(
    segments:      list[_Segment],
    xy:            np.ndarray,
    chainages:     np.ndarray,
    min_radius:    float,
    entry_heading: float | None = None,
    exit_heading:  float | None = None,
) -> list[dict]:
    """
    Like _connect_segments_tangent but forces the heading of the first / last
    primitive to match entry_heading / exit_heading (C1 at window edges).

    For Line primitives: the fitted heading is simply overridden.
    For Arc primitives:  the Kasa-fitted radius is kept; the center is
    relocated so that the tangent at the boundary OSM point equals the
    required heading:
        CCW: center = P + R*(sin φ, −cos φ)
        CW:  center = P + R*(−sin φ, +cos φ)
    """
    if not segments:
        return []

    n = len(segments)

    # ── Step 1: fit geometric primitives ─────────────────────────────────
    fitted: list[dict] = []
    for seg in segments:
        if seg.seg_type == "Arc":
            result = _fit_arc_robust(xy, seg.start_idx, seg.end_idx, min_radius)
            if result is None:
                h = _fit_line_direction(xy, seg.start_idx, seg.end_idx)
                fitted.append({"type": "Line", "heading": h, "seg": seg})
            else:
                cx, cy, R = result
                fitted.append({
                    "type": "Arc", "cx": cx, "cy": cy, "R": R,
                    "rot": seg.rot, "seg": seg,
                })
        else:
            h = _fit_line_direction(xy, seg.start_idx, seg.end_idx)
            fitted.append({"type": "Line", "heading": h, "seg": seg})

    # ── Apply entry_heading constraint ───────────────────────────────────
    if entry_heading is not None and fitted:
        f0 = fitted[0]
        if f0["type"] == "Line":
            f0["heading"] = entry_heading
        else:
            R   = f0["R"]
            rot = f0["rot"]
            sp  = math.sin(entry_heading)
            cp  = math.cos(entry_heading)
            p   = xy[segments[0].start_idx]
            if rot == "ccw":
                f0["cx"] = float(p[0]) + R * sp
                f0["cy"] = float(p[1]) - R * cp
            else:
                f0["cx"] = float(p[0]) - R * sp
                f0["cy"] = float(p[1]) + R * cp

    # ── Apply exit_heading constraint ────────────────────────────────────
    if exit_heading is not None and fitted:
        fm = fitted[-1]
        if fm["type"] == "Line":
            fm["heading"] = exit_heading
        else:
            R   = fm["R"]
            rot = fm["rot"]
            sp  = math.sin(exit_heading)
            cp  = math.cos(exit_heading)
            p   = xy[segments[-1].end_idx]
            if rot == "ccw":
                fm["cx"] = float(p[0]) + R * sp
                fm["cy"] = float(p[1]) - R * cp
            else:
                fm["cx"] = float(p[0]) - R * sp
                fm["cy"] = float(p[1]) + R * cp

    # ── Step 2: compute junction points ──────────────────────────────────
    junctions: list[np.ndarray | None] = [None] * (n + 1)
    junctions[0] = np.array(xy[segments[0].start_idx], dtype=float)
    junctions[n] = np.array(xy[segments[-1].end_idx],  dtype=float)

    for j in range(1, n):
        left  = fitted[j - 1]
        right = fitted[j]

        if left["type"] == "Line" and right["type"] == "Arc":
            phi  = left["heading"]
            prev = junctions[j - 1]
            for _ in range(5):
                jx, jy = _arc_line_tangent_junction(
                    right["cx"], right["cy"], right["R"], right["rot"], phi
                )
                if prev is not None:
                    dx   = jx - float(prev[0])
                    dy   = jy - float(prev[1])
                    dist = math.hypot(dx, dy)
                    if dist > 1e-6:
                        new_phi = math.atan2(dy, dx)
                        if abs(new_phi - phi) < 1e-5:
                            break
                        phi = new_phi
                    else:
                        break
                else:
                    break
            junctions[j] = np.array([jx, jy])

        elif left["type"] == "Arc" and right["type"] == "Line":
            phi = right["heading"]
            jx, jy = _arc_line_tangent_junction(
                left["cx"], left["cy"], left["R"], left["rot"], phi
            )
            junctions[j] = np.array([jx, jy])

        else:
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)

    # ── Step 3: validate junctions ───────────────────────────────────────
    for j in range(1, n):
        if junctions[j] is None:
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)
            continue
        seg_l   = fitted[j - 1]["seg"]
        seg_r   = fitted[j]["seg"]
        all_pts = xy[seg_l.start_idx: seg_r.end_idx + 1]
        if len(all_pts) == 0:
            continue
        margin = 500.0
        if not (all_pts[:, 0].min() - margin <= junctions[j][0] <= all_pts[:, 0].max() + margin
                and all_pts[:, 1].min() - margin <= junctions[j][1] <= all_pts[:, 1].max() + margin):
            bdy = fitted[j - 1]["seg"].end_idx
            junctions[j] = np.array(xy[bdy], dtype=float)

    # ── Step 4: build elements from junctions ────────────────────────────
    elements: list[dict] = []
    sta = 0.0

    for i, f in enumerate(fitted):
        start_pt = junctions[i]
        end_pt   = junctions[i + 1]

        if f["type"] == "Line":
            seg_len = float(np.linalg.norm(end_pt - start_pt))
            if seg_len < 1e-6:
                continue
            heading = math.atan2(
                float(end_pt[1] - start_pt[1]),
                float(end_pt[0] - start_pt[0]),
            )
            elements.append({
                "type":          "Line",
                "sta_start":     sta,
                "length":        seg_len,
                "start":         start_pt.tolist(),
                "end":           end_pt.tolist(),
                "direction_rad": heading,
            })
            sta += seg_len

        else:  # Arc
            cx, cy, R = f["cx"], f["cy"], f["R"]
            rot  = f["rot"]
            sign = 1.0 if rot == "ccw" else -1.0

            a_s = math.atan2(float(start_pt[1]) - cy, float(start_pt[0]) - cx)
            a_e = math.atan2(float(end_pt[1])   - cy, float(end_pt[0])   - cx)
            delta = a_e - a_s
            if rot == "ccw":
                while delta <= 0.0:
                    delta += 2.0 * math.pi
            else:
                while delta >= 0.0:
                    delta -= 2.0 * math.pi

            if abs(delta) > math.pi:
                chord = float(np.linalg.norm(end_pt - start_pt))
                if chord > 1e-6:
                    heading = math.atan2(
                        float(end_pt[1] - start_pt[1]),
                        float(end_pt[0] - start_pt[0]),
                    )
                    elements.append({
                        "type": "Line", "sta_start": sta, "length": chord,
                        "start": start_pt.tolist(), "end": end_pt.tolist(),
                        "direction_rad": heading,
                    })
                    sta += chord
                continue

            arc_len = R * abs(delta)
            chord   = float(np.linalg.norm(end_pt - start_pt))
            elements.append({
                "type":        "Arc",
                "sta_start":   sta,
                "length":      arc_len,
                "start":       start_pt.tolist(),
                "end":         end_pt.tolist(),
                "center":      [cx, cy],
                "radius":      R,
                "rot":         rot,
                "chord":       chord,
                "_deflection": sign * abs(delta),
            })
            sta += arc_len

    return elements


def _mc_window_build(
    xy_win:        np.ndarray,
    chainages_win: np.ndarray,
    entry_heading: float,
    exit_heading:  float,
    max_deviation: float,
    min_radius:    float,
    merge_pct:     float,
    time_budget_s: float,
    seed:          int  = 42,
    max_elements:  int  = 80,
    progress_cb=None,
) -> list[dict]:
    """
    Run the greedy MC insertion loop on a single window.

    Identical mechanics to _progressive_mc_build but operates on local
    xy_win / chainages_win so all boundary indices are 0-relative.
    Final assembly calls _connect_segments_tangent_constrained to enforce
    C1 at both window edges.  Returns elements whose sta_start values
    start from 0.0 (the caller adds the chainage offset when stitching).
    """
    N_win = len(xy_win)
    if N_win < 2:
        return []

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    rng = np.random.default_rng(seed)
    t_start = time.monotonic()

    boundaries: list[int] = [0, N_win - 1]
    types:      list[str] = ["Line"]
    T = max(max_deviation * 2.0, 0.5)

    iteration = 0
    while True:
        elapsed = time.monotonic() - t_start
        if elapsed > time_budget_s:
            break
        if len(boundaries) >= max_elements + 1:
            break

        elements = _build_elements_from_boundaries(
            boundaries, types, xy_win, chainages_win, min_radius
        )
        if not elements:
            break
        worst_dev, worst_idx = _worst_osm_deviation(elements, xy_win, chainages_win)

        n_el = len(boundaries) - 1
        _p(
            f"Iter {iteration + 1}  |  {n_el} element{'s' if n_el != 1 else ''}"
            f"  |  dev {worst_dev:.3f} m"
            f"  |  {elapsed:.0f}/{time_budget_s:.0f} s"
        )

        if worst_dev <= max_deviation:
            break

        seg_k = None
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= worst_idx <= boundaries[k + 1]:
                seg_k = k
                break
        if seg_k is None:
            break

        i0, i1 = boundaries[seg_k], boundaries[seg_k + 1]

        if i1 - i0 < 2:
            if T < 1e-3:
                break
            _do_sa_perturbation(boundaries, types, xy_win, chainages_win, min_radius,
                                worst_dev, rng, T)
            T *= 0.90
            iteration += 1
            continue

        mid = max(i0 + 1, min(i1 - 1, worst_idx))

        best_dev   = worst_dev
        best_bdry  = boundaries[:]
        best_types = types[:]

        # Move A: split into two Lines
        bdry_a = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
        typ_a  = types[:seg_k] + ["Line", "Line"] + types[seg_k + 1:]
        dev_a, _ = _worst_osm_deviation(
            _build_elements_from_boundaries(bdry_a, typ_a, xy_win, chainages_win, min_radius),
            xy_win, chainages_win,
        )
        if dev_a < best_dev:
            best_dev, best_bdry, best_types = dev_a, bdry_a, typ_a

        # Move B: convert segment to Arc
        defl_full = _segment_deflection(xy_win[i0: i1 + 1])
        if (i1 - i0 >= 3
                and abs(defl_full) >= _MIN_ARC_DEFLECTION_RAD
                and types[seg_k] != "Arc"):
            typ_b = types[:seg_k] + ["Arc"] + types[seg_k + 1:]
            dev_b, _ = _worst_osm_deviation(
                _build_elements_from_boundaries(boundaries, typ_b, xy_win, chainages_win, min_radius),
                xy_win, chainages_win,
            )
            if dev_b < best_dev:
                best_dev, best_bdry, best_types = dev_b, boundaries[:], typ_b

        # Move C: Line + Arc
        if mid - i0 >= 3:
            defl_r = _segment_deflection(xy_win[mid: i1 + 1])
            if abs(defl_r) >= _MIN_ARC_DEFLECTION_RAD:
                bdry_c = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
                typ_c  = types[:seg_k] + ["Line", "Arc"] + types[seg_k + 1:]
                dev_c, _ = _worst_osm_deviation(
                    _build_elements_from_boundaries(bdry_c, typ_c, xy_win, chainages_win, min_radius),
                    xy_win, chainages_win,
                )
                if dev_c < best_dev:
                    best_dev, best_bdry, best_types = dev_c, bdry_c, typ_c

        # Move D: Arc + Line
        if i1 - mid >= 3:
            defl_l = _segment_deflection(xy_win[i0: mid + 1])
            if abs(defl_l) >= _MIN_ARC_DEFLECTION_RAD:
                bdry_d = boundaries[:seg_k + 1] + [mid] + boundaries[seg_k + 1:]
                typ_d  = types[:seg_k] + ["Arc", "Line"] + types[seg_k + 1:]
                dev_d, _ = _worst_osm_deviation(
                    _build_elements_from_boundaries(bdry_d, typ_d, xy_win, chainages_win, min_radius),
                    xy_win, chainages_win,
                )
                if dev_d < best_dev:
                    best_dev, best_bdry, best_types = dev_d, bdry_d, typ_d

        boundaries = best_bdry
        types      = best_types

        if iteration % 10 == 9 and T > 1e-3:
            _do_sa_perturbation(boundaries, types, xy_win, chainages_win, min_radius,
                                best_dev, rng, T)
            T *= 0.95

        iteration += 1

    # Final assembly with constrained tangent-point junctions
    segments: list[_Segment] = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        typ  = types[k]
        defl = _segment_deflection(xy_win[i0: i1 + 1])
        rot  = "ccw" if defl >= 0 else "cw"
        R    = math.inf
        if typ == "Arc":
            result = _fit_arc_robust(xy_win, i0, i1, min_radius)
            if result:
                R = result[2]
            else:
                typ = "Line"
        segments.append(_Segment(
            seg_type=typ, start_idx=i0, end_idx=i1,
            R_median=R, rot=rot, deflection=defl,
        ))

    return _connect_segments_tangent_constrained(
        segments, xy_win, chainages_win, min_radius,
        entry_heading=entry_heading,
        exit_heading=exit_heading,
    )


def _progressive_mc_build_piecewise(
    xy:              np.ndarray,
    chainages:       np.ndarray,
    max_deviation:   float,
    min_radius:      float,
    merge_pct:       float = 15.0,
    max_elements:    int   = 80,
    time_budget_s:   float = 60.0,
    division_length: float = 500.0,
    progress_cb=None,
    preview_cb=None,
    preview_interval_s: float = 7.0,
) -> list[dict]:
    """
    Piecewise MC with anchor boundary constraints.

    Divides the OSM polyline into ~division_length windows at the OSM nodes
    closest to each multiple of division_length.  Runs _mc_window_build
    independently in each window with the time budget split evenly.
    C0 + C1 continuity at every inter-window boundary is enforced via
    _connect_segments_tangent_constrained.
    """
    N = len(xy)
    if N < 2:
        return []

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    anchors   = _place_anchors(chainages, division_length)
    n_windows = len(anchors) - 1

    # Compute OSM heading at every anchor node
    tangents = [_anchor_tangent(xy, anchors[k]) for k in range(len(anchors))]

    budget_per_window = time_budget_s / max(n_windows, 1)

    _p(
        f"Piecewise MC: {n_windows} window{'s' if n_windows != 1 else ''},"
        f" {budget_per_window:.0f} s each\u2026"
    )

    all_elements:    list[dict] = []
    t_last_preview = time.monotonic()

    for i in range(n_windows):
        a0, a1 = anchors[i], anchors[i + 1]
        xy_win        = xy[a0: a1 + 1]
        chainages_win = chainages[a0: a1 + 1] - chainages[a0]

        sta0_str = f"{float(chainages[a0]):.0f}"
        sta1_str = f"{float(chainages[a1]):.0f}"
        _p(f"Window {i + 1}/{n_windows}  ({sta0_str}\u2013{sta1_str} m)\u2026")

        def _win_pcb(msg, _wi=i, _nw=n_windows):
            _p(f"  [{_wi + 1}/{_nw}] {msg}")

        win_elements = _mc_window_build(
            xy_win, chainages_win,
            entry_heading = tangents[i],
            exit_heading  = tangents[i + 1],
            max_deviation = max_deviation,
            min_radius    = min_radius,
            merge_pct     = merge_pct,
            time_budget_s = budget_per_window,
            seed          = 42 + i,
            max_elements  = max_elements,
            progress_cb   = _win_pcb,
        )

        # Shift sta_start by the accumulated chainage offset
        sta_off = float(chainages[a0])
        for el in win_elements:
            el["sta_start"] += sta_off

        all_elements.extend(win_elements)

        # Emit preview after each window (or when interval elapsed)
        if preview_cb is not None:
            now = time.monotonic()
            if now - t_last_preview >= preview_interval_s or i == n_windows - 1:
                try:
                    preview_cb(all_elements[:])
                except Exception:
                    pass
                t_last_preview = now

    return all_elements


# ---------------------------------------------------------------------------
# Post-processing: Arc-Line-Arc consolidation
# ---------------------------------------------------------------------------

def _consolidate_arc_line_arc(
    elements:           list[dict],
    xy:                 np.ndarray,
    chainages:          np.ndarray,
    min_tangent_length: float,
    min_radius:         float,
) -> list[dict]:
    """
    Iteratively scan elements for [Arc][short Line][Arc] (same sense only)
    and merge the triple into a single Arc.

    Skips opposite-sense pairs (S-curves) to preserve their return tangent.
    Stops when no further merges are possible or min_tangent_length <= 0.
    """
    if min_tangent_length <= 0:
        return elements

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(elements) - 2:
            A = elements[i]
            L = elements[i + 1]
            B = elements[i + 2]
            if (A.get("type") == "Arc"
                    and L.get("type") == "Line"
                    and B.get("type") == "Arc"
                    and L.get("length", 0.0) < min_tangent_length
                    and A.get("rot") == B.get("rot")):
                merged = _try_merge_arc_line_arc(elements, i, xy, chainages, min_radius)
                if merged is not None:
                    elements = merged
                    changed  = True
                    i        = max(0, i - 1)
                    continue
            i += 1
    return elements


def _try_merge_arc_line_arc(
    elements:   list[dict],
    idx:        int,
    xy:         np.ndarray,
    chainages:  np.ndarray,
    min_radius: float,
) -> list[dict] | None:
    """
    Try to replace elements[idx], elements[idx+1], elements[idx+2] with a
    single merged Arc.  Returns the updated list on success, None otherwise.
    """
    from geometry.alignment import _fit_circle_kasa

    A = elements[idx]
    B = elements[idx + 2]

    sta_start_A = A.get("sta_start", 0.0)
    sta_end_B   = B.get("sta_start", 0.0) + B.get("length", 0.0)

    # Collect OSM points under the triple
    mask    = (chainages >= sta_start_A - 0.1) & (chainages <= sta_end_B + 0.1)
    all_pts = xy[mask]
    if len(all_pts) < 3:
        return None

    # Fit a single circle to the combined point set
    cx, cy, R_fit = _fit_circle_kasa(all_pts)
    if cx is None or R_fit is None or not math.isfinite(float(R_fit)):
        return None
    R_fit = float(R_fit)
    cx, cy = float(cx), float(cy)
    if R_fit < min_radius or R_fit > 1e6:
        return None

    # Determine rotation from total deflection (must be consistent with Arc_A)
    defl = _segment_deflection(all_pts)
    rot  = A.get("rot", "ccw" if defl >= 0 else "cw")

    # Compute left junction (entry into merged arc)
    if idx > 0 and elements[idx - 1].get("type") == "Line":
        left_heading = elements[idx - 1].get("direction_rad", 0.0)
        jx, jy = _arc_line_tangent_junction(cx, cy, R_fit, rot, left_heading)
    else:
        sp = A.get("start", [0.0, 0.0])
        jx, jy = float(sp[0]), float(sp[1])

    # Compute right junction (exit from merged arc)
    if idx + 3 < len(elements) and elements[idx + 3].get("type") == "Line":
        right_heading = elements[idx + 3].get("direction_rad", 0.0)
        jx2, jy2 = _arc_line_tangent_junction(cx, cy, R_fit, rot, right_heading)
    else:
        ep = B.get("end", [0.0, 0.0])
        jx2, jy2 = float(ep[0]), float(ep[1])

    start_pt = np.array([jx,  jy],  dtype=float)
    end_pt   = np.array([jx2, jy2], dtype=float)

    # Build the merged arc's angular span
    a_s   = math.atan2(float(start_pt[1]) - cy, float(start_pt[0]) - cx)
    a_e   = math.atan2(float(end_pt[1])   - cy, float(end_pt[0])   - cx)
    delta = a_e - a_s
    if rot == "ccw":
        while delta <= 0.0:
            delta += 2.0 * math.pi
    else:
        while delta >= 0.0:
            delta -= 2.0 * math.pi

    if abs(delta) > math.pi:
        return None   # pathological geometry — skip

    sign    = 1.0 if rot == "ccw" else -1.0
    arc_len = R_fit * abs(delta)
    chord   = float(np.linalg.norm(end_pt - start_pt))

    merged_el: dict = {
        "type":        "Arc",
        "sta_start":   sta_start_A,
        "length":      arc_len,
        "start":       start_pt.tolist(),
        "end":         end_pt.tolist(),
        "center":      [cx, cy],
        "radius":      R_fit,
        "rot":         rot,
        "chord":       chord,
        "_deflection": sign * abs(delta),
    }

    # Assemble the new element list
    new_elements: list[dict] = elements[:idx] + [merged_el] + elements[idx + 3:]

    # Update the element immediately before if it is a Line
    if idx > 0 and new_elements[idx - 1].get("type") == "Line":
        prev_el = dict(new_elements[idx - 1])
        sp_prev = np.array(prev_el["start"], dtype=float)
        prev_el["end"]    = start_pt.tolist()
        prev_el["length"] = float(np.linalg.norm(start_pt - sp_prev))
        new_elements[idx - 1] = prev_el

    # Update the element immediately after if it is a Line
    after_idx = idx + 1   # merged_el sits at idx; the element after is idx+1
    if after_idx < len(new_elements) and new_elements[after_idx].get("type") == "Line":
        next_el = dict(new_elements[after_idx])
        ep_next = np.array(next_el["end"], dtype=float)
        next_el["start"]  = end_pt.tolist()
        next_el["length"] = float(np.linalg.norm(ep_next - end_pt))
        new_elements[after_idx] = next_el

    # Recompute sta_start for every element from idx onward
    if idx > 0:
        sta_running = (new_elements[idx - 1].get("sta_start", 0.0)
                       + new_elements[idx - 1].get("length",   0.0))
    else:
        sta_running = 0.0
    for k in range(idx, len(new_elements)):
        new_elements[k] = dict(new_elements[k])
        new_elements[k]["sta_start"] = sta_running
        sta_running += new_elements[k].get("length", 0.0)

    return new_elements


# ---------------------------------------------------------------------------
# C1 continuity post-processing
# ---------------------------------------------------------------------------

def _arc_tangent_heading(cx: float, cy: float, px: float, py: float, rot: str) -> float:
    """
    Tangent heading (rad) at point (px, py) on an arc with center (cx, cy).

    CCW arc:  tangent direction = (-sin θ, cos θ)  where θ = atan2(py-cy, px-cx)
              ⟹  atan2(dx, -dy)
    CW  arc:  tangent direction = ( sin θ, -cos θ)
              ⟹  atan2(-dx, dy)
    """
    dx = px - cx
    dy = py - cy
    if rot == "ccw":
        return math.atan2(dx, -dy)
    else:
        return math.atan2(-dx, dy)


def _compute_forced_line_ranges(
    xy:               np.ndarray,
    chainages:        np.ndarray,
    smooth_window:    int,
    min_kappa_radius: float,
    min_kappa_length: float,
) -> list[tuple[float, float]]:
    """
    Return list of (sta_start, sta_end) chainage pairs where smoothed |κ|
    is below 1/min_kappa_radius for at least min_kappa_length metres.

    Returns [] if min_kappa_radius <= 0 or min_kappa_length <= 0.
    """
    from geometry.curvature import compute_curvature, smooth_curvature

    if min_kappa_radius <= 0 or min_kappa_length <= 0:
        return []
    N = len(xy)
    if N < 3:
        return []

    threshold = 1.0 / min_kappa_radius
    kappa        = compute_curvature(xy)
    kappa_smooth = smooth_curvature(kappa, window=smooth_window)

    forced: list[tuple[float, float]] = []
    in_run    = False
    run_start = 0

    for i in range(N):
        if abs(float(kappa_smooth[i])) < threshold:
            if not in_run:
                in_run    = True
                run_start = i
        else:
            if in_run:
                in_run = False
                span = float(chainages[i - 1]) - float(chainages[run_start])
                if span >= min_kappa_length:
                    forced.append((
                        float(chainages[run_start]),
                        float(chainages[i - 1]),
                    ))

    if in_run:
        span = float(chainages[N - 1]) - float(chainages[run_start])
        if span >= min_kappa_length:
            forced.append((
                float(chainages[run_start]),
                float(chainages[N - 1]),
            ))

    return forced


def _merge_adjacent_line_elements(elements: list[dict]) -> list[dict]:
    """
    Merge any two or more consecutive Line elements into a single Line.
    Recomputes length, heading, and sta_start chain after merging.
    """
    if not elements:
        return elements

    merged: list[dict] = []
    for el in elements:
        if (merged
                and el.get("type") == "Line"
                and merged[-1].get("type") == "Line"):
            prev    = merged[-1]
            new_end = el["end"]
            sp      = np.array(prev["start"], dtype=float)
            ep      = np.array(new_end,       dtype=float)
            prev["end"]           = new_end
            prev["length"]        = float(np.linalg.norm(ep - sp))
            prev["direction_rad"] = math.atan2(
                float(ep[1] - sp[1]), float(ep[0] - sp[0])
            )
        else:
            merged.append(dict(el))

    # Recompute sta_start chain
    sta = 0.0
    for el in merged:
        el["sta_start"] = sta
        sta += el.get("length", 0.0)

    return merged


def _insert_line_line_connector_arc(
    elements:   list[dict],
    idx:        int,
    min_radius: float,
) -> list[dict] | None:
    """
    Insert a small circular arc of radius min_radius between the two consecutive
    Line elements at elements[idx] and elements[idx+1] to achieve C1 continuity.

    Returns the modified element list, or None if the arc cannot be fitted
    (collinear lines, or the tangent runout does not fit inside both elements).
    """
    A     = elements[idx]
    B     = elements[idx + 1]
    phi_A = A.get("direction_rad", 0.0)
    phi_B = B.get("direction_rad", 0.0)

    d_phi = phi_B - phi_A
    while d_phi >  math.pi: d_phi -= 2.0 * math.pi
    while d_phi < -math.pi: d_phi += 2.0 * math.pi

    if abs(d_phi) < 1e-4:
        return None   # effectively collinear — no arc needed

    T = min_radius * math.tan(abs(d_phi) / 2.0)
    len_A = A.get("length", 0.0)
    len_B = B.get("length", 0.0)
    if T <= 0 or T >= len_A * 0.45 or T >= len_B * 0.45:
        return None   # arc does not fit within the adjacent elements

    kink    = np.array(A["end"], dtype=float)
    cos_a   = math.cos(phi_A)
    sin_a   = math.sin(phi_A)
    cos_b   = math.cos(phi_B)
    sin_b   = math.sin(phi_B)

    tp_a = kink - T * np.array([cos_a, sin_a])   # tangent point on Line A
    tp_b = kink + T * np.array([cos_b, sin_b])   # tangent point on Line B

    rot = "ccw" if d_phi > 0 else "cw"
    R   = min_radius
    if rot == "ccw":
        center = tp_a + R * np.array([-sin_a,  cos_a])
    else:
        center = tp_a + R * np.array([ sin_a, -cos_a])

    arc_len = R * abs(d_phi)
    chord   = float(np.linalg.norm(tp_b - tp_a))
    sign    = 1.0 if rot == "ccw" else -1.0

    new_A = dict(A)
    new_A["end"]    = tp_a.tolist()
    new_A["length"] = float(np.linalg.norm(tp_a - np.array(A["start"], dtype=float)))

    b_end   = np.array(B["end"], dtype=float)
    new_B   = dict(B)
    new_B["start"]        = tp_b.tolist()
    new_B["length"]       = float(np.linalg.norm(b_end - tp_b))
    new_B["direction_rad"] = math.atan2(
        float(b_end[1] - tp_b[1]), float(b_end[0] - tp_b[0])
    )

    arc_el: dict = {
        "type":        "Arc",
        "sta_start":   new_A["sta_start"] + new_A["length"],
        "length":      arc_len,
        "start":       tp_a.tolist(),
        "end":         tp_b.tolist(),
        "center":      center.tolist(),
        "radius":      R,
        "rot":         rot,
        "chord":       chord,
        "_deflection": sign * abs(d_phi),
    }

    new_elements = elements[:idx] + [new_A, arc_el, new_B] + elements[idx + 2:]

    # Recompute sta_start chain from idx onward
    sta = new_A["sta_start"]
    for k in range(idx, len(new_elements)):
        new_elements[k] = dict(new_elements[k])
        new_elements[k]["sta_start"] = sta
        sta += new_elements[k].get("length", 0.0)

    return new_elements


def _enforce_c1_junctions(
    elements:           list[dict],
    min_radius:         float,
    max_junction_shift: float = 25.0,
    min_kink_rad:       float = 0.004,
) -> list[dict]:
    """
    Multi-pass junction correction to enforce C1 continuity.

    Junction types handled:
      Line→Arc  : move junction to tangent point on arc (tangent = line heading)
      Arc→Line  : same, symmetric
      Arc→Arc   : compute Arc_A tangent at junction; find tangent point on Arc_B
                  with that heading; update if shift < max_junction_shift

    Line→Line kinks are handled by _insert_line_line_connector_arc (called earlier
    in _post_process_elements).

    Runs up to 4 passes; stops early if no junction changed by more than 1e-4 m.
    Recomputes sta_start chain after all passes.
    """
    MAX_PASSES = 4

    def _recompute_arc_length(el: dict) -> float:
        cx   = el["center"][0];  cy  = el["center"][1]
        R    = el["radius"];     rot = el["rot"]
        sp   = el["start"];      ep  = el["end"]
        a_s  = math.atan2(sp[1] - cy, sp[0] - cx)
        a_e  = math.atan2(ep[1] - cy, ep[0] - cx)
        delta = a_e - a_s
        if rot == "ccw":
            while delta <= 0.0: delta += 2.0 * math.pi
        else:
            while delta >= 0.0: delta -= 2.0 * math.pi
        if abs(delta) > math.pi:
            return el.get("length", 0.0)   # pathological — keep old length
        return R * abs(delta)

    for _pass in range(MAX_PASSES):
        any_change = False
        elements   = [dict(e) for e in elements]

        for i in range(len(elements) - 1):
            L    = elements[i]
            R_el = elements[i + 1]
            lt   = L.get("type",   "Line")
            rt   = R_el.get("type","Line")

            if lt == "Line" and rt == "Arc":
                phi  = L.get("direction_rad", 0.0)
                cx   = R_el["center"][0];  cy = R_el["center"][1]
                R_v  = R_el["radius"];     rot = R_el["rot"]
                jx, jy = _arc_line_tangent_junction(cx, cy, R_v, rot, phi)
                cur    = np.array(L["end"], dtype=float)
                shift  = math.hypot(jx - cur[0], jy - cur[1])
                if 1e-4 < shift <= max_junction_shift:
                    new_j = [jx, jy]
                    elements[i]["end"]           = new_j
                    elements[i]["length"]         = float(np.linalg.norm(
                        np.array(new_j) - np.array(L["start"], dtype=float)
                    ))
                    elements[i]["direction_rad"]  = math.atan2(
                        new_j[1] - L["start"][1], new_j[0] - L["start"][0]
                    )
                    elements[i + 1]["start"]      = new_j
                    elements[i + 1]["length"]     = _recompute_arc_length(elements[i + 1])
                    any_change = True

            elif lt == "Arc" and rt == "Line":
                phi  = R_el.get("direction_rad", 0.0)
                cx   = L["center"][0];  cy = L["center"][1]
                R_v  = L["radius"];     rot = L["rot"]
                jx, jy = _arc_line_tangent_junction(cx, cy, R_v, rot, phi)
                cur    = np.array(L["end"], dtype=float)
                shift  = math.hypot(jx - cur[0], jy - cur[1])
                if 1e-4 < shift <= max_junction_shift:
                    new_j = [jx, jy]
                    elements[i]["end"]            = new_j
                    elements[i]["length"]          = _recompute_arc_length(elements[i])
                    elements[i + 1]["start"]       = new_j
                    elements[i + 1]["length"]      = float(np.linalg.norm(
                        np.array(R_el["end"], dtype=float) - np.array(new_j)
                    ))
                    elements[i + 1]["direction_rad"] = math.atan2(
                        R_el["end"][1] - new_j[1], R_el["end"][0] - new_j[0]
                    )
                    any_change = True

            elif lt == "Arc" and rt == "Arc":
                cur_j   = np.array(L["end"], dtype=float)
                cx_a    = L["center"][0];   cy_a = L["center"][1]
                heading_a = _arc_tangent_heading(cx_a, cy_a,
                                                  float(cur_j[0]), float(cur_j[1]),
                                                  L["rot"])
                cx_b   = R_el["center"][0]; cy_b = R_el["center"][1]
                R_b    = R_el["radius"];    rot_b = R_el["rot"]
                jx, jy = _arc_line_tangent_junction(cx_b, cy_b, R_b, rot_b, heading_a)
                shift  = math.hypot(jx - float(cur_j[0]), jy - float(cur_j[1]))
                if 1e-4 < shift <= max_junction_shift:
                    new_j = [jx, jy]
                    elements[i]["end"]        = new_j
                    elements[i]["length"]      = _recompute_arc_length(elements[i])
                    elements[i + 1]["start"]   = new_j
                    elements[i + 1]["length"]  = _recompute_arc_length(elements[i + 1])
                    any_change = True

        if not any_change:
            break

    # Recompute sta_start chain
    if elements:
        sta = elements[0].get("sta_start", 0.0)
        for el in elements:
            el["sta_start"] = sta
            sta += el.get("length", 0.0)

    return elements


def _post_process_elements(
    elements:         list[dict],
    forced_ch_ranges: list[tuple[float, float]],
    min_radius:       float,
    min_kink_rad:     float = 0.004,
) -> list[dict]:
    """
    Apply all C1 post-processing steps uniformly to any algorithm's output:

    1. Demote Arc elements that overlap a forced-line chainage range → Line
    2. Merge adjacent Line elements
    3. Insert connector arcs at Line→Line kinks (heading diff > min_kink_rad)
    4. Enforce C1 at Line→Arc, Arc→Line, Arc→Arc junctions
    5. Final merge of adjacent Lines (forced-line demotion may create new adjacencies)

    Returns a new element list.  Input is not modified.
    """
    if not elements:
        return elements

    # ── Step 1: demote Arcs that overlap forced-line chainage ranges ──────
    if forced_ch_ranges:
        out: list[dict] = []
        for el in elements:
            if el.get("type") == "Arc":
                sta0 = el.get("sta_start", 0.0)
                sta1 = sta0 + el.get("length", 0.0)
                overlap = any(
                    sta0 < r_end and sta1 > r_start
                    for r_start, r_end in forced_ch_ranges
                )
                if overlap:
                    sp = np.array(el["start"], dtype=float)
                    ep = np.array(el["end"],   dtype=float)
                    out.append({
                        "type":          "Line",
                        "sta_start":     sta0,
                        "length":        float(np.linalg.norm(ep - sp)),
                        "start":         el["start"],
                        "end":           el["end"],
                        "direction_rad": math.atan2(
                            float(ep[1] - sp[1]), float(ep[0] - sp[0])
                        ),
                    })
                    continue
            out.append(dict(el))
        elements = out

    # ── Step 2: merge adjacent Lines ──────────────────────────────────────
    elements = _merge_adjacent_line_elements(elements)

    # ── Step 3: insert connector arcs at Line→Line kinks ─────────────────
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(elements) - 1:
            if (elements[i].get("type") == "Line"
                    and elements[i + 1].get("type") == "Line"):
                phi_a = elements[i].get("direction_rad", 0.0)
                phi_b = elements[i + 1].get("direction_rad", 0.0)
                d_phi = phi_b - phi_a
                while d_phi >  math.pi: d_phi -= 2.0 * math.pi
                while d_phi < -math.pi: d_phi += 2.0 * math.pi
                if abs(d_phi) >= min_kink_rad:
                    new_els = _insert_line_line_connector_arc(elements, i, min_radius)
                    if new_els is not None:
                        elements = new_els
                        changed  = True
                        i        = max(0, i - 1)
                        continue
            i += 1

    # ── Step 4: enforce C1 at remaining junctions ────────────────────────
    elements = _enforce_c1_junctions(elements, min_radius, min_kink_rad=min_kink_rad)

    # ── Step 5: final Line merge (cleanup) ────────────────────────────────
    elements = _merge_adjacent_line_elements(elements)

    return elements
