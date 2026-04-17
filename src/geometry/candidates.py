"""
Multi-algorithm candidate alignment generator.

All fitted algorithms (tight / balanced / smooth) produce Lines + Arcs
with **C0 and C1 continuity guaranteed by construction**.

The raw "OSM Polyline" algorithm produces one Line per OSM vertex pair —
no fitting, no continuity requirement, useful as a baseline.

Exported names
--------------
CandidateAlignment  : dataclass holding elements + metrics + display info
evaluate_candidate  : compute max_deviation / rmse / n_elements
CandidateGenerator  : runs all four algorithms
"""

from __future__ import annotations

import math
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
    seg_type:  str    # "Line" | "Arc"
    start_idx: int    # index into xy (inclusive)
    end_idx:   int    # index into xy (inclusive)
    R_median:  float  # Kasa-refined radius; inf for Line
    rot:       str    # "ccw" | "cw"; irrelevant for Line
    deflection: float  # signed total heading change (rad); 0.0 for Line


# Minimum arc deflection below which a curved segment is treated as a line
_MIN_ARC_DEFLECTION_RAD = math.radians(5.0)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    elements: list[dict],
    xy: np.ndarray,
    chainages: np.ndarray,
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
        "tight":    "Tight Segmentation",
        "balanced": "Balanced Merge",
        "smooth":   "Smooth Merge",
        "raw":      "OSM Polyline",
    }

    def __init__(self, xy: np.ndarray, chainages: np.ndarray, settings: dict):
        self.xy             = xy
        self.chainages      = chainages
        self.settings       = settings
        self.min_radius     = settings.get("min_radius",       150.0)
        self.smooth_window  = settings.get("smooth_window",    21)
        self.max_deviation  = settings.get("max_deviation",    0.5)
        self.check_interval = settings.get("check_interval",   5.0)
        self.merge_pct      = settings.get("merge_radius_pct", 15.0)

    def run_all(self) -> list[CandidateAlignment]:
        results = []
        algo_ids = ["tight", "balanced", "smooth", "raw"]
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

    def _run_one(self, algo_id: str) -> CandidateAlignment:
        scale_map = {"tight": 0.5, "balanced": 1.0, "smooth": 2.0}
        if algo_id in scale_map:
            pct = self.merge_pct * scale_map[algo_id]
            elements = _build_candidate(
                self.xy, self.chainages,
                merge_radius_pct=pct,
                min_radius=self.min_radius,
                smooth_window=self.smooth_window,
            )
            metrics = evaluate_candidate(
                elements, self.xy, self.chainages, self.check_interval
            )
            return CandidateAlignment(
                algorithm_id=algo_id,
                label=self.LABELS[algo_id],
                elements=elements,
                **metrics,
            )
        elif algo_id == "raw":
            return self._run_raw()
        else:
            raise ValueError(f"Unknown algorithm: {algo_id!r}")

    # ------------------------------------------------------------------
    # Raw OSM polyline
    # ------------------------------------------------------------------

    def _run_raw(self) -> CandidateAlignment:
        """One Line element per consecutive OSM vertex pair. No fitting, no C1."""
        xy        = self.xy
        chainages = self.chainages
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
        return CandidateAlignment(
            algorithm_id="raw",
            label="OSM Polyline",
            elements=elements,
            **metrics,
        )


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def _signed_radius(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[float, int]:
    """
    Classify a 3-point triplet as a straight or curved micro-element.

    Returns (R, sign) where:
      R    = circle radius in metres; math.inf if the points are collinear
             or the turning angle is below the collinearity threshold.
      sign = +1 for CCW curvature (left-turning), -1 for CW (right-turning).

    Collinearity is detected via the cross product of consecutive edge
    vectors **before** calling the Kasa algebraic fit, to avoid the
    spurious small-radius results that lstsq produces for rank-deficient
    (collinear) inputs.
    """
    from geometry.alignment import _fit_circle_kasa

    e1x = float(p1[0] - p0[0]); e1y = float(p1[1] - p0[1])
    e2x = float(p2[0] - p1[0]); e2y = float(p2[1] - p1[1])

    cross = e1x * e2y - e1y * e2x          # signed area × 2 of the triplet
    L1 = math.hypot(e1x, e1y)
    L2 = math.hypot(e2x, e2y)

    # Collinearity guard: |sin(turning_angle)| < 0.01 rad (~0.57°), or
    # one of the edges is degenerate (duplicate OSM node).
    if L1 < 1e-9 or L2 < 1e-9 or abs(cross) < 0.01 * L1 * L2:
        return math.inf, +1

    sign = +1 if cross >= 0.0 else -1

    pts = np.array([p0, p1, p2], dtype=float)
    cx, cy, r = _fit_circle_kasa(pts)

    if cx is None or r is None or r > 50_000.0 or not math.isfinite(r):
        return math.inf, +1

    return float(r), sign


def _segment_deflection(pts: np.ndarray) -> float:
    """
    Total signed heading change across a point sequence.

    For each consecutive edge pair computes:
        δ_i = atan2(e[i] × e[i+1], e[i] · e[i+1])
    and returns their sum.

    Returns 0.0 for sequences with fewer than 3 points.
    """
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


def _merge_segments_by_sign(
    micro_types: list[str],
    micro_sign:  list[int],
    xy:          np.ndarray,
) -> list[_Segment]:
    """
    Stage-1 merge: group consecutive micro-elements by type + curvature sign only.

    No radius threshold is applied here — that would require a stable radius
    estimate, which is only available after Kasa fitting.  Adjacent Line
    micro-elements always merge; adjacent Arc micro-elements merge when the
    rotation sign is the same.  A sign flip forces a boundary.

    Segments whose total OSM deflection < 5° are demoted to Line.
    Consecutive Line segments (including freshly demoted ones) are merged.
    """
    n_micro = len(micro_types)
    if n_micro == 0:
        return []

    segments: list[_Segment] = []
    seg_start  = 0
    current_type = micro_types[0]
    current_sign = micro_sign[0]

    def _flush(seg_end_micro: int) -> None:
        xi0 = seg_start
        xi1 = min(seg_end_micro + 1, len(xy) - 1)
        pts  = xy[xi0 : xi1 + 1]
        defl = _segment_deflection(pts)
        rot  = "ccw" if current_sign >= 0 else "cw"
        if current_type == "Line" or abs(defl) < _MIN_ARC_DEFLECTION_RAD:
            typ   = "Line"
            R_med = math.inf
        else:
            typ   = "Arc"
            R_med = math.inf   # filled in by Kasa in Stage 2
        segments.append(_Segment(
            seg_type   = typ,
            start_idx  = xi0,
            end_idx    = xi1,
            R_median   = R_med,
            rot        = rot,
            deflection = defl,
        ))

    for m in range(1, n_micro):
        new_type = micro_types[m]
        new_sign = micro_sign[m]
        if new_type != current_type or (new_type == "Arc" and new_sign != current_sign):
            _flush(m - 1)
            seg_start    = m
            current_type = new_type
            current_sign = new_sign
        # else: extend current segment (no action needed)

    _flush(n_micro - 1)

    # Boundary corrections
    if segments and segments[0].start_idx > 0:
        segments[0].start_idx = 0
    if segments and segments[-1].end_idx < len(xy) - 1:
        segments[-1].end_idx = len(xy) - 1

    # Merge consecutive Line segments (demoted arcs break adjacency)
    merged: list[_Segment] = []
    for seg in segments:
        if merged and seg.seg_type == "Line" and merged[-1].seg_type == "Line":
            merged[-1].end_idx  = seg.end_idx
            pts = xy[merged[-1].start_idx : merged[-1].end_idx + 1]
            merged[-1].deflection = _segment_deflection(pts)
        else:
            merged.append(seg)

    return merged


def _merge_arcs_by_radius(
    segments:         list[_Segment],
    xy:               np.ndarray,
    merge_radius_pct: float,
) -> list[_Segment]:
    """
    Stage-2 merge: merge adjacent same-sign Arc segments whose *Kasa* radii
    are within merge_radius_pct of each other.

    Segments must already have R_median set by Kasa fitting.
    """
    tol = merge_radius_pct / 100.0
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
                R_a, R_b = seg.R_median, nxt.R_median
                if R_a > 0 and R_b > 0 and math.isfinite(R_a) and math.isfinite(R_b):
                    rel = abs(R_a - R_b) / max(R_a, R_b)
                    if rel <= tol:
                        # Merge: take the average Kasa radius (weighted by length)
                        L_a = float(np.linalg.norm(xy[seg.end_idx] - xy[seg.start_idx]))
                        L_b = float(np.linalg.norm(xy[nxt.end_idx] - xy[nxt.start_idx]))
                        denom = L_a + L_b if (L_a + L_b) > 0 else 1.0
                        R_merged = (R_a * L_a + R_b * L_b) / denom
                        pts  = xy[seg.start_idx : nxt.end_idx + 1]
                        defl = _segment_deflection(pts)
                        merged.append(_Segment(
                            seg_type   = "Arc",
                            start_idx  = seg.start_idx,
                            end_idx    = nxt.end_idx,
                            R_median   = R_merged,
                            rot        = seg.rot,
                            deflection = defl,
                        ))
                        i += 2
                        changed = True
                        continue
            merged.append(seg)
            i += 1
        segments = merged
    return segments


def _build_elements_c1(
    segments:  list[_Segment],
    xy:        np.ndarray,
    chainages: np.ndarray,
) -> list[dict]:
    """
    Constructive forward build — guaranteed C0 and C1 at every junction.

    Starting from xy[0] with the heading of the first segment, each element
    is placed analytically so that its start equals the previous element's
    end and its entry tangent equals the previous element's exit tangent.

    Arc deflection angles are taken from _Segment.deflection (measured from
    the OSM point cloud), so the fitted alignment follows the real railway
    curvature while maintaining exact geometric continuity.
    """
    if not segments:
        return []

    elements: list[dict] = []
    pos     = np.array(xy[0], dtype=float)
    # Initial heading from first two OSM points
    if len(xy) > 1:
        d = xy[1] - xy[0]
        heading = math.atan2(float(d[1]), float(d[0]))
    else:
        heading = 0.0
    sta = 0.0

    for seg in segments:
        ch0     = float(chainages[seg.start_idx])
        ch1     = float(chainages[seg.end_idx])
        seg_len = max(ch1 - ch0, 1e-6)

        if seg.seg_type == "Line":
            end = pos + seg_len * np.array([math.cos(heading), math.sin(heading)])
            el: dict = {
                "type":          "Line",
                "sta_start":     sta,
                "length":        seg_len,
                "start":         pos.tolist(),
                "end":           end.tolist(),
                "direction_rad": heading,
            }
            pos = end
            # heading unchanged
            sta += seg_len
            elements.append(el)

        else:  # Arc
            R    = seg.R_median
            sign = 1.0 if seg.rot == "ccw" else -1.0
            defl = seg.deflection   # signed

            if abs(defl) < 1e-6 or not math.isfinite(R) or R < 1.0:
                # Degenerate arc: emit as line
                end = pos + seg_len * np.array([math.cos(heading), math.sin(heading)])
                el = {
                    "type":          "Line",
                    "sta_start":     sta,
                    "length":        seg_len,
                    "start":         pos.tolist(),
                    "end":           end.tolist(),
                    "direction_rad": heading,
                }
                pos = end
                sta += seg_len
                elements.append(el)
                continue

            # Center is perpendicular to heading at distance R
            center_angle = heading + sign * math.pi / 2.0
            center = pos + R * np.array([math.cos(center_angle), math.sin(center_angle)])

            # Arc from a_start to a_end (rotate by signed deflection)
            a_start = math.atan2(pos[1] - center[1], pos[0] - center[0])
            a_end   = a_start + sign * abs(defl)
            end     = center + R * np.array([math.cos(a_end), math.sin(a_end)])
            arc_len = R * abs(defl)
            chord   = float(np.linalg.norm(end - pos))

            el = {
                "type":        "Arc",
                "sta_start":   sta,
                "length":      arc_len,
                "start":       pos.tolist(),
                "end":         end.tolist(),
                "center":      center.tolist(),
                "radius":      R,
                "rot":         seg.rot,
                "chord":       chord,
                "_deflection": defl,   # temp field used by spiral cascade
            }
            pos      = end
            heading += sign * abs(defl)
            sta     += arc_len
            elements.append(el)

    return elements


def _arc_sagitta(R: float, defl_rad: float) -> float:
    """Maximum lateral deviation of a circular arc from its chord (metres)."""
    if not math.isfinite(R) or R <= 0 or not math.isfinite(defl_rad):
        return 0.0
    # sagitta = R(1 − cos(|δ|/2))
    return R * (1.0 - math.cos(abs(defl_rad) / 2.0))


def _build_candidate(
    xy:               np.ndarray,
    chainages:        np.ndarray,
    merge_radius_pct: float,
    min_radius:       float,
    smooth_window:    int = 21,
    min_sagitta:      float = 1.5,
) -> list[dict]:
    """
    Full micro-merge pipeline producing a C1-continuous Line+Arc alignment.

    Steps:
      1. Compute smoothed curvature (reuses geometry.curvature) — robust to OSM noise
      2. Classify interior points as Line or Arc with signed curvature
      3. Merge adjacent same-type similar-radius points into _Segment objects
      4. Refine each ARC segment's radius with Kasa fit on all segment points
      5. Constructive forward build → C0+C1 guaranteed
    """
    from geometry.alignment import _fit_circle_kasa
    from geometry.curvature import compute_curvature, smooth_curvature

    N = len(xy)
    if N < 2:
        return []

    if N == 2:
        seg_len = float(chainages[1] - chainages[0])
        heading = math.atan2(float(xy[1, 1] - xy[0, 1]), float(xy[1, 0] - xy[0, 0]))
        return [{
            "type": "Line", "sta_start": 0.0, "length": seg_len,
            "start": xy[0].tolist(), "end": xy[1].tolist(),
            "direction_rad": heading,
        }]

    # Step 1+2 — Smoothed curvature classification
    # Use the same smoothing as the proven curvature pipeline.
    # LINE_TOL=0.001 means R > 1000 m is treated as straight.
    LINE_TOL = 0.001

    kappa        = compute_curvature(xy)
    kappa_smooth = smooth_curvature(kappa, window=smooth_window)

    micro_types: list[str]   = []
    micro_R:     list[float] = []
    micro_sign:  list[int]   = []

    for i in range(1, N - 1):   # interior points only
        k = float(kappa_smooth[i])
        if abs(k) < LINE_TOL:
            micro_types.append("Line")
            micro_R.append(math.inf)
            micro_sign.append(+1)
        else:
            R = max(abs(1.0 / k), float(min_radius))
            micro_types.append("Arc")
            micro_R.append(R)
            micro_sign.append(+1 if k > 0.0 else -1)

    # Step 3a — Stage-1 merge: group by type + sign only (no radius check)
    # Per-point smoothed curvature is too noisy for reliable radius grouping;
    # Kasa over the full segment is far more stable.
    segments = _merge_segments_by_sign(micro_types, micro_sign, xy)

    if not segments:
        return []

    # Step 3b — Kasa refinement on each coarse Arc segment
    for seg in segments:
        if seg.seg_type == "Arc":
            pts = xy[seg.start_idx : seg.end_idx + 1]
            if len(pts) >= 3:
                cx, cy, r_fit = _fit_circle_kasa(pts)
                if (cx is not None and r_fit is not None
                        and 0.0 < r_fit < 1e6 and math.isfinite(r_fit)):
                    seg.R_median = max(float(r_fit), float(min_radius))
                else:
                    # Kasa failed → use median of per-point smoothed radii as fallback
                    seg_pts_idx = range(seg.start_idx, min(seg.end_idx, len(micro_R) - 1) + 1)
                    R_vals = [micro_R[i - 1] for i in seg_pts_idx
                              if 0 < i <= len(micro_R) and math.isfinite(micro_R[i - 1])]
                    if R_vals:
                        seg.R_median = max(float(np.median(R_vals)), float(min_radius))

    # Step 3b-post — Demote Arcs whose sagitta is below min_sagitta to Line.
    # An arc with small sagitta is geometrically indistinguishable from a
    # straight line at the OSM noise level; keeping it produces spurious elements.
    changed = False
    for seg in segments:
        if seg.seg_type == "Arc":
            sag = _arc_sagitta(seg.R_median, seg.deflection)
            if sag < min_sagitta:
                seg.seg_type = "Line"
                seg.R_median = math.inf
                changed = True
    if changed:
        # Re-merge consecutive Lines created by demotion
        merged2: list[_Segment] = []
        for seg in segments:
            if merged2 and seg.seg_type == "Line" and merged2[-1].seg_type == "Line":
                merged2[-1].end_idx = seg.end_idx
                pts = xy[merged2[-1].start_idx : merged2[-1].end_idx + 1]
                merged2[-1].deflection = _segment_deflection(pts)
            else:
                merged2.append(seg)
        segments = merged2

    # Step 3c — Stage-2 merge: merge adjacent same-sign Arcs with similar Kasa radii
    segments = _merge_arcs_by_radius(segments, xy, merge_radius_pct)

    # Step 4 — Re-run Kasa on any freshly merged Arc segments (R_median may be averaged)
    for seg in segments:
        if seg.seg_type == "Arc":
            pts = xy[seg.start_idx : seg.end_idx + 1]
            if len(pts) >= 3:
                cx, cy, r_fit = _fit_circle_kasa(pts)
                if (cx is not None and r_fit is not None
                        and 0.0 < r_fit < 1e6 and math.isfinite(r_fit)):
                    seg.R_median = max(float(r_fit), float(min_radius))

    # Step 5 — Constructive forward build
    elements = _build_elements_c1(segments, xy, chainages)

    return elements
