"""
Horizontal geometry fitting.
Converts a projected 2D polyline into a sequence of geometric primitives:
  Line, Circular Arc, Clothoid (Euler) Spiral.
Output elements are dicts ready for LandXML serialisation.

Accuracy refinement
-------------------
After the initial curvature-based fit, every element is validated by sampling
the fitted geometry at *check_interval* metre spacing and measuring the
distance from each sample to the nearest segment of the original OSM polyline.
Elements whose maximum deviation exceeds *max_deviation* are recursively split
at the midpoint of their OSM-point range and re-fitted, until either:
  • the deviation is within tolerance, or
  • the element is too short to split further (min_*_length applies).

Radius continuity
-----------------
Railway geometry must follow  LINE – SPIRAL – ARC – SPIRAL – LINE.
enforce_continuity() ensures:
  Line  → Spiral : spiral.radius_start = INF
  Spiral → Arc   : spiral.radius_end   = arc.radius
  Arc   → Spiral : spiral.radius_start = arc.radius
  Spiral → Line  : spiral.radius_end   = INF
  Spiral → Spiral: right.radius_start  = left.radius_end

Spirals that still have BOTH radii = INF after this pass are degenerate
(Line-to-Line pseudo-spirals) and are converted to Line elements.
"""

from __future__ import annotations

import math
import numpy as np
from .curvature import (
    compute_curvature,
    smooth_curvature,
    compute_chainages,
    segment_curvature,
    ElementType,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_alignment(
    xy: np.ndarray,
    min_radius: float = 150.0,
    algorithm: str = "tangent",
    angle_threshold_deg: float = 1.0,
    simplify_tolerance: float | None = None,
    use_spirals: bool = True,
    # existing params unchanged:
    smooth_window: int = 21,
    line_tol: float = 0.001,
    arc_tol: float = 0.0002,
    min_element_length: float = 10.0,
    min_line_length: float | None = None,
    min_arc_length: float | None = None,
    min_spiral_length: float | None = None,
    max_deviation: float = 0.5,
    check_interval: float = 5.0,
) -> list[dict]:
    """
    Fit geometric elements to a 2D projected polyline.

    Parameters
    ----------
    xy              : (N, 2) array of projected (x, y) coordinates in metres.
    min_radius      : minimum horizontal curve radius enforced on all arc elements.
    algorithm       : "tangent" (default) or "curvature".
    angle_threshold_deg : minimum turning angle (degrees) to classify as a curve.
    simplify_tolerance  : RDP simplification tolerance in metres (None = auto).
    use_spirals     : if True, insert clothoid spirals at arc transitions.
    smooth_window   : Savitzky-Golay window size for curvature smoothing.
    line_tol        : curvature threshold (1/m) below which a segment is a Line.
    arc_tol         : max kappa variation (1/m) for a constant-radius Arc.
    min_element_length : fallback minimum element length in m (all types).
    min_line_length    : minimum Line element length (overrides fallback).
    min_arc_length     : minimum Arc element length (overrides fallback).
    min_spiral_length  : minimum Spiral element length (overrides fallback).
    max_deviation   : maximum allowed deviation (m) between fitted element and
                      original OSM polyline.  Set <= 0 to skip refinement.
    check_interval  : spacing (m) at which the fitted element is sampled when
                      measuring deviation from the OSM polyline.

    Returns
    -------
    List of element dicts with type-specific fields ready for LandXML output.
    """
    if algorithm == "tangent":
        eff_line   = min_line_length   if min_line_length   is not None else min_element_length
        eff_arc    = min_arc_length    if min_arc_length    is not None else min_element_length
        eff_spiral = min_spiral_length if min_spiral_length is not None else min_element_length
        return _fit_alignment_tangent(
            xy,
            min_radius=min_radius,
            min_line_length=eff_line,
            min_arc_length=eff_arc,
            min_spiral_length=eff_spiral,
            max_deviation=max_deviation,
            check_interval=check_interval,
            angle_threshold_deg=angle_threshold_deg,
            simplify_tolerance=simplify_tolerance,
            use_spirals=use_spirals,
            smooth_window=smooth_window,
            line_tol=line_tol,
            arc_tol=arc_tol,
        )

    # --- curvature algorithm (legacy) ---
    eff_line   = min_line_length   if min_line_length   is not None else min_element_length
    eff_arc    = min_arc_length    if min_arc_length    is not None else min_element_length
    eff_spiral = min_spiral_length if min_spiral_length is not None else min_element_length

    kappa        = compute_curvature(xy)
    kappa_smooth = smooth_curvature(kappa, window=smooth_window)
    chainages    = compute_chainages(xy)

    segments = segment_curvature(
        kappa_smooth, chainages,
        line_tol=line_tol,
        arc_tol=arc_tol,
        min_line_length=eff_line,
        min_arc_length=eff_arc,
        min_spiral_length=eff_spiral,
    )

    elements = []
    for seg in segments:
        el = _fit_element(seg, xy, kappa_smooth, chainages)
        if el is not None:
            elements.append(el)

    elements = enforce_continuity(elements)

    # Accuracy-driven refinement pass
    if max_deviation > 0 and len(xy) >= 3:
        elements = _refine_elements(
            elements, xy, chainages, kappa_smooth,
            max_deviation=max_deviation,
            check_interval=max(check_interval, 0.5),
            min_line_length=eff_line,
            min_arc_length=eff_arc,
            min_spiral_length=eff_spiral,
            line_tol=line_tol,
            arc_tol=arc_tol,
        )
        elements = enforce_continuity(elements)

    # Post-hoc min_radius clamp for curvature path
    if min_radius > 0:
        for el in elements:
            if el.get("type") == "Arc" and el.get("radius", float("inf")) < min_radius:
                el["radius"] = min_radius
        # Recompute clothoid_A for adjacent spirals
        for i, el in enumerate(elements):
            if el.get("type") == "Spiral":
                r_s = el.get("radius_start", float("inf"))
                r_e = el.get("radius_end",   float("inf"))
                finite = [r for r in (r_s, r_e) if not (math.isinf(r) or r > 1e8)]
                if finite:
                    el["clothoid_A"] = float(np.sqrt(min(finite) * el["length"]))

    return elements


def _fit_alignment_tangent(
    xy: np.ndarray,
    min_radius: float,
    min_line_length: float,
    min_arc_length: float,
    min_spiral_length: float,
    max_deviation: float,
    check_interval: float,
    angle_threshold_deg: float,
    simplify_tolerance: float | None,
    use_spirals: bool,
    smooth_window: int = 21,
    line_tol: float = 0.001,
    arc_tol: float = 0.0002,
) -> list[dict]:
    """
    Tangent-continuous alignment fitting.

    Detection pipeline:
      1. Compute curvature profile and smooth it (Savitzky-Golay).
      2. Segment into LINE / ARC / SPIRAL runs with segment_curvature().
      3. Group consecutive non-LINE segments into arc zones via
         _detect_arc_zones_from_segments() — this ensures a smooth curve
         (whose RDP simplification would produce multiple sub-zones) is
         treated as a single zone and fitted with one L-S-A-S-L sequence.
      4. Optimise [R, L_entry, L_exit] per zone with _fit_arc_zone().
      5. Assemble elements using _assemble_from_zones() which uses chainages
         for positioning rather than simplified-polyline vertex positions.
    """
    if len(xy) < 3:
        # Degenerate: single line
        length = float(np.linalg.norm(xy[-1] - xy[0]))
        return [{
            "type": "Line", "sta_start": 0.0, "length": length,
            "start": xy[0].tolist(), "end": xy[-1].tolist(),
            "direction_rad": math.atan2(
                float(xy[-1][1] - xy[0][1]), float(xy[-1][0] - xy[0][0])),
        }]

    chainages = compute_chainages(xy)
    total_len = float(chainages[-1])

    # ── Step 1-2: curvature profile → segments ───────────────────────────
    # Use a SMALL smoothing window for zone detection.  A large window
    # (e.g. default 21) blurs spiral regions whose curvature is lower than
    # line_tol after averaging, creating phantom LINE segments in the middle
    # of a real curve.  Window 5 (or the passed value if smaller) covers ~10 m
    # at 2 m node spacing, which is well below a 50 m minimum spiral.
    kappa          = compute_curvature(xy)
    detect_window  = min(smooth_window, 5)
    kappa_smooth   = smooth_curvature(kappa, window=detect_window)

    segments = segment_curvature(
        kappa_smooth, chainages,
        line_tol=line_tol,
        arc_tol=arc_tol,
        min_line_length=min_line_length,
        min_arc_length=min_arc_length,
        min_spiral_length=min_spiral_length,
    )

    # ── Step 3: group consecutive curved segments into arc zones ─────────
    arc_zones = _detect_arc_zones_from_segments(segments, chainages, xy)

    # ── Step 3b: merge zones separated by phantom straight segments ───────
    # When curvature smoothing mis-classifies parts of a spiral as LINE,
    # a single smooth curve can appear as two separate arc zones with a
    # "straight" gap between them.  Any gap whose points deviate more than
    # 2× max_deviation from their chord is secretly curved → merge the zones.
    arc_zones = _merge_phantom_zones(xy, chainages, arc_zones,
                                     merge_dev_thresh=max(max_deviation * 2.0, 1.0))

    # ── Step 3c: expand zone boundaries into low-curvature spiral tails ──
    # The first (and last) portion of a clothoid spiral has kappa < line_tol
    # and is therefore classified as LINE by segment_curvature.  Without
    # expanding the zone, TC is placed inside the curve, the entry spiral is
    # too short, and max_deviation_element reports a large error.
    # Expand each zone boundary backward / forward in the adjacent straight
    # until kappa_smooth drops below line_tol * 0.25 (noise floor).
    arc_zones = _expand_zone_boundaries(arc_zones, kappa_smooth, chainages, xy,
                                        kappa_expand_tol=line_tol * 0.25)

    # ── Step 3d: annotate zones with kappa-based spiral length estimates ──
    # The optimizer's objective doesn't distinguish a short-spiral + longer-arc
    # from the true LSASL, so we pre-compute estimates from the kappa profile
    # and use them as initial guesses (see spiral_entry_hint in _fit_arc_zone).
    _annotate_spiral_hints_from_kappa(arc_zones, kappa_smooth, chainages)

    if not arc_zones:
        # Entire alignment is effectively straight
        return [{
            "type": "Line", "sta_start": 0.0, "length": total_len,
            "start": xy[0].tolist(), "end": xy[-1].tolist(),
            "direction_rad": math.atan2(
                float(xy[-1][1] - xy[0][1]), float(xy[-1][0] - xy[0][0])),
        }]

    # ── Step 4: fit each zone ─────────────────────────────────────────────
    fitted_zones = []
    for k, zone in enumerate(arc_zones):
        if k == 0:
            avail_entry = zone["chainage_start"]
        else:
            avail_entry = zone["chainage_start"] - arc_zones[k - 1]["chainage_end"]

        if k == len(arc_zones) - 1:
            avail_exit = total_len - zone["chainage_end"]
        else:
            avail_exit = arc_zones[k + 1]["chainage_start"] - zone["chainage_end"]

        avail_entry = max(avail_entry * 0.85, 0.0)
        avail_exit  = max(avail_exit  * 0.85, 0.0)

        fz = _fit_arc_zone(
            zone, min_radius, min_spiral_length, max_deviation,
            use_spirals, avail_entry, avail_exit,
        )
        fitted_zones.append(fz)

    # ── Step 5: assemble elements from chainage-based zones ───────────────
    elements = _assemble_from_zones(
        xy, chainages, arc_zones, fitted_zones,
        min_line_length=min_line_length,
        min_spiral_length=min_spiral_length,
        use_spirals=use_spirals,
    )

    return elements if elements else [{
        "type": "Line", "sta_start": 0.0, "length": total_len,
        "start": xy[0].tolist(), "end": xy[-1].tolist(),
        "direction_rad": math.atan2(
            float(xy[-1][1] - xy[0][1]), float(xy[-1][0] - xy[0][0])),
    }]


# ---------------------------------------------------------------------------
# Element fitting
# ---------------------------------------------------------------------------

def _fit_element(
    seg: dict,
    xy: np.ndarray,
    kappa: np.ndarray,
    chainages: np.ndarray,
) -> dict | None:
    i0   = seg["start_idx"]
    i1   = seg["end_idx"]
    pts  = xy[i0 : i1 + 1]
    sta_start = float(chainages[i0])
    sta_end   = float(chainages[i1])
    length    = sta_end - sta_start

    if length < 0.1 or len(pts) < 2:
        return None

    etype = seg["type"]
    if etype == ElementType.LINE:
        return _fit_line(pts, sta_start, length)
    elif etype == ElementType.ARC:
        return _fit_arc(pts, sta_start, length, seg["mean_kappa"])
    elif etype == ElementType.SPIRAL:
        return _fit_spiral(pts, sta_start, length,
                           seg["kappa_start"], seg["kappa_end"])
    return None


def _fit_line(pts: np.ndarray, sta_start: float, length: float) -> dict:
    """SVD/PCA line fit — more stable than raw OSM node endpoints."""
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    direction_vec = vt[0]

    projections  = (pts - centroid) @ direction_vec
    polyline_dir = pts[-1] - pts[0]
    if np.dot(polyline_dir, direction_vec) < 0:
        direction_vec = -direction_vec
        projections   = -projections

    t_start       = projections[0]
    t_end         = projections[-1]
    start_pt      = (centroid + t_start * direction_vec).tolist()
    end_pt        = (centroid + t_end   * direction_vec).tolist()
    actual_length = abs(t_end - t_start)
    direction     = float(np.arctan2(direction_vec[1], direction_vec[0]))

    return {
        "type":          "Line",
        "sta_start":     sta_start,
        "length":        actual_length if actual_length > 0.1 else length,
        "start":         start_pt,
        "end":           end_pt,
        "direction_rad": direction,
    }


def _fit_arc(
    pts: np.ndarray,
    sta_start: float,
    length: float,
    mean_kappa: float,
) -> dict:
    if abs(mean_kappa) < 1e-9:
        return _fit_line(pts, sta_start, length)

    radius = abs(1.0 / mean_kappa)
    rot    = "ccw" if mean_kappa > 0 else "cw"

    cx, cy, r_fit = _fit_circle_kasa(pts)
    if r_fit is not None and 0 < r_fit < 1e6:
        radius = r_fit

    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    return {
        "type":      "Arc",
        "sta_start": sta_start,
        "length":    length,
        "start":     pts[0].tolist(),
        "end":       pts[-1].tolist(),
        "center":    [float(cx), float(cy)] if cx is not None else None,
        "radius":    radius,
        "rot":       rot,
        "chord":     chord,
    }


def _fit_spiral(
    pts: np.ndarray,
    sta_start: float,
    length: float,
    kappa_start: float,
    kappa_end: float,
) -> dict:
    """Clothoid (Euler spiral): linearly varying kappa(s) = k0 + (k1-k0)*s/L."""
    r_start = abs(1.0 / kappa_start) if abs(kappa_start) > 1e-9 else float("inf")
    r_end   = abs(1.0 / kappa_end)   if abs(kappa_end)   > 1e-9 else float("inf")

    finite = [r for r in (r_start, r_end) if not math.isinf(r)]
    r_min  = min(finite) if finite else 1e9
    A      = float(np.sqrt(r_min * length)) if r_min < 1e8 else 0.0

    mean_k = (kappa_start + kappa_end) / 2
    rot    = "ccw" if mean_k > 0 else "cw"

    return {
        "type":         "Spiral",
        "sta_start":    sta_start,
        "length":       length,
        "start":        pts[0].tolist(),
        "end":          pts[-1].tolist(),
        "radius_start": r_start,
        "radius_end":   r_end,
        "clothoid_A":   A,
        "rot":          rot,
    }


# ---------------------------------------------------------------------------
# Radius continuity enforcement
# ---------------------------------------------------------------------------

def element_end_heading(el: dict, prev_heading: float | None = None) -> float:
    """
    Heading (radians, east=0, CCW positive) at the END of a geometric element.

    Line   : atan2(end[1]-start[1], end[0]-start[0])
    Arc    : atan2(end[1]-cy, end[0]-cx) + sign*pi/2
             sign=+1 for CCW, -1 for CW
    Spiral : prev_heading + sign*(k_s*L + (k_e-k_s)*L/2)
             where k_i = 0 if math.isinf(radius_i) else 1/radius_i
             Falls back to chord direction if prev_heading is None.

    Parameters
    ----------
    el           : element dict (output of fit_alignment)
    prev_heading : heading (rad) at the START of this element.
                   Required for accurate Spiral result; ignored for Line/Arc.

    Returns
    -------
    Heading at end of element in radians (not wrapped).
    """
    etype = el.get("type", "Line")

    if etype == "Line":
        start = el.get("start", [0.0, 0.0])
        end   = el.get("end",   [0.0, 0.0])
        return math.atan2(end[1] - start[1], end[0] - start[0])

    if etype == "Arc":
        center = el.get("center")
        end    = el.get("end", [0.0, 0.0])
        rot    = el.get("rot", "ccw")
        sign   = 1.0 if rot == "ccw" else -1.0
        if center is not None:
            radial = math.atan2(end[1] - center[1], end[0] - center[0])
            return radial + sign * math.pi / 2
        # Fallback: chord direction
        start = el.get("start", [0.0, 0.0])
        return math.atan2(end[1] - start[1], end[0] - start[0])

    if etype == "Spiral":
        L   = float(el.get("length", 0.0))
        r_s = el.get("radius_start", float("inf"))
        r_e = el.get("radius_end",   float("inf"))
        rot  = el.get("rot", "ccw")
        sign = 1.0 if rot == "ccw" else -1.0
        k_s = 0.0 if math.isinf(r_s) else 1.0 / float(r_s)
        k_e = 0.0 if math.isinf(r_e) else 1.0 / float(r_e)
        delta_theta = sign * (k_s * L + (k_e - k_s) * L / 2.0)
        if prev_heading is not None:
            return prev_heading + delta_theta
        # Fallback: chord direction
        start = el.get("start", [0.0, 0.0])
        end   = el.get("end",   [0.0, 0.0])
        return math.atan2(end[1] - start[1], end[0] - start[0])

    # Unknown type: chord fallback
    start = el.get("start", [0.0, 0.0])
    end   = el.get("end",   [0.0, 0.0])
    return math.atan2(end[1] - start[1], end[0] - start[0])


def enforce_continuity(elements: list[dict]) -> list[dict]:
    """
    Enforce radius continuity at every element boundary and remove
    degenerate INF/INF spirals.

    Rules applied in order:
      1. Line  → Spiral  : spiral.radius_start = INF
      2. Spiral → Arc    : spiral.radius_end   = arc.radius
      3. Arc   → Spiral  : spiral.radius_start = arc.radius
      4. Spiral → Line   : spiral.radius_end   = INF
      5. Spiral → Spiral : right.radius_start  = left.radius_end

    After adjusting radii, clothoid_A is recomputed: A = sqrt(R_finite * L).
    """
    if not elements:
        return elements

    for i, el in enumerate(elements):
        if el["type"] != "Spiral":
            continue

        prev = elements[i - 1] if i > 0               else None
        nxt  = elements[i + 1] if i < len(elements)-1 else None

        # radius_start — determined by predecessor
        if prev is not None:
            if prev["type"] == "Line":
                el["radius_start"] = float("inf")
            elif prev["type"] == "Arc":
                el["radius_start"] = prev["radius"]
            elif prev["type"] == "Spiral":
                el["radius_start"] = prev.get("radius_end", el.get("radius_start", float("inf")))

        # radius_end — determined by successor
        if nxt is not None:
            if nxt["type"] == "Line":
                el["radius_end"] = float("inf")
            elif nxt["type"] == "Arc":
                el["radius_end"] = nxt["radius"]
            # Spiral→Spiral: handled when processing the next spiral

        # Recompute clothoid parameter A = sqrt(R_finite * L)
        r_s = el.get("radius_start", float("inf"))
        r_e = el.get("radius_end",   float("inf"))
        finite = [r for r in (r_s, r_e) if not (math.isinf(r) or r > 1e8)]
        el["clothoid_A"] = float(np.sqrt(min(finite) * el["length"])) if finite else 0.0

    return _remove_degenerate_spirals(elements)


def _remove_degenerate_spirals(elements: list[dict]) -> list[dict]:
    """
    Replace any Spiral with radius_start = INF AND radius_end = INF
    with a Line element (no curvature change — geometrically a straight line).
    """
    result = []
    for el in elements:
        if el["type"] == "Spiral":
            r_s = el.get("radius_start", float("inf"))
            r_e = el.get("radius_end",   float("inf"))
            if math.isinf(r_s) and math.isinf(r_e):
                start = el.get("start", [0.0, 0.0])
                end   = el.get("end",   [0.0, 0.0])
                result.append({
                    "type":          "Line",
                    "sta_start":     el["sta_start"],
                    "length":        el["length"],
                    "start":         start,
                    "end":           end,
                    "direction_rad": math.atan2(
                        end[1] - start[1], end[0] - start[0]
                    ),
                })
                continue
        result.append(el)
    return result


# ---------------------------------------------------------------------------
# Accuracy-driven refinement
# ---------------------------------------------------------------------------

def _sample_element_points(el: dict, interval: float) -> list[np.ndarray]:
    """
    Sample the fitted geometric element at approximately *interval* metre spacing.

    Line   — linear interpolation start → end
    Arc    — arc interpolation via center & radius (chord fallback)
    Spiral — linear interpolation along chord (adequate for deviation checking)
    """
    length = el.get("length", 0.0)
    if length < 1e-6:
        return []

    n      = max(2, int(math.ceil(length / interval)) + 1)
    t_vals = np.linspace(0.0, 1.0, n)
    start  = np.array(el["start"], dtype=float)
    end    = np.array(el["end"],   dtype=float)

    if el["type"] == "Arc":
        center = el.get("center")
        radius = el.get("radius", 0.0)
        if center is not None and radius > 0:
            c       = np.array(center, dtype=float)
            a_start = math.atan2(start[1] - c[1], start[0] - c[0])
            a_end   = math.atan2(end[1]   - c[1], end[0]   - c[0])
            if el.get("rot", "ccw") == "cw":
                if a_end > a_start:
                    a_end -= 2 * math.pi
            else:
                if a_end < a_start:
                    a_end += 2 * math.pi
            angles = a_start + t_vals * (a_end - a_start)
            return [np.array([c[0] + radius * math.cos(a),
                               c[1] + radius * math.sin(a)]) for a in angles]

    # Line and Spiral: linear along chord
    return [start + t * (end - start) for t in t_vals]


def _dist_point_to_segment(
    pt: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """Minimum distance from *pt* to finite line segment a–b."""
    ab    = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(pt - a))
    t       = float(np.clip(np.dot(pt - a, ab) / ab_sq, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(pt - closest))


def max_deviation_element(
    el: dict,
    xy_segment: np.ndarray,
    check_interval: float,
) -> float:
    """
    Return the maximum distance (metres) from any point sampled on the fitted
    element to the nearest segment of *xy_segment* (the OSM polyline slice).

    Parameters
    ----------
    el             : fitted element dict
    xy_segment     : (M, 2) OSM polyline nodes covering this element's chainage range
    check_interval : sampling spacing along the element in metres
    """
    if len(xy_segment) < 2:
        return 0.0

    samples = _sample_element_points(el, check_interval)
    if not samples:
        return 0.0

    max_dev = 0.0
    for pt in samples:
        min_d = float("inf")
        for j in range(len(xy_segment) - 1):
            d = _dist_point_to_segment(pt, xy_segment[j], xy_segment[j + 1])
            if d < min_d:
                min_d = d
        if min_d < float("inf"):
            max_dev = max(max_dev, min_d)

    return max_dev


def _refit_range(
    i0: int,
    i1: int,
    xy: np.ndarray,
    kappa: np.ndarray,
    chainages: np.ndarray,
    line_tol: float,
    arc_tol: float,
) -> list[dict]:
    """Fit a single geometric element to the OSM points at indices [i0 … i1]."""
    if i1 <= i0 or i1 >= len(xy):
        return []
    pts       = xy[i0 : i1 + 1]
    sta_start = float(chainages[i0])
    length    = float(chainages[i1] - chainages[i0])
    if length < 0.1 or len(pts) < 2:
        return []

    k_vals  = kappa[i0 : i1 + 1]
    mean_k  = float(np.mean(k_vals))
    k_start = float(k_vals[0])
    k_end   = float(k_vals[-1])

    if abs(mean_k) < line_tol:
        el = _fit_line(pts, sta_start, length)
    elif abs(k_end - k_start) < arc_tol * max(length, 1.0):
        el = _fit_arc(pts, sta_start, length, mean_k)
    else:
        el = _fit_spiral(pts, sta_start, length, k_start, k_end)

    return [el] if el is not None else []


def _refine_elements(
    elements: list[dict],
    xy: np.ndarray,
    chainages: np.ndarray,
    kappa: np.ndarray,
    max_deviation: float,
    check_interval: float,
    min_line_length: float,
    min_arc_length: float,
    min_spiral_length: float,
    line_tol: float,
    arc_tol: float,
    _depth: int = 0,
    _max_depth: int = 8,
) -> list[dict]:
    """
    Recursively split elements whose sampled deviation from the OSM polyline
    exceeds *max_deviation*.

    Each offending element's OSM-point range is bisected and each half is
    re-fitted independently, then enforce_continuity() is applied before the
    next recursion level.
    """
    if _depth >= _max_depth:
        return elements

    min_len_map = {
        "Line":   min_line_length,
        "Arc":    min_arc_length,
        "Spiral": min_spiral_length,
    }

    result: list[dict] = []

    for el in elements:
        sta_s = float(el.get("sta_start", 0.0))
        sta_e = sta_s + float(el.get("length", 0.0))

        i0 = int(np.searchsorted(chainages, sta_s - 1e-4, side="left"))
        i1 = int(np.searchsorted(chainages, sta_e + 1e-4, side="right")) - 1
        i0 = max(0, min(i0, len(xy) - 1))
        i1 = max(i0, min(i1, len(xy) - 1))

        if i1 <= i0 + 1:
            result.append(el)
            continue

        xy_seg  = xy[i0 : i1 + 1]
        dev     = max_deviation_element(el, xy_seg, check_interval)
        min_len = min_len_map.get(el.get("type", "Line"), min_line_length)

        # Within tolerance or too short to split → keep as-is
        if dev <= max_deviation or el.get("length", 0.0) < min_len * 2:
            result.append(el)
            continue

        # Bisect and refit
        i_mid = (i0 + i1) // 2
        if i_mid == i0 or i_mid == i1:
            result.append(el)
            continue

        left_els  = _refit_range(i0, i_mid, xy, kappa, chainages, line_tol, arc_tol)
        right_els = _refit_range(i_mid, i1, xy, kappa, chainages, line_tol, arc_tol)
        sub       = enforce_continuity(left_els + right_els)

        sub = _refine_elements(
            sub, xy, chainages, kappa,
            max_deviation, check_interval,
            min_line_length, min_arc_length, min_spiral_length,
            line_tol, arc_tol,
            _depth=_depth + 1, _max_depth=_max_depth,
        )
        result.extend(sub)

    return result


# ---------------------------------------------------------------------------
# LandXML alignment reconstruction
# ---------------------------------------------------------------------------

def reconstruct_alignment_projected(
    elements: list[dict],
    sample_interval: float = 5.0,
) -> list[tuple[float, float]]:
    """
    Generate a dense sequence of (x, y) points in projected coordinates by
    traversing the fitted geometric elements in order.

    • Line   — linear interpolation along the fitted start→end vector
    • Arc    — arc interpolation using center & radius (chord if center missing)
    • Spiral — numerical integration of the clothoid curvature profile

    The initial heading is taken from the first element and propagated forward
    so that every subsequent element's direction is continuous.

    Parameters
    ----------
    elements        : list of element dicts (output of fit_alignment / enforce_continuity)
    sample_interval : target spacing between output points in metres

    Returns
    -------
    Flat list of (x, y) tuples — dense enough to look smooth on a map.
    """
    all_pts: list[tuple[float, float]] = []
    cur_heading: float | None = None

    for el in elements:
        pts, cur_heading = _sample_element_heading(el, sample_interval, cur_heading)
        if not pts:
            continue
        if all_pts:
            all_pts.extend(pts[1:])   # skip duplicate junction point
        else:
            all_pts.extend(pts)

    return all_pts


def reconstruct_alignment_per_element(
    elements: list[dict],
    sample_interval: float = 2.0,
) -> list[tuple[dict, list[tuple[float, float]]]]:
    """
    Per-element variant of `reconstruct_alignment_projected`.

    Returns a list of ``(element_dict, [(x, y), ...])`` pairs, one entry per
    input element. Adjacent segments share their junction point (no point is
    skipped) so the per-element polylines visually connect on the map.

    Used by the GUI to render per-element coloured polylines with hover
    tooltips containing each element's parameters.

    Parameters
    ----------
    elements        : list of element dicts
    sample_interval : target spacing between output points in metres

    Returns
    -------
    list[(element_dict_ref, list[(x, y)])]
    """
    result: list[tuple[dict, list[tuple[float, float]]]] = []
    cur_heading: float | None = None

    for el in elements:
        pts, cur_heading = _sample_element_heading(el, sample_interval, cur_heading)
        # Defensive: drop NaN/inf samples
        clean = [(float(x), float(y)) for (x, y) in (pts or [])
                 if math.isfinite(x) and math.isfinite(y)]
        result.append((el, clean))

    return result


def _sample_element_heading(
    el: dict,
    interval: float,
    prev_heading: float | None,
) -> tuple[list[tuple[float, float]], float]:
    """
    Sample one element and return (points, end_heading).
    end_heading is used as prev_heading for the next element.
    """
    etype  = el.get("type", "Line")
    start  = np.array(el.get("start", [0.0, 0.0]), dtype=float)
    end    = np.array(el.get("end",   [0.0, 0.0]), dtype=float)
    length = float(el.get("length", 0.0))

    if length < 1e-6:
        h = prev_heading if prev_heading is not None else 0.0
        return [tuple(start)], h

    n      = max(2, int(math.ceil(length / interval)) + 1)
    t_vals = np.linspace(0.0, 1.0, n)

    # ── Line ──────────────────────────────────────────────────────────
    if etype == "Line":
        pts = [tuple(start + t * (end - start)) for t in t_vals]
        h   = math.atan2(float(end[1] - start[1]), float(end[0] - start[0]))
        return pts, h

    # ── Arc ───────────────────────────────────────────────────────────
    if etype == "Arc":
        radius = float(el.get("radius", 0.0))
        rot    = el.get("rot", "ccw")
        sign   = 1.0 if rot == "ccw" else -1.0
        center = el.get("center")

        # Reconstruct center from start + heading if missing
        if (center is None or radius <= 0) and prev_heading is not None and radius > 0:
            perp = prev_heading + sign * math.pi / 2
            center = [float(start[0]) + radius * math.cos(perp),
                      float(start[1]) + radius * math.sin(perp)]

        if center is not None and radius > 0:
            c       = np.array(center, dtype=float)
            a_start = math.atan2(float(start[1] - c[1]), float(start[0] - c[0]))
            a_end   = math.atan2(float(end[1]   - c[1]), float(end[0]   - c[0]))

            # Unwrap so the arc goes in the correct direction
            if rot == "cw":
                if a_end > a_start:
                    a_end -= 2 * math.pi
            else:
                if a_end < a_start:
                    a_end += 2 * math.pi

            angles = a_start + t_vals * (a_end - a_start)
            pts = [(float(c[0] + radius * math.cos(a)),
                    float(c[1] + radius * math.sin(a))) for a in angles]

            # Tangent at end = perpendicular to radius vector
            h_end = a_end + sign * math.pi / 2
            return pts, h_end

        # Fallback: chord interpolation
        pts = [tuple(start + t * (end - start)) for t in t_vals]
        h   = math.atan2(float(end[1] - start[1]), float(end[0] - start[0]))
        return pts, h

    # ── Spiral (Clothoid) ─────────────────────────────────────────────
    if etype == "Spiral":
        return _sample_spiral_heading(el, interval, prev_heading, start, end, length)

    # Fallback for unknown types
    pts = [tuple(start + t * (end - start)) for t in t_vals]
    h   = math.atan2(float(end[1] - start[1]), float(end[0] - start[0]))
    return pts, h


def _sample_spiral_heading(
    el: dict,
    interval: float,
    prev_heading: float | None,
    start: np.ndarray,
    end: np.ndarray,
    length: float,
) -> tuple[list[tuple[float, float]], float]:
    """
    Numerically integrate the clothoid using κ(s) = κ_start + (κ_end−κ_start)·s/L.

    Position is computed by the midpoint rule:
        θ(s) = θ₀ + sign · ∫₀ˢ κ(t) dt
             = θ₀ + sign · [κ_start·s + (κ_end−κ_start)·s²/(2L)]

        x(s) = x₀ + ∫₀ˢ cos θ(t) dt    (midpoint rule)
        y(s) = y₀ + ∫₀ˢ sin θ(t) dt

    The last point is snapped to the declared end coordinate to absorb
    any integration drift caused by OSM-fitting inaccuracies.
    """
    r_s  = el.get("radius_start", float("inf"))
    r_e  = el.get("radius_end",   float("inf"))
    rot  = el.get("rot", "ccw")
    sign = 1.0 if rot == "ccw" else -1.0

    k_s = 0.0 if math.isinf(r_s) else 1.0 / float(r_s)
    k_e = 0.0 if math.isinf(r_e) else 1.0 / float(r_e)

    # Initial heading: use predecessor's end heading, else estimate from chord
    theta_0 = prev_heading if prev_heading is not None else math.atan2(
        float(end[1] - start[1]), float(end[0] - start[0])
    )

    # Fine integration steps (at least 50, at most ~200 per element)
    n_int = max(50, min(200, int(length / max(interval * 0.1, 0.2))))
    ds    = length / n_int

    pts_raw: list[tuple[float, float]] = [tuple(start)]
    x, y = float(start[0]), float(start[1])

    for i in range(1, n_int + 1):
        s_mid    = (i - 0.5) * ds
        theta_m  = theta_0 + sign * (k_s * s_mid +
                                     (k_e - k_s) * s_mid ** 2 / (2.0 * length))
        x += ds * math.cos(theta_m)
        y += ds * math.sin(theta_m)
        pts_raw.append((x, y))

    # Snap last point to the declared endpoint (absorbs integration drift)
    pts_raw[-1] = (float(end[0]), float(end[1]))

    # Downsample to approximately the requested interval
    step = max(1, int(round(interval / ds)))
    pts  = pts_raw[::step]
    if len(pts) < 2 or pts[-1] != pts_raw[-1]:
        pts.append(pts_raw[-1])

    # Analytical end heading
    h_end = theta_0 + sign * (k_s * length + (k_e - k_s) * length / 2.0)
    return pts, h_end


# ---------------------------------------------------------------------------
# Phase B — Geometry core
# ---------------------------------------------------------------------------

def _compute_clothoid_shift(L: float, R: float) -> tuple[float, float]:
    """
    Compute clothoid spiral endpoint in local frame (tangent along +x at s=0).

    The clothoid has κ(s) = s/A² where A² = R*L. The endpoint is:
        x_end = integral_0^L cos(s²/(2A²)) ds    [Fresnel C integral]
        y_end = integral_0^L sin(s²/(2A²)) ds    [Fresnel S integral]

    Parameters
    ----------
    L : spiral arc-length (metres)
    R : circular arc radius (metres)

    Returns
    -------
    (x_end, y_end) : clothoid endpoint in local frame
    """
    from scipy.special import fresnel

    A2 = R * L
    A  = math.sqrt(A2)

    t    = L / (A * math.sqrt(math.pi))
    S, C = fresnel(t)        # note: scipy returns (S, C), i.e. sin-integral first
    scale = A * math.sqrt(math.pi)
    x_end = scale * C
    y_end = scale * S

    return x_end, y_end


def _compute_zone_geometry(
    TC: np.ndarray,
    dir_in: float,
    delta: float,
    R: float,
    L_entry: float,
    L_exit: float,
    n_pts: int = 40,
) -> dict | None:
    """
    Compute the full L-S-A-S-L geometry for one arc zone analytically.

    Tangency is guaranteed by construction.

    Parameters
    ----------
    TC      : (2,) start of entry spiral (world frame)
    dir_in  : heading (rad) of incoming tangent at TC
    delta   : signed total deflection (rad); positive = CCW
    R       : circular arc radius (metres)
    L_entry : entry spiral arc-length (metres, 0 = no spiral)
    L_exit  : exit spiral arc-length (metres, 0 = no spiral)
    n_pts   : number of sample points on each spiral

    Returns
    -------
    dict or None if parameters are geometrically invalid (arc_angle <= 0).
    """
    sign = 1.0 if delta >= 0 else -1.0
    abs_delta = abs(delta)

    # Angles subtended by each spiral = L/(2R)
    theta_entry = L_entry / (2.0 * R) if L_entry > 0 else 0.0
    theta_exit  = L_exit  / (2.0 * R) if L_exit  > 0 else 0.0
    arc_angle   = abs_delta - theta_entry - theta_exit

    if arc_angle < 1e-6:
        return None  # spirals consume all deflection

    # ── Entry spiral ────────────────────────────────────────────────────
    if L_entry > 0:
        x_sp, y_sp = _compute_clothoid_shift(L_entry, R)
        cos_d = math.cos(dir_in)
        sin_d = math.sin(dir_in)
        sp_x = float(TC[0]) + cos_d * x_sp - sign * sin_d * y_sp
        sp_y = float(TC[1]) + sin_d * x_sp + sign * cos_d * y_sp
        arc_start = np.array([sp_x, sp_y])
        dir_arc_start = dir_in + sign * theta_entry
        # Sample spiral points
        spiral_entry_pts = []
        for frac in np.linspace(0, 1, n_pts + 1):
            L_s = frac * L_entry
            xe, ye = _compute_clothoid_shift(L_s, R) if L_s > 0 else (0.0, 0.0)
            wx = float(TC[0]) + cos_d * xe - sign * sin_d * ye
            wy = float(TC[1]) + sin_d * xe + sign * cos_d * ye
            spiral_entry_pts.append([wx, wy])
        spiral_entry_pts = np.array(spiral_entry_pts)
    else:
        arc_start = TC.copy()
        dir_arc_start = dir_in
        spiral_entry_pts = np.array([TC, TC])

    # ── Circular arc ─────────────────────────────────────────────────────
    perp = dir_arc_start + sign * math.pi / 2.0
    center = arc_start + R * np.array([math.cos(perp), math.sin(perp)])

    a_start = math.atan2(arc_start[1] - center[1], arc_start[0] - center[0])
    a_end   = a_start + sign * arc_angle
    arc_end = center + R * np.array([math.cos(a_end), math.sin(a_end)])
    arc_len = R * arc_angle
    dir_arc_end = dir_arc_start + sign * arc_angle

    # ── Exit spiral ──────────────────────────────────────────────────────
    if L_exit > 0:
        x_sp, y_sp = _compute_clothoid_shift(L_exit, R)
        cos_e = math.cos(dir_arc_end)
        sin_e = math.sin(dir_arc_end)
        ct_x = float(arc_end[0]) + cos_e * x_sp - sign * sin_e * y_sp
        ct_y = float(arc_end[1]) + sin_e * x_sp + sign * cos_e * y_sp
        CT = np.array([ct_x, ct_y])
        spiral_exit_pts = []
        for frac in np.linspace(0, 1, n_pts + 1):
            L_s = frac * L_exit
            xe, ye = _compute_clothoid_shift(L_s, R) if L_s > 0 else (0.0, 0.0)
            wx = float(arc_end[0]) + cos_e * xe - sign * sin_e * ye
            wy = float(arc_end[1]) + sin_e * xe + sign * cos_e * ye
            spiral_exit_pts.append([wx, wy])
        spiral_exit_pts = np.array(spiral_exit_pts)
    else:
        CT = arc_end.copy()
        spiral_exit_pts = np.array([arc_end, arc_end])

    dir_out = dir_in + sign * abs_delta  # exact by construction

    rot = "ccw" if delta >= 0 else "cw"

    return {
        "TC":               TC.copy(),
        "arc_start":        arc_start,
        "arc_center":       center,
        "arc_end":          arc_end,
        "CT":               CT,
        "dir_in":           dir_in,
        "dir_out":          dir_out,
        "dir_arc_start":    dir_arc_start,
        "dir_arc_end":      dir_arc_end,
        "arc_angle":        arc_angle,
        "arc_len":          arc_len,
        "rot":              rot,
        "spiral_entry_pts": spiral_entry_pts,
        "spiral_exit_pts":  spiral_exit_pts,
        "R":                R,
        "L_entry":          L_entry,
        "L_exit":           L_exit,
    }


# ---------------------------------------------------------------------------
# Phase C — Detection pipeline
# ---------------------------------------------------------------------------

def _simplify_polyline(xy: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification via Shapely."""
    from shapely.geometry import LineString
    line = LineString(xy.tolist())
    simp = line.simplify(tolerance, preserve_topology=False)
    coords = np.array(simp.coords)
    # Always preserve original first and last point
    if not np.allclose(coords[0], xy[0]):
        coords = np.vstack([xy[0:1], coords])
    if not np.allclose(coords[-1], xy[-1]):
        coords = np.vstack([coords, xy[-1:]])
    return coords


def _find_PI(p0: np.ndarray, dir0: float,
             p1: np.ndarray, dir1: float) -> np.ndarray | None:
    """
    Intersection of two rays: p0 + t*d0 and p1 + s*d1.
    Returns None if rays are nearly parallel (|sin(angle)| < 1e-6).
    """
    d0 = np.array([math.cos(dir0), math.sin(dir0)])
    d1 = np.array([math.cos(dir1), math.sin(dir1)])
    cross = float(d0[0] * d1[1] - d0[1] * d1[0])
    if abs(cross) < 1e-6:
        return None
    diff = p1 - p0
    t = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
    return p0 + t * d0


def _detect_arc_zones(
    simp: np.ndarray,
    angle_threshold_rad: float,
    min_straight_length: float = 50.0,
) -> list[dict]:
    """
    Find arc zones in the simplified polyline.

    An arc zone is an internal vertex of simp with |turning angle| >= threshold.
    Adjacent zones separated by a straight segment shorter than min_straight_length
    are merged into one combined zone.
    """
    n = len(simp)
    zones = []

    for i in range(1, n - 1):
        p_prev = simp[i - 1]
        p_curr = simp[i]
        p_next = simp[i + 1]

        d_in  = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        d_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        len_in  = float(np.linalg.norm(d_in))
        len_out = float(np.linalg.norm(d_out))
        if len_in < 1e-6 or len_out < 1e-6:
            continue

        dir_in  = math.atan2(float(d_in[1]),  float(d_in[0]))
        dir_out = math.atan2(float(d_out[1]), float(d_out[0]))
        # Signed turning angle: positive = left/CCW
        delta = (dir_out - dir_in + math.pi) % (2 * math.pi) - math.pi

        if abs(delta) < angle_threshold_rad:
            continue

        PI = _find_PI(p_prev, dir_in, p_next, dir_out)
        if PI is None:
            PI = p_curr.copy()  # fallback: PI at vertex

        zones.append({
            "entry_pt":       p_prev.copy(),
            "vertex_pt":      p_curr.copy(),
            "exit_pt":        p_next.copy(),
            "dir_in":         dir_in,
            "dir_out":        dir_out,
            "delta":          delta,
            "PI":             PI,
            "simp_idx":       i,
        })

    # Merge adjacent zones with very short straight between them
    if len(zones) < 2:
        return zones

    merged = [zones[0]]
    for z in zones[1:]:
        prev = merged[-1]
        gap = float(np.linalg.norm(z["entry_pt"] - prev["exit_pt"]))
        if gap < min_straight_length:
            combined_delta = prev["delta"] + z["delta"]
            dir_out_new = z["dir_out"]
            PI_new = _find_PI(prev["entry_pt"], prev["dir_in"],
                              z["exit_pt"], dir_out_new)
            if PI_new is None:
                PI_new = z["vertex_pt"]
            merged[-1] = {
                "entry_pt":   prev["entry_pt"],
                "vertex_pt":  prev["vertex_pt"],
                "exit_pt":    z["exit_pt"],
                "dir_in":     prev["dir_in"],
                "dir_out":    dir_out_new,
                "delta":      combined_delta,
                "PI":         PI_new,
                "simp_idx":   prev["simp_idx"],
            }
        else:
            merged.append(z)

    return merged


def _map_osm_to_zones(
    xy: np.ndarray,
    chainages: np.ndarray,
    simp: np.ndarray,
    arc_zones: list[dict],
) -> None:
    """
    For each arc zone, find the OSM points that fall within the zone's
    chainage range and store them in the zone dict (mutates in-place).
    """
    def _nearest_chainage(pt: np.ndarray) -> float:
        dists = np.linalg.norm(xy - pt, axis=1)
        idx = int(np.argmin(dists))
        return float(chainages[idx])

    for zone in arc_zones:
        ch_start = _nearest_chainage(zone["entry_pt"])
        ch_end   = _nearest_chainage(zone["exit_pt"])
        if ch_end < ch_start:
            ch_start, ch_end = ch_end, ch_start
        mask = (chainages >= ch_start) & (chainages <= ch_end)
        zone["osm_pts"]        = xy[mask]
        zone["chainage_start"] = ch_start
        zone["chainage_end"]   = ch_end


# ---------------------------------------------------------------------------
# Phase D — Optimizer
# ---------------------------------------------------------------------------

def _dist_to_polyline_segments(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    For each point in pts, compute distance to the nearest segment of poly.
    Fast vectorised implementation.
    Returns array of shape (len(pts),).
    """
    dists = np.full(len(pts), np.inf)
    for j in range(len(poly) - 1):
        a = poly[j];  b = poly[j + 1]
        ab  = b - a
        len2 = float(np.dot(ab, ab))
        if len2 < 1e-12:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / len2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(pts - proj, axis=1)
        dists = np.minimum(dists, d)
    return dists


def _zone_objective(
    params: np.ndarray,
    dir_in: float,
    delta: float,
    PI: np.ndarray,
    osm_pts: np.ndarray,
    min_radius: float,
    min_spiral_len: float,
    max_Ts_entry: float,
    max_Ts_exit: float,
) -> float:
    """Sum of squared distances from OSM zone points to fitted geometry + penalties."""
    R, L_entry, L_exit = float(params[0]), float(params[1]), float(params[2])
    PENALTY = 1e8

    if R < min_radius * 0.99:
        return PENALTY + (min_radius - R) ** 2 * 1e6
    if L_entry < 0 or L_exit < 0:
        return PENALTY

    abs_delta = abs(delta)
    sign = 1.0 if delta >= 0 else -1.0

    theta_entry = L_entry / (2.0 * R) if L_entry > 0 else 0.0
    theta_exit  = L_exit  / (2.0 * R) if L_exit  > 0 else 0.0

    if theta_entry + theta_exit >= abs_delta - 1e-4:
        return PENALTY  # arc angle would be zero or negative

    # Compute TC from PI
    try:
        x_end_e, y_end_e = (_compute_clothoid_shift(L_entry, R)
                             if L_entry > 0 else (0.0, 0.0))
        theta_s_e = L_entry / (2.0 * R) if L_entry > 0 else 0.0
        k_entry = x_end_e - R * math.sin(theta_s_e) if L_entry > 0 else 0.0
        Ts_entry = R * math.tan(abs_delta / 2.0) + k_entry
    except (ValueError, ZeroDivisionError):
        return PENALTY

    if Ts_entry > max_Ts_entry * 1.05:
        return PENALTY + (Ts_entry - max_Ts_entry) ** 2 * 1e4

    dir_in_hat = np.array([math.cos(dir_in), math.sin(dir_in)])
    TC = PI - Ts_entry * dir_in_hat

    geom = _compute_zone_geometry(TC, dir_in, delta, R, L_entry, L_exit)
    if geom is None:
        return PENALTY

    # Build sample points: spiral_entry + arc_mid + spiral_exit
    sample_pts = []
    if len(geom["spiral_entry_pts"]) > 2:
        sample_pts.append(geom["spiral_entry_pts"])

    center = geom["arc_center"]
    a_s = math.atan2(geom["arc_start"][1] - center[1],
                     geom["arc_start"][0] - center[0])
    n_arc = max(5, int(geom["arc_len"] / 10))
    arc_angles = a_s + np.linspace(0, sign * geom["arc_angle"], n_arc)
    arc_pts = center + geom["R"] * np.column_stack(
        [np.cos(arc_angles), np.sin(arc_angles)]
    )
    sample_pts.append(arc_pts)

    if len(geom["spiral_exit_pts"]) > 2:
        sample_pts.append(geom["spiral_exit_pts"])

    all_pts = np.vstack(sample_pts)

    if len(osm_pts) < 2:
        return 0.0

    # Measure from OSM points to the fitted curve (not vice-versa).
    # This penalises OSM data points that lie far from ANY part of the fitted
    # geometry, including the spiral transition regions.  The symmetric
    # (reversed) direction would allow arc-only solutions to score perfectly
    # by simply not sampling the spiral zones.
    dists = _dist_to_polyline_segments(osm_pts, all_pts)
    return float(np.sum(dists ** 2))


def _annotate_spiral_hints_from_kappa(
    arc_zones: list[dict],
    kappa_smooth: np.ndarray,
    chainages: np.ndarray,
    noise_floor: float = 3e-5,
) -> None:
    """
    Estimate entry/exit spiral lengths for each zone from the curvature profile
    and store them as ``spiral_entry_hint`` / ``spiral_exit_hint`` in the zone dict.

    Algorithm
    ---------
    For the entry spiral:
      1. Find the zone chainage_start_raw (where kappa first exceeded line_tol).
      2. Walk backward in kappa_smooth until kappa drops to *noise_floor* (≈ 0).
         This gives ch_TC_est, the estimated start of the spiral.
      3. Walk forward from ch_TC_est until kappa_smooth plateaus (successive
         changes < 10 % of running mean).  This gives ch_arc_start_est.
      4. entry_hint = ch_arc_start_est − ch_TC_est.

    Symmetric logic for the exit spiral.
    """
    n = len(kappa_smooth)

    def _ch_to_idx(ch: float) -> int:
        return max(0, min(int(np.searchsorted(chainages, ch)), n - 1))

    for zone in arc_zones:
        ch_start_raw = float(zone.get("chainage_start_raw",
                                       zone["chainage_start"]))
        ch_end_raw   = float(zone.get("chainage_end_raw",
                                       zone["chainage_end"]))

        # ── Entry ────────────────────────────────────────────────────────
        idx_cross = _ch_to_idx(ch_start_raw)

        # Walk backward until kappa drops to noise floor
        i = idx_cross
        while i > 0 and abs(kappa_smooth[i]) > noise_floor:
            i -= 1
        ch_TC_est = float(chainages[i])

        # Walk forward from ch_TC_est to find where kappa plateaus
        # (gradient ≈ 0 = end of spiral, start of arc)
        i2 = i
        window = max(3, (idx_cross - i) // 3)  # adaptive window
        kappa_run = abs(kappa_smooth[i2])
        while i2 < n - 1:
            i2 += 1
            kappa_new = abs(kappa_smooth[i2])
            if kappa_new < noise_floor:
                break  # back to straight
            change = abs(kappa_new - kappa_run)
            if kappa_run > noise_floor and change / kappa_run < 0.05:
                # Plateau detected
                break
            kappa_run = kappa_run * 0.7 + kappa_new * 0.3  # EWM
        ch_arc_start_est = float(chainages[i2])
        entry_hint = max(ch_arc_start_est - ch_TC_est, 0.0)

        # ── Exit ─────────────────────────────────────────────────────────
        idx_cross_e = _ch_to_idx(ch_end_raw)

        # Walk forward until kappa drops to noise floor
        j = idx_cross_e
        while j < n - 1 and abs(kappa_smooth[j]) > noise_floor:
            j += 1
        ch_CT_est = float(chainages[j])

        # Walk backward from ch_CT_est to find where kappa plateaus
        j2 = j
        kappa_run2 = abs(kappa_smooth[j2])
        while j2 > 0:
            j2 -= 1
            kappa_new2 = abs(kappa_smooth[j2])
            if kappa_new2 < noise_floor:
                break
            change2 = abs(kappa_new2 - kappa_run2)
            if kappa_run2 > noise_floor and change2 / kappa_run2 < 0.05:
                break
            kappa_run2 = kappa_run2 * 0.7 + kappa_new2 * 0.3
        ch_arc_end_est = float(chainages[j2])
        exit_hint = max(ch_CT_est - ch_arc_end_est, 0.0)

        # Overwrite expansion-based hints with the kappa-profile-based estimates
        # (these are generally more accurate since they directly measure the
        # curvature transition region).
        zone["spiral_entry_hint"] = entry_hint
        zone["spiral_exit_hint"]  = exit_hint


def _expand_zone_boundaries(
    arc_zones: list[dict],
    kappa_smooth: np.ndarray,
    chainages: np.ndarray,
    xy: np.ndarray,
    kappa_expand_tol: float = 0.00025,
) -> list[dict]:
    """
    Extend each arc zone boundary into adjacent "straight" LINE segments to
    capture the low-curvature tails of entry/exit spirals.

    The first (last) ~30% of a clothoid spiral has kappa < line_tol and is
    therefore classified as LINE by segment_curvature.  Without this expansion,
    TC is placed inside the spiral, the fitted spiral is too short, and
    max_deviation_element reports large errors.

    For the entry boundary: walk backward from chainage_start while
    kappa_smooth > kappa_expand_tol, but stop before the previous zone's end.

    For the exit boundary: walk forward from chainage_end while
    kappa_smooth > kappa_expand_tol, but stop before the next zone's start.

    dir_in / dir_out are recomputed from the truly-straight portion that
    remains BEFORE the expanded entry (or AFTER the expanded exit).
    """
    if not arc_zones:
        return arc_zones

    n = len(kappa_smooth)

    def _ch_to_idx(ch: float) -> int:
        return max(0, min(int(np.searchsorted(chainages, ch)), n - 1))

    new_zones = []
    for k, zone in enumerate(arc_zones):
        zone = dict(zone)  # shallow copy so we can mutate

        # ── Expand entry backward ────────────────────────────────────────
        # Hard lower bound: previous zone's end (or alignment start)
        lo_limit = arc_zones[k - 1]["chainage_end"] if k > 0 else 0.0
        idx = _ch_to_idx(zone["chainage_start"])
        while idx > 0 and float(chainages[idx]) > lo_limit:
            if abs(kappa_smooth[idx - 1]) < kappa_expand_tol:
                break
            idx -= 1
        new_ch_start = float(chainages[idx])

        # Recompute dir_in from the flat region BEFORE the extended entry.
        # Use a window of up to 10 % of the straight length for stability.
        flat_start_idx = _ch_to_idx(lo_limit)
        flat_end_idx   = max(flat_start_idx, idx - 1)
        mid_idx = flat_start_idx + max(1, (flat_end_idx - flat_start_idx) // 2)
        mid_idx = min(mid_idx, flat_end_idx)
        if flat_end_idx > flat_start_idx and mid_idx < flat_end_idx:
            new_dir_in = math.atan2(
                float(xy[flat_end_idx, 1] - xy[mid_idx, 1]),
                float(xy[flat_end_idx, 0] - xy[mid_idx, 0]),
            )
        else:
            new_dir_in = zone["dir_in"]

        # ── Expand exit forward ──────────────────────────────────────────
        hi_limit = arc_zones[k + 1]["chainage_start"] if k < len(arc_zones) - 1 else float(chainages[-1])
        idx2 = _ch_to_idx(zone["chainage_end"])
        while idx2 < n - 1 and float(chainages[idx2]) < hi_limit:
            if abs(kappa_smooth[idx2 + 1]) < kappa_expand_tol:
                break
            idx2 += 1
        new_ch_end = float(chainages[idx2])

        # Recompute dir_out from the flat region AFTER the extended exit.
        flat_start_idx2 = max(idx2 + 1, 0)
        flat_end_idx2   = _ch_to_idx(hi_limit)
        mid_idx2 = flat_start_idx2 + max(1, (flat_end_idx2 - flat_start_idx2) // 2)
        mid_idx2 = min(mid_idx2, flat_end_idx2)
        if flat_end_idx2 > flat_start_idx2 and flat_start_idx2 < flat_end_idx2:
            new_dir_out = math.atan2(
                float(xy[flat_end_idx2, 1] - xy[mid_idx2, 1]),
                float(xy[flat_end_idx2, 0] - xy[mid_idx2, 0]),
            )
        else:
            new_dir_out = zone["dir_out"]

        # Recompute delta and PI
        new_delta = (new_dir_out - new_dir_in + math.pi) % (2 * math.pi) - math.pi

        # Compute PI from points that are CLEARLY on the straight (flat) regions,
        # not from inside the spiral/curved zone.  The curved zone boundary
        # (zone["entry_pt"]) is at ch_start_raw which is already inside the
        # spiral — using it shifts PI off the true tangent line, placing TC
        # several metres inside the curve.
        #
        # flat_end_idx   = last point BEFORE the entry expansion  (y ≈ 0 for straight)
        # flat_start_idx2 = first point AFTER the exit  expansion  (on exit straight)
        if flat_end_idx > flat_start_idx:
            orig_entry_pt = xy[flat_end_idx].copy()
        else:
            orig_entry_pt = xy[idx].copy()  # fallback: expanded boundary itself

        if flat_start_idx2 < flat_end_idx2:
            orig_exit_pt = xy[flat_start_idx2].copy()
        else:
            orig_exit_pt = xy[idx2].copy()  # fallback

        PI_new = _find_PI(orig_entry_pt, new_dir_in, orig_exit_pt, new_dir_out)
        if PI_new is None:
            PI_new = (orig_entry_pt + orig_exit_pt) / 2.0

        entry_pt = xy[_ch_to_idx(new_ch_start)].copy()
        exit_pt  = xy[_ch_to_idx(new_ch_end)].copy()

        # Combine full osm_pts for the expanded zone
        i0 = _ch_to_idx(new_ch_start)
        i1 = _ch_to_idx(new_ch_end)
        osm_combined = xy[i0 : i1 + 1].copy()

        # Expansion lengths — used as spiral-length hints in _fit_arc_zone.
        # kappa crosses line_tol at ch_start_raw, and drops to expand_tol at
        # ch_start (new).  The full entry spiral is approximately:
        #   L_entry_hint ≈ entry_expansion / (1 - line_tol / max_kappa_in_zone)
        # We use a simpler heuristic: extrapolate assuming kappa is linear from
        # 0 at ch=TC_est to 1/R at ch=arc_start.  Since kappa at ch_start is
        # ~expand_tol and at ch_start_raw is ~line_tol:
        #   (ch_start_raw - ch_start) / L_spiral = (line_tol - expand_tol) / (1/R)
        # We don't know R yet, so we store the raw expansion length and scale
        # by factor = line_tol / expand_tol = 4 → full spiral hint ≈ 4× expansion.
        entry_expansion = zone.get("chainage_start_raw", new_ch_start) - new_ch_start
        exit_expansion  = new_ch_end - zone.get("chainage_end_raw", new_ch_end)
        scale = max(kappa_expand_tol, 1e-9)
        approx_scale = 1.0 / scale  # 1/expand_tol; multiplied by expand_tol gives "1"
        # Simpler: hint = expansion * (line_tol/expand_tol + 1) which gives the
        # fraction of spiral that crosses line_tol.  For line_tol = 4*expand_tol:
        #   hint ≈ expansion * 5 → rough estimate of total spiral.
        lt = kappa_expand_tol * 4.0   # approximate line_tol (may differ)
        entry_hint = entry_expansion * (lt / kappa_expand_tol + 1.0)
        exit_hint  = exit_expansion  * (lt / kappa_expand_tol + 1.0)

        zone.update({
            "chainage_start": new_ch_start,
            "chainage_end":   new_ch_end,
            "dir_in":         new_dir_in,
            "dir_out":        new_dir_out,
            "delta":          new_delta,
            "PI":             PI_new,
            "entry_pt":       entry_pt,
            "exit_pt":        exit_pt,
            "osm_pts":        osm_combined,
            "spiral_entry_hint": max(entry_hint, 0.0),
            "spiral_exit_hint":  max(exit_hint,  0.0),
        })
        new_zones.append(zone)

    return new_zones


def _merge_phantom_zones(
    xy: np.ndarray,
    chainages: np.ndarray,
    arc_zones: list[dict],
    merge_dev_thresh: float = 1.0,
) -> list[dict]:
    """
    Merge pairs of adjacent arc zones when the straight gap between them is
    actually curved (i.e. the OSM points in the gap deviate more than
    *merge_dev_thresh* from the chord connecting the gap endpoints).

    This corrects the case where curvature smoothing mis-classifies part of a
    spiral as LINE, splitting one smooth curve into two separate zones.
    """
    if len(arc_zones) <= 1:
        return arc_zones

    changed = True
    while changed:
        changed = False
        new_zones: list[dict] = []
        i = 0
        while i < len(arc_zones):
            if i == len(arc_zones) - 1:
                new_zones.append(arc_zones[i])
                i += 1
                continue

            za = arc_zones[i]
            zb = arc_zones[i + 1]

            # Extract OSM points in the gap between zones
            ch0 = za["chainage_end"]
            ch1 = zb["chainage_start"]
            mask = (chainages >= ch0) & (chainages <= ch1)
            gap_pts = xy[mask]

            # Compute maximum perpendicular deviation of the gap from its chord
            phantom = False
            if len(gap_pts) >= 3:
                chord = gap_pts[-1] - gap_pts[0]
                chord_len = float(np.linalg.norm(chord))
                if chord_len > 1e-6:
                    # Unit normal to chord (perpendicular direction)
                    normal = np.array([-chord[1], chord[0]]) / chord_len
                    perp_devs = np.abs((gap_pts - gap_pts[0]) @ normal)
                    if float(np.max(perp_devs)) > merge_dev_thresh:
                        phantom = True

            if phantom:
                # Merge za and zb into one zone
                new_delta = (
                    (zb["dir_out"] - za["dir_in"] + math.pi) % (2 * math.pi) - math.pi
                )
                PI_new = _find_PI(za["entry_pt"], za["dir_in"],
                                  zb["exit_pt"], zb["dir_out"])
                if PI_new is None:
                    PI_new = (za["entry_pt"] + zb["exit_pt"]) / 2.0

                # Combine OSM points (zone a + gap + zone b)
                osm_combined = np.vstack([
                    za.get("osm_pts", np.empty((0, 2))),
                    gap_pts,
                    zb.get("osm_pts", np.empty((0, 2))),
                ])

                merged: dict = {
                    "entry_pt":       za["entry_pt"],
                    "exit_pt":        zb["exit_pt"],
                    "dir_in":         za["dir_in"],
                    "dir_out":        zb["dir_out"],
                    "delta":          new_delta,
                    "PI":             PI_new,
                    "osm_pts":        osm_combined,
                    "chainage_start": za["chainage_start"],
                    "chainage_end":   zb["chainage_end"],
                    "simp_idx":       za.get("simp_idx", 0),
                }
                new_zones.append(merged)
                i += 2
                changed = True
            else:
                new_zones.append(za)
                i += 1

        arc_zones = new_zones

    return arc_zones


def _detect_arc_zones_from_segments(
    segments: list[dict],
    chainages: np.ndarray,
    xy: np.ndarray,
) -> list[dict]:
    """
    Group consecutive non-LINE curvature segments into a single arc zone dict
    that is compatible with _fit_arc_zone.

    Uses the last few points of the preceding Line and the first few points of
    the following Line as the tangent direction at each zone boundary.

    Returns list of zone dicts with keys:
        entry_pt, exit_pt, dir_in, dir_out, delta, PI,
        osm_pts, chainage_start, chainage_end, simp_idx (placeholder)
    """
    from .curvature import ElementType

    zones: list[dict] = []
    n = len(segments)
    i = 0

    while i < n:
        if segments[i]["type"] == ElementType.LINE:
            i += 1
            continue

        # Find extent of this curved run
        j = i
        while j < n and segments[j]["type"] != ElementType.LINE:
            j += 1
        # segments[i:j] form one arc zone

        i0 = int(segments[i]["start_idx"])
        i1 = int(segments[j - 1]["end_idx"])
        i1 = min(i1, len(xy) - 1)

        # ── Incoming tangent direction ─────────────────────────────────────
        # Use the FIRST half of the preceding LINE segment (not the last
        # few points, which may be within the spiral due to curvature
        # smoothing causing some spiral points to be misclassified as LINE).
        if i > 0 and segments[i - 1]["type"] == ElementType.LINE:
            s0 = int(segments[i - 1]["start_idx"])
            s1 = int(segments[i - 1]["end_idx"])
            # Use first 50 % of the LINE to stay in the clearly-straight zone
            s_mid = s0 + max(2, (s1 - s0) // 2)
            s_mid = min(s_mid, s1)
            dir_in = math.atan2(float(xy[s_mid, 1] - xy[s0, 1]),
                                float(xy[s_mid, 0] - xy[s0, 0]))
        else:
            idx_s = max(0, i0 - 3)
            dir_in = math.atan2(float(xy[i0, 1] - xy[idx_s, 1]),
                                float(xy[i0, 0] - xy[idx_s, 0]))

        # ── Outgoing tangent direction ─────────────────────────────────────
        # Use the LAST half of the following LINE segment (not the first
        # few points, which may be within the exit spiral).
        if j < n and segments[j]["type"] == ElementType.LINE:
            s0 = int(segments[j]["start_idx"])
            s1 = int(segments[j]["end_idx"])
            # Use last 50 % of the LINE to stay in the clearly-straight zone
            s_mid = s1 - max(2, (s1 - s0) // 2)
            s_mid = max(s_mid, s0)
            dir_out = math.atan2(float(xy[s1, 1] - xy[s_mid, 1]),
                                 float(xy[s1, 0] - xy[s_mid, 0]))
        else:
            idx_e = min(i1 + 3, len(xy) - 1)
            dir_out = math.atan2(float(xy[idx_e, 1] - xy[i1, 1]),
                                 float(xy[idx_e, 0] - xy[i1, 0]))

        delta = (dir_out - dir_in + math.pi) % (2 * math.pi) - math.pi

        entry_pt = xy[i0].copy()
        exit_pt  = xy[i1].copy()

        PI = _find_PI(entry_pt, dir_in, exit_pt, dir_out)
        if PI is None:
            PI = (entry_pt + exit_pt) / 2.0

        ch_start = float(chainages[i0])
        ch_end   = float(chainages[i1])

        zones.append({
            "entry_pt":           entry_pt,
            "exit_pt":            exit_pt,
            "dir_in":             dir_in,
            "dir_out":            dir_out,
            "delta":              delta,
            "PI":                 PI,
            "osm_pts":            xy[i0 : i1 + 1].copy(),
            "chainage_start":     ch_start,
            "chainage_end":       ch_end,
            "chainage_start_raw": ch_start,  # before boundary expansion
            "chainage_end_raw":   ch_end,    # before boundary expansion
            "simp_idx":           i,         # placeholder — not used in assembly
        })

        i = j

    return zones


def _initial_radius_estimate(osm_pts: np.ndarray, min_radius: float) -> float:
    """Estimate arc radius from OSM points via Kasa circle fit."""
    if len(osm_pts) < 3:
        return min_radius * 2.0
    try:
        _cx, _cy, r = _fit_circle_kasa(osm_pts)
        if r is None or r <= 0 or not math.isfinite(r):
            return min_radius * 2.0
        return float(np.clip(r, min_radius, 5000.0))
    except Exception:
        return min_radius * 2.0


def _fit_arc_zone(
    zone: dict,
    min_radius: float,
    min_spiral_length: float,
    max_deviation: float,
    use_spirals: bool,
    available_entry: float,
    available_exit: float,
) -> dict:
    """
    Optimise [R, L_entry, L_exit] for one arc zone.
    Returns {"R", "L_entry", "L_exit", "geom", "converged"}.
    """
    from scipy.optimize import minimize

    osm_pts  = zone.get("osm_pts", np.empty((0, 2)))
    delta    = zone["delta"]
    PI       = zone["PI"]
    dir_in   = zone["dir_in"]
    abs_delta = abs(delta)

    # Initial guess
    R0       = _initial_radius_estimate(osm_pts, min_radius)

    # Kappa-profile-based spiral length estimates — these are directly measured
    # from the curvature profile and are far more reliable than the optimizer
    # alone, because the objective is nearly flat w.r.t. L_entry/L_exit (a 60m
    # vs 80m clothoid fitting a 60m window look identical from OSM data alone).
    entry_hint = float(zone.get("spiral_entry_hint", 0.0))
    exit_hint  = float(zone.get("spiral_exit_hint",  0.0))

    # Compute max_Ts as the projection distance from PI to the alignment
    # boundary along dir_in / dir_out.  Using available_entry directly as
    # max_Ts was wrong because available_entry is the CHAINAGE of the zone
    # boundary, not the distance PI→alignment_start along the tangent.
    PI_np = np.asarray(PI, dtype=float)
    dir_in_hat = np.array([math.cos(dir_in), math.sin(dir_in)])
    pi_proj = float(np.dot(PI_np, dir_in_hat))
    max_Ts_entry = max(pi_proj * 0.98, available_entry * 1.5)

    dir_out_rad = float(zone.get("dir_out", dir_in + delta))
    dir_out_hat = np.array([math.cos(dir_out_rad), math.sin(dir_out_rad)])
    pi_proj_exit = float(np.dot(PI_np, dir_out_hat))
    max_Ts_exit  = max(pi_proj_exit * 0.98, available_exit * 1.5)

    obj_args = (dir_in, delta, PI, osm_pts, min_radius,
                min_spiral_length, max_Ts_entry, max_Ts_exit)

    def obj(p): return _zone_objective(p, *obj_args)

    if use_spirals and entry_hint > min_spiral_length and exit_hint > min_spiral_length:
        # ── Kappa-guided mode: fix L_entry / L_exit from curvature profile ──
        # The optimizer objective is nearly flat w.r.t. spiral lengths, so the
        # kappa-profile estimates are more reliable than a free optimisation.
        # Fix L at the hint values and optimise only R (1D).
        L_e_fixed = min(entry_hint, available_entry * 0.95)
        L_x_fixed = min(exit_hint,  available_exit  * 0.95)
        # Ensure arc_angle > 0
        max_deflect = abs_delta - 0.01
        L_e_fixed = min(L_e_fixed, R0 * max_deflect / 2.5)
        L_x_fixed = min(L_x_fixed, R0 * max_deflect / 2.5)
        L_e_fixed = max(L_e_fixed, min_spiral_length)
        L_x_fixed = max(L_x_fixed, min_spiral_length)

        def obj_r(r_arr):
            return obj([float(r_arr[0]), L_e_fixed, L_x_fixed])

        r_bounds = [(min_radius, 5000.0)]
        r_starts = [R0, min_radius, min_radius * 2, R0 * 1.5]
        best_r = None
        for r_start in r_starts:
            try:
                res_r = minimize(obj_r, [r_start], method="L-BFGS-B",
                                 bounds=r_bounds, options={"maxiter": 200})
                if best_r is None or res_r.fun < best_r.fun:
                    best_r = res_r
            except Exception:
                pass
        R_opt = max(float(best_r.x[0]) if best_r is not None else R0, min_radius)
        L_e_opt = L_e_fixed
        L_x_opt = L_x_fixed

    else:
        # ── Free optimisation (no kappa hints available) ──────────────────
        max_spiral = min(available_entry, available_exit,
                         R0 * abs_delta / 2.0 - 1.0) if use_spirals else 0.0
        max_spiral = max(max_spiral, 0.0)

        if use_spirals:
            bounds = [(min_radius, 5000.0),
                      (0.0, max_spiral),
                      (0.0, max_spiral)]
            x0_v = [R0, 0.0, 0.0]
        else:
            bounds = [(min_radius, 5000.0), (0.0, 0.0), (0.0, 0.0)]
            x0_v = [R0, 0.0, 0.0]

        res = minimize(obj, x0_v, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-8})
        best = res
        for R_try in [min_radius, min_radius * 2, R0 * 1.5]:
            x_try = [R_try, 0.0, 0.0]
            res2 = minimize(obj, x_try, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 200, "ftol": 1e-8})
            if res2.fun < best.fun:
                best = res2

        R_opt  = max(float(best.x[0]), min_radius)
        L_e_opt = float(best.x[1])
        L_x_opt = float(best.x[2])

    # Compute final geometry
    abs_delta = abs(delta)
    sign = 1.0 if delta >= 0 else -1.0
    try:
        x_end_e, _ = _compute_clothoid_shift(L_e_opt, R_opt) if L_e_opt > 0 else (0.0, 0.0)
        theta_s_e  = L_e_opt / (2.0 * R_opt) if L_e_opt > 0 else 0.0
        k_entry    = x_end_e - R_opt * math.sin(theta_s_e) if L_e_opt > 0 else 0.0
        Ts_entry   = R_opt * math.tan(abs_delta / 2.0) + k_entry
    except Exception:
        Ts_entry = R_opt * math.tan(abs_delta / 2.0)

    dir_in_hat = np.array([math.cos(dir_in), math.sin(dir_in)])
    TC = PI - Ts_entry * dir_in_hat
    geom = _compute_zone_geometry(TC, dir_in, delta, R_opt, L_e_opt, L_x_opt)

    # Fallback: reduce spirals if still None
    if geom is None:
        L_e_opt = 0.0; L_x_opt = 0.0
        TC = PI - R_opt * math.tan(abs_delta / 2.0) * dir_in_hat
        geom = _compute_zone_geometry(TC, dir_in, delta, R_opt, 0.0, 0.0)

    return {
        "R":        R_opt,
        "L_entry":  L_e_opt,
        "L_exit":   L_x_opt,
        "geom":     geom,
        "converged": True,
        "TC":        TC,
    }


# ---------------------------------------------------------------------------
# Phase E — Assembly
# ---------------------------------------------------------------------------

def _assemble_alignment(
    xy: np.ndarray,
    chainages: np.ndarray,
    simp: np.ndarray,
    arc_zones: list[dict],
    fitted_zones: list[dict],
    min_line_length: float,
    use_spirals: bool,
) -> list[dict]:
    """
    Assemble element list from straight segments and fitted arc zones.
    Traverses the alignment from start to end.
    """
    elements = []
    sta = 0.0  # running station

    zone_by_simp_idx = {z["simp_idx"]: (z, fz)
                        for z, fz in zip(arc_zones, fitted_zones)}

    n_simp = len(simp)
    i = 0
    cur_pos = np.array(simp[0], dtype=float)
    cur_dir = math.atan2(float(simp[1][1] - simp[0][1]),
                         float(simp[1][0] - simp[0][0]))

    while i < n_simp - 1:
        if (i + 1) in zone_by_simp_idx:
            zone, fz = zone_by_simp_idx[i + 1]
            geom = fz["geom"]
            if geom is None:
                # Degenerate: emit straight to exit_pt then continue
                end_pos = np.array(zone["exit_pt"])
                seg_len = float(np.linalg.norm(end_pos - cur_pos))
                if seg_len >= min_line_length / 2:
                    elements.append({
                        "type": "Line", "sta_start": sta, "length": seg_len,
                        "start": cur_pos.tolist(), "end": end_pos.tolist(),
                        "direction_rad": math.atan2(
                            float(end_pos[1]-cur_pos[1]),
                            float(end_pos[0]-cur_pos[0])),
                    })
                    sta += seg_len
                cur_pos = end_pos
                i += 2
                continue

            TC = geom["TC"]
            CT = geom["CT"]

            # ── Line from cur_pos to TC ───────────────────────────────
            tc_dist = float(np.linalg.norm(TC - cur_pos))
            if tc_dist >= min_line_length / 2:
                elements.append({
                    "type": "Line", "sta_start": sta, "length": tc_dist,
                    "start": cur_pos.tolist(), "end": TC.tolist(),
                    "direction_rad": math.atan2(
                        float(TC[1]-cur_pos[1]), float(TC[0]-cur_pos[0])),
                })
                sta += tc_dist

            # ── Entry spiral ──────────────────────────────────────────
            if use_spirals and fz["L_entry"] >= 1.0:
                L_e = fz["L_entry"]
                rot  = geom["rot"]
                R    = geom["R"]
                r_s  = float("inf")
                r_e  = R
                A    = math.sqrt(R * L_e)
                elements.append({
                    "type": "Spiral", "sta_start": sta, "length": L_e,
                    "start": TC.tolist(), "end": geom["arc_start"].tolist(),
                    "radius_start": r_s, "radius_end": r_e,
                    "clothoid_A": A, "rot": rot,
                })
                sta += L_e

            # ── Circular arc ──────────────────────────────────────────
            arc_start = geom["arc_start"]
            arc_end   = geom["arc_end"]
            arc_len   = geom["arc_len"]
            R         = geom["R"]
            rot       = geom["rot"]
            center    = geom["arc_center"]
            chord     = float(np.linalg.norm(arc_end - arc_start))
            elements.append({
                "type": "Arc", "sta_start": sta, "length": arc_len,
                "start":  arc_start.tolist(), "end": arc_end.tolist(),
                "center": center.tolist(), "radius": R,
                "rot": rot, "chord": chord,
            })
            sta += arc_len

            # ── Exit spiral ───────────────────────────────────────────
            if use_spirals and fz["L_exit"] >= 1.0:
                L_x = fz["L_exit"]
                rot  = geom["rot"]
                R    = geom["R"]
                r_s  = R
                r_e  = float("inf")
                A    = math.sqrt(R * L_x)
                elements.append({
                    "type": "Spiral", "sta_start": sta, "length": L_x,
                    "start": arc_end.tolist(), "end": CT.tolist(),
                    "radius_start": r_s, "radius_end": r_e,
                    "clothoid_A": A, "rot": rot,
                })
                sta += L_x

            cur_pos = CT
            cur_dir = geom["dir_out"]
            i += 1   # next simp vertex is the exit_pt vertex

        else:
            # Straight segment to simp[i+1]
            end_pos = np.array(simp[i + 1])
            seg_len = float(np.linalg.norm(end_pos - cur_pos))
            if seg_len >= min_line_length / 2:
                elements.append({
                    "type": "Line", "sta_start": sta, "length": seg_len,
                    "start": cur_pos.tolist(), "end": end_pos.tolist(),
                    "direction_rad": math.atan2(
                        float(end_pos[1]-cur_pos[1]),
                        float(end_pos[0]-cur_pos[0])),
                })
                sta += seg_len
            cur_pos = end_pos
            i += 1

    return elements


# ---------------------------------------------------------------------------
# Circle fitting (Kåsa algebraic method)
# ---------------------------------------------------------------------------

def _assemble_from_zones(
    xy: np.ndarray,
    chainages: np.ndarray,
    arc_zones: list[dict],
    fitted_zones: list[dict],
    min_line_length: float,
    min_spiral_length: float,
    use_spirals: bool,
) -> list[dict]:
    """
    Walk along the alignment using chainage boundaries of arc zones.

    For each gap between zones (or start/end of alignment) emit a Line.
    For each zone emit (optional Spiral) + Arc + (optional Spiral).

    This function replaces the simp-vertex-based _assemble_alignment for the
    curvature-based zone detection path.
    """
    elements: list[dict] = []
    sta = 0.0
    total_len = float(chainages[-1])

    def _nearest_xy(ch: float) -> np.ndarray:
        """Return the OSM point nearest to chainage ch."""
        idx = int(np.searchsorted(chainages, ch))
        idx = max(0, min(idx, len(xy) - 1))
        return xy[idx].copy()

    cur_ch  = 0.0
    cur_pos = np.array(xy[0], dtype=float)

    spiral_thresh = max(min_spiral_length * 0.5, 0.5)

    for zone, fz in zip(arc_zones, fitted_zones):
        geom = fz.get("geom")

        if geom is None:
            # Degenerate: span the zone as a straight line
            end_ch  = zone["chainage_end"]
            end_pos = _nearest_xy(end_ch)
            seg_len = end_ch - cur_ch
            if seg_len >= min_line_length / 2:
                elements.append({
                    "type": "Line", "sta_start": sta, "length": seg_len,
                    "start": cur_pos.tolist(), "end": end_pos.tolist(),
                    "direction_rad": math.atan2(
                        float(end_pos[1] - cur_pos[1]),
                        float(end_pos[0] - cur_pos[0])),
                })
                sta += seg_len
            cur_ch  = end_ch
            cur_pos = end_pos
            continue

        TC  = np.array(geom["TC"],  dtype=float)
        R   = float(geom["R"])
        rot = geom["rot"]

        # Decide actual spiral lengths (0 if below threshold)
        L_e_raw = float(fz["L_entry"])
        L_x_raw = float(fz["L_exit"])
        L_e = L_e_raw if (use_spirals and L_e_raw >= spiral_thresh) else 0.0
        L_x = L_x_raw if (use_spirals and L_x_raw >= spiral_thresh) else 0.0

        # If actual L_e / L_x differ from the optimised values, recompute geometry
        # so that arc_center and arc_end are consistent with the real spiral lengths.
        if L_e != L_e_raw or L_x != L_x_raw:
            geom = _compute_zone_geometry(TC, float(geom["dir_in"]),
                                          float(zone["delta"]), R, L_e, L_x)
            if geom is None:
                # Fallback: pure arc with no spirals
                geom = _compute_zone_geometry(TC, float(zone["dir_in"]),
                                              float(zone["delta"]), R, 0.0, 0.0)
            if geom is None:
                # Truly degenerate — skip this zone
                cur_ch = zone["chainage_end"]
                continue

        CT = np.array(geom["CT"], dtype=float)

        # ── Line from current position to TC ──────────────────────────────
        tc_dist = float(np.linalg.norm(TC - cur_pos))
        if tc_dist >= min_line_length / 2:
            elements.append({
                "type": "Line", "sta_start": sta, "length": tc_dist,
                "start": cur_pos.tolist(), "end": TC.tolist(),
                "direction_rad": math.atan2(
                    float(TC[1] - cur_pos[1]), float(TC[0] - cur_pos[0])),
            })
            sta += tc_dist

        # ── Entry spiral ──────────────────────────────────────────────────
        if L_e >= spiral_thresh:
            arc_start_pt = np.array(geom["arc_start"], dtype=float)
            elements.append({
                "type": "Spiral", "sta_start": sta, "length": L_e,
                "start": TC.tolist(), "end": arc_start_pt.tolist(),
                "radius_start": float("inf"), "radius_end": R,
                "clothoid_A": math.sqrt(R * L_e), "rot": rot,
            })
            sta += L_e
        else:
            arc_start_pt = TC  # no spiral: arc starts at TC

        # ── Circular arc ──────────────────────────────────────────────────
        arc_end_pt = np.array(geom["arc_end"],    dtype=float)
        center_pt  = np.array(geom["arc_center"], dtype=float)
        arc_len    = float(geom["arc_len"])
        chord      = float(np.linalg.norm(arc_end_pt - arc_start_pt))
        elements.append({
            "type": "Arc", "sta_start": sta, "length": arc_len,
            "start":  arc_start_pt.tolist(),
            "end":    arc_end_pt.tolist(),
            "center": center_pt.tolist(),
            "radius": R, "rot": rot, "chord": chord,
        })
        sta += arc_len

        # ── Exit spiral ───────────────────────────────────────────────────
        if L_x >= spiral_thresh:
            elements.append({
                "type": "Spiral", "sta_start": sta, "length": L_x,
                "start": arc_end_pt.tolist(), "end": CT.tolist(),
                "radius_start": R, "radius_end": float("inf"),
                "clothoid_A": math.sqrt(R * L_x), "rot": rot,
            })
            sta += L_x

        cur_pos  = CT
        cur_ch   = zone["chainage_end"]
        cur_dout = float(geom["dir_out"])  # exit tangent heading for this zone

    # ── Final straight to end of alignment ───────────────────────────────
    # Project xy[-1] onto the exit tangent line through cur_pos so that the
    # Line's geometric direction equals the zone's exit heading exactly.
    # This ensures tangency continuity at the Arc/Spiral → Line boundary.
    end_pos_raw = np.array(xy[-1], dtype=float)
    if 'cur_dout' in dir():
        d_out = np.array([math.cos(cur_dout), math.sin(cur_dout)])
        t = float(np.dot(end_pos_raw - cur_pos, d_out))
        if t > 0:
            end_pos = cur_pos + t * d_out
        else:
            end_pos = end_pos_raw
    else:
        end_pos = end_pos_raw
    final_len = float(np.linalg.norm(end_pos - cur_pos))
    if final_len >= min_line_length / 2:
        elements.append({
            "type": "Line", "sta_start": sta, "length": final_len,
            "start": cur_pos.tolist(), "end": end_pos.tolist(),
            "direction_rad": math.atan2(
                float(end_pos[1] - cur_pos[1]),
                float(end_pos[0] - cur_pos[0])),
        })

    return elements


def _fit_circle_kasa(
    pts: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Algebraic circle fit. Returns (cx, cy, radius) or (None, None, None)."""
    if len(pts) < 3:
        return None, None, None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    cx = result[0] / 2
    cy = result[1] / 2
    r  = float(np.sqrt(result[2] + cx ** 2 + cy ** 2))
    return float(cx), float(cy), r
