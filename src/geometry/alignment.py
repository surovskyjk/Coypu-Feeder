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

    return elements


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
# Circle fitting (Kåsa algebraic method)
# ---------------------------------------------------------------------------

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
