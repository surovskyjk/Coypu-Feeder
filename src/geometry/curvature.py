"""
Curvature analysis of a 2D polyline.
Computes signed curvature κ at each vertex, smooths it,
then segments the profile into Line / Arc / Spiral regions.
"""

import numpy as np
from scipy.signal import savgol_filter
from enum import Enum


class ElementType(Enum):
    LINE = "Line"
    ARC = "Arc"
    SPIRAL = "Spiral"


def compute_curvature(xy: np.ndarray) -> np.ndarray:
    """
    Compute signed curvature at each point of a 2D polyline.
    Uses the three-point circle formula.
    xy: (N, 2) array of (x, y) coordinates.
    Returns: (N,) array of signed curvature values (1/m).
    """
    n = len(xy)
    kappa = np.zeros(n)

    for i in range(1, n - 1):
        p0, p1, p2 = xy[i - 1], xy[i], xy[i + 1]
        # Vectors
        d1 = p1 - p0
        d2 = p2 - p1
        # Cross product (signed area)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        # Lengths
        l1 = np.linalg.norm(d1)
        l2 = np.linalg.norm(d2)
        l12 = np.linalg.norm(p2 - p0)
        denom = l1 * l2 * l12
        if denom < 1e-12:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * cross / denom

    # Extrapolate endpoints
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def smooth_curvature(kappa: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth the curvature profile."""
    if len(kappa) < window + 2:
        return kappa
    # Window must be odd
    if window % 2 == 0:
        window += 1
    return savgol_filter(kappa, window_length=window, polyorder=2)


def compute_chainages(xy: np.ndarray) -> np.ndarray:
    """Cumulative chord lengths (stationing) along the polyline."""
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def segment_curvature(
    kappa: np.ndarray,
    chainages: np.ndarray,
    line_tol: float = 0.001,    # 1/m — below this is a straight line
    arc_tol: float = 0.0002,    # 1/m — max κ variation for constant arc
    min_length: float = 5.0,    # m  — merge segments shorter than this
) -> list[dict]:
    """
    Segment the curvature profile into element types.
    Returns list of dicts: {type, start_idx, end_idx, mean_kappa, kappa_start, kappa_end}
    """
    n = len(kappa)
    segments = []
    i = 0

    while i < n:
        # Grow a segment from index i
        k0 = kappa[i]
        seg_type = _classify_point(k0, line_tol)
        j = i + 1

        while j < n:
            k1 = kappa[j]
            if seg_type == ElementType.LINE:
                if abs(k1) < line_tol:
                    j += 1
                    continue
                break
            elif seg_type == ElementType.ARC:
                # Check constant curvature
                if abs(k1 - k0) < arc_tol and abs(k1) >= line_tol:
                    j += 1
                    continue
                # Might be a spiral — linear κ transition?
                dk = abs(k1 - kappa[i])
                di = chainages[j] - chainages[i]
                if di > 0:
                    rate = dk / di
                    if rate > 0 and abs(k1) >= line_tol:
                        seg_type = ElementType.SPIRAL
                        j += 1
                        continue
                break
            elif seg_type == ElementType.SPIRAL:
                # Linearly varying κ
                dk = abs(k1 - kappa[i])
                di = chainages[j] - chainages[i]
                if di > 0 and abs(k1) >= line_tol:
                    j += 1
                    continue
                break

        seg_chainages = chainages[i:j]
        length = seg_chainages[-1] - seg_chainages[0] if len(seg_chainages) > 1 else 0
        segments.append({
            "type": seg_type,
            "start_idx": i,
            "end_idx": j - 1,
            "mean_kappa": float(np.mean(kappa[i:j])),
            "kappa_start": float(kappa[i]),
            "kappa_end": float(kappa[j - 1]),
            "length": length,
        })
        i = j

    return _merge_short_segments(segments, chainages, min_length)


def _classify_point(k: float, line_tol: float) -> ElementType:
    return ElementType.LINE if abs(k) < line_tol else ElementType.ARC


def _merge_short_segments(
    segments: list[dict], chainages: np.ndarray, min_length: float
) -> list[dict]:
    """Merge segments shorter than min_length into their neighbour."""
    if not segments:
        return segments

    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg["length"] < min_length and len(segments) > 1:
                # Absorb into previous if exists, else next
                if merged:
                    prev = merged[-1]
                    prev["end_idx"] = seg["end_idx"]
                    prev["length"] = chainages[prev["end_idx"]] - chainages[prev["start_idx"]]
                    prev["mean_kappa"] = (prev["mean_kappa"] + seg["mean_kappa"]) / 2
                    prev["kappa_end"] = seg["kappa_end"]
                    changed = True
                elif i + 1 < len(segments):
                    nxt = segments[i + 1]
                    nxt["start_idx"] = seg["start_idx"]
                    nxt["kappa_start"] = seg["kappa_start"]
                    nxt["length"] = chainages[nxt["end_idx"]] - chainages[nxt["start_idx"]]
                    changed = True
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
            i += 1
        segments = merged

    return segments
