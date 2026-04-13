"""
Multi-algorithm candidate alignment generator.
All algorithms produce Lines + Arcs only (no spirals).

Exported names
--------------
CandidateAlignment  : dataclass holding elements + metrics + display info
evaluate_candidate  : compute max_deviation / rmse / n_elements for an element list
CandidateGenerator  : runs curvature / RANSAC / greedy algorithms
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
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
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    elements: list[dict],
    xy: np.ndarray,
    chainages: np.ndarray,
    check_interval: float = 5.0,
) -> dict:
    """
    Compute quality metrics for a list of fitted elements against the OSM polyline.

    Returns
    -------
    dict with keys: max_deviation (float), rmse (float), n_elements (int)
    """
    from geometry.alignment import max_deviation_element

    if not elements or len(xy) < 2:
        return {"max_deviation": 0.0, "rmse": 0.0, "n_elements": len(elements)}

    total_len = float(chainages[-1])
    max_dev = 0.0
    sq_sum  = 0.0
    sq_cnt  = 0

    for el in elements:
        sta0 = el.get("sta_start", 0.0)
        sta1 = sta0 + el.get("length", 0.0)

        # Find OSM points in this element's chainage range (with a small margin)
        mask   = (chainages >= sta0 - 0.1) & (chainages <= sta1 + 0.1)
        xy_seg = xy[mask]

        if len(xy_seg) < 2:
            continue

        dev = max_deviation_element(el, xy_seg, check_interval)
        if dev > max_dev:
            max_dev = dev

    # RMSE: for every OSM point find its distance to the nearest element
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
    """Distance from point pt to the nearest point on a Line or Arc element."""
    el_type = el.get("type", "Line")
    if el_type == "Line":
        start = np.array(el["start"])
        end   = np.array(el["end"])
        seg   = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return float(np.linalg.norm(pt - start))
        t = float(np.dot(pt - start, seg)) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        nearest = start + t * seg
        return float(np.linalg.norm(pt - nearest))
    elif el_type == "Arc":
        center = np.array(el["center"])
        r      = el.get("radius", 1.0)
        dist_to_center = float(np.linalg.norm(pt - center))
        return abs(dist_to_center - r)
    else:
        # Spiral or unknown — fall back to chord distance
        start = np.array(el.get("start", [0.0, 0.0]))
        end   = np.array(el.get("end",   [0.0, 0.0]))
        seg   = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return float(np.linalg.norm(pt - start))
        t = float(np.dot(pt - start, seg)) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        nearest = start + t * seg
        return float(np.linalg.norm(pt - nearest))


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------

class CandidateGenerator:
    """Run all three candidate algorithms on a projected XY polyline."""

    COLORS = ["#ff9800", "#66bb6a", "#42a5f5"]
    LABELS = {
        "curvature": "Curvature Heuristic",
        "ransac":    "RANSAC Arc Fit",
        "greedy":    "Greedy Split",
    }

    def __init__(self, xy: np.ndarray, chainages: np.ndarray, settings: dict):
        self.xy           = xy
        self.chainages    = chainages
        self.settings     = settings
        self.min_radius   = settings.get("min_radius",    150.0)
        self.smooth_window = settings.get("smooth_window", 21)
        self.max_deviation = settings.get("max_deviation", 0.5)
        self.check_interval = settings.get("check_interval", 5.0)

    def run_all(self) -> list[CandidateAlignment]:
        results = []
        for algo_id, color in zip(["curvature", "ransac", "greedy"], self.COLORS):
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
        if algo_id == "curvature":
            return self._run_curvature()
        elif algo_id == "ransac":
            return self._run_ransac()
        else:
            return self._run_greedy()

    # ------------------------------------------------------------------
    # Algorithm 1 — Curvature Heuristic
    # ------------------------------------------------------------------

    def _run_curvature(self) -> CandidateAlignment:
        from geometry.alignment import fit_alignment
        elements = fit_alignment(
            self.xy,
            algorithm="curvature",
            use_spirals=False,
            min_radius=self.min_radius,
            smooth_window=self.smooth_window,
            max_deviation=self.max_deviation,
            check_interval=self.check_interval,
        )
        metrics = evaluate_candidate(elements, self.xy, self.chainages,
                                     self.check_interval)
        return CandidateAlignment(
            algorithm_id="curvature",
            label="Curvature Heuristic",
            elements=elements,
            **metrics,
        )

    # ------------------------------------------------------------------
    # Algorithm 2 — RANSAC Arc Fit
    # ------------------------------------------------------------------

    def _run_ransac(self) -> CandidateAlignment:
        from geometry.curvature import smooth_curvature, segment_curvature, compute_curvature
        from geometry.alignment import _fit_line

        xy        = self.xy
        chainages = self.chainages
        kappa       = compute_curvature(xy)
        kappa_smooth = smooth_curvature(kappa, window=self.smooth_window)

        segments = segment_curvature(
            kappa_smooth, chainages,
            line_tol=0.001, arc_tol=0.0002,
            min_line_length=10.0, min_arc_length=10.0, min_spiral_length=10.0,
        )

        elements = []
        sta = 0.0

        for seg in segments:
            i0, i1 = int(seg["start_idx"]), int(seg["end_idx"])
            pts     = xy[i0:i1 + 1]
            seg_len = float(chainages[i1] - chainages[i0])

            if seg["type"].name == "LINE" or len(pts) < 3:
                el = _fit_line(pts, sta, seg_len)
                el["type"] = "Line"
                elements.append(el)
                sta += el["length"]
            else:
                arc_el = self._ransac_fit_arc(pts, sta, seg_len)
                elements.append(arc_el)
                sta += arc_el["length"]

        # Enforce min_radius
        for el in elements:
            if el.get("type") == "Arc" and el.get("radius", float("inf")) < self.min_radius:
                el["radius"] = self.min_radius

        metrics = evaluate_candidate(elements, xy, chainages, self.check_interval)
        return CandidateAlignment(
            algorithm_id="ransac",
            label="RANSAC Arc Fit",
            elements=elements,
            **metrics,
        )

    def _ransac_fit_arc(self, pts: np.ndarray, sta: float, seg_len: float) -> dict:
        from geometry.alignment import _fit_circle_kasa, _fit_line
        from geometry.curvature import compute_curvature

        n = len(pts)
        if n < 3:
            return _fit_line(pts, sta, seg_len)

        ransac_tol  = max(self.max_deviation * 2.0, 1.0)
        n_iter      = min(100, max(20, n * 2))
        rng         = np.random.default_rng(42)

        best_inlier_count = 0
        best_cx = best_cy = best_r = None

        for _ in range(n_iter):
            idxs   = rng.choice(n, 3, replace=False)
            sample = pts[idxs]
            cx, cy, r = _fit_circle_kasa(sample)
            if cx is None or r is None or r <= 0 or not math.isfinite(r):
                continue
            if r < self.min_radius * 0.5:
                continue

            dists        = np.abs(np.linalg.norm(pts - np.array([cx, cy]), axis=1) - r)
            inlier_mask  = dists < ransac_tol
            inlier_count = int(np.sum(inlier_mask))

            if inlier_count > best_inlier_count:
                inliers = pts[inlier_mask]
                if len(inliers) >= 3:
                    cx_r, cy_r, r_r = _fit_circle_kasa(inliers)
                    if cx_r is not None and r_r is not None and r_r > 0 and math.isfinite(r_r):
                        best_cx, best_cy, best_r = cx_r, cy_r, r_r
                        best_inlier_count = inlier_count

        if best_r is None or best_r < self.min_radius:
            cx_all, cy_all, r_all = _fit_circle_kasa(pts)
            if cx_all is not None and r_all is not None and r_all > 0:
                best_cx, best_cy, best_r = cx_all, cy_all, r_all
            else:
                return _fit_line(pts, sta, seg_len)

        best_r = max(best_r, self.min_radius)

        kappa_pts = compute_curvature(pts)
        mean_k    = float(np.mean(kappa_pts))
        rot       = "ccw" if mean_k >= 0 else "cw"
        chord     = float(np.linalg.norm(pts[-1] - pts[0]))

        return {
            "type":      "Arc",
            "sta_start": sta,
            "length":    seg_len,
            "start":     pts[0].tolist(),
            "end":       pts[-1].tolist(),
            "center":    [float(best_cx), float(best_cy)],
            "radius":    best_r,
            "rot":       rot,
            "chord":     chord,
        }

    # ------------------------------------------------------------------
    # Algorithm 3 — Greedy Iterative Splitting
    # ------------------------------------------------------------------

    def _run_greedy(self) -> CandidateAlignment:
        from geometry.curvature import smooth_curvature, segment_curvature, compute_curvature
        from geometry.alignment import _fit_line

        xy        = self.xy
        chainages = self.chainages
        kappa       = compute_curvature(xy)
        kappa_smooth = smooth_curvature(kappa, window=self.smooth_window)

        segments = segment_curvature(
            kappa_smooth, chainages,
            line_tol=0.001, arc_tol=0.0002,
            min_line_length=10.0, min_arc_length=10.0, min_spiral_length=10.0,
        )

        elements = []
        sta = 0.0

        for seg in segments:
            i0, i1   = int(seg["start_idx"]), int(seg["end_idx"])
            pts      = xy[i0:i1 + 1]
            seg_chs  = chainages[i0:i1 + 1]
            seg_len  = float(chainages[i1] - chainages[i0])

            if seg["type"].name == "LINE" or len(pts) < 3:
                el = _fit_line(pts, sta, seg_len)
                el["type"] = "Line"
                elements.append(el)
                sta += el["length"]
            else:
                sub_els = self._greedy_fit(pts, seg_chs, sta)
                for el in sub_els:
                    elements.append(el)
                    sta += el["length"]

        for el in elements:
            if el.get("type") == "Arc" and el.get("radius", float("inf")) < self.min_radius:
                el["radius"] = self.min_radius

        metrics = evaluate_candidate(elements, xy, chainages, self.check_interval)
        return CandidateAlignment(
            algorithm_id="greedy",
            label="Greedy Split",
            elements=elements,
            **metrics,
        )

    def _greedy_fit(
        self,
        pts: np.ndarray,
        chs: np.ndarray,
        sta: float,
        depth: int = 0,
    ) -> list[dict]:
        from geometry.alignment import _fit_circle_kasa, _fit_line, max_deviation_element
        from geometry.curvature import compute_curvature

        MIN_ARC_PTS = 4
        MAX_DEPTH   = 6

        n       = len(pts)
        seg_len = float(chs[-1] - chs[0])

        if n < MIN_ARC_PTS or seg_len < 10.0 or depth >= MAX_DEPTH:
            kappa_pts = compute_curvature(pts) if n >= 3 else np.array([0.0])
            if abs(float(np.mean(kappa_pts))) < 0.001:
                el = _fit_line(pts, sta, seg_len)
                el["type"] = "Line"
            else:
                cx, cy, r = _fit_circle_kasa(pts)
                if cx is None or r is None or r <= 0:
                    el = _fit_line(pts, sta, seg_len)
                    el["type"] = "Line"
                else:
                    r       = max(r, self.min_radius)
                    mean_k  = float(np.mean(compute_curvature(pts)))
                    rot     = "ccw" if mean_k >= 0 else "cw"
                    chord   = float(np.linalg.norm(pts[-1] - pts[0]))
                    el = {
                        "type":      "Arc",
                        "sta_start": sta,
                        "length":    seg_len,
                        "start":     pts[0].tolist(),
                        "end":       pts[-1].tolist(),
                        "center":    [float(cx), float(cy)],
                        "radius":    r,
                        "rot":       rot,
                        "chord":     chord,
                    }
            return [el]

        # Fit arc to all points
        cx, cy, r = _fit_circle_kasa(pts)
        if cx is None or r is None or r <= 0:
            el = _fit_line(pts, sta, seg_len)
            el["type"] = "Line"
            return [el]

        r      = max(r, self.min_radius)
        mean_k = float(np.mean(compute_curvature(pts)))
        rot    = "ccw" if mean_k >= 0 else "cw"
        chord  = float(np.linalg.norm(pts[-1] - pts[0]))
        el = {
            "type":      "Arc",
            "sta_start": sta,
            "length":    seg_len,
            "start":     pts[0].tolist(),
            "end":       pts[-1].tolist(),
            "center":    [float(cx), float(cy)],
            "radius":    r,
            "rot":       rot,
            "chord":     chord,
        }

        dev = max_deviation_element(el, pts, self.check_interval)

        if dev <= self.max_deviation:
            return [el]

        # Split at worst-fitting point
        dists     = np.abs(np.linalg.norm(pts - np.array([cx, cy]), axis=1) - r)
        split_idx = int(np.argmax(dists))
        split_idx = max(MIN_ARC_PTS - 1, min(split_idx, n - MIN_ARC_PTS))

        left_pts  = pts[:split_idx + 1]
        left_chs  = chs[:split_idx + 1]
        right_pts = pts[split_idx:]
        right_chs = chs[split_idx:]

        left_els     = self._greedy_fit(left_pts, left_chs, sta, depth + 1)
        left_sta_end = sta + sum(e["length"] for e in left_els)
        right_els    = self._greedy_fit(right_pts, right_chs, left_sta_end, depth + 1)

        return left_els + right_els
