"""
Tests for geometry/alignment.py

Synthetic polylines with known geometry are used so we can measure exact
deviations and verify element types and radius continuity.

Polyline generators
-------------------
make_straight  — straight line along x-axis
make_arc       — arc of given radius and swept angle
make_lal       — Line – Arc – Line joined tangentially
make_lsal      — Line – Spiral – Arc – Line (one-sided transition)
make_lsasl     — Line – Spiral – Arc – Spiral – Line (full railway sequence)
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Make sure src/ is on the path when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry.alignment import (
    fit_alignment,
    enforce_continuity,
    max_deviation_element,
    _remove_degenerate_spirals,
    _sample_element_points,
    _dist_point_to_segment,
)
from geometry.curvature import compute_chainages


# ---------------------------------------------------------------------------
# Polyline generators
# ---------------------------------------------------------------------------

def make_straight(length: float = 500.0, n: int = 100) -> np.ndarray:
    """Perfectly straight line along x-axis."""
    x = np.linspace(0.0, length, n)
    return np.column_stack([x, np.zeros(n)])


def make_arc(
    radius: float,
    sweep_deg: float,
    start_xy: tuple[float, float] = (0.0, 0.0),
    start_dir_deg: float = 0.0,
    n: int = 60,
) -> np.ndarray:
    """
    Arc of given *radius* sweeping *sweep_deg* degrees (left = positive = ccw).
    *start_dir_deg* is the tangent direction at the start in degrees from east.
    Returns (n, 2) array in world coordinates.
    """
    sweep_rad     = math.radians(sweep_deg)
    start_dir_rad = math.radians(start_dir_deg)

    # Centre of the arc is to the left of the travel direction (ccw) or right (cw)
    ccw = sweep_deg >= 0
    perp_angle = start_dir_rad + (math.pi / 2 if ccw else -math.pi / 2)
    cx = start_xy[0] + radius * math.cos(perp_angle)
    cy = start_xy[1] + radius * math.sin(perp_angle)

    # Starting angle from centre to start point
    a0 = start_dir_rad - (math.pi / 2 if ccw else -math.pi / 2)
    angles = a0 + np.linspace(0.0, sweep_rad if ccw else -abs(sweep_rad), n)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])


def make_lal(
    radius: float = 500.0,
    sweep_deg: float = 30.0,
    straight_len: float = 200.0,
    n_each: int = 50,
) -> np.ndarray:
    """Straight – Arc – Straight joined tangentially."""
    s1 = make_straight(straight_len, n_each)

    arc = make_arc(radius, sweep_deg,
                   start_xy=(s1[-1, 0], s1[-1, 1]),
                   start_dir_deg=0.0, n=n_each)

    # Tangent direction at end of arc
    end_dir_rad = math.radians(sweep_deg)  # for ccw arc starting eastward
    dx = math.cos(end_dir_rad)
    dy = math.sin(end_dir_rad)
    t  = np.linspace(0.0, straight_len, n_each)
    s2 = arc[-1] + np.outer(t, [dx, dy])

    return np.vstack([s1[:-1], arc[:-1], s2])


def make_lsasl(
    radius: float = 400.0,
    sweep_deg: float = 40.0,
    spiral_len: float = 80.0,
    straight_len: float = 300.0,
    n_each: int = 40,
) -> np.ndarray:
    """
    Full railway sequence: Straight – Spiral – Arc – Spiral – Straight.

    The spirals are approximated by linearly interpolating the curvature from
    0 (at the straight end) to 1/radius (at the arc end).  The arc in the
    middle sweeps sweep_deg – 2*(spiral contribution) degrees.
    """
    # Each spiral contributes sweep_deg/2 of the total turning
    spiral_sweep_rad = spiral_len / (2 * radius)   # approx: A² = R*L
    arc_sweep_rad    = math.radians(sweep_deg) - 2 * spiral_sweep_rad
    if arc_sweep_rad < 0.01:
        arc_sweep_rad = 0.01

    pts = []

    # Segment 1 — straight
    s1 = make_straight(straight_len, n_each)
    pts.append(s1)
    cur_xy  = (s1[-1, 0], s1[-1, 1])
    cur_dir = 0.0   # degrees

    # Segment 2 — entry spiral (curvature 0 → 1/radius)
    t_vals = np.linspace(0.0, 1.0, n_each)
    # Approximate spiral path by incrementally integrating direction
    spiral1 = [np.array(cur_xy)]
    step     = spiral_len / (n_each - 1)
    angle    = math.radians(cur_dir)
    for k in range(1, n_each):
        kappa_k   = (k / (n_each - 1)) / radius   # linearly increasing
        angle    += kappa_k * step
        spiral1.append(spiral1[-1] + step * np.array([math.cos(angle), math.sin(angle)]))
    spiral1 = np.array(spiral1)
    pts.append(spiral1[1:])
    cur_xy  = (float(spiral1[-1, 0]), float(spiral1[-1, 1]))
    cur_dir = math.degrees(angle)

    # Segment 3 — arc
    arc = make_arc(radius, math.degrees(arc_sweep_rad),
                   start_xy=cur_xy, start_dir_deg=cur_dir, n=n_each)
    pts.append(arc[1:])
    cur_xy  = (float(arc[-1, 0]), float(arc[-1, 1]))
    cur_dir += math.degrees(arc_sweep_rad)

    # Segment 4 — exit spiral (curvature 1/radius → 0)
    spiral2 = [np.array(cur_xy)]
    angle    = math.radians(cur_dir)
    for k in range(1, n_each):
        kappa_k   = (1.0 - k / (n_each - 1)) / radius
        angle    += kappa_k * step
        spiral2.append(spiral2[-1] + step * np.array([math.cos(angle), math.sin(angle)]))
    spiral2 = np.array(spiral2)
    pts.append(spiral2[1:])
    cur_xy  = (float(spiral2[-1, 0]), float(spiral2[-1, 1]))
    cur_dir = math.degrees(angle)

    # Segment 5 — final straight
    dx = math.cos(math.radians(cur_dir))
    dy = math.sin(math.radians(cur_dir))
    t  = np.linspace(0.0, straight_len, n_each)
    s2 = np.array(cur_xy) + np.outer(t, [dx, dy])
    pts.append(s2[1:])

    return np.vstack(pts)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_no_inf_inf_spiral(elements: list[dict]) -> None:
    """Assert that no Spiral element has both radii equal to INF."""
    for i, el in enumerate(elements):
        if el["type"] == "Spiral":
            r_s = el.get("radius_start", float("inf"))
            r_e = el.get("radius_end",   float("inf"))
            assert not (math.isinf(r_s) and math.isinf(r_e)), (
                f"Element {i} is a Spiral with both radii = INF "
                f"(radius_start={r_s}, radius_end={r_e})"
            )


def _check_boundary_continuity(elements: list[dict]) -> None:
    """
    Assert radius continuity at every element boundary.
    At each Line/Spiral or Spiral/Arc boundary the touching radius must match.
    """
    for i in range(len(elements) - 1):
        a = elements[i]
        b = elements[i + 1]

        # Line → Spiral : spiral.radius_start must be INF
        if a["type"] == "Line" and b["type"] == "Spiral":
            assert math.isinf(b.get("radius_start", 0.0)), (
                f"Boundary {i}: Line→Spiral but radius_start={b.get('radius_start')}"
            )

        # Spiral → Line : spiral.radius_end must be INF
        if a["type"] == "Spiral" and b["type"] == "Line":
            assert math.isinf(a.get("radius_end", 0.0)), (
                f"Boundary {i}: Spiral→Line but radius_end={a.get('radius_end')}"
            )

        # Spiral → Arc : spiral.radius_end == arc.radius (within 1 %)
        if a["type"] == "Spiral" and b["type"] == "Arc":
            r_sp  = a.get("radius_end", float("inf"))
            r_arc = b["radius"]
            assert abs(r_sp - r_arc) / max(r_arc, 1.0) < 0.01, (
                f"Boundary {i}: Spiral.radius_end={r_sp:.2f} != Arc.radius={r_arc:.2f}"
            )

        # Arc → Spiral : spiral.radius_start == arc.radius (within 1 %)
        if a["type"] == "Arc" and b["type"] == "Spiral":
            r_arc = a["radius"]
            r_sp  = b.get("radius_start", float("inf"))
            assert abs(r_sp - r_arc) / max(r_arc, 1.0) < 0.01, (
                f"Boundary {i}: Arc.radius={r_arc:.2f} != Spiral.radius_start={r_sp:.2f}"
            )


# ===========================================================================
# Tests: enforce_continuity  &  _remove_degenerate_spirals
# ===========================================================================

class TestEnforceContinuity:

    def test_line_spiral_arc_spiral_line_radii(self):
        """Classic L-S-A-S-L: check all boundary radii."""
        R = 400.0
        elements = [
            {"type": "Line",   "sta_start": 0.0,   "length": 100.0,
             "start": [0, 0], "end": [100, 0], "direction_rad": 0.0},
            {"type": "Spiral", "sta_start": 100.0, "length": 80.0,
             "start": [100, 0], "end": [180, 5], "radius_start": float("inf"),
             "radius_end": R, "clothoid_A": math.sqrt(R * 80), "rot": "ccw"},
            {"type": "Arc",    "sta_start": 180.0, "length": 120.0,
             "start": [180, 5], "end": [300, 10], "center": None,
             "radius": R, "rot": "ccw", "chord": 100.0},
            {"type": "Spiral", "sta_start": 300.0, "length": 80.0,
             "start": [300, 10], "end": [380, 5], "radius_start": R,
             "radius_end": float("inf"), "clothoid_A": math.sqrt(R * 80), "rot": "ccw"},
            {"type": "Line",   "sta_start": 380.0, "length": 100.0,
             "start": [380, 5], "end": [480, 5], "direction_rad": 0.0},
        ]
        result = enforce_continuity(elements)
        _check_no_inf_inf_spiral(result)
        _check_boundary_continuity(result)

    def test_spiral_inf_inf_converted_to_line(self):
        """A Spiral between two Lines must be replaced with a Line."""
        elements = [
            {"type": "Line",   "sta_start": 0.0,  "length": 100.0,
             "start": [0, 0], "end": [100, 0], "direction_rad": 0.0},
            {"type": "Spiral", "sta_start": 100.0, "length": 50.0,
             "start": [100, 0], "end": [150, 0],
             "radius_start": float("inf"), "radius_end": float("inf"),
             "clothoid_A": 0.0, "rot": "ccw"},
            {"type": "Line",   "sta_start": 150.0, "length": 100.0,
             "start": [150, 0], "end": [250, 0], "direction_rad": 0.0},
        ]
        result = enforce_continuity(elements)
        types = [e["type"] for e in result]
        assert "Spiral" not in types, (
            f"Expected all INF/INF Spirals converted to Lines, got {types}"
        )
        assert types.count("Line") == 3

    def test_spiral_spiral_boundary_radius_propagated(self):
        """When two Spirals are adjacent, radius_start of the 2nd equals
        radius_end of the 1st."""
        R1, R2 = 500.0, 300.0
        elements = [
            {"type": "Arc",    "sta_start": 0.0,   "length": 200.0,
             "start": [0, 0], "end": [200, 20], "center": None,
             "radius": R1, "rot": "ccw", "chord": 180.0},
            {"type": "Spiral", "sta_start": 200.0, "length": 60.0,
             "start": [200, 20], "end": [260, 30],
             "radius_start": R1, "radius_end": R2,
             "clothoid_A": math.sqrt(R2 * 60), "rot": "ccw"},
            {"type": "Spiral", "sta_start": 260.0, "length": 60.0,
             "start": [260, 30], "end": [320, 35],
             "radius_start": R2, "radius_end": float("inf"),
             "clothoid_A": math.sqrt(R2 * 60), "rot": "ccw"},
            {"type": "Line",   "sta_start": 320.0, "length": 100.0,
             "start": [320, 35], "end": [420, 35], "direction_rad": 0.0},
        ]
        result = enforce_continuity(elements)
        _check_no_inf_inf_spiral(result)
        _check_boundary_continuity(result)

    def test_clothoid_a_recomputed(self):
        """A = sqrt(R_finite * L) must be recomputed after radius correction."""
        R = 600.0
        L = 90.0
        elements = [
            {"type": "Line",   "sta_start": 0.0,   "length": 100.0,
             "start": [0, 0], "end": [100, 0], "direction_rad": 0.0},
            {"type": "Spiral", "sta_start": 100.0, "length": L,
             "start": [100, 0], "end": [190, 3],
             # deliberately wrong radius_start (will be corrected to INF)
             "radius_start": 999.0, "radius_end": R,
             "clothoid_A": 0.0, "rot": "ccw"},
            {"type": "Arc",    "sta_start": 190.0, "length": 200.0,
             "start": [190, 3], "end": [390, 30], "center": None,
             "radius": R, "rot": "ccw", "chord": 180.0},
        ]
        result = enforce_continuity(elements)
        sp = next(e for e in result if e["type"] == "Spiral")
        expected_A = math.sqrt(R * L)
        assert abs(sp["clothoid_A"] - expected_A) < 0.1, (
            f"clothoid_A={sp['clothoid_A']:.3f}, expected {expected_A:.3f}"
        )
        assert math.isinf(sp["radius_start"]), "radius_start should be INF (predecessor is Line)"
        assert abs(sp["radius_end"] - R) < 0.01, f"radius_end should be {R}"


# ===========================================================================
# Tests: _sample_element_points
# ===========================================================================

class TestSampleElement:

    def test_line_sample_count(self):
        el = {"type": "Line", "length": 100.0,
              "start": [0.0, 0.0], "end": [100.0, 0.0]}
        pts = _sample_element_points(el, 10.0)
        assert len(pts) >= 11   # at least one per 10 m

    def test_line_endpoints_correct(self):
        el = {"type": "Line", "length": 50.0,
              "start": [0.0, 0.0], "end": [50.0, 0.0]}
        pts = _sample_element_points(el, 5.0)
        np.testing.assert_allclose(pts[0],  [0.0,  0.0], atol=1e-9)
        np.testing.assert_allclose(pts[-1], [50.0, 0.0], atol=1e-9)

    def test_arc_sample_on_circle(self):
        R      = 300.0
        center = np.array([0.0, R])
        el = {
            "type":   "Arc",
            "length": math.radians(30) * R,
            "start":  [0.0, 0.0],
            "end":    [R * math.sin(math.radians(30)),
                       R * (1 - math.cos(math.radians(30)))],
            "center": center.tolist(),
            "radius": R,
            "rot":    "ccw",
        }
        pts = _sample_element_points(el, 20.0)
        for pt in pts:
            dist = float(np.linalg.norm(pt - center))
            assert abs(dist - R) < 1.0, (
                f"Sample point not on circle: dist={dist:.3f}, R={R}"
            )

    def test_zero_length_returns_empty(self):
        el = {"type": "Line", "length": 0.0,
              "start": [0.0, 0.0], "end": [0.0, 0.0]}
        assert _sample_element_points(el, 5.0) == []


# ===========================================================================
# Tests: _dist_point_to_segment
# ===========================================================================

class TestDistPointToSegment:

    def test_perpendicular(self):
        pt = np.array([5.0, 3.0])
        a  = np.array([0.0, 0.0])
        b  = np.array([10.0, 0.0])
        assert abs(_dist_point_to_segment(pt, a, b) - 3.0) < 1e-9

    def test_past_end(self):
        pt = np.array([15.0, 0.0])
        a  = np.array([0.0,  0.0])
        b  = np.array([10.0, 0.0])
        assert abs(_dist_point_to_segment(pt, a, b) - 5.0) < 1e-9

    def test_before_start(self):
        pt = np.array([-3.0, 4.0])
        a  = np.array([0.0,  0.0])
        b  = np.array([10.0, 0.0])
        assert abs(_dist_point_to_segment(pt, a, b) - 5.0) < 1e-9

    def test_on_segment(self):
        pt = np.array([5.0, 0.0])
        a  = np.array([0.0, 0.0])
        b  = np.array([10.0, 0.0])
        assert _dist_point_to_segment(pt, a, b) < 1e-9


# ===========================================================================
# Tests: max_deviation_element
# ===========================================================================

class TestMaxDeviation:

    def test_line_on_itself_zero_deviation(self):
        """A Line element sampled from the same polyline should have ~0 deviation."""
        xy = make_straight(200.0, 100)
        el = {
            "type":   "Line",
            "length": 200.0,
            "start":  xy[0].tolist(),
            "end":    xy[-1].tolist(),
        }
        dev = max_deviation_element(el, xy, check_interval=5.0)
        assert dev < 1e-6, f"Expected ~0 deviation, got {dev:.6f} m"

    def test_line_offset_deviation(self):
        """A Line shifted 2 m perpendicular to a straight polyline should report ~2 m."""
        xy_osm = make_straight(200.0, 100)
        el = {
            "type":   "Line",
            "length": 200.0,
            "start":  [0.0,   2.0],   # shifted 2 m in y
            "end":    [200.0, 2.0],
        }
        dev = max_deviation_element(el, xy_osm, check_interval=5.0)
        assert abs(dev - 2.0) < 0.05, f"Expected ~2.0 m deviation, got {dev:.4f} m"

    def test_arc_on_itself_zero_deviation(self):
        """Arc element sampled from an exact arc polyline has ~0 deviation."""
        R  = 500.0
        xy = make_arc(R, 20.0, n=80)
        # Use chord endpoints and centre from the generator
        cx = 0.0; cy = R  # from make_arc defaults
        el = {
            "type":   "Arc",
            "length": math.radians(20.0) * R,
            "start":  xy[0].tolist(),
            "end":    xy[-1].tolist(),
            "center": [cx, cy],
            "radius": R,
            "rot":    "ccw",
        }
        dev = max_deviation_element(el, xy, check_interval=10.0)
        assert dev < 1.0, f"Arc on itself should have small deviation, got {dev:.4f} m"


# ===========================================================================
# Tests: fit_alignment (integration)
# ===========================================================================

class TestFitAlignment:

    def test_straight_line_produces_line_elements(self):
        """A perfectly straight polyline should produce only Line elements."""
        xy       = make_straight(1000.0, 200)
        elements = fit_alignment(xy, smooth_window=11, max_deviation=0.0)
        types    = [e["type"] for e in elements]
        assert all(t == "Line" for t in types), \
            f"Straight polyline produced non-Line elements: {types}"

    def test_no_inf_inf_spiral_straight(self):
        xy       = make_straight(800.0, 150)
        elements = fit_alignment(xy, max_deviation=0.0)
        _check_no_inf_inf_spiral(elements)

    def test_no_inf_inf_spiral_lal(self):
        """L-A-L polyline: no spiral should have both radii INF."""
        xy       = make_lal(radius=500.0, sweep_deg=30.0, straight_len=300.0)
        elements = fit_alignment(xy, smooth_window=15, max_deviation=0.0)
        _check_no_inf_inf_spiral(elements)

    def test_boundary_continuity_lal(self):
        xy       = make_lal(radius=500.0, sweep_deg=30.0, straight_len=300.0)
        elements = fit_alignment(xy, smooth_window=15, max_deviation=0.0)
        _check_boundary_continuity(elements)

    def test_boundary_continuity_lsasl(self):
        """Full L-S-A-S-L sequence: boundary radii must match."""
        xy       = make_lsasl(radius=400.0, sweep_deg=40.0, spiral_len=80.0)
        elements = fit_alignment(xy, smooth_window=15, max_deviation=0.0)
        _check_no_inf_inf_spiral(elements)
        _check_boundary_continuity(elements)

    def test_nonempty_output(self):
        xy       = make_lal(radius=400.0, sweep_deg=25.0)
        elements = fit_alignment(xy, smooth_window=11)
        assert len(elements) > 0

    def test_stationing_monotone(self):
        """sta_start must be strictly increasing across elements."""
        xy       = make_lal(radius=500.0, sweep_deg=30.0)
        elements = fit_alignment(xy, smooth_window=11, max_deviation=0.0)
        stations = [e["sta_start"] for e in elements]
        for i in range(1, len(stations)):
            assert stations[i] >= stations[i - 1], \
                f"sta_start not monotone at index {i}: {stations[i-1]:.2f} → {stations[i]:.2f}"

    def test_deviation_within_threshold(self):
        """
        After fitting with max_deviation=0.5 m, every element should report
        a deviation ≤ 0.5 m against the OSM polyline (with some tolerance for
        the check_interval discretisation).
        """
        THRESH         = 0.5
        CHECK_INTERVAL = 5.0
        xy             = make_lsasl(radius=400.0, sweep_deg=40.0, spiral_len=80.0)
        chainages      = compute_chainages(xy)
        elements       = fit_alignment(
            xy,
            smooth_window=15,
            max_deviation=THRESH,
            check_interval=CHECK_INTERVAL,
        )
        for el in elements:
            sta_s = float(el.get("sta_start", 0.0))
            sta_e = sta_s + float(el.get("length", 0.0))
            i0    = int(np.searchsorted(chainages, sta_s - 1e-4))
            i1    = int(np.searchsorted(chainages, sta_e + 1e-4))
            i1    = min(i1, len(xy) - 1)
            xy_seg = xy[i0 : i1 + 1]
            if len(xy_seg) < 2:
                continue
            from geometry.alignment import max_deviation_element
            dev = max_deviation_element(el, xy_seg, CHECK_INTERVAL)
            assert dev <= THRESH * 1.5, (  # 1.5x tolerance for sampling artefacts
                f"Element {el['type']} at sta {sta_s:.0f}–{sta_e:.0f} m: "
                f"deviation {dev:.4f} m exceeds {THRESH * 1.5:.4f} m"
            )

    def test_deviation_tighter_means_more_elements(self):
        """
        Using a tighter max_deviation should produce >= as many elements as
        a looser threshold (more splits allowed).
        """
        xy      = make_lsasl(radius=400.0, sweep_deg=50.0, spiral_len=100.0)
        els_loose  = fit_alignment(xy, smooth_window=15, max_deviation=2.0)
        els_tight  = fit_alignment(xy, smooth_window=15, max_deviation=0.2)
        assert len(els_tight) >= len(els_loose), (
            f"Tighter threshold gave fewer elements: "
            f"{len(els_tight)} < {len(els_loose)}"
        )

    def test_all_elements_have_required_fields(self):
        """Every element must have the fields expected by the LandXML builder."""
        xy       = make_lal(radius=600.0, sweep_deg=20.0)
        elements = fit_alignment(xy, smooth_window=11)
        for el in elements:
            assert "type"      in el
            assert "sta_start" in el
            assert "length"    in el
            assert "start"     in el
            assert "end"       in el
            if el["type"] == "Arc":
                assert "radius" in el
                assert "rot"    in el
            if el["type"] == "Spiral":
                assert "radius_start" in el
                assert "radius_end"   in el
                assert "clothoid_A"   in el
                assert "rot"          in el


# ===========================================================================
# Tests: LandXML builder — spirals serialised correctly
# ===========================================================================

class TestLandXMLSpiral:

    def test_spiral_radii_never_both_inf_in_xml(self):
        """
        After a full pipeline run the LandXML should not contain a Spiral
        whose radiusStart AND radiusEnd are both 'INF'.
        """
        from lxml import etree
        from landxml.builder import build_landxml

        xy       = make_lsasl(radius=400.0, sweep_deg=40.0, spiral_len=80.0)
        elements = fit_alignment(xy, smooth_window=15, max_deviation=0.0)

        aln_data = [{
            "name":     "TestLine",
            "elements": elements,
            "vertical": [],
            "sta_start": 0.0,
        }]
        root = build_landxml(aln_data, output_epsg=32633)
        xml  = etree.tostring(root, pretty_print=True).decode()

        # Find all <Spiral> elements
        ns     = "http://www.landxml.org/schema/LandXML-1.2"
        spirals = root.findall(f".//{{{ns}}}Spiral")
        for sp in spirals:
            r_s = sp.get("radiusStart", "0")
            r_e = sp.get("radiusEnd",   "0")
            assert not (r_s == "INF" and r_e == "INF"), (
                f"Spiral in LandXML has both radiusStart=INF and radiusEnd=INF\n{xml}"
            )
