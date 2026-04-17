"""
Spiral (Euler clothoid) insertion at Line–Arc transitions.

Public API
----------
find_transitions(elements) -> list[Transition]
    Scan a list of Line/Arc elements and return every junction where a
    clothoid spiral could be inserted (or has already been inserted).

insert_spiral(elements, arc_idx, side, L_spiral) -> list[dict]
    Return a new element list with a spiral inserted at the requested
    Line–Arc or Arc–Line junction.  The adjacent Line is shortened by
    L_spiral metres; the Arc geometry is recomputed from the spiral exit
    so that C0 and C1 continuity are preserved.

remove_spiral(elements, arc_idx, side) -> list[dict]
    Remove a previously inserted spiral, restore the Line, and recompute
    the Arc from the original entry tangent.

enforce_c1_chain(elements) -> list[dict]
    Walk elements in order, re-propagating (pos, heading) so that every
    junction is exactly C0 and C1 continuous.  Mutates in-place and
    returns the list.

auto_suggest_L(R, available_L, fraction=0.25) -> float
    Rule-of-thumb spiral length: R/6, capped to fraction of the available
    straight, rounded to nearest 5 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Describes a Line–Arc or Arc–Line junction."""
    arc_idx:     int    # index of the Arc element in the elements list
    side:        str    # "entry" (Line→Arc) | "exit" (Arc→Line)
    dir_in:      float  # tangent heading (rad) at the junction, into the Arc
    R:           float  # arc radius (metres)
    rot:         str    # "ccw" | "cw"
    junction_pt: list   # [x, y] projected — current Line/Arc meeting point
    available_L: float  # metres of adjacent straight available to shorten
    has_spiral:  bool   # True when a Spiral is already present at this junction
    label:       str    # human label e.g. "Arc #2 — entry"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_transitions(elements: list[dict]) -> list[Transition]:
    """
    Scan a Line/Arc/Spiral element list and return every junction where a
    clothoid spiral can be inserted or has already been inserted.

    Only Line–Arc and Arc–Line adjacencies are considered.
    """
    transitions: list[Transition] = []
    n         = len(elements)
    arc_human = 0

    for i, el in enumerate(elements):
        if el.get("type") != "Arc":
            continue

        arc_human += 1
        R   = float(el.get("radius", 300.0))
        rot = el.get("rot", "ccw")

        # ── Entry side ────────────────────────────────────────────────
        if i > 0:
            prev      = elements[i - 1]
            prev_type = prev.get("type", "Line")

            if prev_type == "Line":
                dir_in  = float(prev.get("direction_rad", _chord_heading(prev)))
                junc_pt = list(prev.get("end", el.get("start", [0.0, 0.0])))
                avail_L = float(prev.get("length", 0.0)) * 0.9
                transitions.append(Transition(
                    arc_idx=i, side="entry",
                    dir_in=dir_in, R=R, rot=rot,
                    junction_pt=junc_pt, available_L=avail_L,
                    has_spiral=False,
                    label=f"Arc #{arc_human} — entry",
                ))

            elif prev_type == "Spiral":
                dir_in  = float(prev.get("direction_rad", _chord_heading(prev)))
                junc_pt = list(prev.get("end", el.get("start", [0.0, 0.0])))
                transitions.append(Transition(
                    arc_idx=i, side="entry",
                    dir_in=dir_in, R=R, rot=rot,
                    junction_pt=junc_pt, available_L=0.0,
                    has_spiral=True,
                    label=f"Arc #{arc_human} — entry",
                ))

        # ── Exit side ─────────────────────────────────────────────────
        if i < n - 1:
            nxt      = elements[i + 1]
            nxt_type = nxt.get("type", "Line")

            if nxt_type == "Line":
                dir_out = _arc_end_heading(el)
                junc_pt = list(el.get("end", nxt.get("start", [0.0, 0.0])))
                avail_L = float(nxt.get("length", 0.0)) * 0.9
                transitions.append(Transition(
                    arc_idx=i, side="exit",
                    dir_in=dir_out, R=R, rot=rot,
                    junction_pt=junc_pt, available_L=avail_L,
                    has_spiral=False,
                    label=f"Arc #{arc_human} — exit",
                ))

            elif nxt_type == "Spiral":
                dir_out = _arc_end_heading(el)
                junc_pt = list(el.get("end", nxt.get("start", [0.0, 0.0])))
                transitions.append(Transition(
                    arc_idx=i, side="exit",
                    dir_in=dir_out, R=R, rot=rot,
                    junction_pt=junc_pt, available_L=0.0,
                    has_spiral=True,
                    label=f"Arc #{arc_human} — exit",
                ))

    return transitions


def insert_spiral(
    elements: list[dict],
    arc_idx:  int,
    side:     str,
    L_spiral: float,
) -> list[dict]:
    """
    Insert an Euler spiral of length *L_spiral* at the Line–Arc boundary
    identified by (*arc_idx*, *side*).

    The adjacent Line is shortened by *L_spiral* metres.  The Arc geometry
    is **recomputed** from the spiral exit point using the stored deflection
    angle, so C0 and C1 continuity are preserved.

    Returns a new element list.  The input list is not mutated.
    Raises ValueError if the insertion is not feasible.
    """
    if L_spiral <= 0:
        raise ValueError("L_spiral must be positive")

    els = [dict(e) for e in elements]
    arc = els[arc_idx]
    R   = float(arc.get("radius", 300.0))
    rot = arc.get("rot", "ccw")
    sign = 1.0 if rot == "ccw" else -1.0

    if side == "entry":
        if arc_idx == 0:
            raise ValueError("No element before Arc — cannot insert entry spiral")
        prev_idx = arc_idx - 1
        prev = els[prev_idx]
        if prev.get("type") != "Line":
            raise ValueError(
                f"Element before Arc is '{prev.get('type')}', not Line — "
                "cannot insert entry spiral (already has one?)"
            )
        line_len = float(prev.get("length", 0.0))
        if L_spiral > line_len - 1.0:
            raise ValueError(
                f"Spiral length {L_spiral:.1f} m exceeds available "
                f"Line length {line_len:.1f} m"
            )

        dir_rad = float(prev.get("direction_rad", _chord_heading(prev)))
        line_start = np.array(prev.get("start", [0.0, 0.0]), dtype=float)
        dir_hat    = np.array([math.cos(dir_rad), math.sin(dir_rad)])

        # TC = step back L_spiral from the line's current end
        line_end = np.array(prev.get("end", [0.0, 0.0]), dtype=float)
        TC = line_end - L_spiral * dir_hat

        # Spiral exit position via Fresnel integrals (in local tangent frame)
        from geometry.alignment import _compute_clothoid_shift
        x_sp, y_sp = _compute_clothoid_shift(L_spiral, R)
        cos_d = math.cos(dir_rad)
        sin_d = math.sin(dir_rad)
        new_arc_start = np.array([
            float(TC[0]) + cos_d * x_sp - sign * sin_d * y_sp,
            float(TC[1]) + sin_d * x_sp + sign * cos_d * y_sp,
        ])

        # Exit heading of spiral = entry + sign * L/(2R)
        exit_heading = dir_rad + sign * L_spiral / (2.0 * R)

        # Shorten Line
        new_line = dict(prev)
        new_line["end"]    = TC.tolist()
        new_line["length"] = line_len - L_spiral

        # Spiral element
        A_cloth = math.sqrt(R * L_spiral)
        spiral = {
            "type":         "Spiral",
            "sta_start":    new_line["sta_start"] + new_line["length"],
            "length":       L_spiral,
            "start":        TC.tolist(),
            "end":          new_arc_start.tolist(),
            "radius_start": float("inf"),
            "radius_end":   R,
            "clothoid_A":   A_cloth,
            "rot":          rot,
        }

        # Recompute Arc from spiral exit — preserve deflection angle
        defl      = _arc_deflection(arc)
        new_center_angle = exit_heading + sign * math.pi / 2.0
        new_center = new_arc_start + R * np.array([
            math.cos(new_center_angle), math.sin(new_center_angle)
        ])
        a_start = math.atan2(
            new_arc_start[1] - new_center[1],
            new_arc_start[0] - new_center[0],
        )
        a_end   = a_start + sign * abs(defl)
        new_arc_end = new_center + R * np.array([math.cos(a_end), math.sin(a_end)])

        arc["start"]       = new_arc_start.tolist()
        arc["end"]         = new_arc_end.tolist()
        arc["center"]      = new_center.tolist()
        arc["chord"]       = float(np.linalg.norm(new_arc_end - new_arc_start))
        arc["length"]      = R * abs(defl)
        arc["_deflection"] = defl

        els[prev_idx] = new_line
        els.insert(arc_idx, spiral)   # Arc shifts to arc_idx + 1

    elif side == "exit":
        if arc_idx >= len(els) - 1:
            raise ValueError("No element after Arc — cannot insert exit spiral")
        next_idx = arc_idx + 1
        nxt = els[next_idx]
        if nxt.get("type") != "Line":
            raise ValueError(
                f"Element after Arc is '{nxt.get('type')}', not Line — "
                "cannot insert exit spiral (already has one?)"
            )
        line_len = float(nxt.get("length", 0.0))
        if L_spiral > line_len - 1.0:
            raise ValueError(
                f"Spiral length {L_spiral:.1f} m exceeds available "
                f"Line length {line_len:.1f} m"
            )

        # Heading at arc exit = tangent to circle at Arc.end
        arc_exit_heading = _arc_end_heading(arc)
        arc_end          = np.array(arc.get("end", [0.0, 0.0]), dtype=float)

        # Spiral start = current Arc.end; compute spiral exit (CT) via Fresnel
        from geometry.alignment import _compute_clothoid_shift
        x_sp, y_sp = _compute_clothoid_shift(L_spiral, R)
        cos_d = math.cos(arc_exit_heading)
        sin_d = math.sin(arc_exit_heading)
        CT = np.array([
            float(arc_end[0]) + cos_d * x_sp - sign * sin_d * y_sp,
            float(arc_end[1]) + sin_d * x_sp + sign * cos_d * y_sp,
        ])

        # CT heading (entry of shortened Line)
        CT_heading = arc_exit_heading + sign * L_spiral / (2.0 * R)

        A_cloth = math.sqrt(R * L_spiral)
        spiral = {
            "type":         "Spiral",
            "sta_start":    arc.get("sta_start", 0.0) + arc.get("length", 0.0),
            "length":       L_spiral,
            "start":        arc_end.tolist(),
            "end":          CT.tolist(),
            "radius_start": R,
            "radius_end":   float("inf"),
            "clothoid_A":   A_cloth,
            "rot":          rot,
        }

        # Shorten Line from its front, update direction
        line_dir = float(nxt.get("direction_rad", _chord_heading(nxt)))
        line_end = np.array(nxt.get("end", [0.0, 0.0]), dtype=float)
        new_line = dict(nxt)
        new_line["start"]        = CT.tolist()
        new_line["length"]       = line_len - L_spiral
        new_line["direction_rad"] = CT_heading

        els[next_idx] = new_line
        els.insert(next_idx, spiral)   # Line shifts to next_idx + 1

    else:
        raise ValueError(f"side must be 'entry' or 'exit', got {side!r}")

    _recompute_chainages(els)
    enforce_c1_chain(els)
    return els


def remove_spiral(
    elements: list[dict],
    arc_idx:  int,
    side:     str,
) -> list[dict]:
    """
    Remove a previously inserted spiral and restore the adjacent Line.

    The Arc geometry is recomputed from the restored Line exit tangent so
    that C0 and C1 continuity are preserved.

    Returns a new element list.
    """
    els = [dict(e) for e in elements]

    if side == "entry":
        arc_pos = _find_arc_near(els, arc_idx)
        if arc_pos is None:
            raise ValueError(f"No Arc found near index {arc_idx}")
        spiral_pos = arc_pos - 1
        if spiral_pos < 0 or els[spiral_pos].get("type") != "Spiral":
            raise ValueError("No Spiral before Arc at expected position")
        prev_pos = spiral_pos - 1
        if prev_pos < 0 or els[prev_pos].get("type") != "Line":
            raise ValueError("No Line before Spiral — unexpected structure")

        spiral_len = float(els[spiral_pos].get("length", 0.0))
        new_line   = dict(els[prev_pos])
        # Restore Line: extend end back to where the arc now starts
        dir_rad    = float(new_line.get("direction_rad", 0.0))
        dir_hat    = np.array([math.cos(dir_rad), math.sin(dir_rad)])
        old_end    = np.array(new_line.get("end", [0.0, 0.0]), dtype=float)
        restored_end = old_end + spiral_len * dir_hat
        new_line["end"]    = restored_end.tolist()
        new_line["length"] = float(new_line.get("length", 0.0)) + spiral_len

        arc = els[arc_pos]
        els[prev_pos] = new_line
        del els[spiral_pos]   # spiral removed; arc is now at prev arc_pos - 1

        # Recompute Arc from restored entry tangent
        _recompute_arc_from_entry(els, arc_pos - 1, dir_rad)

    elif side == "exit":
        arc_pos = _find_arc_near(els, arc_idx)
        if arc_pos is None:
            raise ValueError(f"No Arc found near index {arc_idx}")
        spiral_pos = arc_pos + 1
        if spiral_pos >= len(els) or els[spiral_pos].get("type") != "Spiral":
            raise ValueError("No Spiral after Arc at expected position")
        next_pos = spiral_pos + 1
        if next_pos >= len(els) or els[next_pos].get("type") != "Line":
            raise ValueError("No Line after Spiral — unexpected structure")

        spiral_len = float(els[spiral_pos].get("length", 0.0))
        new_line   = dict(els[next_pos])
        # Restore Line: move start back by spiral_len in the line's direction
        dir_rad    = float(new_line.get("direction_rad", _chord_heading(new_line)))
        dir_hat    = np.array([math.cos(dir_rad), math.sin(dir_rad)])
        old_start  = np.array(new_line.get("start", [0.0, 0.0]), dtype=float)
        restored_start = old_start - spiral_len * dir_hat
        new_line["start"]  = restored_start.tolist()
        new_line["length"] = float(new_line.get("length", 0.0)) + spiral_len

        els[next_pos] = new_line
        del els[spiral_pos]

    else:
        raise ValueError(f"side must be 'entry' or 'exit', got {side!r}")

    _recompute_chainages(els)
    enforce_c1_chain(els)
    return els


def enforce_c1_chain(elements: list[dict]) -> list[dict]:
    """
    Walk elements in order, re-propagating (pos, heading) through each element.

    For every element:
      - el["start"] is set to the current propagated position.
      - el["end"], el["center"] (Arc), and el["direction_rad"] (Line) are
        recomputed from the stored length / radius / deflection.
      - The exit position and heading become the input for the next element.

    Mutates elements in-place and returns the list.
    """
    from geometry.alignment import _compute_clothoid_shift

    if not elements:
        return elements

    # Anchor: the very first start point is ground truth
    pos     = np.array(elements[0]["start"], dtype=float)
    # Derive initial heading from first element
    heading = _element_entry_heading(elements[0])

    for el in elements:
        el["start"] = pos.tolist()
        etype = el.get("type", "Line")

        if etype == "Line":
            length  = float(el.get("length", 0.0))
            end     = pos + length * np.array([math.cos(heading), math.sin(heading)])
            el["end"]           = end.tolist()
            el["direction_rad"] = heading
            pos = end
            # heading unchanged

        elif etype == "Arc":
            R    = float(el.get("radius", 300.0))
            rot  = el.get("rot", "ccw")
            sign = 1.0 if rot == "ccw" else -1.0
            defl = el.get("_deflection") or _arc_deflection(el)

            if abs(defl) < 1e-9 or R < 1e-3:
                # Degenerate: treat as line
                length = float(el.get("length", 0.0))
                end    = pos + length * np.array([math.cos(heading), math.sin(heading)])
                el["end"] = end.tolist()
                pos = end
                continue

            center_angle = heading + sign * math.pi / 2.0
            center       = pos + R * np.array([
                math.cos(center_angle), math.sin(center_angle)
            ])
            a_start = math.atan2(pos[1] - center[1], pos[0] - center[0])
            a_end   = a_start + sign * abs(defl)
            end     = center + R * np.array([math.cos(a_end), math.sin(a_end)])

            el["center"]      = center.tolist()
            el["end"]         = end.tolist()
            el["chord"]       = float(np.linalg.norm(end - pos))
            el["length"]      = R * abs(defl)
            el["_deflection"] = defl

            pos      = end
            heading += sign * abs(defl)

        elif etype == "Spiral":
            L        = float(el.get("length", 0.0))
            r_start  = el.get("radius_start", float("inf"))
            r_end    = el.get("radius_end",   float("inf"))
            rot      = el.get("rot", "ccw")
            sign     = 1.0 if rot == "ccw" else -1.0

            # Determine which end connects to a circle
            if math.isinf(r_start) and not math.isinf(r_end):
                R = float(r_end)   # entry spiral: Line→Arc
            elif not math.isinf(r_start) and math.isinf(r_end):
                R = float(r_start) # exit spiral:  Arc→Line
            elif not math.isinf(r_start) and not math.isinf(r_end):
                R = min(float(r_start), float(r_end))
            else:
                R = 300.0  # degenerate

            if L < 1e-6 or R < 1e-3:
                el["end"] = pos.tolist()
                continue

            x_sp, y_sp = _compute_clothoid_shift(L, R)
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            end = np.array([
                pos[0] + cos_h * x_sp - sign * sin_h * y_sp,
                pos[1] + sin_h * x_sp + sign * cos_h * y_sp,
            ])
            el["end"] = end.tolist()

            # Heading change = sign * L / (2R)
            heading += sign * L / (2.0 * R)
            pos = end

    _recompute_chainages(elements)
    return elements


def auto_suggest_L(R: float, available_L: float, fraction: float = 0.25) -> float:
    """
    Suggest a spiral length based on arc radius and available straight.

    Rule of thumb:  L ≈ R / 6  (empirical for mainline railways).
    Capped to *fraction* of the available adjacent straight.
    Returns a multiple of 5 m, minimum 10 m.
    """
    suggested = min(R / 6.0, available_L * fraction)
    suggested = max(suggested, 10.0)
    return round(suggested / 5.0) * 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chord_heading(el: dict) -> float:
    start = el.get("start", [0.0, 0.0])
    end   = el.get("end",   [0.0, 0.0])
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _arc_end_heading(el: dict) -> float:
    """Tangent heading at the end of an Arc element."""
    center = el.get("center")
    end    = el.get("end", [0.0, 0.0])
    rot    = el.get("rot", "ccw")
    sign   = 1.0 if rot == "ccw" else -1.0
    if center is not None:
        radial = math.atan2(end[1] - center[1], end[0] - center[0])
        return radial + sign * math.pi / 2.0
    return _chord_heading(el)


def _arc_deflection(arc: dict) -> float:
    """
    Return the stored _deflection if present, otherwise reconstruct it from
    the Arc's center, start, and end coordinates.
    """
    stored = arc.get("_deflection")
    if stored is not None:
        return float(stored)

    center = arc.get("center")
    start  = arc.get("start", [0.0, 0.0])
    end    = arc.get("end",   [0.0, 0.0])
    rot    = arc.get("rot", "ccw")
    sign   = 1.0 if rot == "ccw" else -1.0

    if center is None:
        # Estimate from arc length and radius
        L = float(arc.get("length", 0.0))
        R = float(arc.get("radius", 300.0))
        return sign * (L / R) if R > 0 else 0.0

    a_s = math.atan2(start[1] - center[1], start[0] - center[0])
    a_e = math.atan2(end[1]   - center[1], end[0]   - center[0])
    delta = a_e - a_s
    # Unwrap to the correct sign direction
    if sign > 0 and delta < 0:
        delta += 2.0 * math.pi
    elif sign < 0 and delta > 0:
        delta -= 2.0 * math.pi
    return delta


def _element_entry_heading(el: dict) -> float:
    """Estimate the entry heading of the very first element."""
    etype = el.get("type", "Line")
    if etype == "Line":
        return float(el.get("direction_rad", _chord_heading(el)))
    elif etype == "Arc":
        # Entry tangent is perpendicular to center→start vector
        center = el.get("center")
        start  = el.get("start", [0.0, 0.0])
        rot    = el.get("rot", "ccw")
        sign   = 1.0 if rot == "ccw" else -1.0
        if center is not None:
            radial = math.atan2(start[1] - center[1], start[0] - center[0])
            return radial + sign * math.pi / 2.0
        return _chord_heading(el)
    else:
        return _chord_heading(el)


def _recompute_chainages(els: list[dict]) -> None:
    """Update sta_start for every element in-place from cumulative lengths."""
    sta = 0.0
    for el in els:
        el["sta_start"] = sta
        sta += float(el.get("length", 0.0))


def _recompute_arc_from_entry(els: list[dict], arc_idx: int, entry_heading: float) -> None:
    """
    Recompute the Arc element at *arc_idx* using *entry_heading* as the
    tangent direction at Arc.start.  Preserves the deflection angle.
    """
    if arc_idx < 0 or arc_idx >= len(els):
        return
    arc  = els[arc_idx]
    R    = float(arc.get("radius", 300.0))
    rot  = arc.get("rot", "ccw")
    sign = 1.0 if rot == "ccw" else -1.0
    defl = _arc_deflection(arc)

    start        = np.array(arc.get("start", [0.0, 0.0]), dtype=float)
    center_angle = entry_heading + sign * math.pi / 2.0
    new_center   = start + R * np.array([math.cos(center_angle), math.sin(center_angle)])
    a_start      = math.atan2(start[1] - new_center[1], start[0] - new_center[0])
    a_end        = a_start + sign * abs(defl)
    new_end      = new_center + R * np.array([math.cos(a_end), math.sin(a_end)])

    arc["center"]      = new_center.tolist()
    arc["end"]         = new_end.tolist()
    arc["chord"]       = float(np.linalg.norm(new_end - start))
    arc["length"]      = R * abs(defl)
    arc["_deflection"] = defl


def _find_arc_near(els: list[dict], target_idx: int) -> int | None:
    """Find the nearest Arc at or within ±2 positions of *target_idx*."""
    for offset in range(3):
        for delta in (offset, -offset):
            idx = target_idx + delta
            if 0 <= idx < len(els) and els[idx].get("type") == "Arc":
                return idx
    return None
