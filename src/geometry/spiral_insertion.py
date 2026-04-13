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
    L_spiral metres; the Arc geometry is left unchanged.

remove_spiral(elements, arc_idx, side) -> list[dict]
    Remove a previously inserted spiral and restore the Line to its
    original length.

auto_suggest_L(R, available_L, fraction=0.25) -> float
    Rule-of-thumb spiral length suggestion: fraction of the available
    straight, capped to avoid consuming the arc's deflection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Describes a Line–Arc or Arc–Line junction."""
    arc_idx:     int     # index of the Arc element in the elements list
    side:        str     # "entry" (Line→Arc) | "exit" (Arc→Line)
    dir_in:      float   # tangent heading (rad) at the junction, pointing into the Arc
    R:           float   # arc radius (metres)
    rot:         str     # "ccw" | "cw"
    junction_pt: list    # [x, y] projected — the current Line/Arc meeting point
    available_L: float   # metres of adjacent straight we can shorten
    has_spiral:  bool    # True when a Spiral is already present at this junction
    label:       str     # human label, e.g. "Arc #2 — entry"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_transitions(elements: list[dict]) -> list[Transition]:
    """
    Scan a Line/Arc/Spiral element list and return every junction where a
    clothoid spiral can be inserted or has already been inserted.

    Only Line–Arc and Arc–Line adjacencies are considered.  Spiral–Arc
    and Arc–Spiral boundaries are noted as ``has_spiral=True``.
    """
    transitions: list[Transition] = []
    arc_counter: dict[int, int] = {}   # arc element index → human number

    n = len(elements)
    arc_human = 0

    for i, el in enumerate(elements):
        if el.get("type") != "Arc":
            continue

        arc_human += 1
        arc_counter[i] = arc_human
        R   = float(el.get("radius", 300.0))
        rot = el.get("rot", "ccw")

        # ── Entry side: element before this Arc ───────────────────────────
        if i > 0:
            prev = elements[i - 1]
            prev_type = prev.get("type", "Line")

            if prev_type == "Line":
                # Heading at end of Line = direction of the Line
                dir_in    = float(prev.get("direction_rad",
                                           _chord_heading(prev)))
                junc_pt   = list(prev.get("end", el.get("start", [0.0, 0.0])))
                avail_L   = float(prev.get("length", 0.0)) * 0.9  # keep 10% buffer
                transitions.append(Transition(
                    arc_idx=i, side="entry",
                    dir_in=dir_in, R=R, rot=rot,
                    junction_pt=junc_pt,
                    available_L=avail_L,
                    has_spiral=False,
                    label=f"Arc #{arc_human} — entry",
                ))

            elif prev_type == "Spiral":
                dir_in  = float(prev.get("direction_rad",
                                         _chord_heading(prev)))
                junc_pt = list(prev.get("end", el.get("start", [0.0, 0.0])))
                transitions.append(Transition(
                    arc_idx=i, side="entry",
                    dir_in=dir_in, R=R, rot=rot,
                    junction_pt=junc_pt,
                    available_L=0.0,
                    has_spiral=True,
                    label=f"Arc #{arc_human} — entry",
                ))

        # ── Exit side: element after this Arc ─────────────────────────────
        if i < n - 1:
            nxt = elements[i + 1]
            nxt_type = nxt.get("type", "Line")

            if nxt_type == "Line":
                # Heading at end of Arc = tangent to circle at Arc.end
                dir_out   = _arc_end_heading(el)
                junc_pt   = list(el.get("end", nxt.get("start", [0.0, 0.0])))
                avail_L   = float(nxt.get("length", 0.0)) * 0.9
                transitions.append(Transition(
                    arc_idx=i, side="exit",
                    dir_in=dir_out, R=R, rot=rot,
                    junction_pt=junc_pt,
                    available_L=avail_L,
                    has_spiral=False,
                    label=f"Arc #{arc_human} — exit",
                ))

            elif nxt_type == "Spiral":
                dir_out = _arc_end_heading(el)
                junc_pt = list(el.get("end", nxt.get("start", [0.0, 0.0])))
                transitions.append(Transition(
                    arc_idx=i, side="exit",
                    dir_in=dir_out, R=R, rot=rot,
                    junction_pt=junc_pt,
                    available_L=0.0,
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

    The adjacent Line is shortened by *L_spiral* metres.  The Arc element
    is left geometrically unchanged (its start/end coordinates stay fixed;
    the spiral bridges any small C1 gap).

    Returns a new element list (the input list is not mutated).
    Raises ``ValueError`` if the insertion is not feasible (e.g. Line too
    short, already has a spiral).

    Parameters
    ----------
    elements  : current list of element dicts
    arc_idx   : index of the Arc element in *elements*
    side      : "entry" (preceding Line → Arc) or "exit" (Arc → following Line)
    L_spiral  : clothoid arc-length in metres (> 0)
    """
    if L_spiral <= 0:
        raise ValueError("L_spiral must be positive")

    els = [dict(e) for e in elements]   # shallow copy each element dict
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

        # Points
        line_start = np.array(prev.get("start", [0.0, 0.0]), dtype=float)
        line_end   = np.array(prev.get("end",   [0.0, 0.0]), dtype=float)
        arc_start  = np.array(arc.get("start",  [0.0, 0.0]), dtype=float)

        dir_rad = float(prev.get("direction_rad", _chord_heading(prev)))
        dir_hat = np.array([math.cos(dir_rad), math.sin(dir_rad)])

        # TC = step backward L_spiral along Line
        TC = line_end - L_spiral * dir_hat

        # Shorten Line
        new_line = dict(prev)
        new_line["end"]    = TC.tolist()
        new_line["length"] = line_len - L_spiral

        # Spiral: radius_start = ∞, radius_end = R
        r_end = R
        A     = math.sqrt(R * L_spiral)
        sp_end = arc_start   # spiral ends where arc currently starts
        spiral = {
            "type":         "Spiral",
            "sta_start":    new_line["sta_start"] + new_line["length"],
            "length":       L_spiral,
            "start":        TC.tolist(),
            "end":          sp_end.tolist(),
            "radius_start": float("inf"),
            "radius_end":   r_end,
            "clothoid_A":   A,
            "rot":          rot,
        }

        # Rebuild list
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

        arc_end    = np.array(arc.get("end",    [0.0, 0.0]), dtype=float)
        line_start = np.array(nxt.get("start",  [0.0, 0.0]), dtype=float)
        line_end   = np.array(nxt.get("end",    [0.0, 0.0]), dtype=float)

        dir_rad = float(nxt.get("direction_rad", _chord_heading(nxt)))
        dir_hat = np.array([math.cos(dir_rad), math.sin(dir_rad)])

        # CT = step forward L_spiral along Line from its start
        CT = line_start + L_spiral * dir_hat

        # Spiral: radius_start = R, radius_end = ∞
        A  = math.sqrt(R * L_spiral)
        spiral = {
            "type":         "Spiral",
            "sta_start":    arc.get("sta_start", 0.0) + arc.get("length", 0.0),
            "length":       L_spiral,
            "start":        arc_end.tolist(),   # spiral starts at current Arc.end
            "end":          CT.tolist(),
            "radius_start": R,
            "radius_end":   float("inf"),
            "clothoid_A":   A,
            "rot":          rot,
        }

        # Shorten Line (from front)
        new_line = dict(nxt)
        new_line["start"]  = CT.tolist()
        new_line["length"] = line_len - L_spiral

        els[next_idx] = new_line
        els.insert(next_idx, spiral)   # Line shifts to next_idx + 1

    else:
        raise ValueError(f"side must be 'entry' or 'exit', got {side!r}")

    # Recompute sta_start for all elements sequentially
    _recompute_chainages(els)
    return els


def remove_spiral(
    elements: list[dict],
    arc_idx:  int,
    side:     str,
) -> list[dict]:
    """
    Remove a previously inserted spiral and restore the adjacent Line to
    its original length.

    *arc_idx* refers to the Arc element index **in the original list
    before insertion** — i.e. after an entry-spiral insert the Arc is at
    arc_idx+1; after an exit-spiral insert the Arc is still at arc_idx.
    To avoid ambiguity, this function searches for the Arc by identity
    around the expected position.

    Returns a new element list.
    """
    els = [dict(e) for e in elements]

    if side == "entry":
        # Arc may have shifted by +1 (an entry spiral was inserted before it)
        arc_pos = _find_arc_at_or_after(els, arc_idx)
        if arc_pos is None:
            raise ValueError(f"No Arc found near index {arc_idx}")
        spiral_pos = arc_pos - 1
        if spiral_pos < 0 or els[spiral_pos].get("type") != "Spiral":
            raise ValueError("No Spiral found before Arc at the expected position")
        prev_pos = spiral_pos - 1
        if prev_pos < 0 or els[prev_pos].get("type") != "Line":
            raise ValueError("No Line found before Spiral — unexpected structure")

        # Restore Line by extending its end back to Arc.start
        spiral_len = float(els[spiral_pos].get("length", 0.0))
        new_line   = dict(els[prev_pos])
        new_line["end"]    = els[arc_pos].get("start", new_line["end"])
        new_line["length"] = float(new_line.get("length", 0.0)) + spiral_len

        els[prev_pos] = new_line
        del els[spiral_pos]   # remove Spiral

    elif side == "exit":
        arc_pos = _find_arc_at_or_after(els, arc_idx)
        if arc_pos is None:
            raise ValueError(f"No Arc found near index {arc_idx}")
        spiral_pos = arc_pos + 1
        if spiral_pos >= len(els) or els[spiral_pos].get("type") != "Spiral":
            raise ValueError("No Spiral found after Arc at the expected position")
        next_pos = spiral_pos + 1
        if next_pos >= len(els) or els[next_pos].get("type") != "Line":
            raise ValueError("No Line found after Spiral — unexpected structure")

        spiral_len = float(els[spiral_pos].get("length", 0.0))
        new_line   = dict(els[next_pos])
        new_line["start"]  = els[arc_pos].get("end", new_line["start"])
        new_line["length"] = float(new_line.get("length", 0.0)) + spiral_len

        els[next_pos] = new_line
        del els[spiral_pos]

    else:
        raise ValueError(f"side must be 'entry' or 'exit', got {side!r}")

    _recompute_chainages(els)
    return els


def auto_suggest_L(R: float, available_L: float, fraction: float = 0.25) -> float:
    """
    Suggest a spiral length based on arc radius and available straight.

    Rule of thumb:  L ≈ R / 6  (empirical for mainline railways)
    Capped to *fraction* of the available adjacent straight.

    Returns a round number (multiple of 5 m).
    """
    suggested = min(R / 6.0, available_L * fraction)
    suggested = max(suggested, 10.0)       # minimum 10 m
    # Round to nearest 5 m
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


def _recompute_chainages(els: list[dict]) -> None:
    """Update sta_start for every element in-place based on cumulative lengths."""
    sta = 0.0
    for el in els:
        el["sta_start"] = sta
        sta += float(el.get("length", 0.0))


def _find_arc_at_or_after(els: list[dict], target_idx: int) -> int | None:
    """Find the nearest Arc at or after *target_idx* (within ±2 positions)."""
    for offset in range(3):
        for delta in (offset, -offset):
            idx = target_idx + delta
            if 0 <= idx < len(els) and els[idx].get("type") == "Arc":
                return idx
    return None
