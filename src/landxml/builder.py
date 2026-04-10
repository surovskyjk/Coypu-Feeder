"""
LandXML 1.2 builder.
Assembles horizontal + vertical geometry into a valid LandXML file.
"""

import math
from datetime import datetime
from lxml import etree
from typing import Optional

LANDXML_NS = "http://www.landxml.org/schema/LandXML-1.2"
XSD_LOC = "http://www.landxml.org/schema/LandXML-1.2 http://www.landxml.org/schema/LandXML1.2/LandXML-1.2.xsd"

# Coordinate format: 6 decimal places is sub-millimetre for metric CRS
COORD_FMT = "{:.6f}"
DIST_FMT = "{:.4f}"
ANGLE_FMT = "{:.10f}"


def build_landxml(
    alignments: list[dict],
    output_epsg: int,
    project_name: str = "Railway Alignment",
) -> etree._Element:
    """
    Build a LandXML 1.2 ElementTree root from a list of alignment dicts.

    Each alignment dict:
      {
        "name": str,
        "elements": list[dict],        # from alignment.fit_alignment()
        "vertical": list[dict],        # from elevation.fit_vertical_geometry()
        "sta_start": float,
      }
    """
    XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
    nsmap = {None: LANDXML_NS, "xsi": XSI_NS}
    root = etree.Element("LandXML", nsmap=nsmap)
    root.set("version", "1.2")
    root.set(f"{{{XSI_NS}}}schemaLocation", XSD_LOC)
    root.set("date", datetime.utcnow().strftime("%Y-%m-%d"))
    root.set("time", datetime.utcnow().strftime("%H:%M:%S"))
    root.set("readOnly", "false")
    root.set("language", "English")

    # <Project>
    proj_el = etree.SubElement(root, "Project")
    proj_el.set("name", project_name)

    # <Units>
    units_el = etree.SubElement(root, "Units")
    metric_el = etree.SubElement(units_el, "Metric")
    metric_el.set("linearUnit", "meter")
    metric_el.set("areaUnit", "squareMeter")
    metric_el.set("volumeUnit", "cubicMeter")
    metric_el.set("angularUnit", "radians")
    metric_el.set("directionUnit", "radians")

    # <CoordinateSystem>
    cs_el = etree.SubElement(root, "CoordinateSystem")
    cs_el.set("epsgCode", str(output_epsg))
    cs_el.set("desc", f"EPSG:{output_epsg}")

    # <Alignments>
    aligns_el = etree.SubElement(root, "Alignments")

    for aln in alignments:
        _add_alignment(aligns_el, aln)

    return root


def write_landxml(root: etree._Element, filepath: str) -> None:
    tree = etree.ElementTree(root)
    tree.write(
        filepath,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )


# ---------------------------------------------------------------------------
# Alignment element
# ---------------------------------------------------------------------------

def _add_alignment(parent: etree._Element, aln: dict) -> None:
    elements = aln.get("elements", [])
    vertical = aln.get("vertical", [])
    sta_start = aln.get("sta_start", 0.0)

    total_length = sum(el.get("length", 0) for el in elements)

    el = etree.SubElement(parent, "Alignment")
    el.set("name", aln["name"])
    el.set("length", DIST_FMT.format(total_length))
    el.set("staStart", DIST_FMT.format(sta_start))

    # Horizontal geometry
    coord_geom = etree.SubElement(el, "CoordGeom")
    coord_geom.set("desc", aln["name"])
    for geom_el in elements:
        _add_geom_element(coord_geom, geom_el)

    # Vertical geometry
    if vertical:
        profile_el = etree.SubElement(el, "Profile")
        prof_align = etree.SubElement(profile_el, "ProfAlign")
        prof_align.set("name", aln["name"] + "_VAlign")
        _add_vertical_elements(prof_align, vertical)


# ---------------------------------------------------------------------------
# Horizontal elements
# ---------------------------------------------------------------------------

def _add_geom_element(parent: etree._Element, el: dict) -> None:
    etype = el.get("type")
    if etype == "Line":
        _add_line(parent, el)
    elif etype == "Arc":
        _add_curve(parent, el)
    elif etype == "Spiral":
        _add_spiral(parent, el)


def _add_line(parent: etree._Element, el: dict) -> None:
    line = etree.SubElement(parent, "Line")
    line.set("length", DIST_FMT.format(el["length"]))
    line.set("dir", ANGLE_FMT.format(el.get("direction_rad", 0.0)))
    _add_point(line, "Start", el["start"])
    _add_point(line, "End", el["end"])


def _add_curve(parent: etree._Element, el: dict) -> None:
    curve = etree.SubElement(parent, "Curve")
    curve.set("length", DIST_FMT.format(el["length"]))
    curve.set("radius", DIST_FMT.format(el["radius"]))
    curve.set("chord", DIST_FMT.format(el.get("chord", el["length"])))
    curve.set("rot", el.get("rot", "ccw"))
    _add_point(curve, "Start", el["start"])
    _add_point(curve, "End", el["end"])
    if el.get("center"):
        _add_point(curve, "Center", el["center"])


def _add_spiral(parent: etree._Element, el: dict) -> None:
    spiral = etree.SubElement(parent, "Spiral")
    spiral.set("length", DIST_FMT.format(el["length"]))
    spiral.set("rot", el.get("rot", "ccw"))
    spiral.set("spiType", "clothoid")

    r_start = el.get("radius_start", float("inf"))
    r_end = el.get("radius_end", float("inf"))
    spiral.set("radiusStart", "INF" if math.isinf(r_start) else DIST_FMT.format(r_start))
    spiral.set("radiusEnd", "INF" if math.isinf(r_end) else DIST_FMT.format(r_end))
    spiral.set("theta", ANGLE_FMT.format(_spiral_theta(el["length"], r_start, r_end)))
    spiral.set("totalX", DIST_FMT.format(el["length"]))
    spiral.set("totalY", DIST_FMT.format(el.get("clothoid_A", 0.0) ** 2 / (6 * max(min(r_start, r_end), 1))))

    _add_point(spiral, "Start", el["start"])
    _add_point(spiral, "End", el["end"])


def _spiral_theta(length: float, r_start: float, r_end: float) -> float:
    """Deflection angle of a clothoid transition."""
    r = min(r_start, r_end)
    if math.isinf(r) or r < 1e-6:
        return 0.0
    return length / (2.0 * r)


def _add_point(parent: etree._Element, tag: str, xy: list[float]) -> None:
    pt = etree.SubElement(parent, tag)
    pt.text = f"{COORD_FMT.format(xy[1])} {COORD_FMT.format(xy[0])}"


# ---------------------------------------------------------------------------
# Vertical elements
# ---------------------------------------------------------------------------

def _add_vertical_elements(parent: etree._Element, vertical: list[dict]) -> None:
    for item in vertical:
        if item["type"] == "PVI":
            pvi = etree.SubElement(parent, "PVI")
            pvi.text = f"{DIST_FMT.format(item['station'])} {DIST_FMT.format(item['elevation'])}"
        elif item["type"] == "ParaCurve":
            pc = etree.SubElement(parent, "ParaCurve")
            pc.set("length", DIST_FMT.format(item["length"]))
            pc.text = f"{DIST_FMT.format(item['station'])} {DIST_FMT.format(item['elevation'])}"
