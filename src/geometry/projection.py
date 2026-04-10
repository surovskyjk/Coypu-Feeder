"""
WGS84 ↔ projected CRS transformations using pyproj.
The internal working CRS is always a metric flat plane (auto-selected UTM or user-chosen).
"""

import math
from pyproj import Transformer, CRS


def utm_epsg_for_lon(lon: float) -> int:
    """Return the EPSG code of the UTM zone covering the given longitude."""
    zone = int((lon + 180) / 6) + 1
    # Assume northern hemisphere for simplicity; railway tool can handle both
    return 32600 + zone  # WGS84 UTM North


def make_transformer(source_epsg: int, target_epsg: int) -> Transformer:
    return Transformer.from_crs(source_epsg, target_epsg, always_xy=True)


def wgs84_to_projected(
    coords_latlon: list[tuple[float, float]],
    target_epsg: int,
) -> list[tuple[float, float]]:
    """
    Project (lat, lon) pairs to (easting, northing) in target_epsg.
    Returns (x, y) = (easting, northing).
    """
    transformer = make_transformer(4326, target_epsg)
    result = []
    for lat, lon in coords_latlon:
        x, y = transformer.transform(lon, lat)
        result.append((x, y))
    return result


def projected_to_wgs84(
    coords_xy: list[tuple[float, float]],
    source_epsg: int,
) -> list[tuple[float, float]]:
    """Inverse: (x, y) → (lat, lon)."""
    transformer = make_transformer(source_epsg, 4326)
    result = []
    for x, y in coords_xy:
        lon, lat = transformer.transform(x, y)
        result.append((lat, lon))
    return result


def auto_utm_epsg(coords_latlon: list[tuple[float, float]]) -> int:
    """Pick a UTM zone based on the centroid of the track."""
    lats = [c[0] for c in coords_latlon]
    lons = [c[1] for c in coords_latlon]
    centre_lon = sum(lons) / len(lons)
    centre_lat = sum(lats) / len(lats)
    zone = int((centre_lon + 180) / 6) + 1
    if centre_lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


# Preset CRS options shown in the GUI
CRS_PRESETS: list[tuple[str, int]] = [
    ("WGS 84 (EPSG:4326)", 4326),
    ("S-JTSK / Krovak East North (EPSG:5514)", 5514),
    ("UTM zone 32N (EPSG:32632)", 32632),
    ("UTM zone 33N (EPSG:32633)", 32633),
    ("UTM zone 34N (EPSG:32634)", 32634),
    ("ETRS89 / UTM zone 32N (EPSG:25832)", 25832),
    ("ETRS89 / UTM zone 33N (EPSG:25833)", 25833),
    ("Auto UTM (from track centroid)", -1),
]
