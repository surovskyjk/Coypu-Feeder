"""
Elevation querying via OpenTopoData (Copernicus GLO-30, 30 m resolution).
Samples elevations along an alignment and fits a vertical geometry profile.
"""

import math
import numpy as np
import requests
from typing import Optional

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/copernicus30"
BATCH_SIZE = 100  # API limit per request
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# DEM querying
# ---------------------------------------------------------------------------

def sample_elevations(
    coords_latlon: list[tuple[float, float]],
) -> list[Optional[float]]:
    """
    Query OpenTopoData for elevations at each (lat, lon).
    Returns a list of elevations (m) in the same order; None where unavailable.
    """
    elevations = []
    for batch_start in range(0, len(coords_latlon), BATCH_SIZE):
        batch = coords_latlon[batch_start:batch_start + BATCH_SIZE]
        locations = "|".join(f"{lat},{lon}" for lat, lon in batch)
        try:
            resp = requests.get(
                OPENTOPODATA_URL,
                params={"locations": locations},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            for result in data.get("results", []):
                elevations.append(result.get("elevation"))
        except Exception:
            elevations.extend([None] * len(batch))
    return elevations


def interpolate_along_alignment(
    coords_latlon: list[tuple[float, float]],
    chainages: np.ndarray,
    sample_interval: float = 20.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Resample alignment at regular chainage intervals for DEM querying.
    Returns (sample_chainages, sample_latlon_coords).
    """
    total_length = float(chainages[-1])
    n_samples = max(2, int(total_length / sample_interval) + 1)
    target_chainages = np.linspace(0, total_length, n_samples)

    sample_coords = []
    for s in target_chainages:
        idx = np.searchsorted(chainages, s, side="right") - 1
        idx = int(np.clip(idx, 0, len(chainages) - 2))
        t = (s - chainages[idx]) / max(chainages[idx + 1] - chainages[idx], 1e-9)
        lat = coords_latlon[idx][0] + t * (coords_latlon[idx + 1][0] - coords_latlon[idx][0])
        lon = coords_latlon[idx][1] + t * (coords_latlon[idx + 1][1] - coords_latlon[idx][1])
        sample_coords.append((lat, lon))

    return target_chainages, sample_coords


# ---------------------------------------------------------------------------
# Vertical geometry fitting
# ---------------------------------------------------------------------------

def fit_vertical_geometry(
    chainages: np.ndarray,
    elevations: list[Optional[float]],
    min_grade_length: float = 50.0,
    vc_length: float = 100.0,
) -> list[dict]:
    """
    Fit a piecewise-linear grade model to the elevation profile, then insert
    parabolic vertical curves at each grade break.

    Returns a list of vertical element dicts:
      {"type": "PVI", "station": ..., "elevation": ...}
      {"type": "ParaCurve", "station": ..., "length": ..., "elevation": ...}
    """
    # Filter out None elevations
    valid_mask = np.array([e is not None for e in elevations])
    if valid_mask.sum() < 2:
        return []

    s = chainages[valid_mask]
    z = np.array([e for e in elevations if e is not None], dtype=float)

    # Smooth elevations to reduce DEM noise
    z_smooth = _smooth_elevations(s, z)

    # Detect grade break points via second-derivative thresholding
    pvs = _detect_grade_breaks(s, z_smooth, min_grade_length)

    # Build PVI list: start, breaks, end
    pvi_list = []
    for station, elevation in pvs:
        pvi_list.append({"type": "PVI", "station": float(station), "elevation": float(elevation)})

    # Insert parabolic vertical curves at interior PVIs
    elements = []
    for i, pvi in enumerate(pvi_list):
        elements.append(pvi)
        if 0 < i < len(pvi_list) - 1:
            # Insert ParaCurve centred on this PVI
            elements.append({
                "type": "ParaCurve",
                "station": float(pvi["station"]),
                "length": float(vc_length),
                "elevation": float(pvi["elevation"]),
            })

    return elements


def _smooth_elevations(s: np.ndarray, z: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(z) < window:
        return z
    kernel = np.ones(window) / window
    z_pad = np.pad(z, window // 2, mode="edge")
    return np.convolve(z_pad, kernel, mode="valid")[: len(z)]


def _detect_grade_breaks(
    s: np.ndarray,
    z: np.ndarray,
    min_length: float,
) -> list[tuple[float, float]]:
    """
    Detect significant grade changes using a sliding window gradient comparison.
    Returns a list of (station, elevation) PVI points including endpoints.
    """
    if len(s) < 3:
        return [(s[0], z[0]), (s[-1], z[-1])]

    grades = np.diff(z) / np.maximum(np.diff(s), 1e-9)
    grade_changes = np.abs(np.diff(grades))

    # Threshold: significant grade change > 0.5%
    threshold = 0.005
    break_indices = np.where(grade_changes > threshold)[0] + 1

    # Ensure minimum spacing between breaks
    filtered = [0]
    for idx in break_indices:
        if s[idx] - s[filtered[-1]] >= min_length:
            filtered.append(int(idx))
    if filtered[-1] != len(s) - 1:
        filtered.append(len(s) - 1)

    return [(float(s[i]), float(z[i])) for i in filtered]
