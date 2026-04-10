"""
Overpass API queries for railway data.
Supports search by name, relation ID, and bounding box.
"""

import time
import requests
from typing import Optional

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
TIMEOUT = 60


def _run_query(query: str) -> dict:
    last_exc = None
    for url in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(url, data={"data": query}, timeout=TIMEOUT)
            if response.status_code == 429:
                time.sleep(5)
                response = requests.post(url, data={"data": query}, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code in (429, 504):
                time.sleep(3)
                continue
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
    raise last_exc


def search_railways_by_name(name: str) -> list[dict]:
    """
    Search for railway route relations matching a name.
    Returns a list of dicts with keys: id, name, network, operator, from, to.
    """
    query = f"""
[out:json][timeout:30];
relation["type"="route"]["route"="railway"]["name"~"{name}",i];
out tags;
"""
    data = _run_query(query)
    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        results.append({
            "id": el["id"],
            "name": tags.get("name", ""),
            "network": tags.get("network", ""),
            "operator": tags.get("operator", ""),
            "from": tags.get("from", ""),
            "to": tags.get("to", ""),
        })
    return results


def fetch_relation_ways(relation_id: int) -> dict:
    """
    Fetch all ways and nodes belonging to a railway route relation.
    Returns raw Overpass JSON with nodes and ways.
    """
    query = f"""
[out:json][timeout:{TIMEOUT}];
relation({relation_id});
way(r);
(._; >;);
out body;
"""
    return _run_query(query)


def fetch_bbox_ways(south: float, west: float, north: float, east: float) -> dict:
    """
    Fetch all railway ways within a bounding box.
    Returns raw Overpass JSON with nodes and ways.
    """
    query = f"""
[out:json][timeout:{TIMEOUT}];
(
  way["railway"="rail"]({south},{west},{north},{east});
  way["railway"="light_rail"]({south},{west},{north},{east});
  way["railway"="subway"]({south},{west},{north},{east});
);
(._; >;);
out body;
"""
    return _run_query(query)


def fetch_relation_metadata(relation_id: int) -> Optional[dict]:
    """Fetch tags for a single relation."""
    query = f"""
[out:json][timeout:15];
relation({relation_id});
out tags;
"""
    data = _run_query(query)
    elements = data.get("elements", [])
    if not elements:
        return None
    tags = elements[0].get("tags", {})
    return {
        "id": relation_id,
        "name": tags.get("name", f"Relation {relation_id}"),
        "network": tags.get("network", ""),
        "operator": tags.get("operator", ""),
        "from": tags.get("from", ""),
        "to": tags.get("to", ""),
    }
