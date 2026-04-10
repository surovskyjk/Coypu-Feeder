"""
Parse raw Overpass JSON into ordered node sequences (tracks).
Each 'track' is a list of (lat, lon) tuples forming a continuous polyline.
Ways are chained by shared endpoints.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    way_ids: list[int] = field(default_factory=list)
    nodes: list[tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    tags: dict = field(default_factory=dict)
    name: str = ""


def _build_node_index(elements: list[dict]) -> dict[int, tuple[float, float]]:
    return {
        el["id"]: (el["lat"], el["lon"])
        for el in elements
        if el["type"] == "node"
    }


def _extract_ways(elements: list[dict]) -> list[dict]:
    return [el for el in elements if el["type"] == "way"]


def _chain_ways(ways: list[dict], node_index: dict) -> list[list[int]]:
    """
    Chain ways into continuous sequences by matching endpoints.
    Returns a list of node-ID chains.
    """
    # Build adjacency: endpoint node_id → list of way indices
    endpoint_map: dict[int, list[int]] = {}
    for i, way in enumerate(ways):
        nodes = way.get("nodes", [])
        if len(nodes) < 2:
            continue
        for endpoint in (nodes[0], nodes[-1]):
            endpoint_map.setdefault(endpoint, []).append(i)

    used = set()
    chains = []

    for start_idx in range(len(ways)):
        if start_idx in used:
            continue
        way = ways[start_idx]
        nodes = way.get("nodes", [])
        if len(nodes) < 2:
            used.add(start_idx)
            continue

        chain = list(nodes)
        used.add(start_idx)

        # Extend forward
        while True:
            tail = chain[-1]
            candidates = [i for i in endpoint_map.get(tail, []) if i not in used]
            if not candidates:
                break
            next_idx = candidates[0]
            next_nodes = ways[next_idx].get("nodes", [])
            if next_nodes[0] == tail:
                chain.extend(next_nodes[1:])
            else:
                chain.extend(reversed(next_nodes[:-1]))
            used.add(next_idx)

        # Extend backward
        while True:
            head = chain[0]
            candidates = [i for i in endpoint_map.get(head, []) if i not in used]
            if not candidates:
                break
            next_idx = candidates[0]
            next_nodes = ways[next_idx].get("nodes", [])
            if next_nodes[-1] == head:
                chain = next_nodes + chain[1:]
            else:
                chain = list(reversed(next_nodes)) + chain[1:]
            used.add(next_idx)

        chains.append(chain)

    return chains


def _group_by_track(ways: list[dict]) -> list[list[dict]]:
    """
    Group ways by track number (OSM tag 'railway:track' or positional proximity).
    Falls back to treating all ways as one group if no track tags exist.
    """
    track_map: dict[str, list[dict]] = {}
    for way in ways:
        tags = way.get("tags", {})
        track_key = tags.get("railway:track", tags.get("track", "1"))
        track_map.setdefault(track_key, []).append(way)

    # If only one implicit group, don't split further
    if len(track_map) == 1:
        return list(track_map.values())

    return list(track_map.values())


def parse_tracks(overpass_data: dict) -> list[Track]:
    """
    Parse Overpass JSON into a list of Track objects.
    Multiple parallel tracks produce separate Track entries.
    """
    elements = overpass_data.get("elements", [])
    node_index = _build_node_index(elements)
    ways = _extract_ways(elements)

    if not ways:
        return []

    way_groups = _group_by_track(ways)
    tracks = []

    for group_idx, group in enumerate(way_groups):
        chains = _chain_ways(group, node_index)
        for chain_idx, node_ids in enumerate(chains):
            coords = []
            for nid in node_ids:
                if nid in node_index:
                    coords.append(node_index[nid])

            if len(coords) < 2:
                continue

            # Collect representative tags from the first way in group
            tags = group[0].get("tags", {}) if group else {}
            track_num = tags.get("railway:track", tags.get("track", str(group_idx + 1)))
            name = f"Track {track_num}" if len(way_groups) > 1 else "Track 1"
            if len(chains) > 1:
                name += f" (segment {chain_idx + 1})"

            track = Track(
                way_ids=[w["id"] for w in group],
                nodes=coords,
                tags=tags,
                name=name,
            )
            tracks.append(track)

    return tracks


def get_track_names(tracks: list[Track]) -> list[str]:
    return [t.name for t in tracks]
