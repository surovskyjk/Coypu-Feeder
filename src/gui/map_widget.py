"""
Map widget: QWebEngineView hosting a Leaflet.js map.
Python ↔ JavaScript via QWebChannel.

Tiles:
  Dark  — CARTO Dark Matter base + OpenRailwayMap overlay (coloured railways)
  Light — CARTO Voyager base   + OpenRailwayMap overlay

Custom Leaflet panes keep the alignment overlay on top of tracks:
  tracksPane        zIndex 410
  alignmentGlowPane zIndex 640
  alignmentPane     zIndex 650  ← exported line, always visible
"""

from __future__ import annotations

import json
from collections import deque

from PySide6.QtCore import QObject, Signal, Slot, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

# ---------------------------------------------------------------------------
# Track colours
# ---------------------------------------------------------------------------
TRACK_COLORS = [
    "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
    "#ce93d8", "#80cbc4", "#fff176", "#ff8a65",
]

# ---------------------------------------------------------------------------
# Inline HTML / JavaScript
# ---------------------------------------------------------------------------

MAP_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body, #map { width:100%; height:100%; margin:0; padding:0; }
</style>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map', {zoomControl: true}).setView([50.05, 14.42], 7);

/* Custom panes for z-ordering */
map.createPane('tracksPane').style.zIndex      = 410;
map.createPane('alignmentGlowPane').style.zIndex = 640;
map.createPane('alignmentPane').style.zIndex   = 650;

var baseLayer      = null;
var railOverlay    = null;
var trackLayers    = [];
var alignmentLayers = [];
var backend        = null;

/* ── Theme / tile switching ─────────────────────────────────────── */
function setTheme(dark) {
  if (baseLayer)   { map.removeLayer(baseLayer);   baseLayer   = null; }
  if (railOverlay) { map.removeLayer(railOverlay); railOverlay = null; }

  var tileUrl = dark
    ? 'https://{s}.basemaps.cartocdn.com/dark_matter/{z}/{x}/{y}.png'
    : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png';
  var tileAttr =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' +
    ' &copy; <a href="https://carto.com/attributions">CARTO</a>';

  baseLayer = L.tileLayer(tileUrl, {
    attribution: tileAttr,
    subdomains: 'abcd',
    maxZoom: 20
  }).addTo(map);

  /* OpenRailwayMap — colours railways by type/status */
  railOverlay = L.tileLayer(
    'https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
    {
      attribution: '&copy; <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a>',
      subdomains: 'abc',
      maxZoom: 19,
      opacity: dark ? 0.85 : 0.65
    }
  ).addTo(map);
}

/* ── QWebChannel bootstrap ────────────────────────────────────────*/
new QWebChannel(qt.webChannelTransport, function(channel) {
  backend = channel.objects.backend;
  backend.on_ready();
});

/* ── Map bounds (for "search in view") ───────────────────────────*/
function getMapBounds() {
  var b = map.getBounds();
  if (backend) backend.on_bounds_ready(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
}

/* ── Track display ───────────────────────────────────────────────*/
function showTracks(jsonStr) {
  trackLayers.forEach(function(l) { map.removeLayer(l); });
  trackLayers = [];
  var tracks = JSON.parse(jsonStr);
  var all = [];
  tracks.forEach(function(t) {
    var color   = t.color || '#4fc3f7';
    var latlngs = t.nodes.map(function(n) { return [n[0], n[1]]; });
    all = all.concat(latlngs);
    var pl = L.polyline(latlngs, {
      color: color, weight: 3, opacity: 0.85, pane: 'tracksPane'
    });
    pl.options._baseColor = color;
    pl.addTo(map);
    trackLayers.push(pl);
  });
  if (all.length > 0) {
    map.fitBounds(L.latLngBounds(all), {padding: [20, 20]});
  }
}

function highlightTrack(idx) {
  trackLayers.forEach(function(l, i) {
    var base = l.options._baseColor || l.options.color;
    if (idx < 0) {
      l.setStyle({color: base, weight: 3, opacity: 0.85});
    } else if (i === idx) {
      l.setStyle({color: '#ffeb3b', weight: 6, opacity: 1.0});
      l.bringToFront();
    } else {
      l.setStyle({color: base, weight: 2, opacity: 0.3});
    }
  });
}

function flyToTracks() {
  if (trackLayers.length === 0) return;
  var all = [];
  trackLayers.forEach(function(l) { all = all.concat(l.getLatLngs()); });
  if (all.length > 0)
    map.flyToBounds(L.latLngBounds(all), {padding: [30, 30], duration: 0.8});
}

/* ── Exported alignment overlay ──────────────────────────────────*/
function showAlignment(jsonStr) {
  clearAlignment();
  var alns = JSON.parse(jsonStr);
  var all  = [];
  alns.forEach(function(aln) {
    var latlngs = aln.nodes.map(function(n) { return [n[0], n[1]]; });
    if (latlngs.length < 2) return;
    all = all.concat(latlngs);
    /* Glow: wide semi-transparent halo */
    var glow = L.polyline(latlngs, {
      color: '#ffffff', weight: 12, opacity: 0.25, pane: 'alignmentGlowPane'
    }).addTo(map);
    /* Solid bright red line on top */
    var bright = L.polyline(latlngs, {
      color: '#ff1744', weight: 4, opacity: 1.0, pane: 'alignmentPane'
    }).addTo(map);
    alignmentLayers.push(glow, bright);
  });
  if (all.length > 0)
    map.flyToBounds(L.latLngBounds(all), {padding: [30, 30], duration: 0.8});
}

function flyToAlignment() {
  if (alignmentLayers.length === 0) return;
  var all = [];
  alignmentLayers.forEach(function(l) {
    try { all = all.concat(l.getLatLngs()); } catch(e) {}
  });
  if (all.length > 0)
    map.flyToBounds(L.latLngBounds(all), {padding: [30, 30], duration: 0.8});
}

function clearAlignment() {
  alignmentLayers.forEach(function(l) { map.removeLayer(l); });
  alignmentLayers = [];
}

function clearAll() {
  showTracks('[]');
  clearAlignment();
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Python ↔ JS bridge
# ---------------------------------------------------------------------------

class MapBridge(QObject):
    bounds_ready = Signal(float, float, float, float)  # s, w, n, e
    ready        = Signal()

    @Slot(float, float, float, float)
    def on_bounds_ready(self, s: float, w: float, n: float, e: float):
        self.bounds_ready.emit(s, w, n, e)

    @Slot()
    def on_ready(self):
        self.ready.emit()


# ---------------------------------------------------------------------------
# MapWidget
# ---------------------------------------------------------------------------

class MapWidget(QWidget):
    """Leaflet map hosted inside a QWebEngineView."""

    bounds_ready = Signal(float, float, float, float)   # s, w, n, e

    def __init__(self, parent=None):
        super().__init__(parent)
        from gui.theme import is_dark_mode
        self._dark = is_dark_mode()
        self._map_ready = False
        self._js_queue: deque[str] = deque()
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = QWebEngineView()
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._bridge  = MapBridge()
        self._channel = QWebChannel()
        self._channel.registerObject("backend", self._bridge)
        self._view.page().setWebChannel(self._channel)

        self._bridge.ready.connect(self._on_map_ready)
        self._bridge.bounds_ready.connect(self.bounds_ready)

        self._view.setHtml(MAP_HTML, QUrl("qrc:///"))
        layout.addWidget(self._view)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_map_ready(self):
        self._map_ready = True
        # Set initial theme first, then flush queued calls
        self._view.page().runJavaScript(
            f"setTheme({'true' if self._dark else 'false'})"
        )
        while self._js_queue:
            self._view.page().runJavaScript(self._js_queue.popleft())

    def _run_js(self, js: str):
        if self._map_ready:
            self._view.page().runJavaScript(js)
        else:
            self._js_queue.append(js)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_theme(self, dark: bool):
        """Switch base tiles and OpenRailwayMap opacity for dark/light mode."""
        self._dark = dark
        self._run_js(f"setTheme({'true' if dark else 'false'})")

    def show_tracks(self, tracks):
        payload = [
            {
                "nodes": [[n[0], n[1]] for n in t.nodes],
                "color": TRACK_COLORS[i % len(TRACK_COLORS)],
                "name":  t.name,
            }
            for i, t in enumerate(tracks)
        ]
        self._run_js(f"showTracks({json.dumps(payload)})")

    def highlight_track(self, idx: int):
        self._run_js(f"highlightTrack({idx})")

    def fly_to_tracks(self):
        self._run_js("flyToTracks()")

    def show_alignment(self, alignments: list):
        """
        alignments: list of node lists.
        Each node list is [[lat, lon], ...] — one per exported alignment/track.
        """
        payload = [{"nodes": nodes} for nodes in alignments]
        self._run_js(f"showAlignment({json.dumps(payload)})")

    def fly_to_alignment(self):
        self._run_js("flyToAlignment()")

    def clear_alignment(self):
        self._run_js("clearAlignment()")

    def request_bounds(self):
        """Ask the map for its current view bounds; result arrives via bounds_ready."""
        self._run_js("getMapBounds()")

    def clear_all(self):
        self._run_js("clearAll()")
