"""
Map widget: QWebEngineView hosting a Leaflet.js map.
Python ↔ JavaScript via QWebChannel.

The HTML page is served from a local http://127.0.0.1 server so the page
has a real HTTP origin — this is the only reliable way to allow Chromium
(QtWebEngine) to fetch remote tile URLs without CORS/mixed-content blocks.

Leaflet JS + CSS are served from the same local server (src/gui/static/).
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import socket
import threading
from collections import deque

from PySide6.QtCore import QObject, Signal, Slot, QUrl, QTimer
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox, QSizePolicy,
)

# ---------------------------------------------------------------------------
# Tile providers
# ---------------------------------------------------------------------------

TILE_PROVIDERS = [
    {
        "label":  "OpenStreetMap",
        "url":    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "subs":   "",
        "attr":   "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
        "maxZoom": 19,
    },
    {
        "label":  "CARTO Dark Matter",
        "url":    "https://{s}.basemaps.cartocdn.com/dark_matter/{z}/{x}/{y}.png",
        "subs":   "abcd",
        "attr":   "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> © <a href='https://carto.com/attributions'>CARTO</a>",
        "maxZoom": 20,
    },
    {
        "label":  "CARTO Voyager",
        "url":    "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "subs":   "abcd",
        "attr":   "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> © <a href='https://carto.com/attributions'>CARTO</a>",
        "maxZoom": 20,
    },
    {
        "label":  "OpenTopoMap",
        "url":    "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "subs":   "abc",
        "attr":   "© <a href='https://opentopomap.org'>OpenTopoMap</a> © OpenStreetMap contributors",
        "maxZoom": 17,
    },
    {
        "label":  "Esri Satellite",
        "url":    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "subs":   "",
        "attr":   "Tiles © Esri — Source: Esri, DigitalGlobe, GeoEye, Earthstar Geographics",
        "maxZoom": 19,
    },
    {
        "label":  "OpenRailwayMap (railway only)",
        "url":    "https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png",
        "subs":   "abc",
        "attr":   "© <a href='https://www.openrailwaymap.org/'>OpenRailwayMap</a> © OpenStreetMap contributors",
        "maxZoom": 19,
    },
]

# Default provider index
_DEFAULT_PROVIDER = 0   # OpenStreetMap

# Track colours
TRACK_COLORS = [
    "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
    "#ce93d8", "#80cbc4", "#fff176", "#ff8a65",
]

# ---------------------------------------------------------------------------
# Inline map HTML  (served as map.html from the local HTTP server)
# ---------------------------------------------------------------------------

MAP_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body, #map { width:100%; height:100%; margin:0; padding:0; background:#1a1a1a; }
</style>
<link rel="stylesheet" href="leaflet.css"/>
<script src="leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head>
<body>
<div id="map"></div>
<script>
"use strict";
var map = L.map('map', {zoomControl: true}).setView([50.05, 14.42], 7);

/* Custom panes for z-ordering */
map.createPane('tracksPane').style.zIndex      = '410';
map.createPane('osmRefPane').style.zIndex      = '630';
map.createPane('candidatesPane').style.zIndex  = '620';
map.createPane('alignmentGlowPane').style.zIndex = '640';
map.createPane('alignmentPane').style.zIndex   = '650';
map.createPane('crossSectionPane').style.zIndex = '625';

var baseLayer       = null;
var railOverlay     = null;
var trackLayers     = [];
var osmRefLayers    = [];
var alignmentLayers = [];
var candidateLayers = [];
var csLeftLayer     = null;
var csRightLayer    = null;
var backend         = null;
var _showRailOverlay = true;

/* ── Tile provider ─────────────────────────────────────────────── */
function setTileProvider(url, subs, attr, maxZoom) {
  if (baseLayer) { map.removeLayer(baseLayer); baseLayer = null; }
  var opts = { attribution: attr, maxZoom: maxZoom || 19 };
  if (subs) opts.subdomains = subs;
  baseLayer = L.tileLayer(url, opts).addTo(map);
  /* Keep rail overlay on top if active */
  if (railOverlay) railOverlay.bringToFront();
}

/* ── OpenRailwayMap overlay ────────────────────────────────────── */
function setRailOverlay(show) {
  _showRailOverlay = show;
  if (show) {
    if (!railOverlay) {
      railOverlay = L.tileLayer(
        'https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
        { subdomains:'abc', maxZoom:19, opacity:0.8,
          attribution:'© <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a>' }
      ).addTo(map);
    }
  } else {
    if (railOverlay) { map.removeLayer(railOverlay); railOverlay = null; }
  }
}

/* Load default tiles immediately */
setTileProvider(
  'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  '', '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', 19
);
setRailOverlay(true);

/* ── QWebChannel bootstrap ──────────────────────────────────────  */
new QWebChannel(qt.webChannelTransport, function(channel) {
  backend = channel.objects.backend;
  backend.on_ready();
});

/* ── Map bounds ─────────────────────────────────────────────────  */
function getMapBounds() {
  var b = map.getBounds();
  backend.on_bounds_ready(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
}

/* ── Track display ──────────────────────────────────────────────  */
function showTracks(tracks) {
  trackLayers.forEach(function(l) { map.removeLayer(l); });
  trackLayers = [];
  var all = [];
  tracks.forEach(function(t) {
    var color   = t.color || '#4fc3f7';
    var latlngs = t.nodes.map(function(n) { return [n[0], n[1]]; });
    if (!latlngs.length) return;
    all = all.concat(latlngs);
    var pl = L.polyline(latlngs, {color:color, weight:4, opacity:0.9, pane:'tracksPane'});
    pl._baseColor = color;
    pl.addTo(map);
    trackLayers.push(pl);
  });
  if (all.length)
    try { map.fitBounds(L.latLngBounds(all), {padding:[20,20], maxZoom:14}); } catch(e){}
}

function highlightTrack(idx) {
  trackLayers.forEach(function(l, i) {
    var base = l._baseColor || '#4fc3f7';
    if (idx < 0)       l.setStyle({color:base,    weight:4, opacity:0.9});
    else if (i===idx) { l.setStyle({color:'#ffeb3b', weight:7, opacity:1.0}); l.bringToFront(); }
    else               l.setStyle({color:base,    weight:2, opacity:0.25});
  });
}

function flyToTracks() {
  var all = [];
  trackLayers.forEach(function(l){ all = all.concat(l.getLatLngs()); });
  if (all.length)
    try { map.flyToBounds(L.latLngBounds(all), {padding:[30,30], duration:0.8, maxZoom:14}); } catch(e){}
}

/* ── OSM reference overlay (dashed cyan) ────────────────────────  */
function showOSMReference(alns) {
  clearOSMReference();
  alns.forEach(function(aln) {
    var latlngs = aln.nodes.map(function(n){ return [n[0],n[1]]; });
    if (latlngs.length < 2) return;
    var pl = L.polyline(latlngs, {
      color: '#00e5ff', weight: 3, opacity: 0.85,
      dashArray: '8 6', pane: 'osmRefPane'
    }).addTo(map);
    osmRefLayers.push(pl);
  });
}

function clearOSMReference() {
  osmRefLayers.forEach(function(l){ try{ map.removeLayer(l); }catch(e){} });
  osmRefLayers = [];
}

/* ── Fitted alignment overlay (solid red) ───────────────────────  */
function showAlignment(alns) {
  clearAlignment();
  var all = [];
  alns.forEach(function(aln) {
    var latlngs = aln.nodes.map(function(n){ return [n[0],n[1]]; });
    if (latlngs.length < 2) return;
    all = all.concat(latlngs);
    L.polyline(latlngs, {color:'#ffffff', weight:14, opacity:0.2, pane:'alignmentGlowPane'}).addTo(map);
    var bright = L.polyline(latlngs, {color:'#ff1744', weight:5, opacity:1.0, pane:'alignmentPane'}).addTo(map);
    alignmentLayers.push(bright);
  });
  if (all.length)
    try { map.flyToBounds(L.latLngBounds(all), {padding:[30,30], duration:0.8}); } catch(e){}
}

function flyToAlignment() {
  var all = [];
  alignmentLayers.forEach(function(l){ try{ all=all.concat(l.getLatLngs()); }catch(e){} });
  if (all.length)
    try { map.flyToBounds(L.latLngBounds(all), {padding:[30,30], duration:0.8}); } catch(e){}
}

function clearAlignment() {
  alignmentLayers.forEach(function(l){ try{ map.removeLayer(l); }catch(e){} });
  alignmentLayers = [];
}

/* ── Candidate overlays (one per algorithm) ──────────────────── */
function showCandidates(candidates) {
  clearCandidates();
  candidates.forEach(function(c) {
    var latlngs = c.nodes.map(function(n){ return [n[0], n[1]]; });
    if (latlngs.length < 2) return;
    /* Glow */
    L.polyline(latlngs, {color:'#ffffff', weight:10, opacity:0.15,
                          pane:'candidatesPane'}).addTo(map);
    /* Main line */
    var pl = L.polyline(latlngs, {
      color:   c.color || '#ffffff',
      weight:  4,
      opacity: 0.9,
      pane:    'candidatesPane'
    }).addTo(map);
    pl.bindTooltip(c.label || '', {sticky: true});
    candidateLayers.push(pl);
  });
}

function clearCandidates() {
  candidateLayers.forEach(function(l){
    try{ map.removeLayer(l); }catch(e){}
  });
  candidateLayers = [];
}

/* ── Cross-section overlays (left / right, colour-coded) ──────── */
function showCrossSection(leftPts, rightPts) {
  clearCrossSection();
  function buildLayer(pts) {
    var layer = L.layerGroup();
    for (var i = 0; i < pts.length - 1; i++) {
      L.polyline(
        [[pts[i][0], pts[i][1]], [pts[i+1][0], pts[i+1][1]]],
        {color: pts[i][2], weight: 4, opacity: 0.85, pane: 'crossSectionPane'}
      ).addTo(layer);
    }
    return layer;
  }
  if (leftPts  && leftPts.length  > 0) csLeftLayer  = buildLayer(leftPts).addTo(map);
  if (rightPts && rightPts.length > 0) csRightLayer = buildLayer(rightPts).addTo(map);
}

function clearCrossSection() {
  if (csLeftLayer)  { try { map.removeLayer(csLeftLayer);  } catch(e) {} csLeftLayer  = null; }
  if (csRightLayer) { try { map.removeLayer(csRightLayer); } catch(e) {} csRightLayer = null; }
}

function clearAll() {
  showTracks([]);
  clearOSMReference();
  clearAlignment();
  clearCandidates();
  clearCrossSection();
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Local HTTP server (serves static/ directory so tiles can load freely)
# ---------------------------------------------------------------------------

class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that suppresses access log output."""
    def log_message(self, fmt, *args):
        pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_static_server(static_dir: str) -> int:
    """Serve *static_dir* over HTTP on a random local port. Returns the port."""
    port = _find_free_port()
    handler = functools.partial(_SilentHandler, directory=static_dir)
    server  = http.server.HTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port


# ---------------------------------------------------------------------------
# Debug page
# ---------------------------------------------------------------------------

class DebugPage(QWebEnginePage):
    js_error = Signal(str)

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):
        ERROR = QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel
        WARN  = QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel
        INFO  = QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel
        names = {ERROR: "ERROR", WARN: "WARN", INFO: "INFO"}
        print(f"[MapJS {names.get(level,'LOG')}] {message}  (line {line_number})")
        if level == ERROR:
            self.js_error.emit(f"Map JS error: {message}")


# ---------------------------------------------------------------------------
# Python ↔ JS bridge
# ---------------------------------------------------------------------------

class MapBridge(QObject):
    bounds_ready = Signal(float, float, float, float)
    ready        = Signal()

    @Slot(float, float, float, float)
    def on_bounds_ready(self, s, w, n, e):
        self.bounds_ready.emit(s, w, n, e)

    @Slot()
    def on_ready(self):
        self.ready.emit()


# ---------------------------------------------------------------------------
# MapWidget
# ---------------------------------------------------------------------------

class MapWidget(QWidget):
    bounds_ready = Signal(float, float, float, float)
    js_error     = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from gui.theme import is_dark_mode
        self._dark = is_dark_mode()
        self._map_ready  = False
        self._js_queue: deque[str] = deque()
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(6)

        toolbar.addWidget(QLabel("Map:"))
        self._provider_combo = QComboBox()
        for p in TILE_PROVIDERS:
            self._provider_combo.addItem(p["label"])
        self._provider_combo.setCurrentIndex(_DEFAULT_PROVIDER)
        self._provider_combo.setToolTip("Choose base map tile provider")
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self._provider_combo.setFixedWidth(200)
        toolbar.addWidget(self._provider_combo)

        self._rail_chk = QCheckBox("🚂 Railway overlay")
        self._rail_chk.setChecked(True)
        self._rail_chk.setToolTip("Show OpenRailwayMap overlay on top of base tiles")
        self._rail_chk.toggled.connect(self._on_rail_overlay_toggled)
        toolbar.addWidget(self._rail_chk)

        toolbar.addStretch()

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        toolbar_widget.setFixedHeight(30)
        layout.addWidget(toolbar_widget)

        # ── Web view ──────────────────────────────────────────────────
        page = DebugPage()
        page.js_error.connect(self.js_error)

        self._view = QWebEngineView()
        self._view.setPage(page)
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._bridge  = MapBridge()
        self._channel = QWebChannel()
        self._channel.registerObject("backend", self._bridge)
        self._view.page().setWebChannel(self._channel)

        self._bridge.ready.connect(self._on_map_ready)
        self._bridge.bounds_ready.connect(self.bounds_ready)

        # Write map.html into static/ and serve the whole directory locally.
        # Using http://127.0.0.1 as origin removes ALL cross-origin tile blocks.
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        html_path  = os.path.join(static_dir, "map.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(MAP_HTML)

        self._server_port = _start_static_server(static_dir)
        self._view.setUrl(QUrl(f"http://127.0.0.1:{self._server_port}/map.html"))

        layout.addWidget(self._view)

        # Safety fallback: if QWebChannel never fires (slow startup), unblock after 12 s
        self._ready_timer = QTimer(self)
        self._ready_timer.setSingleShot(True)
        self._ready_timer.timeout.connect(self._on_ready_timeout)
        self._ready_timer.start(12_000)

    # ------------------------------------------------------------------
    # Ready / queue
    # ------------------------------------------------------------------

    def _on_map_ready(self):
        self._ready_timer.stop()
        if self._map_ready:
            return
        self._map_ready = True
        # Apply the correct theme-based provider
        self._apply_current_provider()
        while self._js_queue:
            self._view.page().runJavaScript(self._js_queue.popleft())

    def _on_ready_timeout(self):
        if not self._map_ready:
            print("[MapWidget] QWebChannel timeout — forcing ready.")
            self._map_ready = True
            self._apply_current_provider()
            while self._js_queue:
                self._view.page().runJavaScript(self._js_queue.popleft())

    def _run_js(self, js: str):
        if self._map_ready:
            self._view.page().runJavaScript(js)
        else:
            self._js_queue.append(js)

    # ------------------------------------------------------------------
    # Tile provider helpers
    # ------------------------------------------------------------------

    def _apply_current_provider(self):
        idx = self._provider_combo.currentIndex()
        p = TILE_PROVIDERS[idx]
        self._run_js(
            f"setTileProvider({json.dumps(p['url'])}, {json.dumps(p['subs'])}, "
            f"{json.dumps(p['attr'])}, {p['maxZoom']})"
        )
        show_rail = self._rail_chk.isChecked()
        self._run_js(f"setRailOverlay({'true' if show_rail else 'false'})")

    def _on_provider_changed(self, idx: int):
        p = TILE_PROVIDERS[idx]
        self._run_js(
            f"setTileProvider({json.dumps(p['url'])}, {json.dumps(p['subs'])}, "
            f"{json.dumps(p['attr'])}, {p['maxZoom']})"
        )

    def _on_rail_overlay_toggled(self, checked: bool):
        self._run_js(f"setRailOverlay({'true' if checked else 'false'})")

    # ------------------------------------------------------------------
    # Public API (called from App)
    # ------------------------------------------------------------------

    def set_theme(self, dark: bool):
        self._dark = dark
        # Auto-switch to a sensible provider for the new theme
        target = "CARTO Dark Matter" if dark else "CARTO Voyager"
        for i, p in enumerate(TILE_PROVIDERS):
            if p["label"] == target:
                self._provider_combo.setCurrentIndex(i)
                break

    def show_tracks(self, tracks):
        payload = [
            {"nodes": [[n[0], n[1]] for n in t.nodes],
             "color": TRACK_COLORS[i % len(TRACK_COLORS)],
             "name":  t.name}
            for i, t in enumerate(tracks)
        ]
        self._run_js(f"showTracks({json.dumps(payload)})")

    def highlight_track(self, idx: int):
        self._run_js(f"highlightTrack({idx})")

    def fly_to_tracks(self):
        self._run_js("flyToTracks()")

    def show_osm_reference(self, alignments: list):
        """Draw the raw OSM polyline as a dashed cyan reference overlay."""
        payload = [{"nodes": nodes} for nodes in alignments]
        self._run_js(f"showOSMReference({json.dumps(payload)})")

    def clear_osm_reference(self):
        self._run_js("clearOSMReference()")

    def show_alignment(self, alignments: list):
        payload = [{"nodes": nodes} for nodes in alignments]
        self._run_js(f"showAlignment({json.dumps(payload)})")

    def show_alignment_segmented(self, segments: list):
        """
        Draw the alignment as a sequence of per-element coloured polylines.

        Parameters
        ----------
        segments : list of dicts, each shaped:
            {"type":   "Line" | "Arc" | "Spiral",
             "params": {length: …, sta_start: …, radius: …, ...},
             "points": [[lat, lon], [lat, lon], ...]}

        The JS side picks a colour by element type (Line=blue, Arc=red,
        Spiral=green) and binds a sticky tooltip listing the element's
        parameters.
        """
        self._run_js(f"showAlignmentSegmented({json.dumps(segments)})")

    def fly_to_alignment(self):
        self._run_js("flyToAlignment()")

    def clear_alignment(self):
        self._run_js("clearAlignment()")

    def request_bounds(self):
        self._run_js("getMapBounds()")

    def show_candidates(self, candidates: list):
        """
        Show multiple candidate alignments as coloured polylines.

        Parameters
        ----------
        candidates : list of dicts with keys: nodes [[lat,lon],...], color, label
        """
        self._run_js(f"showCandidates({json.dumps(candidates)})")

    def clear_candidates(self):
        """Remove all candidate alignment overlays."""
        self._run_js("clearCandidates()")

    def show_cross_section(self, left_pts: list, right_pts: list):
        """
        Draw colour-coded cross-section overlays.

        Parameters
        ----------
        left_pts / right_pts : list of [lat, lon, color_hex]
            One entry per sampled station; consecutive entries are connected
            by polyline segments coloured by the first entry's colour.
        """
        self._run_js(
            f"showCrossSection({json.dumps(left_pts)}, {json.dumps(right_pts)})"
        )

    def clear_cross_section(self):
        """Remove cross-section overlays."""
        self._run_js("clearCrossSection()")

    def clear_all(self):
        self._run_js("clearAll()")   # clears tracks + osmRef + alignment + candidates + cross-section
