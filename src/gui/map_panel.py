"""
Map panel — displays the railway route and allows bounding box drawing.
Uses tkintermapview with OpenStreetMap tiles.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional
import customtkinter as ctk

try:
    from tkintermapview import TkinterMapView
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


class MapPanel(ctk.CTkFrame):
    def __init__(self, parent, on_bbox: Callable):
        super().__init__(parent)
        self._on_bbox = on_bbox
        self._bbox_start: Optional[tuple] = None
        self._bbox_rect = None
        self._bbox_mode = False
        self._path_widget = None
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        toolbar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(toolbar, text="Map", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=5
        )

        self._bbox_btn = ctk.CTkButton(
            toolbar, text="Draw BBox", width=100, command=self._toggle_bbox_mode
        )
        self._bbox_btn.grid(row=0, column=2, padx=5)

        self._clear_btn = ctk.CTkButton(
            toolbar, text="Clear", width=60, command=self._clear_bbox, fg_color="gray"
        )
        self._clear_btn.grid(row=0, column=3, padx=(0, 5))

        if MAP_AVAILABLE:
            self._map = TkinterMapView(self, corner_radius=6)
            self._map.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
            self._map.set_position(50.0, 14.5)  # Default: central Europe
            self._map.set_zoom(6)
            self._map.add_right_click_menu_command(
                label="Set bbox corner here",
                command=self._map_right_click,
                pass_coords=True,
            )
        else:
            ctk.CTkLabel(
                self,
                text="Map unavailable.\nInstall tkintermapview to enable.",
                text_color="gray",
            ).grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # ------------------------------------------------------------------
    # Bbox drawing
    # ------------------------------------------------------------------

    def _toggle_bbox_mode(self):
        self._bbox_mode = not self._bbox_mode
        if self._bbox_mode:
            self._bbox_btn.configure(text="Cancel BBox", fg_color="orange")
            self._bbox_start = None
            if MAP_AVAILABLE:
                self._map.add_left_click_map_command(self._map_click)
        else:
            self._bbox_btn.configure(text="Draw BBox", fg_color=["#3B8ED0", "#1F6AA5"])
            if MAP_AVAILABLE:
                self._map.add_left_click_map_command(None)

    def _map_click(self, coords):
        """Handle left-click for bbox drawing."""
        if not self._bbox_mode:
            return
        lat, lon = coords
        if self._bbox_start is None:
            self._bbox_start = (lat, lon)
        else:
            lat0, lon0 = self._bbox_start
            south = min(lat0, lat)
            north = max(lat0, lat)
            west = min(lon0, lon)
            east = max(lon0, lon)
            self._bbox_start = None
            self._bbox_mode = False
            self._bbox_btn.configure(text="Draw BBox", fg_color=["#3B8ED0", "#1F6AA5"])
            if MAP_AVAILABLE:
                self._map.add_left_click_map_command(None)
            self._on_bbox((south, west, north, east))

    def _map_right_click(self, coords):
        self._map_click(coords)

    def _clear_bbox(self):
        self._bbox_start = None
        if MAP_AVAILABLE:
            self._map.add_left_click_map_command(None)
        self._bbox_mode = False
        self._bbox_btn.configure(text="Draw BBox", fg_color=["#3B8ED0", "#1F6AA5"])

    # ------------------------------------------------------------------
    # Route display
    # ------------------------------------------------------------------

    def show_relation(self, overpass_data: dict):
        """Draw the railway track on the map."""
        if not MAP_AVAILABLE:
            return

        if self._path_widget:
            try:
                self._path_widget.delete()
            except Exception:
                pass
            self._path_widget = None

        node_index = {
            el["id"]: (el["lat"], el["lon"])
            for el in overpass_data.get("elements", [])
            if el["type"] == "node"
        }

        all_coords = []
        for el in overpass_data.get("elements", []):
            if el["type"] != "way":
                continue
            coords = [node_index[n] for n in el.get("nodes", []) if n in node_index]
            if len(coords) >= 2:
                all_coords.extend(coords)

        if not all_coords:
            return

        # Draw path
        try:
            self._path_widget = self._map.set_path(all_coords, color="red", width=3)
        except Exception:
            pass

        # Fit view to track
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        centre_lat = (min(lats) + max(lats)) / 2
        centre_lon = (min(lons) + max(lons)) / 2
        self._map.set_position(centre_lat, centre_lon)

        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        span = max(lat_span, lon_span)
        zoom = max(5, min(14, int(8 - span * 2)))
        self._map.set_zoom(zoom)
