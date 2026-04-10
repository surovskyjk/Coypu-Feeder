"""
Main application window.
Three-column layout: Search | Map | Settings+Export
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from .search_panel import SearchPanel
from .map_panel import MapPanel
from .track_panel import TrackPanel
from .export_panel import ExportPanel

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Coypu-Feeder — OSM Railway to LandXML")
        self.geometry("1280x780")
        self.minsize(1000, 650)

        self._tracks = []          # parsed Track objects
        self._elements = {}        # track name → fitted horizontal elements
        self._vertical = {}        # track name → fitted vertical elements
        self._overpass_data = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=300)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Left column: search + track selection
        left_frame = ctk.CTkFrame(self, width=300)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_frame.grid_propagate(False)
        left_frame.grid_rowconfigure(0, weight=2)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.search_panel = SearchPanel(left_frame, on_result=self._on_railway_selected)
        self.search_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 2))

        self.track_panel = TrackPanel(left_frame)
        self.track_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=(2, 5))

        # Centre column: map
        self.map_panel = MapPanel(self, on_bbox=self._on_bbox_selected)
        self.map_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)

        # Right column: export settings
        self.export_panel = ExportPanel(self, on_export=self._on_export)
        self.export_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)

        # Bottom status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ctk.CTkLabel(
            self, textvariable=self.status_var,
            anchor="w", font=("Helvetica", 11),
        )
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 6))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_railway_selected(self, overpass_data: dict, relation_info: dict):
        """Called when the user selects a railway and data is fetched."""
        self._overpass_data = overpass_data
        self._parse_and_display_tracks(overpass_data)
        self.map_panel.show_relation(overpass_data)

    def _on_bbox_selected(self, bbox: tuple[float, float, float, float]):
        """Called when the user finishes drawing a bounding box on the map."""
        self.search_panel.set_bbox(bbox)

    def _on_export(self, settings: dict):
        """Called when the user clicks Export."""
        if not self._tracks:
            messagebox.showwarning("No data", "Please search and select a railway first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("LandXML files", "*.xml"), ("All files", "*.*")],
            title="Save LandXML file",
        )
        if not filepath:
            return

        selected_tracks = self.track_panel.get_selected_tracks(self._tracks)
        if not selected_tracks:
            messagebox.showwarning("No tracks", "Please select at least one track to export.")
            return

        self._run_export_thread(selected_tracks, settings, filepath)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _parse_and_display_tracks(self, overpass_data: dict):
        from ..osm.parser import parse_tracks
        self._tracks = parse_tracks(overpass_data)
        self.track_panel.populate(self._tracks)
        n = len(self._tracks)
        self._set_status(f"Found {n} track{'s' if n != 1 else ''}. Select tracks and configure export.")

    def _run_export_thread(self, selected_tracks, settings: dict, filepath: str):
        self.export_panel.set_busy(True)
        self._set_status("Processing — querying elevation and fitting geometry…")

        def worker():
            try:
                self._process_and_export(selected_tracks, settings, filepath)
                self.after(0, lambda: self._set_status(f"Export complete: {filepath}"))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Export failed", str(exc)))
                self.after(0, lambda: self._set_status("Export failed."))
            finally:
                self.after(0, lambda: self.export_panel.set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _process_and_export(self, selected_tracks, settings: dict, filepath: str):
        import numpy as np
        from ..geometry.projection import wgs84_to_projected, auto_utm_epsg
        from ..geometry.alignment import fit_alignment
        from ..geometry.elevation import (
            interpolate_along_alignment, sample_elevations, fit_vertical_geometry,
        )
        from ..geometry.curvature import compute_chainages
        from ..landxml.builder import build_landxml, write_landxml

        epsg = settings["epsg"]
        smooth_window = settings["smooth_window"]
        sample_interval = settings["sample_interval"]
        vc_length = settings["vc_length"]
        project_name = settings.get("project_name", "Railway Alignment")

        alignments = []

        for track in selected_tracks:
            latlon = track.nodes
            if epsg == -1:
                work_epsg = auto_utm_epsg(latlon)
            else:
                work_epsg = epsg

            xy = np.array(wgs84_to_projected(latlon, work_epsg))
            chainages = compute_chainages(xy)

            # Horizontal fitting
            h_elements = fit_alignment(
                xy,
                smooth_window=smooth_window,
                min_element_length=settings.get("min_element_length", 10.0),
            )

            # Elevation sampling
            self.after(0, lambda: self._set_status(f"Querying DEM for {track.name}…"))
            sample_ch, sample_latlon = interpolate_along_alignment(
                latlon, chainages, sample_interval=sample_interval
            )
            elevs = sample_elevations(sample_latlon)

            # Vertical fitting
            v_elements = fit_vertical_geometry(
                sample_ch, elevs, vc_length=vc_length
            )

            alignments.append({
                "name": track.name,
                "elements": h_elements,
                "vertical": v_elements,
                "sta_start": 0.0,
            })

        root = build_landxml(alignments, output_epsg=epsg if epsg != -1 else work_epsg,
                             project_name=project_name)
        write_landxml(root, filepath)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self.status_var.set(msg)
