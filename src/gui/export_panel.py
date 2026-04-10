"""
Export settings panel — CRS selection, geometry precision, export button.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Callable
import customtkinter as ctk
from ..geometry.projection import CRS_PRESETS


class ExportPanel(ctk.CTkFrame):
    def __init__(self, parent, on_export: Callable):
        super().__init__(parent)
        self._on_export = on_export
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        row = 0

        ctk.CTkLabel(self, text="Export Settings", font=("Helvetica", 14, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=(10, 8)
        )
        row += 1

        # --- Project name ---
        ctk.CTkLabel(self, text="Project name:").grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        self._project_name = ctk.CTkEntry(self, placeholder_text="Railway Alignment")
        self._project_name.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 10))
        row += 1

        # --- CRS ---
        ctk.CTkLabel(self, text="Output CRS:").grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        preset_labels = [label for label, _ in CRS_PRESETS]
        self._crs_var = tk.StringVar(value=preset_labels[0])
        self._crs_menu = ctk.CTkOptionMenu(
            self, values=preset_labels, variable=self._crs_var,
            command=self._on_crs_change, dynamic_resizing=False,
        )
        self._crs_menu.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 4))
        row += 1

        ctk.CTkLabel(self, text="Custom EPSG (overrides above):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._custom_epsg = ctk.CTkEntry(self, placeholder_text="e.g. 5514")
        self._custom_epsg.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 10))
        row += 1

        # --- Geometry settings ---
        ctk.CTkLabel(self, text="─" * 30, text_color="gray").grid(row=row, column=0, pady=2)
        row += 1

        ctk.CTkLabel(self, text="Geometry Settings", font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=(6, 4)
        )
        row += 1

        # Smoothing window
        ctk.CTkLabel(self, text="Curvature smoothing window (pts):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._smooth_var = tk.IntVar(value=21)
        smooth_frame = ctk.CTkFrame(self, fg_color="transparent")
        smooth_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 6))
        smooth_frame.grid_columnconfigure(0, weight=1)
        self._smooth_slider = ctk.CTkSlider(
            smooth_frame, from_=5, to=51, number_of_steps=23,
            variable=self._smooth_var, command=self._on_smooth_change,
        )
        self._smooth_slider.grid(row=0, column=0, sticky="ew")
        self._smooth_label = ctk.CTkLabel(smooth_frame, text="21", width=30)
        self._smooth_label.grid(row=0, column=1, padx=(4, 0))
        row += 1

        # DEM sample interval
        ctk.CTkLabel(self, text="Elevation sample interval (m):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._sample_entry = ctk.CTkEntry(self, placeholder_text="20")
        self._sample_entry.insert(0, "20")
        self._sample_entry.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 6))
        row += 1

        # Min element length
        ctk.CTkLabel(self, text="Min element length (m):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._min_len_entry = ctk.CTkEntry(self, placeholder_text="10")
        self._min_len_entry.insert(0, "10")
        self._min_len_entry.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 6))
        row += 1

        # Vertical curve length
        ctk.CTkLabel(self, text="Vertical curve length (m):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._vc_entry = ctk.CTkEntry(self, placeholder_text="100")
        self._vc_entry.insert(0, "100")
        self._vc_entry.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 10))
        row += 1

        # --- Export button ---
        ctk.CTkLabel(self, text="─" * 30, text_color="gray").grid(row=row, column=0, pady=2)
        row += 1

        self._export_btn = ctk.CTkButton(
            self, text="Export LandXML", height=40,
            font=("Helvetica", 14, "bold"),
            command=self._do_export,
        )
        self._export_btn.grid(row=row, column=0, sticky="ew", padx=10, pady=(8, 10))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_crs_change(self, value: str):
        # Clear custom entry if a preset is chosen
        self._custom_epsg.delete(0, tk.END)

    def _on_smooth_change(self, value):
        v = int(float(value))
        if v % 2 == 0:
            v += 1
        self._smooth_var.set(v)
        self._smooth_label.configure(text=str(v))

    def _do_export(self):
        settings = self._collect_settings()
        if settings is None:
            return
        self._on_export(settings)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self._export_btn.configure(
            state=state,
            text="Exporting…" if busy else "Export LandXML",
        )

    def _collect_settings(self) -> dict | None:
        # Resolve EPSG
        custom = self._custom_epsg.get().strip()
        if custom:
            if not custom.isdigit():
                messagebox.showwarning("Invalid EPSG", "Custom EPSG must be a number.")
                return None
            epsg = int(custom)
        else:
            label = self._crs_var.get()
            epsg = next((code for lbl, code in CRS_PRESETS if lbl == label), 4326)

        # Parse numeric fields
        def _float(entry, default):
            try:
                return float(entry.get().strip() or default)
            except ValueError:
                return default

        project_name = self._project_name.get().strip() or "Railway Alignment"

        return {
            "epsg": epsg,
            "project_name": project_name,
            "smooth_window": int(self._smooth_var.get()),
            "sample_interval": _float(self._sample_entry, 20.0),
            "min_element_length": _float(self._min_len_entry, 10.0),
            "vc_length": _float(self._vc_entry, 100.0),
        }
