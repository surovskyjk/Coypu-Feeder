"""
Search panel — railway name search, relation ID entry, bbox display.
"""

import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from typing import Callable, Optional


class SearchPanel(ctk.CTkFrame):
    def __init__(self, parent, on_result: Callable):
        super().__init__(parent)
        self._on_result = on_result
        self._results: list[dict] = []
        self._bbox: Optional[tuple] = None
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        row = 0

        ctk.CTkLabel(self, text="Search Railway", font=("Helvetica", 14, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=(10, 4)
        )
        row += 1

        # --- Name search ---
        ctk.CTkLabel(self, text="By name:").grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        self._name_entry = ctk.CTkEntry(self, placeholder_text="e.g. Praha–Brno")
        self._name_entry.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 4))
        self._name_entry.bind("<Return>", lambda e: self._search_by_name())
        row += 1

        self._search_btn = ctk.CTkButton(self, text="Search", command=self._search_by_name)
        self._search_btn.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 8))
        row += 1

        # Results list
        ctk.CTkLabel(self, text="Results:").grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        self._listbox_frame = ctk.CTkScrollableFrame(self, height=130)
        self._listbox_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._listbox_frame.grid_columnconfigure(0, weight=1)
        self._result_buttons: list[ctk.CTkButton] = []
        row += 1

        # Divider
        ctk.CTkLabel(self, text="─" * 30, text_color="gray").grid(
            row=row, column=0, pady=2
        )
        row += 1

        # --- Relation ID ---
        ctk.CTkLabel(self, text="By relation ID:").grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        rel_frame = ctk.CTkFrame(self, fg_color="transparent")
        rel_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 8))
        rel_frame.grid_columnconfigure(0, weight=1)
        self._rel_entry = ctk.CTkEntry(rel_frame, placeholder_text="e.g. 123456")
        self._rel_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._rel_entry.bind("<Return>", lambda e: self._fetch_by_relation())
        ctk.CTkButton(rel_frame, text="Fetch", width=60, command=self._fetch_by_relation).grid(
            row=0, column=1
        )
        row += 1

        # --- BBox display ---
        ctk.CTkLabel(self, text="By bounding box (draw on map):").grid(
            row=row, column=0, sticky="w", padx=10
        )
        row += 1
        self._bbox_label = ctk.CTkLabel(
            self, text="No bbox selected", text_color="gray", font=("Helvetica", 10)
        )
        self._bbox_label.grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        self._bbox_btn = ctk.CTkButton(
            self, text="Fetch from bbox", state="disabled", command=self._fetch_by_bbox
        )
        self._bbox_btn.grid(row=row, column=0, sticky="ew", padx=10, pady=(4, 10))

    # ------------------------------------------------------------------
    # Search actions
    # ------------------------------------------------------------------

    def _search_by_name(self):
        name = self._name_entry.get().strip()
        if not name:
            return
        self._search_btn.configure(state="disabled", text="Searching…")
        self._clear_results()

        def worker():
            try:
                from ..osm.query import search_railways_by_name
                results = search_railways_by_name(name)
                self.after(0, lambda: self._populate_results(results))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Search error", str(exc)))
            finally:
                self.after(0, lambda: self._search_btn.configure(state="normal", text="Search"))

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_by_relation(self):
        rel_text = self._rel_entry.get().strip()
        if not rel_text.isdigit():
            messagebox.showwarning("Invalid ID", "Please enter a numeric OSM relation ID.")
            return
        self._do_fetch(int(rel_text))

    def _fetch_by_bbox(self):
        if not self._bbox:
            return
        south, west, north, east = self._bbox
        self._set_busy(True)

        def worker():
            try:
                from ..osm.query import fetch_bbox_ways
                data = fetch_bbox_ways(south, west, north, east)
                info = {"id": None, "name": "Bbox selection", "network": "", "operator": ""}
                self.after(0, lambda: self._on_result(data, info))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Fetch error", str(exc)))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _do_fetch(self, relation_id: int):
        self._set_busy(True)

        def worker():
            try:
                from ..osm.query import fetch_relation_ways, fetch_relation_metadata
                data = fetch_relation_ways(relation_id)
                info = fetch_relation_metadata(relation_id) or {"id": relation_id, "name": str(relation_id)}
                self.after(0, lambda: self._on_result(data, info))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Fetch error", str(exc)))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------

    def _populate_results(self, results: list[dict]):
        self._results = results
        self._clear_results()
        if not results:
            ctk.CTkLabel(self._listbox_frame, text="No results found.", text_color="gray").grid(
                row=0, column=0, pady=4
            )
            return
        for i, r in enumerate(results):
            label = r["name"] or f"Relation {r['id']}"
            if r.get("from") and r.get("to"):
                label += f"\n{r['from']} → {r['to']}"
            btn = ctk.CTkButton(
                self._listbox_frame,
                text=label,
                anchor="w",
                font=("Helvetica", 11),
                command=lambda rid=r["id"]: self._do_fetch(rid),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=2)
            self._result_buttons.append(btn)

    def _clear_results(self):
        for w in self._listbox_frame.winfo_children():
            w.destroy()
        self._result_buttons.clear()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_bbox(self, bbox: tuple[float, float, float, float]):
        """Called by MapPanel when user finishes drawing a bbox."""
        self._bbox = bbox
        s, w, n, e = bbox
        self._bbox_label.configure(
            text=f"S:{s:.4f} W:{w:.4f}\nN:{n:.4f} E:{e:.4f}",
            text_color="white",
        )
        self._bbox_btn.configure(state="normal")

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self._search_btn.configure(state=state)
        self._bbox_btn.configure(state=state if self._bbox else "disabled")
