"""
Track selection panel — shown after a railway is fetched.
User can choose to export all tracks or select specific ones.
"""

import tkinter as tk
import customtkinter as ctk


class TrackPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self._tracks = []
        self._check_vars: list[tk.BooleanVar] = []
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Track Selection", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4)
        )

        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        mode_frame.grid_columnconfigure(0, weight=1)
        mode_frame.grid_columnconfigure(1, weight=1)

        self._mode_var = tk.StringVar(value="all")
        ctk.CTkRadioButton(
            mode_frame, text="All tracks", variable=self._mode_var,
            value="all", command=self._on_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkRadioButton(
            mode_frame, text="Select tracks", variable=self._mode_var,
            value="select", command=self._on_mode_change,
        ).grid(row=0, column=1, sticky="w")

        self._scroll = ctk.CTkScrollableFrame(self, height=100)
        self._scroll.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._scroll.grid_columnconfigure(0, weight=1)

        self._empty_label = ctk.CTkLabel(
            self._scroll, text="No railway loaded.", text_color="gray", font=("Helvetica", 11)
        )
        self._empty_label.grid(row=0, column=0, pady=4)

    def populate(self, tracks):
        """Populate the track list from parsed Track objects."""
        self._tracks = tracks
        self._check_vars.clear()
        for w in self._scroll.winfo_children():
            w.destroy()

        if not tracks:
            ctk.CTkLabel(
                self._scroll, text="No tracks found.", text_color="gray"
            ).grid(row=0, column=0, pady=4)
            return

        for i, track in enumerate(tracks):
            var = tk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(
                self._scroll,
                text=f"{track.name}  ({len(track.nodes)} nodes)",
                variable=var,
                font=("Helvetica", 11),
            )
            cb.grid(row=i, column=0, sticky="w", pady=1)
            self._check_vars.append(var)

        self._on_mode_change()

    def _on_mode_change(self):
        mode = self._mode_var.get()
        state = "normal" if mode == "select" else "disabled"
        for w in self._scroll.winfo_children():
            if isinstance(w, ctk.CTkCheckBox):
                w.configure(state=state)

    def get_selected_tracks(self, tracks):
        """Return the Track objects selected for export."""
        if self._mode_var.get() == "all":
            return tracks
        return [t for t, var in zip(tracks, self._check_vars) if var.get()]
