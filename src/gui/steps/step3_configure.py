"""
Step 3 — Configure geometry settings.
CRS / output projection is chosen later in Step 6 (Export).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QDoubleSpinBox, QScrollArea, QFormLayout, QGroupBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class Step3Configure(QWidget):
    config_confirmed = Signal(dict)
    back_requested   = Signal()   # user wants to go back to Select Section

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)

        # ── Project ──────────────────────────────────────────────────
        proj_group = QGroupBox("Project")
        pf = QFormLayout(proj_group)
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("Railway Alignment")
        pf.addRow("Project name:", self._project_edit)
        v.addWidget(proj_group)

        # ── Geometry ─────────────────────────────────────────────────
        geo_group = QGroupBox("Geometry Settings")
        gf = QVBoxLayout(geo_group)
        gform = QFormLayout()

        # Smoothing
        smooth_row = QHBoxLayout()
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(5, 51)
        self._smooth_slider.setSingleStep(2)
        self._smooth_slider.setPageStep(2)
        self._smooth_slider.setValue(21)
        self._smooth_lbl = QLabel("21")
        self._smooth_lbl.setFixedWidth(28)
        self._smooth_slider.valueChanged.connect(self._on_smooth)
        smooth_row.addWidget(self._smooth_slider)
        smooth_row.addWidget(self._smooth_lbl)
        gform.addRow("Curvature smooth window:", smooth_row)

        # Sample interval
        self._sample_spin = QDoubleSpinBox()
        self._sample_spin.setRange(1.0, 500.0)
        self._sample_spin.setSingleStep(5.0)
        self._sample_spin.setValue(20.0)
        self._sample_spin.setSuffix(" m")
        gform.addRow("Elevation sample interval:", self._sample_spin)

        # Vertical curve length
        self._vc_spin = QDoubleSpinBox()
        self._vc_spin.setRange(10.0, 2000.0)
        self._vc_spin.setSingleStep(10.0)
        self._vc_spin.setValue(100.0)
        self._vc_spin.setSuffix(" m")
        gform.addRow("Vertical curve length:", self._vc_spin)

        gf.addLayout(gform)

        # Min element lengths (per-type)
        min_group = QGroupBox("Minimum Element Lengths")
        mf = QFormLayout(min_group)

        self._min_line_spin = QDoubleSpinBox()
        self._min_line_spin.setRange(1.0, 500.0)
        self._min_line_spin.setValue(10.0)
        self._min_line_spin.setSuffix(" m")
        mf.addRow("Minimum Line length:", self._min_line_spin)

        self._min_arc_spin = QDoubleSpinBox()
        self._min_arc_spin.setRange(1.0, 500.0)
        self._min_arc_spin.setValue(10.0)
        self._min_arc_spin.setSuffix(" m")
        mf.addRow("Minimum Arc (Curve) length:", self._min_arc_spin)

        self._min_spiral_spin = QDoubleSpinBox()
        self._min_spiral_spin.setRange(1.0, 500.0)
        self._min_spiral_spin.setValue(10.0)
        self._min_spiral_spin.setSuffix(" m")
        mf.addRow("Minimum Spiral length:", self._min_spiral_spin)

        gf.addWidget(min_group)
        v.addWidget(geo_group)

        # ── Alignment accuracy ────────────────────────────────────────
        acc_group = QGroupBox("Alignment Accuracy")
        acc_group.setToolTip(
            "Controls how closely the fitted geometric elements must follow\n"
            "the original OSM polyline. After the initial curvature-based fit,\n"
            "any element whose maximum deviation exceeds the threshold is\n"
            "recursively split and re-fitted."
        )
        af = QFormLayout(acc_group)

        self._max_dev_spin = QDoubleSpinBox()
        self._max_dev_spin.setRange(0.01, 5.0)
        self._max_dev_spin.setSingleStep(0.05)
        self._max_dev_spin.setValue(0.50)
        self._max_dev_spin.setSuffix(" m")
        self._max_dev_spin.setToolTip(
            "Maximum allowed deviation between the fitted alignment element\n"
            "and the original OSM polyline.\n"
            "Smaller values → more elements, higher accuracy.\n"
            "Typical range: 0.05 m (cm accuracy) to 2.0 m (rough fit)."
        )
        af.addRow("Max deviation from OSM line:", self._max_dev_spin)

        self._min_radius_spin = QDoubleSpinBox()
        self._min_radius_spin.setRange(50.0, 10000.0)
        self._min_radius_spin.setSingleStep(25.0)
        self._min_radius_spin.setDecimals(0)
        self._min_radius_spin.setValue(150.0)
        self._min_radius_spin.setSuffix(" m")
        self._min_radius_spin.setToolTip(
            "Minimum horizontal curve radius enforced on all arc elements.\n"
            "Mainline railway: ≥ 300 m  |  Secondary line: ≥ 150 m\n"
            "Tramway / light rail: 50–100 m  |  High-speed rail: ≥ 1500 m"
        )
        af.addRow("Minimum curve radius:", self._min_radius_spin)

        self._check_interval_spin = QDoubleSpinBox()
        self._check_interval_spin.setRange(1.0, 50.0)
        self._check_interval_spin.setSingleStep(1.0)
        self._check_interval_spin.setValue(5.0)
        self._check_interval_spin.setSuffix(" m")
        self._check_interval_spin.setToolTip(
            "Sampling interval along each fitted element for deviation checking.\n"
            "Smaller values catch localised deviations more precisely but are\n"
            "slower. 5 m is a good default for typical railway OSM data."
        )
        af.addRow("Deviation check interval:", self._check_interval_spin)

        v.addWidget(acc_group)

        # ── Candidate Generation ──────────────────────────────────────
        cand_group = QGroupBox("Candidate Generation")
        cf = QFormLayout(cand_group)

        self._merge_pct_spin = QDoubleSpinBox()
        self._merge_pct_spin.setRange(5.0, 40.0)
        self._merge_pct_spin.setSingleStep(5.0)
        self._merge_pct_spin.setDecimals(0)
        self._merge_pct_spin.setValue(15.0)
        self._merge_pct_spin.setSuffix(" %")
        self._merge_pct_spin.setToolTip(
            "Controls how aggressively consecutive arcs are merged.\n"
            "Low (5 %): many small, precise arc segments — Tight candidate.\n"
            "High (40 %): fewer, smoother arcs — Smooth candidate.\n"
            "The Balanced candidate uses this value directly;\n"
            "Tight uses half of it; Smooth uses double."
        )
        cf.addRow("Radius merge tolerance:", self._merge_pct_spin)

        v.addWidget(cand_group)

        v.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Navigation row (outside scroll)
        nav_row = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setFixedWidth(80)
        self._back_btn.clicked.connect(self.back_requested.emit)
        nav_row.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next →  Candidates")
        self._next_btn.setMinimumHeight(38)
        self._next_btn.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._next_btn)
        outer.addLayout(nav_row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_smooth(self, val: int):
        # Force odd
        if val % 2 == 0:
            val += 1
            self._smooth_slider.setValue(val)
        self._smooth_lbl.setText(str(val))

    def _on_next(self):
        self.config_confirmed.emit({
            "project_name":      self._project_edit.text().strip() or "Railway Alignment",
            "smooth_window":     self._smooth_slider.value(),
            "sample_interval":   self._sample_spin.value(),
            "vc_length":         self._vc_spin.value(),
            "min_line_length":   self._min_line_spin.value(),
            "min_arc_length":    self._min_arc_spin.value(),
            "min_spiral_length": self._min_spiral_spin.value(),
            "max_deviation":     self._max_dev_spin.value(),
            "check_interval":    self._check_interval_spin.value(),
            "min_radius":        self._min_radius_spin.value(),
            "merge_radius_pct":  self._merge_pct_spin.value(),
        })
