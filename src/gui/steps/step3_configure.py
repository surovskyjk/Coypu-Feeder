"""
Step 3 — Configure geometry settings.
CRS / output projection is chosen later in Step 6 (Export).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QDoubleSpinBox, QSpinBox, QScrollArea, QFormLayout, QGroupBox,
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

        # ── Export Settings ───────────────────────────────────────────
        exp_group = QGroupBox("Export Settings")
        exp_group.setToolTip(
            "These settings control the LandXML output — elevation sampling\n"
            "density and vertical curve shape.  They do not affect how the\n"
            "horizontal geometry candidates are computed."
        )
        ef = QFormLayout(exp_group)

        self._sample_spin = QDoubleSpinBox()
        self._sample_spin.setRange(1.0, 500.0)
        self._sample_spin.setSingleStep(5.0)
        self._sample_spin.setValue(20.0)
        self._sample_spin.setSuffix(" m")
        self._sample_spin.setToolTip(
            "Distance between elevation sampling points along the alignment.\n"
            "Smaller = more elevation detail in the LandXML output."
        )
        ef.addRow("Elevation sample interval:", self._sample_spin)

        self._vc_spin = QDoubleSpinBox()
        self._vc_spin.setRange(10.0, 2000.0)
        self._vc_spin.setSingleStep(10.0)
        self._vc_spin.setValue(100.0)
        self._vc_spin.setSuffix(" m")
        self._vc_spin.setToolTip(
            "Length of fitted parabolic vertical curves (sags and crests)\n"
            "in the LandXML output.  Mainline: 100–500 m."
        )
        ef.addRow("Vertical curve length:", self._vc_spin)

        v.addWidget(exp_group)

        # ── Alignment Accuracy ────────────────────────────────────────
        acc_group = QGroupBox("Alignment Accuracy")
        acc_group.setToolTip(
            "Controls how closely the fitted geometric elements must follow\n"
            "the original OSM polyline.\n\n"
            "Typical values:\n"
            "  0.10 m — centimetre accuracy (slow, many elements)\n"
            "  0.50 m — balanced (default)\n"
            "  2.00 m — coarse / fast"
        )
        af = QFormLayout(acc_group)

        self._max_dev_spin = QDoubleSpinBox()
        self._max_dev_spin.setRange(0.01, 5.0)
        self._max_dev_spin.setSingleStep(0.05)
        self._max_dev_spin.setValue(0.50)
        self._max_dev_spin.setSuffix(" m")
        self._max_dev_spin.setToolTip(
            "Maximum allowed perpendicular distance between any fitted\n"
            "element and the OSM polyline.\n"
            "All three algorithms use this as their convergence target."
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

        v.addWidget(acc_group)

        # ── Candidate Generation ──────────────────────────────────────
        cand_group = QGroupBox("Candidate Generation")
        cf = QFormLayout(cand_group)

        # Curvature smooth window — Segment & Fit only
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
        smooth_lbl_row = QLabel("Curvature smooth window\n(Segment & Fit):")
        smooth_lbl_row.setToolTip(
            "Savitzky-Golay smoothing window applied to the curvature profile\n"
            "before segmenting into Line / Arc zones.\n"
            "Used by the 'Segment & Fit' algorithm only.\n"
            "Larger = smoother, fewer segments; smaller = more sensitive to noise."
        )
        cf.addRow(smooth_lbl_row, smooth_row)

        self._merge_pct_spin = QDoubleSpinBox()
        self._merge_pct_spin.setRange(5.0, 40.0)
        self._merge_pct_spin.setSingleStep(5.0)
        self._merge_pct_spin.setDecimals(0)
        self._merge_pct_spin.setValue(15.0)
        self._merge_pct_spin.setSuffix(" %")
        self._merge_pct_spin.setToolTip(
            "Controls how aggressively adjacent arc segments are merged.\n"
            "  5 %  → many small, precise arcs\n"
            " 15 %  → balanced (default)\n"
            " 30 %  → fewer, longer arcs\n\n"
            "Also used as the DP regularisation strength and MC perturbation range."
        )
        cf.addRow("Radius merge tolerance:", self._merge_pct_spin)

        self._time_budget_spin = QSpinBox()
        self._time_budget_spin.setRange(10, 300)
        self._time_budget_spin.setSingleStep(10)
        self._time_budget_spin.setValue(60)
        self._time_budget_spin.setSuffix(" s")
        self._time_budget_spin.setToolTip(
            "Maximum wall-clock time allowed per algorithm.\n"
            "Applies to 'DP Segmentation' and 'Progressive MC'.\n"
            "Larger value → more iterations, better accuracy, longer wait.\n"
            "30 s is usually sufficient for sections under 10 km."
        )
        cf.addRow("Max computing time:", self._time_budget_spin)

        self._division_spin = QDoubleSpinBox()
        self._division_spin.setRange(100.0, 2000.0)
        self._division_spin.setSingleStep(100.0)
        self._division_spin.setDecimals(0)
        self._division_spin.setValue(500.0)
        self._division_spin.setSuffix(" m")
        self._division_spin.setToolTip(
            "Length of each MC window (Progressive MC only).\n"
            "The OSM polyline is subdivided into windows of approximately\n"
            "this length; each window is optimised independently with C1\n"
            "continuity enforced at every boundary.\n"
            "Smaller windows = tighter local control, more stitching overhead.\n"
            "Typical range: 300\u20131000 m."
        )
        cf.addRow("MC window length:", self._division_spin)

        self._min_tangent_spin = QDoubleSpinBox()
        self._min_tangent_spin.setRange(0.0, 200.0)
        self._min_tangent_spin.setSingleStep(5.0)
        self._min_tangent_spin.setDecimals(0)
        self._min_tangent_spin.setValue(30.0)
        self._min_tangent_spin.setSuffix(" m")
        self._min_tangent_spin.setToolTip(
            "Minimum length of a tangent Line element between two same-sense Arcs.\n"
            "After Progressive MC finishes, any Line shorter than this that sits\n"
            "between two arcs of the same rotation sense is merged into a single Arc.\n"
            "Set to 0 to disable consolidation."
        )
        cf.addRow("Min tangent length:", self._min_tangent_spin)

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
            "project_name":    self._project_edit.text().strip() or "Railway Alignment",
            "smooth_window":   self._smooth_slider.value(),
            "sample_interval": self._sample_spin.value(),
            "vc_length":       self._vc_spin.value(),
            "max_deviation":   self._max_dev_spin.value(),
            "min_radius":      self._min_radius_spin.value(),
            "merge_radius_pct": self._merge_pct_spin.value(),
            "time_budget_s":    float(self._time_budget_spin.value()),
            "division_length":   self._division_spin.value(),
            "min_tangent_length": self._min_tangent_spin.value(),
            # Keep these with defaults so downstream code that still references
            # them (e.g. old ExportWorker path) does not KeyError.
            "check_interval":  5.0,
            "min_line_length":   10.0,
            "min_arc_length":    10.0,
            "min_spiral_length": 10.0,
        })
