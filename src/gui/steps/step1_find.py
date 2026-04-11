"""
Step 1 — Find Railway.

Two tabs:
  • Search       — ref / name / number-in-name / direct relation ID
  • Lines in View — search for all railway lines visible in the current map view
                   (available only when the view is ≤ ~20 km wide/tall)
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem, QTabWidget,
    QFrame, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from gui.worker import SearchWorker, FetchWorker


class Step1Find(QWidget):
    railway_fetched          = Signal(object, dict)   # (overpass_data, relation_info)
    search_in_view_requested = Signal()               # → App requests map bounds

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers: list = []
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._tabs = QTabWidget()
        self._tab_search = self._tabs.addTab(self._build_search_tab(), "Search")
        self._tab_view   = self._tabs.addTab(self._build_view_tab(),   "Lines in View")
        layout.addWidget(self._tabs)

    # ── Search tab ────────────────────────────────────────────────────

    def _build_search_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        mode_lbl = QLabel("Search by:")
        mode_lbl.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        v.addWidget(mode_lbl)

        self._mode_group = QButtonGroup(self)
        self._radio_ref  = QRadioButton("Timetable line number  (ref tag)")
        self._radio_name = QRadioButton("Name")
        self._radio_num  = QRadioButton("Number in relation name")
        self._radio_ref.setChecked(True)
        for i, r in enumerate([self._radio_ref, self._radio_name, self._radio_num]):
            self._mode_group.addButton(r, i)
            v.addWidget(r)

        self._num_hint = QLabel(
            "Searches the number inside the name field.\n"
            "e.g. '212' → '212 – Čerčany – Světlá nad Sázavou'"
        )
        self._num_hint.setStyleSheet("color:#888; font-size:10px;")
        self._num_hint.setVisible(False)
        v.addWidget(self._num_hint)
        self._radio_num.toggled.connect(self._num_hint.setVisible)

        # Query row
        row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Enter search term…")
        self._search_edit.returnPressed.connect(self._do_search)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._do_search)
        row.addWidget(self._search_edit)
        row.addWidget(self._search_btn)
        v.addLayout(row)

        # Results
        self._results_list = QListWidget()
        self._results_list.setAlternatingRowColors(True)
        self._results_list.itemDoubleClicked.connect(self._on_result_double_clicked)
        v.addWidget(self._results_list, stretch=1)

        fetch_row = QHBoxLayout()
        self._fetch_btn = QPushButton("Fetch selected")
        self._fetch_btn.clicked.connect(self._fetch_selected)
        fetch_row.addStretch()
        fetch_row.addWidget(self._fetch_btn)
        v.addLayout(fetch_row)

        # Divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;")
        v.addWidget(sep)

        # Direct relation ID
        rel_lbl = QLabel("By relation ID:")
        rel_lbl.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        v.addWidget(rel_lbl)

        rel_row = QHBoxLayout()
        self._rel_edit = QLineEdit()
        self._rel_edit.setPlaceholderText("e.g. 3128446")
        self._rel_edit.returnPressed.connect(self._fetch_by_relation)
        self._rel_fetch_btn = QPushButton("Fetch")
        self._rel_fetch_btn.clicked.connect(self._fetch_by_relation)
        rel_row.addWidget(self._rel_edit)
        rel_row.addWidget(self._rel_fetch_btn)
        v.addLayout(rel_row)

        return w

    # ── Lines in View tab ─────────────────────────────────────────────

    def _build_view_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        v.addWidget(QLabel("Find all railway relations in the current map view."))

        hint = QLabel(
            "Zoom into an area of roughly 20 × 20 km or smaller, then click\n"
            "the button below. Searching over a large area is blocked to avoid\n"
            "slow or rate-limited Overpass requests."
        )
        hint.setStyleSheet("color:#888; font-size:10px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self._view_search_btn = QPushButton("🔍  Search Railway Lines in Current View")
        self._view_search_btn.setMinimumHeight(36)
        self._view_search_btn.setStyleSheet(
            "QPushButton { background:#d35400; color:#fff; border-radius:4px; padding:5px; }"
            "QPushButton:hover { background:#e67e22; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._view_search_btn.clicked.connect(self.search_in_view_requested.emit)
        v.addWidget(self._view_search_btn)

        self._view_status = QLabel("")
        self._view_status.setStyleSheet("color:#aaa; font-size:10px;")
        self._view_status.setWordWrap(True)
        v.addWidget(self._view_status)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;")
        v.addWidget(sep)

        v.addWidget(QLabel("Results:"))

        self._view_results_list = QListWidget()
        self._view_results_list.setAlternatingRowColors(True)
        self._view_results_list.itemDoubleClicked.connect(self._on_view_result_double_clicked)
        v.addWidget(self._view_results_list, stretch=1)

        view_fetch_row = QHBoxLayout()
        self._view_fetch_btn = QPushButton("Fetch selected")
        self._view_fetch_btn.clicked.connect(self._fetch_view_selected)
        view_fetch_row.addStretch()
        view_fetch_row.addWidget(self._view_fetch_btn)
        v.addLayout(view_fetch_row)

        return w

    # ------------------------------------------------------------------
    # Search actions
    # ------------------------------------------------------------------

    def _do_search(self):
        term = self._search_edit.text().strip()
        if not term:
            return
        self._search_btn.setEnabled(False)
        self._search_btn.setText("Searching…")
        self._results_list.clear()

        mode = ["ref", "name", "number_in_name"][self._mode_group.checkedId()]
        worker = SearchWorker(mode, term, self)
        worker.results_ready.connect(self._on_search_results)
        worker.failed.connect(lambda e: QMessageBox.critical(self, "Search error", e))
        worker.finished.connect(lambda: self._search_btn.setText("Search"))
        worker.finished.connect(lambda: self._search_btn.setEnabled(True))
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_search_results(self, results: list):
        self._populate_list(self._results_list, results)

    def _on_result_double_clicked(self, item: QListWidgetItem):
        r = item.data(Qt.ItemDataRole.UserRole)
        if r:
            self._do_fetch(r["id"])

    def _fetch_selected(self):
        item = self._results_list.currentItem()
        if item:
            r = item.data(Qt.ItemDataRole.UserRole)
            if r:
                self._do_fetch(r["id"])

    def _fetch_by_relation(self):
        text = self._rel_edit.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "Invalid ID",
                                "Please enter a numeric OSM relation ID.")
            return
        self._do_fetch(int(text))

    # ── Lines in View actions ─────────────────────────────────────────

    def _on_view_result_double_clicked(self, item: QListWidgetItem):
        r = item.data(Qt.ItemDataRole.UserRole)
        if r:
            self._do_fetch(r["id"])

    def _fetch_view_selected(self):
        item = self._view_results_list.currentItem()
        if item:
            r = item.data(Qt.ItemDataRole.UserRole)
            if r:
                self._do_fetch(r["id"])

    # ── Shared fetch ──────────────────────────────────────────────────

    def _do_fetch(self, relation_id: int):
        self.setEnabled(False)
        worker = FetchWorker(relation_id, self)
        worker.data_ready.connect(self._on_data_ready)
        worker.failed.connect(lambda e: QMessageBox.critical(self, "Fetch error", e))
        worker.finished.connect(lambda: self.setEnabled(True))
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_data_ready(self, data, info: dict):
        self.railway_fetched.emit(data, info)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_list(self, lst: QListWidget, results: list):
        lst.clear()
        if not results:
            item = QListWidgetItem("No results found.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            lst.addItem(item)
            return
        for r in results:
            label = r.get("name") or f"Relation {r['id']}"
            sub   = ""
            if r.get("from") and r.get("to"):
                sub = f"  {r['from']} → {r['to']}"
            elif r.get("network"):
                sub = f"  {r['network']}"
            item = QListWidgetItem(label + sub)
            item.setData(Qt.ItemDataRole.UserRole, r)
            lst.addItem(item)

    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    # ------------------------------------------------------------------
    # Public API (called from App)
    # ------------------------------------------------------------------

    def show_view_results(self, results: list, status: str = ""):
        """Populate the 'Lines in View' tab and switch to it."""
        self._populate_list(self._view_results_list, results)
        if status:
            self._view_status.setText(status)
        self._tabs.setCurrentIndex(1)

    def set_view_search_busy(self, busy: bool):
        self._view_search_btn.setEnabled(not busy)
        self._view_search_btn.setText(
            "Searching…" if busy
            else "🔍  Search Railway Lines in Current View"
        )
        if not busy:
            pass  # status cleared by caller

    def populate_results(self, results: list):
        """Compatibility: populate Search tab results list."""
        self._populate_list(self._results_list, results)
