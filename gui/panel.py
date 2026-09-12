#!/usr/bin/env python3
"""SimPanel — the redesigned simulation form for the MicroTomo app.

Layout (top → bottom):
  1 · Simulation preset — one-click scenario selector
  2 · Chamber — diameter + boundary note
  3 · Objects — add/remove, position/size/permittivity with mm labels + tooltips
  4 · Transceivers — mode, count, positions + preset fillers
  5 · Advanced — hidden by default; freq count, scatter density, noise, seed
  6 · Reconstruction — grid, Run button, results

Maps its controls onto a gui.state.SimState via get_state()/set_state().
Emits stateChanged (live scene refresh) and runRequested (start a run).
"""
from __future__ import annotations

import copy

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.param_meta import DIM_META, PARAM_META, get_object_meta, Param
from gui.presets import (
    PRESET_ORDER,
    get_preset,
    preset_names,
    preset_summary,
    preset_to_dict,
    state_matches_preset,
    apply_preset,
)
from gui.state import (
    DIM_LABELS,
    ObjectSpec,
    SimState,
    corners_tx,
    default_state,
    random_tx,
)

KIND_CHOICES = ["ellipsoid", "cylinder", "pepper"]


class SimPanel(QWidget):
    """Form that edits a SimState. Debounced stateChanged on any edit."""

    stateChanged = pyqtSignal()
    runRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state: SimState = default_state()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._emit_changed)

        # ---- objects ----------------------------------------------------
        self.obj_list = QListWidget(self)
        self.add_kind = QComboBox(self)
        self.add_kind.addItems(KIND_CHOICES)
        btn_add = QPushButton("+ add", self)
        btn_add.clicked.connect(self._add_object)
        btn_del = QPushButton("remove", self)
        btn_del.clicked.connect(self._remove_object)

        self.kind_cb = QComboBox(self)
        self.kind_cb.addItems(KIND_CHOICES)
        self.cx = self._mm_spin(0.0, 0.0, 500.0, 1.0, "x")
        self.cy = self._mm_spin(0.0, 0.0, 500.0, 1.0, "y")
        self.cz = self._mm_spin(0.0, 0.0, 500.0, 1.0, "z")
        self.d1 = self._mm_spin(20.0, 1.0, 250.0, 1.0, "a")
        self.d2 = self._mm_spin(16.0, 1.0, 250.0, 1.0, "b")
        self.d3 = self._mm_spin(25.0, 1.0, 250.0, 1.0, "c")
        self.eps = self._spin(4.5, 1.2, 80.0, 0.5, "eps")
        self.eps.setSuffix(" εr")
        self.tilt_z = self._spin(0.0, 0.0, 360.0, 5.0, "rotz")
        self.tilt_z.setSuffix(" °")

        # ---- transceivers -----------------------------------------------
        self.mode_cb = QComboBox(self)
        self.mode_cb.addItems(["static", "scan"])
        self.n_tx = QSpinBox(self)
        self.n_tx.setRange(1, 16)
        self.n_tx.setSuffix(" tx")
        self.n_tx.setValue(4)
        self.n_stations = QSpinBox(self)
        self.n_stations.setRange(2, 48)
        self.n_stations.setSuffix(" pos")
        self.n_stations.setValue(24)
        self.chamber_cb = QComboBox(self)
        self.chamber_cb.addItems(["10cm", "50cm"])
        self.tx_table = QTableWidget(0, 3, self)
        self.tx_table.setHorizontalHeaderLabels(["x (m)", "y (m)", "z (m)"])
        self.tx_table.setMinimumWidth(230)
        btn_ring = QPushButton("ring", self)
        btn_ring.clicked.connect(lambda: self._fill("ring"))
        btn_corners = QPushButton("corners", self)
        btn_corners.clicked.connect(lambda: self._fill("corners"))
        btn_random = QPushButton("random", self)
        btn_random.clicked.connect(lambda: self._fill("random"))

        # ---- advanced (hidden by default) -------------------------------
        self.n_freq = QSpinBox(self)
        self.n_freq.setRange(8, 128)
        self.n_freq.setSuffix(" freqs")
        self.n_freq.setValue(48)
        self.scatter_sp = QSpinBox(self)
        self.scatter_sp.setRange(50, 50000)
        self.scatter_sp.setSuffix(" pts")
        self.scatter_sp.setValue(1500)
        self.snr = QDoubleSpinBox(self)
        self.snr.setRange(0.0, 60.0)
        self.snr.setValue(30.0)
        self.snr.setSuffix(" dB")
        self.seed_sb = QSpinBox(self)
        self.seed_sb.setRange(0, 99999)
        self.seed_sb.setValue(42)

        # ---- reconstruction & run ---------------------------------------
        self.grid = QSpinBox(self)
        self.grid.setRange(4, 64)
        self.grid.setSuffix(" pts/axis")
        self.grid.setValue(20)
        self.btn_run = QPushButton("Run simulation", self)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self.runRequested.emit)
        self.results = QLabel("no run yet", self)
        self.results.setWordWrap(True)
        self.results.setStyleSheet("color: #2b7;")
        self.expected_runtime = QLabel("", self)
        self.expected_runtime.setStyleSheet("color: #888; font-size: 11px;")

        self._build_ui()
        self._wire()
        self.set_state(self._state)

    # ------------------------------------------------------------------ ui
    def _spin(self, val, lo, hi, step, label) -> QDoubleSpinBox:
        s = QDoubleSpinBox(self)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.setValue(val)
        s.setPrefix(label + " ")
        s.setMinimumWidth(150)
        return s

    def _mm_spin(self, val, lo, hi, step, label) -> QDoubleSpinBox:
        s = QDoubleSpinBox(self)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(1)
        s.setValue(val)
        s.setPrefix(label + " ")
        s.setSuffix(" mm")
        s.setMinimumWidth(150)
        return s

    def _triple_row(self, a, b, c) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(a)
        row.addWidget(b)
        row.addWidget(c)
        w = QWidget(self)
        w.setLayout(row)
        return w

    def _mk(self, name, fn, parent) -> QPushButton:
        b = QPushButton(name, parent)
        b.clicked.connect(fn)
        return b

    def _apply_tooltip(self, widget, meta: Param):
        """Set a widget's tooltip from param metadata."""
        if meta.unit:
            widget.setToolTip(f"({meta.unit}) {meta.help}")
        else:
            widget.setToolTip(meta.help)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        hint = QLabel("← drag to rotate  ·  scroll to zoom  ·  right-click to pan")
        hint.setStyleSheet("color: #666; font-size: 11px; padding-bottom: 4px;")
        outer.addWidget(hint)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        form = QWidget()
        form_l = QVBoxLayout(form)
        form_l.setContentsMargins(0, 0, 0, 0)

        # ---- 1. Simulation preset --------------------------------------
        sec1 = QGroupBox("1 · Simulation preset")
        s1l = QFormLayout(sec1)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + list(PRESET_ORDER))
        s1l.addRow("scenario", self.preset_combo)
        self.preset_desc = QLabel("", sec1)
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet("color: #555; font-size: 12px;")
        self.preset_expected = QLabel("", sec1)
        self.preset_expected.setStyleSheet("color: #888; font-size: 11px;")
        s1l.addRow(self.preset_desc)
        s1l.addRow("expect", self.preset_expected)
        form_l.addWidget(sec1)

        # ---- 2. Chamber -------------------------------------------------
        sec2 = QGroupBox("2 · Chamber")
        s2l = QFormLayout(sec2)
        s2l.addRow(PARAM_META["chamber_m"].label, self.chamber_cb)
        self._apply_tooltip(self.chamber_cb, PARAM_META["chamber_m"])
        wall_note = QLabel("Boundary: absorbing (default, cleanest signal)")
        wall_note.setStyleSheet("color: #555; font-size: 12px;")
        s2l.addRow(wall_note)
        form_l.addWidget(sec2)

        # ---- 3. Objects -------------------------------------------------
        sec3 = QGroupBox("3 · Objects")
        s3l = QVBoxLayout(sec3)
        # list + add/remove row
        list_row = QHBoxLayout()
        list_row.addWidget(self.obj_list, stretch=1)
        list_row.addWidget(self.add_kind)
        list_row.addWidget(btn_add := self._mk("+ add", self._add_object, sec3))
        list_row.addWidget(btn_del := self._mk("remove", self._remove_object, sec3))
        list_w = QWidget(self)
        list_w.setLayout(list_row)
        s3l.addWidget(list_w)

        # editor
        ed = QFormLayout()
        ed.addRow("shape", self.kind_cb)
        om = get_object_meta()
        for spin, meta_key in (
            (self.cx, "center"), (self.cy, "center"), (self.cz, "center"),
        ):
            self._apply_tooltip(spin, om[meta_key])
        ed.addRow("position (x / y / z)", self._triple_row(self.cx, self.cy, self.cz))
        for spin in (self.d1, self.d2, self.d3):
            self._apply_tooltip(spin, Param(label="", unit="mm", help="Object extent in this axis."))
        ed.addRow("size (a / b / c)", self._triple_row(self.d1, self.d2, self.d3))
        self._apply_tooltip(self.eps, om["eps"])
        self._apply_tooltip(self.tilt_z, om["tilt_deg"])
        ed.addRow("permittivity / rotation", self._triple_row(self.eps, self.tilt_z, QLabel("")))
        ed_w = QWidget(self)
        ed_w.setLayout(ed)
        s3l.addWidget(ed_w)
        form_l.addWidget(sec3)

        # ---- 4. Transceivers -------------------------------------------
        sec4 = QGroupBox("4 · Transceivers")
        s4l = QVBoxLayout(sec4)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("scheme"))
        mrow.addWidget(self.mode_cb)
        mrow.addWidget(QLabel("count"))
        mrow.addWidget(self.n_tx)
        mrow.addWidget(QLabel("stations"))
        mrow.addWidget(self.n_stations)
        mrow_w = QWidget(self)
        mrow_w.setLayout(mrow)
        s4l.addWidget(mrow_w)
        s4l.addWidget(self.chamber_cb)
        s4l.addWidget(self.tx_table)
        brow = QHBoxLayout()
        for label in ("ring", "corners", "random"):
            fn = {"ring": lambda: self._fill("ring"),
                  "corners": lambda: self._fill("corners"),
                  "random": lambda: self._fill("random")}[label]
            brow.addWidget(self._mk(label, fn, sec4))
        brow_w = QWidget(self)
        brow_w.setLayout(brow)
        s4l.addWidget(brow_w)
        tx_help = QLabel(
            "Presets fill the position table. Edit cells directly to place "
            "antennas at custom spots. Values are metres within the chamber."
        )
        tx_help.setStyleSheet("color: #555; font-size: 12px;")
        tx_help.setWordWrap(True)
        s4l.addWidget(tx_help)
        form_l.addWidget(sec4)

        # ---- 5. Advanced (hidden by default) ----------------------------
        self.advanced_box = QGroupBox("5 · Advanced measurement")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(False)
        a5l = QFormLayout(self.advanced_box)
        for spin, key in (
            (self.n_freq, "n_freq"),
            (self.scatter_sp, "n_scatter_per_object"),
            (self.snr, "snr_db"),
            (self.seed_sb, "seed"),
        ):
            m = PARAM_META[key]
            label_txt = f"{m.label}" + (f" ({m.unit})" if m.unit else "")
            a5l.addRow(label_txt, spin)
            self._apply_tooltip(spin, m)
        form_l.addWidget(self.advanced_box)

        # ---- 6. Reconstruction & run ------------------------------------
        sec6 = QGroupBox("6 · Reconstruction & run")
        s6l = QFormLayout(sec6)
        m6 = PARAM_META["grid"]
        s6l.addRow(f"{m6.label} ({m6.unit})", self.grid)
        self._apply_tooltip(self.grid, m6)
        s6l.addWidget(self.expected_runtime)
        s6l.addRow(self.btn_run)
        s6l.addRow(self.results)
        form_l.addWidget(sec6)

        form_l.addStretch()
        scroll.setWidget(form)
        outer.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------ state
    def set_state(self, state: SimState) -> None:
        """Populate all controls from a state (signals blocked)."""
        self._block(True)
        try:
            self.mode_cb.setCurrentText(state.mode)
            self.n_tx.setValue(state.n_tx)
            self.n_stations.setValue(state.n_stations)
            self.chamber_cb.setCurrentText(
                "10cm" if state.chamber_m <= 0.11 else "50cm")
            self.grid.setValue(state.grid)
            self.n_freq.setValue(state.n_freq)
            self.snr.setValue(state.snr_db)
            self.seed_sb.setValue(state.seed)
            self.scatter_sp.setValue(state.n_scatter_per_object)
            # object editor: sync first object or clear
            self.obj_list.clear()
            for o in state.objects:
                self.obj_list.addItem(f"{o.label or o.kind} ({o.kind})")
            if state.objects:
                self.obj_list.setCurrentRow(0)
            self._sync_object_editor()
            self._populate_tx_table(state.physical_positions())
            # preset combo sync
            self._sync_preset_combo(state)
        finally:
            self._block(False)
        self._state = state

    def get_state(self) -> SimState:
        """Return a deep copy carrying the current widget values."""
        i = self.obj_list.currentRow()
        if 0 <= i < len(self._state.objects):
            self._state.objects[i] = self._read_object_editor(
                self._state.objects[i], i)
        st = copy.deepcopy(self._state)
        st.chamber_m = 0.10 if self.chamber_cb.currentText() == "10cm" else 0.50
        st.mode = self.mode_cb.currentText()
        st.n_tx = self.n_tx.value()
        st.n_stations = self.n_stations.value()
        st.grid = self.grid.value()
        st.n_freq = self.n_freq.value()
        st.snr_db = self.snr.value()
        st.seed = self.seed_sb.value()
        st.n_scatter_per_object = self.scatter_sp.value()
        st.tx_positions = self._read_tx_table()
        return st

    # ---- preset sync ---------------------------------------------------
    def _sync_preset_combo(self, state: SimState):
        matched = None
        for key in PRESET_ORDER:
            if state_matches_preset(state, key):
                matched = key
                break
        if matched:
            self.preset_combo.setCurrentText(matched)
            p = get_preset(matched)
            self.preset_desc.setText(p.description)
            self.preset_expected.setText(p.expected)
        else:
            self.preset_combo.setCurrentText("Custom")
            self.preset_desc.setText(
                "Tweak any knob and this switches to 'Custom'.")
            self.preset_expected.setText("")

    def _apply_preset(self, preset_key: str) -> None:
        if preset_key in ("", "Custom"):
            return
        new_st = apply_preset(self._state, preset_key)
        p = get_preset(preset_key)
        self._block(True)
        try:
            self.grid.setValue(p.grid)
            self.n_freq.setValue(p.n_freq)
            self.scatter_sp.setValue(p.n_scatter_per_object)
            self.n_tx.setValue(p.n_tx)
            self.mode_cb.setCurrentText(p.mode)
            self.n_stations.setValue(p.n_stations)
            self.snr.setValue(p.snr_db)
            self.expected_runtime.setText(p.expected)
        finally:
            self._block(False)
        self._state = new_st
        self.preset_desc.setText(p.description)
        self.preset_expected.setText(p.expected)
        self._emit_changed()

    # ---- object editor helpers -----------------------------------------
    def _enable_dim_fields(self) -> None:
        n = len(DIM_LABELS.get(self.kind_cb.currentText(), ()))
        self.d2.setEnabled(n >= 2)
        self.d3.setEnabled(n >= 3)

    def _sync_object_editor(self):
        i = self.obj_list.currentRow()
        if i < 0 or i >= len(self._state.objects):
            return
        o = self._state.objects[i]
        self.kind_cb.setCurrentText(o.kind)
        # centre in mm
        c = [0.0] * 3 + list(o.center)
        self.cx.setValue(c[0] * 1000)
        self.cy.setValue(c[1] * 1000)
        self.cz.setValue(c[2] * 1000)
        # dims in mm
        d = [0.0] * 3 + list(o.dim)
        self.d1.setValue(d[0] * 1000)
        self.d2.setValue(d[1] * 1000)
        self.d3.setValue(d[2] * 1000)
        self.eps.setValue(o.eps)
        self.tilt_z.setValue(o.tilt_deg[2] if len(o.tilt_deg) > 2 else 0.0)
        self._enable_dim_fields()

    def _read_object_editor(self, base: ObjectSpec, i: int) -> ObjectSpec:
        kind = self.kind_cb.currentText()
        n = len(DIM_LABELS.get(kind, ()))
        dims: list = []
        for k, spin in ((1, self.d1), (2, self.d2), (3, self.d3)):
            if n >= k:
                dims.append(spin.value() / 1000)       # mm → m
        return ObjectSpec(
            kind=kind,
            center=[self.cx.value() / 1000,
                    self.cy.value() / 1000,
                    self.cz.value() / 1000],
            dim=dims,
            tilt_deg=[0.0, 0.0, self.tilt_z.value()],
            eps=self.eps.value(),
            label=base.label or f"object{i + 1}",
        )

    def _add_object(self) -> None:
        i = len(self._state.objects)
        self._state.objects.append(
            ObjectSpec(kind=self.add_kind.currentText(),
                       label=f"{self.add_kind.currentText()}{i + 1}"))
        self.set_state(self._state)
        self.obj_list.setCurrentRow(i)
        self._emit_changed()

    def _remove_object(self) -> None:
        i = self.obj_list.currentRow()
        if 0 <= i < len(self._state.objects):
            self._state.objects.pop(i)
            self.set_state(self._state)
            self._emit_changed()

    # ---- transceiver table ---------------------------------------------
    def _populate_tx_table(self, pos: np.ndarray) -> None:
        pos = np.asarray(pos, float)
        self.tx_table.setRowCount(len(pos))
        for r, p in enumerate(pos):
            for c in range(3):
                it = QTableWidgetItem(f"{p[c]:.4f}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                self.tx_table.setItem(r, c, it)

    def _read_tx_table(self) -> np.ndarray | None:
        rows = self.tx_table.rowCount()
        if rows == 0 or self.tx_table.item(0, 0) is None:
            return None
        chamber = 0.10 if self.chamber_cb.currentText() == "10cm" else 0.50
        out = []
        for r in range(rows):
            vals = []
            for c in range(3):
                it = self.tx_table.item(r, c)
                try:
                    v = float(it.text()) if it is not None else 0.0
                except ValueError:
                    v = 0.0
                vals.append(min(chamber, max(0.0, v)))
            out.append(vals)
        return np.asarray(out, float)

    def _fill(self, kind: str) -> None:
        st = self.get_state()
        if kind == "ring":
            st.tx_positions = st.default_ring()
        elif kind == "corners":
            st.tx_positions = corners_tx(st.n_tx, st.chamber_m)
        else:
            st.tx_positions = random_tx(st.n_tx, st.chamber_m)
        self._state = st
        self._populate_tx_table(st.tx_positions)
        self._emit_changed()

    # ---- wiring --------------------------------------------------------
    def _wire(self) -> None:
        # presets
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        # basic fields
        for w in (self.n_tx, self.n_stations, self.grid, self.n_freq, self.snr,
                  self.seed_sb, self.cx, self.cy, self.cz,
                  self.d1, self.d2, self.d3, self.eps, self.tilt_z,
                  self.scatter_sp):
            w.valueChanged.connect(self._debounced)
        for w in (self.mode_cb, self.chamber_cb, self.kind_cb):
            w.currentTextChanged.connect(self._debounced)
        self.tx_table.itemChanged.connect(self._debounced)
        self.obj_list.currentRowChanged.connect(
            lambda _i: self._sync_object_editor())

    def _debounced(self, *_a) -> None:
        self._debounce.start()

    def _emit_changed(self) -> None:
        self._state = self.get_state()
        self.stateChanged.emit()

    def _block(self, on: bool) -> None:
        self._debounce.stop()
        for w in (self.n_tx, self.n_stations, self.grid, self.n_freq, self.snr,
                  self.seed_sb, self.cx, self.cy, self.cz,
                  self.d1, self.d2, self.d3, self.eps, self.tilt_z,
                  self.scatter_sp):
            w.blockSignals(on)
        for w in (self.mode_cb, self.chamber_cb, self.kind_cb):
            w.blockSignals(on)

    def set_results(self, text: str,
                    ok: bool = True) -> None:
        self.results.setText(text)
        colour = "#2b7" if ok else "#c44"
        self.results.setStyleSheet(f"color: {colour};")
