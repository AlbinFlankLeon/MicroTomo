#!/usr/bin/env python3
"""SimPanel — the editable simulation setup/form for the MicroTomo app.

Pure Qt widget: objects (add/remove/edit kind+place+size), transceiver
placement (mode, count, editable table + geometry presets), and run options.
Maps its controls onto a gui.state.SimState via get_state()/set_state().
Emits stateChanged (live scene refresh) and runRequested (start a run).
"""
from __future__ import annotations

import copy

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.state import DIM_LABELS, ObjectSpec, SimState, default_state

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

        # objects
        self.obj_list = QListWidget(self)
        self.add_kind = QComboBox(self)
        self.add_kind.addItems(KIND_CHOICES)
        btn_add = QPushButton("+ add", self)
        btn_add.clicked.connect(self._add_object)
        btn_del = QPushButton("remove", self)
        btn_del.clicked.connect(self._remove_object)

        self.kind_cb = QComboBox(self)
        self.kind_cb.addItems(KIND_CHOICES)
        self.cx = self._spin(0.0, 0.0, 0.5, 0.005, "x")
        self.cy = self._spin(0.0, 0.0, 0.5, 0.005, "y")
        self.cz = self._spin(0.0, 0.0, 0.5, 0.005, "z")
        self.d1 = self._spin(0.0, 0.001, 0.25, 0.002, "a")
        self.d2 = self._spin(0.0, 0.001, 0.25, 0.002, "b")
        self.d3 = self._spin(0.0, 0.001, 0.25, 0.002, "c")
        self.eps = self._spin(4.5, 1.2, 80.0, 0.5, "eps")
        self.tilt_z = self._spin(0.0, 0.0, 360.0, 5.0, "rotz")

        # transceivers
        self.mode_cb = QComboBox(self)
        self.mode_cb.addItems(["static", "scan"])
        self.n_tx = QSpinBox(self)
        self.n_tx.setRange(1, 16)
        self.n_tx.setValue(4)
        self.n_stations = QSpinBox(self)
        self.n_stations.setRange(2, 48)
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

        # run options
        self.grid = QSpinBox(self)
        self.grid.setRange(4, 48)
        self.grid.setValue(20)
        self.n_freq = QSpinBox(self)
        self.n_freq.setRange(8, 128)
        self.n_freq.setValue(48)
        self.snr = QDoubleSpinBox(self)
        self.snr.setRange(0.0, 60.0)
        self.snr.setValue(30.0)
        self.snr.setSuffix(" dB")
        self.seed_sb = QSpinBox(self)
        self.seed_sb.setRange(0, 99999)
        self.seed_sb.setValue(42)
        btn_run = QPushButton("Run simulation", self)
        btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_run.clicked.connect(self.runRequested.emit)
        self.results = QLabel("no run yet", self)
        self.results.setWordWrap(True)
        self.results.setStyleSheet("color: #2b7;")

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

    def _triple_row(self, a, b, c) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(a)
        row.addWidget(b)
        row.addWidget(c)
        w = QWidget(self)
        w.setLayout(row)
        return w

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        obj_box = QGroupBox("Objects", self)
        b = QVBoxLayout(obj_box)
        b.addWidget(self.obj_list)
        row = QHBoxLayout()
        row.addWidget(self.add_kind)
        for name, fn in (("+ add", self._add_object), ("remove", self._remove_object)):
            btn = QPushButton(name, obj_box)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        b.addLayout(row)

        ed = QFormLayout()
        ed.addRow("type", self.kind_cb)
        ed.addRow("centre", self._triple_row(self.cx, self.cy, self.cz))
        ed.addRow("size", self._triple_row(self.d1, self.d2, self.d3))
        ed.addRow("eps / rotz", self._triple_row(self.eps, self.tilt_z, QLabel("")))
        b.addLayout(ed)
        outer.addWidget(obj_box)

        tx_box = QGroupBox("Transceivers", self)
        form = QFormLayout(tx_box)
        mrow = QHBoxLayout()
        mrow.addWidget(self.mode_cb)
        mrow.addWidget(self.n_tx)
        mrow.addWidget(QLabel("tx"))
        mrow.addWidget(self.n_stations)
        mrow.addWidget(QLabel("stations"))
        form.addRow("mode / count", mrow)
        form.addRow("chamber", self.chamber_cb)
        form.addRow("positions", self.tx_table)
        brow = QHBoxLayout()
        brow.addWidget(self._mk("ring", lambda: self._fill("ring"), tx_box))
        brow.addWidget(self._mk("corners", lambda: self._fill("corners"), tx_box))
        brow.addWidget(self._mk("random", lambda: self._fill("random"), tx_box))
        form.addRow("presets", brow)
        outer.addWidget(tx_box)

        run_box = QGroupBox("Reconstruction", self)
        r = QFormLayout(run_box)
        rrow = QHBoxLayout()
        rrow.addWidget(self.grid)
        rrow.addWidget(QLabel("grid"))
        rrow.addWidget(self.n_freq)
        rrow.addWidget(QLabel("freq"))
        r.addRow("options", rrow)
        srow = QHBoxLayout()
        srow.addWidget(self.snr)
        srow.addWidget(self.seed_sb)
        srow.addWidget(QLabel("seed"))
        r.addRow("noise", srow)
        btn_run = QPushButton("Run simulation", run_box)
        btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_run.clicked.connect(self.runRequested.emit)
        r.addRow(btn_run)
        r.addRow(self.results)
        outer.addWidget(run_box)

    def _mk(self, name, fn, parent) -> QPushButton:
        b = QPushButton(name, parent)
        b.clicked.connect(fn)
        return b

    # ------------------------------------------------------------------ state
    def set_state(self, state: SimState) -> None:
        """Populate all controls from a state (signals blocked)."""
        self._block(True)
        try:
            self.mode_cb.setCurrentText(state.mode)
            self.n_tx.setValue(state.n_tx)
            self.n_stations.setValue(state.n_stations)
            self.chamber_cb.setCurrentText("10cm" if state.chamber_m <= 0.11 else "50cm")
            self.grid.setValue(state.grid)
            self.n_freq.setValue(state.n_freq)
            self.snr.setValue(state.snr_db)
            self.seed_sb.setValue(state.seed)
            self.obj_list.clear()
            for o in state.objects:
                self.obj_list.addItem(f"{o.label or o.kind} ({o.kind})")
            if state.objects:
                self.obj_list.setCurrentRow(0)
            self._populate_tx_table(state.physical_positions())
        finally:
            self._block(False)
        self._state = state
        self._sync_object_editor()

    def get_state(self) -> SimState:
        """Return a deep copy carrying the current widget values."""
        i = self.obj_list.currentRow()
        if 0 <= i < len(self._state.objects):
            self._state.objects[i] = self._read_object_editor(self._state.objects[i], i)
        st = copy.deepcopy(self._state)
        st.chamber_m = 0.10 if self.chamber_cb.currentText() == "10cm" else 0.50
        st.mode = self.mode_cb.currentText()
        st.n_tx = self.n_tx.value()
        st.n_stations = self.n_stations.value()
        st.grid = self.grid.value()
        st.n_freq = self.n_freq.value()
        st.snr_db = self.snr.value()
        st.seed = self.seed_sb.value()
        st.tx_positions = self._read_tx_table()
        return st

    # ---- object editor ----
    def _sync_object_editor(self) -> None:
        i = self.obj_list.currentRow()
        if i < 0 or i >= len(self._state.objects):
            return
        o = self._state.objects[i]
        self.kind_cb.setCurrentText(o.kind)
        self.cx.setValue(o.center[0])
        self.cy.setValue(o.center[1])
        self.cz.setValue(o.center[2])
        dims = (o.dim[:3] + [0.0, 0.0, 0.0])[:3]
        self.d1.setValue(dims[0])
        self.d2.setValue(dims[1])
        self.d3.setValue(dims[2])
        self.eps.setValue(o.eps)
        self.tilt_z.setValue(o.tilt_deg[2] if len(o.tilt_deg) > 2 else 0.0)
        self._enable_dim_fields()

    def _enable_dim_fields(self) -> None:
        n = len(DIM_LABELS.get(self.kind_cb.currentText(), ()))
        self.d2.setEnabled(n >= 2)
        self.d3.setEnabled(n >= 3)

    def _read_object_editor(self, base: ObjectSpec, i: int) -> ObjectSpec:
        kind = self.kind_cb.currentText()
        n = len(DIM_LABELS.get(kind, ()))
        dims: list = []
        for k, spin in ((1, self.d1), (2, self.d2), (3, self.d3)):
            if n >= k:
                dims.append(spin.value())
        return ObjectSpec(
            kind=kind,
            center=[self.cx.value(), self.cy.value(), self.cz.value()],
            dim=dims,
            tilt_deg=[0.0, 0.0, self.tilt_z.value()],
            eps=self.eps.value(),
            label=base.label or f"object{i + 1}",
        )

    def _add_object(self) -> None:
        i = len(self._state.objects)
        self._state.objects.append(ObjectSpec(kind=self.add_kind.currentText(),
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

    # ---- transceiver table ----
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
            st.tx_positions = _corners(st.n_tx, st.chamber_m)
        else:
            st.tx_positions = _random_tx(st.n_tx, st.chamber_m)
        self._state = st
        self._populate_tx_table(st.tx_positions)
        self._emit_changed()

    # ---- wiring ----
    def _wire(self) -> None:
        for w in (self.n_tx, self.n_stations, self.grid, self.n_freq, self.snr,
                  self.seed_sb, self.cx, self.cy, self.cz,
                  self.d1, self.d2, self.d3, self.eps, self.tilt_z):
            w.valueChanged.connect(self._debounced)
        for w in (self.mode_cb, self.chamber_cb, self.kind_cb):
            w.currentTextChanged.connect(self._debounced)
        self.tx_table.itemChanged.connect(self._debounced)
        self.obj_list.currentRowChanged.connect(lambda _i: self._sync_object_editor())

    def _debounced(self, *_a) -> None:
        self._debounce.start()

    def _emit_changed(self) -> None:
        self._state = self.get_state()
        self.stateChanged.emit()

    def _block(self, on: bool) -> None:
        self._debounce.stop()
        for w in (self.n_tx, self.n_stations, self.grid, self.n_freq, self.snr,
                  self.seed_sb, self.cx, self.cy, self.cz,
                  self.d1, self.d2, self.d3, self.eps, self.tilt_z):
            w.blockSignals(on)
        for w in (self.mode_cb, self.chamber_cb, self.kind_cb):
            w.blockSignals(on)

    def set_results(self, text: str, ok: bool = True) -> None:
        self.results.setText(text)
        self.results.setStyleSheet("color: #2b7;" if ok else "color: #e44;")


def _corners(n: int, chamber_m: float) -> np.ndarray:
    import itertools

    m = chamber_m * 0.16
    lo, hi = m, chamber_m - m
    pool = list(itertools.product((lo, hi), repeat=3))
    sel = [pool[i % len(pool)] for i in range(n)]
    return np.asarray(sel, float)


def _random_tx(n: int, chamber_m: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    lo, hi = 0.1 * chamber_m, 0.9 * chamber_m
    return rng.uniform(lo, hi, size=(n, 3)).astype(float)