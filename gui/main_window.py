#!/usr/bin/env python3
"""MainWindow — the interactive MicroTomo editor.

Left: 3D viewport (true surface, reconstructed cloud, transceiver markers)
with a true / recon / both toggle. Right: the editable SimPanel. The panel's
edits rebuild the scene live; "Run simulation" executes forward + DAS and
shows the reconstruction + metrics. State persists to ~/.config/microtomo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))) / "microtomo"
CONFIG_FILE = CONFIG_DIR / "sim_state.json"


class MainWindow:
    """QMainWindow subclassing done in build() to keep imports Qt-lazy."""

    def build(self):
        from PyQt6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QRadioButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )
        from pyvistaqt import QtInteractor

        from gui.main import View, _draw_scene
        from gui.panel import SimPanel

        win = QMainWindow()
        win.setWindowTitle(f"MicroTomo — simulation editor ({_version()})")
        win.resize(1440, 860)

        central = QWidget()
        outer = QHBoxLayout(central)

        # ---- left: viewport + view toggle ----
        left = QVBoxLayout()
        plotter = QtInteractor(central)
        left.addWidget(plotter.interactor, stretch=1)

        toggle_row = QHBoxLayout()
        self.rb_true = QRadioButton("true geometry")
        self.rb_recon = QRadioButton("reconstruction")
        self.rb_both = QRadioButton("both")
        self.rb_both.setChecked(True)
        toggle_row.addWidget(QLabel("view: "))
        toggle_row.addWidget(self.rb_true)
        toggle_row.addWidget(self.rb_recon)
        toggle_row.addWidget(self.rb_both)
        toggle_row.addStretch(1)
        left.addLayout(toggle_row)

        self.status = QLabel("ready — edit the scene or hit Run simulation")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        # ---- right: editable panel ----
        right = QVBoxLayout()
        top = QHBoxLayout()
        for name, fn in (("Defaults", self._defaults), ("Save", self._save_config),
                         ("Load", self._load_config)):
            b = QPushButton(name, central)
            b.clicked.connect(fn)
            top.addWidget(b)
        top.addStretch(1)
        right.addLayout(top)

        panel = SimPanel()
        scroll = QScrollArea(central)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        scroll.setWidget(panel)
        right.addWidget(scroll, stretch=1)
        outer.addLayout(left, stretch=1)
        outer.addLayout(right)
        win.setCentralWidget(central)

        self.win = win
        self.plotter = plotter
        self.panel = panel
        self.recon = None
        self.metrics: dict = {}

        panel.stateChanged.connect(self._on_scene_edit)
        panel.runRequested.connect(self.run_simulation)
        for rb in (self.rb_true, self.rb_recon, self.rb_both):
            rb.toggled.connect(lambda _c, _rb=rb: self.apply_view_mode())
        self.refresh_scene()
        return win

    # ------------------------------------------------------------------ state
    @property
    def state(self):
        return self.panel.get_state()

    # ------------------------------------------------------------------ paint
    def _on_scene_edit(self) -> None:
        """Any edit invalidates the previous reconstruction (geometry changed)."""
        if self.recon is not None:
            self.recon = None
            self.metrics = {}
            self.panel.set_results("scene edited — run again to reconstruct")
        self.refresh_scene()

    def refresh_scene(self) -> None:
        try:
            st = self.state
            scatter = st.scatter()
            view = View(scatter=scatter, chamber_m=st.chamber_m, recon=self.recon,
                        transceivers=st.physical_positions())
            self.plotter.clear()
            _draw_scene(self.plotter, view, name="scene")
            self.plotter.reset_camera()
            self.apply_view_mode()
            self.plotter.render()
            n_recon = 0 if view.recon is None else len(view.recon)
            note = f" · recon {n_recon} pts (none — abortive threshold?)" if n_recon == 0 else ""
            self.status.setText(
                f"true geometry: {len(scatter)} surface points · "
                f"{len(st.objects)} objects · {len(view.transceivers)} transceivers"
                f"{note}")
        except Exception as e:  # keep the app alive on bad geometry
            print(f"[microtomo] refresh_scene failed: {e!r}", file=sys.stderr)
            self.panel.set_results(f"refresh error: {e}", ok=False)
            self.status.setText(f"refresh error: {e}")

    def apply_view_mode(self) -> None:
        actors = {a.name: a for a in self.plotter.renderer.actors.values()
                  if hasattr(a, "name")}
        scatter = actors.get("scene_scatter")
        recon = actors.get("recon_cloud")
        if scatter is not None:
            scatter.visibility = self.rb_true.isChecked() or self.rb_both.isChecked()
        if recon is not None:
            recon.visibility = self.rb_recon.isChecked() or self.rb_both.isChecked()
        if hasattr(self, "plotter") and self.plotter is not None:
            self.plotter.render()

    # ------------------------------------------------------------------ run
    def run_simulation(self) -> None:
        from PyQt6.QtWidgets import QApplication

        st = self.state
        self.status.setText("running forward model + DAS …")
        QApplication.processEvents()
        try:
            recon, metrics, wall = st.run_pipeline()
        except Exception as e:
            self.panel.set_results(f"run failed: {e}", ok=False)
            self.status.setText(f"run failed: {e}")
            return
        self.recon = recon
        self.metrics = metrics
        self.rb_both.setChecked(True)          # always reveal the reconstruction
        self.refresh_scene()
        n = len(recon)
        if n == 0:
            self.panel.set_results(
                "run finished but reconstruction is EMPTY — raise grid / "
                "add transceivers / check object placement", ok=False)
            self.status.setText("run produced an empty reconstruction — see panel")
            return
        self.panel.set_results(
            f"{metrics['mode']} {metrics['n_tx']}×{metrics.get('n_stations', 1)} · "
            f"median {metrics['chamfer_median_cm']:.2f} cm · "
            f"{metrics['pct_within_1cm']:.1f}% ≤1 cm · "
            f"{len(recon)} pts · {metrics['wall_time_s']:.1f}s")
        self.status.setText(
            f"run done — median chamfer {metrics['chamfer_median_cm']:.2f} cm, "
            f"{metrics['pct_within_1cm']:.1f}% of surface within 1 cm")
        self.apply_view_mode()

    # ------------------------------------------------------------------ config
    def _defaults(self) -> None:
        from gui.state import default_state

        self.recon = None
        self.metrics = {}
        self.panel._block(True)
        self.panel.set_state(default_state())
        self.panel._block(False)
        self.refresh_scene()

    def _save_config(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(self.state.to_dict(), indent=2))
        self.status.setText(f"saved configuration → {CONFIG_FILE}")

    def _load_config(self) -> None:
        from gui.state import SimState

        if not CONFIG_FILE.exists():
            self.status.setText("no saved configuration yet")
            return
        st = SimState.from_dict(json.loads(CONFIG_FILE.read_text()))
        self.panel.set_state(st)
        self.refresh_scene()
        self.status.setText(f"loaded configuration ← {CONFIG_FILE}")


def _version() -> str:
    """Short git hash, or 'dev' when not in a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or "dev"
    except Exception:
        return "dev"


def build_editor_window(app):
    return MainWindow().build()