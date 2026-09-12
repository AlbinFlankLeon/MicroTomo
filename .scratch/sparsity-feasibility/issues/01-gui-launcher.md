# T1 — GUI skeleton + launcher

- **Blocks:** none (first tracer bullet)
- **Depends on:** [TBD none]

## Goal
A launchable MicroTomo GUI: PyQt6 + PyVista/pyvistaqt desktop window with an empty 3D viewport (chamber-placeholder), opened by a simple `./launch_gui.sh` one-liner. Proves the GUI stack works on this machine before any scene data exists.

## Deliverables
- `gui/main.py` — Qt app + main window + 3D viewport widget (pv.Plotter embedded in QWidget; placeholder sphere/wireframe so it visibly renders).
- `gui/requirements.txt` or explicit deps: PyQt6, pyvista, pyvistaqt (installed into MicroTomo `.venv`).
- `launch_gui.sh` — trivial launcher: activates `.venv`, runs `python gui/main.py`.
- Optionally `--headless-smoke` flag for CI (offscreen render + import check).

## Verification
1. `./launch_gui.sh --smoke` exits 0 and prints "GUI OK".
2. `./launch_gui.sh` opens a window with a renderable 3D viewport (manual check).

## Notes
- Fresh code; do NOT copy GPRForce code (no license). Borrow only the *module split* idea (main window / viewport / controls separated).
- Ship as git commit.