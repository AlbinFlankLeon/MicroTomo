# T8 — GUI scene view (3D viewport + A-scans + run button)

- **Blocks:** none (user-facing viewer; final product)
- **Depends on:** T1 (skeleton), T6 (data produced), T4 (contour to show)

## Goal
Make the GUI useful: load any HDF5 scene written by the study/forward model and render chamber wireframe + transceiver positions (+ scan path) + phantom mesh/point cloud + reconstructed contour overlay. Add a per-channel A-scan view and a single-scene run button bound to the study modules.

## Deliverables
- `gui/scene_view.py`: HDF5 loader → PyVista actors (chamber box, antennas as markers, phantom geometry voxels → Mesh, contour points from DAS, error coloring optional).
- `gui/ascan_view.py`: matplotlib A-scan per Tx·Rx channel (x vs range).
- `gui/controls.py`: combo boxes (chamber, transceivers, static/scan, seed) + Run button → calls `sparsity_study.py` for ONE scene, refreshes viewport/scan.
- `launch_gui.sh` unchanged (main window now loads).

## Verification
- Open a generated demo scene → all actors visible; clicking a channel shows its A-scan.
- Run button produces a new scene and refreshes the viewport.
- Headless smoke (`--smoke`) still passes.