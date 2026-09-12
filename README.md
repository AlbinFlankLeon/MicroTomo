# MicroTomo

60 GHz surface-contour microwave tomography simulator and visualiser.

Edit 3-D object scenes, place transceivers on a cylindrical scanning surface,
run a fast analytical forward model + delay-and-sum reconstruction, and
interactively compare the true geometry against the DAS result.

## Quick install (Linux)

```bash
# download the latest release
wget https://github.com/AlbinFlankLeon/MicroTomo/releases/latest/download/microtomo-0.1.0.tar.gz
tar -xzf microtomo-0.1.0.tar.gz && cd microtomo-0.1.0
bash install.sh
./launch_gui.sh
```

Or clone the source and build from there:

```bash
git clone https://github.com/AlbinFlankLeon/MicroTomo.git
cd MicroTomo
bash install.sh        # venv, deps, demo dataset, app-menu entry
```

## Verify an existing install

```bash
.venv/bin/python scripts/install_check.py
```

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | **Live / stable** — what the download page points to |
| `dev`  | Active development — new features land here and are merged to `main` when stable |

## Download page

See **[docs/download.html](docs/download.html)** for the version history, direct
download links, and install instructions. When hosted on GitHub Pages it serves
as a self-contained release landing page.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest tests/ -q          # should pass (GL may be skipped in CI)
bash scripts/build_release.sh      # builds dist/ tarball + docs/download.html
```

Optional ML loader (not required by the GUI): `pip install -r requirements-ml.txt`

## Uninstall the app-menu entry

```bash
bash install.sh --remove
```
