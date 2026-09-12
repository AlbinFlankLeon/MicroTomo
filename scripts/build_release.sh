#!/usr/bin/env bash
# MicroTomo — release builder.
#
# Produces a distributable installer tarball, checksums, and regenerates the
# version download page (docs/download.html) so the latest release is always
# discoverable.
#
# Usage:
#   ./scripts/build_release.sh            # build + checksums + download page
#   ./scripts/build_release.sh --test     # run the test suite first
#   ./scripts/build_release.sh --dry-run  # show what would be packaged, no output
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VERSION="$(cat VERSION)"
DIST_DIR="$ROOT/dist"
# Where the public project will live (used by the download page links).
# Update this once the GitHub repository is created.
GITHUB_USER="AlbinFlankLeon"
GITHUB_REPO="MicroTomo"
REPO_URL="https://github.com/$GITHUB_USER/$GITHUB_REPO"
RELEASE_URL="$REPO_URL/releases/download/v$VERSION"
TARBALL="microtomo-$VERSION.tar.gz"

[[ "${1:-}" == "--test" ]] && { echo ">> running test suite…"; .venv/bin/python -m pytest tests/ -q; }

echo ">> MicroTomo $VERSION release build"
mkdir -p "$DIST_DIR"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "packaging: $(tar --exclude=.venv --exclude=.git --exclude=__pycache__ \
        --exclude='*.pyc' --exclude=dist --exclude='datasets/h5/*.h5' \
        --exclude='datasets/raw' --exclude='sim/gprmax/*.out' \
        -cf - . | wc -c) bytes (estimated)"
    exit 0
fi

echo ">> building $TARBALL…"
# Ship exactly what belongs in the project: tracked + non-ignored files.
git ls-files -co --exclude-standard -z |
    tar --null --no-recursion -czf "$DIST_DIR/$TARBALL" -T -
chmod +x install.sh launch_gui.sh scripts/install_app.sh
echo ">> checksums…"
( cd "$DIST_DIR" && sha256sum "$TARBALL" > SHA256SUMS )

echo ">> generating download page…"
mkdir -p docs
python3 - "$VERSION" "$RELEASE_URL/$TARBALL" "$TARBALL" "$REPO_URL" <<'PY' > docs/download.html
import sys
ver, asset_url, asset_name, repo_url = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MicroTomo — Downloads</title>
<style>
  :root { --fg:#232136; --muted:#6e6a86; --accent:#6c4bb4; --bg:#ece9f7; }
  * { box-sizing: border-box; }
  body { margin:0; font: 15px/1.6 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 28px; margin-bottom: 4px; }
  .tag { color: var(--accent); font-weight: 600; }
  .card { background:#fff; border:1px solid #ddd5ec; border-radius:12px; padding:24px; margin:20px 0; }
  .btn { display:inline-block; background:var(--accent); color:#fff; text-decoration:none;
         padding:12px 22px; border-radius:8px; font-weight:600; }
  .btn:hover { opacity:.9; }
  .btn.secondary { background:#fff; color:var(--accent); border:1px solid var(--accent); }
  code { background:#f1eef9; padding:2px 6px; border-radius:4px; font-size:13px; }
  .muted { color:var(--muted); font-size:13px; }
  table { width:100%%; border-collapse:collapse; margin-top:8px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #ece7f6; }
  a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>MicroTomo <span class="tag">v%s</span></h1>
  <p class="muted">60&nbsp;GHz surface-contour simulator — edit scenes, place transceivers, run a DAS reconstruction, and rotate the 3D result.</p>

  <div class="card">
    <h2>Latest release — v%s</h2>
    <p style="margin-top:6px">
      <a class="btn" href="%s">Download %s</a>
      &nbsp;
      <a class="btn secondary" href="%s/releases">All releases</a>
    </p>
    <p class="muted">Linux · ~2–5&nbsp;min install · requires Python 3.10+ and a 64-bit system with OpenGL.</p>
    <h3>Install</h3>
    <pre><code>tar -xzf %s &amp;&amp; cd microtomo-%s
bash install.sh
./launch_gui.sh</code></pre>
    <h3>Verify</h3>
    <p class="muted">Checksum: <code>dist/SHA256SUMS</code> in the tarball, or run <code>sha256sum %s</code>.</p>
  </div>

  <div class="card">
    <h2>Version history</h2>
    <table>
      <tr><th>Version</th><th>Notes</th><th>Download</th></tr>
      <tr><td>v%s</td><td>First public release — editor GUI, DAS reconstruction, visualisation, demo dataset.</td>
          <td><a href="%s">tar.gz</a></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Development</h2>
    <p class="muted">Source is on GitHub: <a href="%s">github.com/AlbinFlankLeon/MicroTomo</a>.
       The <code>main</code> branch is the live stable tree; active work happens on <code>dev</code>.</p>
  </div>
</div>
</body>
</html>
""" % (ver, ver, asset_url, asset_name, repo_url, asset_name, ver, asset_name, ver, asset_url, repo_url)
sys.stdout.write(html)
PY

echo ">> wrote docs/download.html"
echo "------------------------------"
echo "artifacts:"
ls -lh "$DIST_DIR/$TARBALL" "$DIST_DIR/SHA256SUMS" docs/download.html
echo
echo "publish with (after git push):"
echo "  gh release create v$VERSION $DIST_DIR/$TARBALL --title 'MicroTomo $VERSION' --notes \"...\""