#!/usr/bin/env bash
# Install dependencies for the check-github-render skill.
# Idempotent: re-running is cheap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

echo "[install] Node deps (mathjax-full)..."
cd "$SKILL_DIR"
if [ ! -d node_modules/mathjax-full ]; then
  npm install --silent --no-audit --no-fund
fi

echo "[install] Python deps (markdown-it-py for the bold-flanking check)..."
# Drives the emphasis-flanking check; same CommonMark engine CLAUDE.md uses.
# weasyprint is only needed for --visual; left to that path / the
# markdown-to-pdf skill so a text-only check stays lightweight.
python3 -c "import markdown_it" 2>/dev/null || pip install -q markdown-it-py

echo "[install] Done."
