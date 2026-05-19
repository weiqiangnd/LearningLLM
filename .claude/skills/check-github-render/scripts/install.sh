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

echo "[install] Done."
