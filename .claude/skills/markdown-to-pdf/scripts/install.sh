#!/usr/bin/env bash
# Install dependencies for the markdown-to-pdf skill.
# Idempotent: re-running is cheap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$SKILL_DIR/assets"

echo "[install] Python deps (weasyprint, markdown-it-py, plugins, pygments)..."
pip3 install --quiet --upgrade \
  weasyprint \
  markdown-it-py \
  mdit-py-plugins \
  linkify-it-py \
  pygments

echo "[install] Node deps (katex)..."
cd "$SKILL_DIR"
if [ ! -d node_modules/katex ]; then
  npm install --silent --no-audit --no-fund
fi

# Refresh KaTeX CSS + fonts from node_modules every run (cheap, idempotent).
# Only woff2 is kept — the inlining step in render.py turns the first woff2
# url() in each @font-face into a data URI, so the woff/ttf fallbacks are
# never used.
mkdir -p "$ASSETS_DIR/fonts"
if [ -d "$SKILL_DIR/node_modules/katex/dist" ]; then
  echo "[install] Copying KaTeX CSS + woff2 fonts into assets/..."
  cp "$SKILL_DIR/node_modules/katex/dist/katex.min.css" "$ASSETS_DIR/katex.min.css"
  cp "$SKILL_DIR/node_modules/katex/dist/fonts/"*.woff2 "$ASSETS_DIR/fonts/"
fi

# Refresh github-markdown.css (light theme) if available locally.
GHCSS_SRC=""
if [ -f "$SKILL_DIR/node_modules/github-markdown-css/github-markdown-light.css" ]; then
  GHCSS_SRC="$SKILL_DIR/node_modules/github-markdown-css/github-markdown-light.css"
fi
if [ -n "$GHCSS_SRC" ] && [ ! -s "$ASSETS_DIR/github-markdown.css" ]; then
  cp "$GHCSS_SRC" "$ASSETS_DIR/github-markdown.css"
fi

echo "[install] CJK fonts (LXGW WenKai + Noto CJK fallback + emoji)..."
if command -v apt-get >/dev/null 2>&1; then
  # LXGW WenKai (霞鹜文楷) — readable kai-style face for body Chinese.
  if ! fc-list | grep -qi "lxgw"; then
    apt-get install -y --no-install-recommends fonts-lxgw-wenkai >/dev/null 2>&1 || true
  fi
  # Noto CJK as a fallback for any glyphs LXGW lacks.
  if ! fc-list | grep -qi "noto.*cjk"; then
    apt-get install -y --no-install-recommends fonts-noto-cjk >/dev/null 2>&1 || true
  fi
  if ! fc-list | grep -qi "noto color emoji"; then
    apt-get install -y --no-install-recommends fonts-noto-color-emoji >/dev/null 2>&1 || true
  fi
fi
# Refresh fontconfig cache.
fc-cache -f >/dev/null 2>&1 || true

echo "[install] Done."
