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
  # Emoji (✅ ❌ ⚠ …) MUST come from an *outline* font — Symbola.
  # Noto Color Emoji is CBDT/CBLC bitmap-only; WeasyPrint can't place bitmap
  # glyphs (they end up microscopic at the page origin → looks like the glyph
  # vanished, while the advance width still reserves blank space). fontconfig
  # prefers the "emoji" generic family for emoji-presentation codepoints, so
  # merely installing Symbola is not enough — Noto Color Emoji must be absent.
  if ! fc-list | grep -qi "symbola"; then
    apt-get install -y --no-install-recommends fonts-symbola >/dev/null 2>&1 || true
  fi
  if fc-list | grep -qi "noto color emoji"; then
    apt-get remove -y fonts-noto-color-emoji >/dev/null 2>&1 || true
  fi
fi

# Second line of defense: a fontconfig rejectfont rule. The apt remove above
# has been observed to fail *silently* in sandboxed environments (all output
# discarded + `|| true`), which once shipped PDFs whose ✅/❌ were invisible.
# With this rule, even if the package survives or gets reinstalled by other
# tooling, fontconfig refuses to expose the font, so Pango can never pick it.
REJECT_CONF=/etc/fonts/conf.d/99-reject-bitmap-emoji.conf
if [ ! -f "$REJECT_CONF" ]; then
  mkdir -p "$(dirname "$REJECT_CONF")" 2>/dev/null || true
  cat > "$REJECT_CONF" 2>/dev/null <<'XML' || true
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<!-- WeasyPrint/Pango cannot draw CBDT/CBLC bitmap glyphs: emoji like U+2705
     get blank space instead of a visible glyph. Reject the bitmap emoji font
     at the fontconfig layer so it can never be selected, even if the package
     is (re)installed by other tooling. Outline emoji come from Symbola. -->
<fontconfig>
  <selectfont>
    <rejectfont>
      <pattern>
        <patelt name="family"><string>Noto Color Emoji</string></patelt>
      </pattern>
    </rejectfont>
  </selectfont>
</fontconfig>
XML
fi

# Refresh fontconfig cache.
fc-cache -f >/dev/null 2>&1 || true

# Verify loudly (no more silent failure): after both defenses, fontconfig
# must NOT see the bitmap font. render.py re-checks this at runtime too.
if fc-list 2>/dev/null | grep -qi "noto color emoji"; then
  echo "[install] ERROR: 'Noto Color Emoji' is still visible to fontconfig." >&2
  echo "          WeasyPrint cannot draw this bitmap font — ✅/❌ would be" >&2
  echo "          INVISIBLE in generated PDFs. Remove the package manually" >&2
  echo "          (apt-get remove fonts-noto-color-emoji) or make" >&2
  echo "          $REJECT_CONF writable, then re-run install.sh." >&2
  exit 1
fi

echo "[install] Done."
