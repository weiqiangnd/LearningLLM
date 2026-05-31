#!/usr/bin/env bash
# Install dependencies for the check-pdf-formula skill.
# Idempotent: re-running is cheap.
#
# 这条 skill 复用 markdown-to-pdf 的渲染管线（WeasyPrint + KaTeX + 字体），
# 所以先把那条 skill 的依赖装齐，再补上本 skill 特有的 PyMuPDF / Pillow。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MD2PDF_INSTALL="$SKILLS_DIR/markdown-to-pdf/scripts/install.sh"
if [ -f "$MD2PDF_INSTALL" ]; then
  echo "[install] 复用 markdown-to-pdf 的依赖（WeasyPrint / KaTeX / 字体）…"
  bash "$MD2PDF_INSTALL"
else
  echo "[install][warn] 找不到 markdown-to-pdf 的 install.sh，单独装 weasyprint…"
  pip3 install --quiet --upgrade weasyprint
fi

echo "[install] 本 skill 特有依赖（PyMuPDF / Pillow）…"
pip3 install --quiet --upgrade pymupdf pillow

echo "[install] Done."
