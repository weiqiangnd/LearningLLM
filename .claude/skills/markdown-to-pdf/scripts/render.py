#!/usr/bin/env python3
"""Render a project markdown file to GitHub-style PDF under ./dist.

Usage:
    python3 render.py <input.md> [more.md ...]

For each input file `dir/name.md`, writes `dist/name.pdf` (relative to CWD).
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

SKILL_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_DIR / "assets"

# Markdown -> HTML (GFM-flavored) via markdown-it-py.
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.deflist import deflist_plugin
from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


# ---------- math extraction ----------

# Token markers that pass through markdown unchanged (pure alphanumerics).
_BLOCK_MARK = "zZkAtExBlOcK{idx}MaRkEr"
_INLINE_MARK = "zZkAtExInLiNe{idx}MaRkEr"
_CODE_MARK = "zZcOdEbLoCk{idx}MaRkEr"
_ICODE_MARK = "zZiNlInEcOdE{idx}MaRkEr"

_FENCED_RE = re.compile(r"(^|\n)(?P<fence>```+|~~~+)[^\n]*\n.*?\n(?P=fence)[ \t]*(?=\n|$)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(`+)(?!`)(.+?)\1", re.DOTALL)
_BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)\$\$", re.DOTALL)
# Inline math: single $...$, must not contain newline or unescaped $. Avoid
# matching digit-flanked currency like "$5" by requiring non-space adjacent
# to the $ on each side.
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\d)")


def _unescape_markdown_in_tex(tex: str) -> str:
    r"""Reverse the markdown-level escapes that are only meaningful before
    markdown parsing, so KaTeX sees real LaTeX.

    Background: in this repo's source markdown, authors write `\_` (and
    occasionally `\*` / `\$`) inside `$...$` / `$$...$$` to keep markdown
    from interpreting `_` as emphasis when the surrounding paragraph
    contains other `_` / `*`. GitHub renders math via MathJax *after*
    markdown has unescaped those — so on github.com `\_{...}` correctly
    renders as `_{...}` (subscript). Our pipeline pulls math out *before*
    markdown sees the surrounding paragraph, so the escapes survive into
    KaTeX, which treats `\_` as a literal underscore glyph — and the
    intended subscript turns into `E_τ∼π_θ`-shaped garbage.

    These backslash-escapes are unambiguous to undo: in LaTeX neither `\_`
    nor `\*` nor `\$` appears in regular math notation (the literal
    underscore in LaTeX is `\underline{}` / `\_{}` in text mode but not
    math), so collapsing them to the bare char is safe.
    """
    # \_  -> _   (subscript: \mathbb{E}\_{...} -> \mathbb{E}_{...})
    # \*  -> *   (rare; emphasis escape)
    # \$  -> $   (dollar escape — but `$` inside math is illegal anyway)
    return (
        tex.replace(r"\_", "_")
           .replace(r"\*", "*")
           .replace(r"\$", "$")
    )


def mask_code_and_math(md_text: str):
    """Replace code blocks, inline code, and math with placeholders.

    Returns (masked_text, code_blocks, inline_codes, block_math, inline_math).
    """
    code_blocks: list[str] = []
    inline_codes: list[str] = []
    block_math: list[str] = []
    inline_math: list[str] = []

    def repl_fenced(m: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(m.group(0))
        # Keep the leading newline so markdown still sees a paragraph break.
        return (m.group(1) or "") + _CODE_MARK.format(idx=idx)

    text = _FENCED_RE.sub(repl_fenced, md_text)

    def repl_inline_code(m: re.Match) -> str:
        idx = len(inline_codes)
        inline_codes.append(m.group(0))
        return _ICODE_MARK.format(idx=idx)

    text = _INLINE_CODE_RE.sub(repl_inline_code, text)

    def repl_block_math(m: re.Match) -> str:
        idx = len(block_math)
        block_math.append(_unescape_markdown_in_tex(m.group(1).strip()))
        # Surround with blank lines so markdown treats it as its own paragraph.
        return "\n\n" + _BLOCK_MARK.format(idx=idx) + "\n\n"

    text = _BLOCK_MATH_RE.sub(repl_block_math, text)

    def repl_inline_math(m: re.Match) -> str:
        idx = len(inline_math)
        inline_math.append(_unescape_markdown_in_tex(m.group(1).strip()))
        return _INLINE_MARK.format(idx=idx)

    text = _INLINE_MATH_RE.sub(repl_inline_math, text)

    return text, code_blocks, inline_codes, block_math, inline_math


def render_math_via_katex(items: list[tuple[str, bool]]) -> list[str]:
    """Render [(tex, display), ...] -> [html, ...] via Node + KaTeX."""
    if not items:
        return []
    node_script = SKILL_DIR / "scripts" / "render_math.js"
    payload = "\n".join(
        json.dumps({"tex": tex, "display": display}, ensure_ascii=False)
        for tex, display in items
    )
    proc = subprocess.run(
        ["node", str(node_script)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"KaTeX renderer failed:\n{proc.stderr}")
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(out_lines) != len(items):
        raise RuntimeError(
            f"KaTeX renderer returned {len(out_lines)} lines, expected {len(items)}"
        )
    results: list[str] = []
    for ln, (tex, _) in zip(out_lines, items):
        parsed = json.loads(ln)
        if isinstance(parsed, dict) and "error" in parsed:
            # Fall back to <code> so the doc still renders.
            results.append(f'<code class="math-error">{tex}</code>')
        else:
            results.append(parsed)
    return results


# ---------- code highlighting ----------

_FORMATTER = HtmlFormatter(nowrap=False, cssclass="highlight")
CODE_CSS = _FORMATTER.get_style_defs(".highlight")


def highlight_code(code: str, lang: str | None) -> str:
    # No language specified → render as plain text. Don't call guess_lexer:
    # Pygments will happily pick the highest-confidence match for an ASCII
    # directory tree / log output / etc., then flag the unrecognized parts
    # as Token.Error which gets a red 1px border in the default style. GitHub
    # itself never auto-guesses; we match that behavior.
    if not lang:
        return f'<div class="highlight"><pre><code>{escape_html(code)}</code></pre></div>'
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        return f'<div class="highlight"><pre><code>{escape_html(code)}</code></pre></div>'
    return pyg_highlight(code, lexer, _FORMATTER)


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------- markdown -> html ----------

def build_md() -> MarkdownIt:
    md = MarkdownIt("gfm-like", {"html": True, "linkify": True, "typographer": False})
    md.use(anchors_plugin, max_level=6, permalink=False, slug_func=github_slug)
    md.use(footnote_plugin)
    md.use(tasklists_plugin, enabled=True)
    md.use(deflist_plugin)
    md.enable("table")
    md.enable("strikethrough")

    def fence_render(self, tokens, idx, options, env):
        token = tokens[idx]
        return highlight_code(token.content, (token.info or "").strip().split(" ")[0])

    md.add_render_rule("fence", fence_render)
    return md


# GitHub heading slug algorithm: lowercase, drop punctuation, replace
# whitespace with dashes, strip non-word chars (keep unicode letters/digits).
_PUNCT_RE = re.compile(r"[^\w一-鿿\- ]+", re.UNICODE)


def github_slug(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", "-", s)
    return s


# ---------- image inlining ----------

_IMG_TAG_RE = re.compile(r'<img\b([^>]*?)\bsrc="([^"]+)"([^>]*)>', re.IGNORECASE)

# Modest cap to avoid eating an unbounded remote response into a PDF.
_REMOTE_IMG_MAX_BYTES = 4 * 1024 * 1024  # 4 MB

# Map well-known remote badges to local assets so PDFs render the same
# even when the network policy blocks the upstream host.
_REMOTE_IMAGE_LOCAL_MAP = {
    "https://colab.research.google.com/assets/colab-badge.svg": ASSETS_DIR / "colab-badge.svg",
}


def _fetch_remote_image(url: str) -> tuple[bytes, str] | None:
    """Fetch a remote image. Returns (bytes, mime) or None on failure."""
    import urllib.request

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "markdown-to-pdf-skill/0.1 (+local PDF render)",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(_REMOTE_IMG_MAX_BYTES + 1)
            if len(data) > _REMOTE_IMG_MAX_BYTES:
                print(f"[warn] remote image > {_REMOTE_IMG_MAX_BYTES} bytes, skipping: {url}", file=sys.stderr)
                return None
            mime = resp.headers.get("Content-Type") or ""
            mime = mime.split(";", 1)[0].strip()
            if not mime:
                # Fall back to URL extension.
                guessed, _ = mimetypes.guess_type(url)
                mime = guessed or "application/octet-stream"
            return data, mime
    except Exception as e:  # noqa: BLE001 — best-effort fetch
        print(f"[warn] failed to fetch remote image {url}: {e}", file=sys.stderr)
        return None


def inline_images(html: str, base_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        pre, src, post = m.group(1), m.group(2), m.group(3)
        parsed = urlparse(src)
        if parsed.scheme == "data":
            return m.group(0)
        if parsed.scheme in ("http", "https"):
            # First, see if this is a known badge we ship locally.
            local_override = _REMOTE_IMAGE_LOCAL_MAP.get(src)
            if local_override and local_override.is_file():
                mime, _ = mimetypes.guess_type(local_override.name)
                mime = mime or "image/svg+xml"
                b64 = base64.b64encode(local_override.read_bytes()).decode("ascii")
                return f'<img{pre} src="data:{mime};base64,{b64}"{post}>'
            fetched = _fetch_remote_image(src)
            if fetched is None:
                return m.group(0)
            data, mime = fetched
            b64 = base64.b64encode(data).decode("ascii")
            return f'<img{pre} src="data:{mime};base64,{b64}"{post}>'
        # Local file. Resolve relative to base_dir.
        local = (base_dir / unquote(src)).resolve()
        if not local.is_file():
            print(f"[warn] image not found: {src} (resolved to {local})", file=sys.stderr)
            return m.group(0)
        mime, _ = mimetypes.guess_type(local.name)
        mime = mime or "application/octet-stream"
        b64 = base64.b64encode(local.read_bytes()).decode("ascii")
        return f'<img{pre} src="data:{mime};base64,{b64}"{post}>'

    return _IMG_TAG_RE.sub(repl, html)


# ---------- KaTeX CSS: inline @font-face fonts as base64 ----------

_FONT_URL_RE = re.compile(r'url\(([^)]+)\)')


def inline_font_urls(css: str) -> str:
    """Rewrite `url(fonts/xxx.woff2)` references in CSS to base64 data URIs."""
    fonts_dir = ASSETS_DIR / "fonts"

    def repl(m: re.Match) -> str:
        raw = m.group(1).strip().strip("\"'")
        path_part = raw.split("?", 1)[0].split("#", 1)[0]
        rel = path_part
        if rel.startswith("fonts/"):
            font_path = ASSETS_DIR / rel
        else:
            font_path = fonts_dir / Path(rel).name
        if not font_path.is_file():
            return m.group(0)
        # Only inline woff2 (smallest). Non-woff2 refs in @font-face stay as
        # broken relative paths — harmless since WeasyPrint will fall back to
        # the working data URI listed in the same `src:`.
        if not font_path.name.endswith(".woff2"):
            return m.group(0)
        mime = "font/woff2"
        b64 = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return f'url(data:{mime};base64,{b64})'

    return _FONT_URL_RE.sub(repl, css)


def load_katex_css() -> str:
    css_path = ASSETS_DIR / "katex.min.css"
    if not css_path.is_file():
        raise FileNotFoundError(
            f"KaTeX CSS missing at {css_path}. Run install.sh first."
        )
    return inline_font_urls(css_path.read_text(encoding="utf-8"))


def load_print_css() -> str:
    return inline_font_urls(load_text(PRINT_CSS_PATH))


GITHUB_MARKDOWN_CSS_PATH = ASSETS_DIR / "github-markdown.css"
PRINT_CSS_PATH = ASSETS_DIR / "print.css"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# ---------- main pipeline ----------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{github_css}
</style>
<style>
{katex_css}
</style>
<style>
{code_css}
</style>
<style>
{print_css}
</style>
</head>
<body>
<article class="markdown-body">
{body}
</article>
</body>
</html>
"""


def render_markdown_to_html(md_path: Path) -> str:
    md_text = md_path.read_text(encoding="utf-8")
    masked, code_blocks, inline_codes, block_math, inline_math = mask_code_and_math(md_text)

    md = build_md()
    html = md.render(masked)

    # Render math through KaTeX in one Node invocation.
    math_items = [(t, True) for t in block_math] + [(t, False) for t in inline_math]
    rendered = render_math_via_katex(math_items)
    rendered_block = rendered[: len(block_math)]
    rendered_inline = rendered[len(block_math) :]

    # Restore block math (likely wrapped in <p>...</p> by markdown). Wrap in a
    # display-math div for cleaner spacing.
    for idx, html_math in enumerate(rendered_block):
        marker = _BLOCK_MARK.format(idx=idx)
        wrapped = f'<div class="katex-block">{html_math}</div>'
        # Replace whole paragraph if marker is alone in a <p>.
        html = re.sub(
            rf"<p>\s*{re.escape(marker)}\s*</p>", wrapped, html
        )
        html = html.replace(marker, wrapped)

    for idx, html_math in enumerate(rendered_inline):
        marker = _INLINE_MARK.format(idx=idx)
        html = html.replace(marker, html_math)

    # Restore inline code: render via markdown-it's inline rule manually since
    # we masked them out before parsing. Each placeholder becomes the original
    # `code` snippet wrapped in <code>...</code>.
    for idx, raw in enumerate(inline_codes):
        marker = _ICODE_MARK.format(idx=idx)
        inner = _INLINE_CODE_RE.match(raw)
        body = inner.group(2) if inner else raw
        html = html.replace(marker, f"<code>{escape_html(body)}</code>")

    # Restore fenced code blocks: render via Pygments.
    for idx, raw in enumerate(code_blocks):
        marker = _CODE_MARK.format(idx=idx)
        # Strip leading newline we kept in mask step.
        stripped = raw.lstrip("\n")
        m = re.match(r"(```+|~~~+)([^\n]*)\n(.*?)\n\1[ \t]*$", stripped, re.DOTALL)
        if m:
            lang = m.group(2).strip().split(" ")[0]
            code = m.group(3)
            replacement = highlight_code(code, lang)
        else:
            replacement = f"<pre><code>{escape_html(stripped)}</code></pre>"
        html = re.sub(
            rf"<p>\s*{re.escape(marker)}\s*</p>", replacement, html
        )
        html = html.replace(marker, replacement)

    # Inline local images.
    html = inline_images(html, md_path.parent)
    return html


def md_to_pdf(md_path: Path, dist_dir: Path) -> Path:
    from weasyprint import HTML, CSS

    body = render_markdown_to_html(md_path)
    full_html = HTML_TEMPLATE.format(
        title=md_path.stem,
        github_css=load_text(GITHUB_MARKDOWN_CSS_PATH),
        katex_css=load_katex_css(),
        code_css=CODE_CSS,
        print_css=load_print_css(),
        body=body,
    )
    dist_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dist_dir / f"{md_path.stem}.pdf"
    HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    return pdf_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Render markdown to GitHub-style PDF.")
    parser.add_argument("inputs", nargs="+", help="Markdown file(s) or glob patterns.")
    parser.add_argument(
        "--dist", default="dist", help="Output directory (default: ./dist relative to CWD)."
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    dist_dir = (cwd / args.dist).resolve()

    paths: list[Path] = []
    for pattern in args.inputs:
        if any(ch in pattern for ch in "*?["):
            expanded = sorted(glob.glob(pattern))
            if not expanded:
                print(f"[warn] no files match {pattern}", file=sys.stderr)
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    rc = 0
    for path in paths:
        if not path.is_file():
            print(f"[error] not a file: {path}", file=sys.stderr)
            rc = 1
            continue
        try:
            pdf = md_to_pdf(path, dist_dir)
            print(f"[ok] {path}  ->  {pdf.relative_to(cwd) if pdf.is_relative_to(cwd) else pdf}")
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
