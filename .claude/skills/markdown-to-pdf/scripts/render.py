#!/usr/bin/env python3
"""Render a project markdown file to dist/ as a {md, html, pdf} triplet.

Usage:
    python3 render.py <input.md> [more.md ...]

For each input file `dir/<name>.md`, writes (all under ./dist/):
    dist/<name>.md     -- KaTeX-compatible markdown (math regions unescaped)
    dist/<name>.html   -- self-contained HTML, KaTeX rendered server-side
    dist/<name>.pdf    -- the same HTML run through WeasyPrint

Between producing the HTML and the PDF, the pipeline runs a math consistency
check that compares math regions across the three representations and aborts
PDF generation if any anomaly is detected (lost subscript, leftover
backslash-escape, KaTeX parse error, MathML/source mismatch).
"""
from __future__ import annotations

import argparse
import base64
import glob
import html as html_lib
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

# Combined regex used by `extract_math_in_order` to walk math regions in
# document order while keeping block-vs-inline distinguishable.
_ANY_MATH_RE = re.compile(
    r"(?<!\\)\$\$(?P<block>.+?)\$\$"
    r"|"
    r"(?<![\\$])\$(?!\s)(?P<inline>[^\n$]+?)(?<!\s)\$(?!\d)",
    re.DOTALL,
)


# Source TeX uses `\ne` (renders correctly on GitHub via MathJax). KaTeX
# *parses* `\ne` fine too, but renders it as a slash `\rlap`-overlaid on `=`,
# which WeasyPrint mis-positions in the PDF — slash leaks, `=` flung off or
# dropped, in scriptstyle subscripts the trailing `=0` even drops to a new line.
# Rewriting to `\mathrel{\char"2260}` pulls the U+2260 glyph straight from the
# font and bypasses the rlap, rendering a clean ≠ in the PDF. MathJax never
# sees this rewrite (the substitution happens only on the PDF pipeline), so
# GitHub keeps using `\ne` as written.
#
# Scope is intentionally narrow: only the canonical `\ne` / `\neq` forms.
# Non-canonical writings (`\not=`, literal `≠`, `\unicode{x2260}`, `\char...`)
# are flagged by check-github-render's `ne-non-canonical` rule and should be
# fixed in source rather than silently rewritten here. The `(?![A-Za-z])`
# lookahead keeps `\nearrow` and friends safe.
_NE_TO_CHAR_RE = re.compile(r"\\neq?(?![A-Za-z])")


def _ne_to_char_glyph(tex: str) -> str:
    """`\\ne` / `\\neq` → `\\mathrel{\\char"2260}` so KaTeX renders ≠ as a
    single glyph (and WeasyPrint can typeset it) instead of as the rlap'd
    slash-over-`=` it normally builds."""
    return _NE_TO_CHAR_RE.sub(r'\\mathrel{\\char"2260}', tex)


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
    # Then rewrite `\ne` / `\neq` → `\mathrel{\char"2260}` so the PDF gets a
    # clean ≠ glyph instead of KaTeX's WeasyPrint-breaking rlap. See
    # `_ne_to_char_glyph` above.
    return _ne_to_char_glyph(
        tex.replace(r"\_", "_")
           .replace(r"\*", "*")
           .replace(r"\$", "$")
    )


def _mask_code(md_text: str) -> tuple[str, list[str], list[str]]:
    """Replace fenced code blocks and inline code with opaque placeholders.

    Used by both `mask_code_and_math` (full pipeline) and `unescape_math_in_md`
    / `extract_math_in_order` (verification helpers) so neither touches `$`
    that happens to live inside a code span.
    """
    code_blocks: list[str] = []
    inline_codes: list[str] = []

    def repl_fenced(m: re.Match) -> str:
        idx = len(code_blocks)
        # The fenced-block regex anchors on the leading `^` or `\n` so it only
        # fires on a new line. We keep that newline in the masked text so
        # markdown still sees a paragraph break, and strip it from the saved
        # body so `_restore_code` doesn't reintroduce it — otherwise the dist
        # md ends up with an extra blank line before every code fence.
        leading = m.group(1) or ""
        body = m.group(0)[len(leading):]
        code_blocks.append(body)
        return leading + _CODE_MARK.format(idx=idx)

    text = _FENCED_RE.sub(repl_fenced, md_text)

    def repl_inline_code(m: re.Match) -> str:
        idx = len(inline_codes)
        inline_codes.append(m.group(0))
        return _ICODE_MARK.format(idx=idx)

    text = _INLINE_CODE_RE.sub(repl_inline_code, text)
    return text, code_blocks, inline_codes


def _restore_code(text: str, code_blocks: list[str], inline_codes: list[str]) -> str:
    for i, raw in enumerate(inline_codes):
        text = text.replace(_ICODE_MARK.format(idx=i), raw)
    for i, raw in enumerate(code_blocks):
        text = text.replace(_CODE_MARK.format(idx=i), raw)
    return text


def unescape_math_in_md(md_text: str) -> str:
    """Produce a KaTeX-compatible variant of `md_text` by unescaping `\\_`,
    `\\*`, `\\$` *inside math regions only*. Prose and code are byte-for-byte
    unchanged.

    Why this is a separate step now (instead of unescaping inline during the
    HTML pipeline as before): we emit this string to `dist/<name>.md` so the
    verification step can compare math regions across three representations
    (original md → KaTeX-compatible md → rendered HTML annotations). Keeping
    the unescape as a single, inspectable transformation also makes it
    auditable — you can diff `src/...md` vs `dist/...md` and the only
    differences should be `\\_ -> _` etc. within `$...$` blocks.
    """
    text, code_blocks, inline_codes = _mask_code(md_text)
    text = _BLOCK_MATH_RE.sub(
        lambda m: "$$" + _unescape_markdown_in_tex(m.group(1)) + "$$",
        text,
    )
    text = _INLINE_MATH_RE.sub(
        lambda m: "$" + _unescape_markdown_in_tex(m.group(1)) + "$",
        text,
    )
    return _restore_code(text, code_blocks, inline_codes)


def extract_math_in_order(md_text: str, unescape: bool = False) -> list[tuple[str, bool]]:
    """Walk `md_text` (after masking code) and return all math regions in
    document order as `[(tex, is_display), ...]`. The two helpers above are
    used to build the lists compared in `verify_math_consistency`."""
    text, _, _ = _mask_code(md_text)
    out: list[tuple[str, bool]] = []
    for m in _ANY_MATH_RE.finditer(text):
        if m.group("block") is not None:
            tex = m.group("block").strip()
            is_display = True
        else:
            tex = m.group("inline").strip()
            is_display = False
        if unescape:
            tex = _unescape_markdown_in_tex(tex)
        out.append((tex, is_display))
    return out


def mask_code_and_math(md_text: str):
    """Replace code blocks, inline code, and math with placeholders.

    Returns (masked_text, code_blocks, inline_codes, block_math, inline_math).
    Math content is unescaped (`\\_` -> `_` etc.) before being collected — by
    the time this is called in the HTML pipeline the input is the already
    KaTeX-compatible md, so the unescape is a no-op safety net.
    """
    text, code_blocks, inline_codes = _mask_code(md_text)
    block_math: list[str] = []
    inline_math: list[str] = []

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


# ---------- table cell post-processing ----------

# Cells that are "short and atomic" — short plain text with no whitespace
# and no break-opportunity punctuation — get `class="nowrap"`. This is the
# only reliable way to keep WeasyPrint's table auto-layout from laddering
# "状态空间" down a column: with `word-break: normal` every CJK char is a
# break opportunity (and WeasyPrint does NOT honor `word-break: keep-all`),
# so a column whose neighbours hold long prose collapses to ~1 char wide
# and the short label wraps to one char per line.
#
# We can't use a blanket CSS rule like `td:not(:last-child)` because some
# chapters have wide tables (6 cols, multiple long-prose non-last cells)
# where forcing all non-last cells to nowrap blows past the page width.
# The right discriminator is "does the cell content have any natural break
# points?" — which is per-cell, not per-column.

_CELL_RE = re.compile(r'<(td|th)\b([^>]*)>(.*?)</\1>', re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r'<[^>]+>')
# Whitespace + Western punctuation + CJK punctuation that all introduce a
# legitimate line-break opportunity in mixed CJK/Latin text.
# Note: ASCII hyphen `-` is intentionally NOT a break char — keep kebab-case
# identifiers like "encoder-only" / "decoder-only" together. Em-dash `—`
# and en-dash `–` remain break points because they're used as sentence-
# level dashes in the prose.
_BREAK_CHAR_RE = re.compile(
    r'[\s,.;:/+=\(\)\[\]\\<>'                    # ASCII punctuation (no `-`)
    r'，。、；：／（）【】「」『』《》〈〉—–·]'  # CJK punctuation + em/en dash
)
# Plain-text cells longer than this still get nowrap suppressed — at some
# point a "single word" is wide enough that nowrapping it would crowd out
# the rest of the table. 14 CJK chars ≈ 56mm at 14px, leaves room on A4.
_NOWRAP_MAX_CHARS = 14


def _cell_plain_text(inner_html: str) -> str:
    # Strip tags (including KaTeX math spans — their text content is the
    # rendered glyph stream, which is fine for length estimation, but it
    # has no spaces/punctuation that would mark a break point anyway, so
    # the cell still reads as "atomic" the way we want).
    no_tags = _TAG_RE.sub('', inner_html)
    return html_lib.unescape(no_tags).strip()


def _should_nowrap(plain: str) -> bool:
    if not plain:
        # Cell is image / pure math / empty — atomic, safe to nowrap.
        return True
    if len(plain) > _NOWRAP_MAX_CHARS:
        return False
    return not _BREAK_CHAR_RE.search(plain)


def annotate_short_table_cells(html: str) -> str:
    """Tag <td>/<th> cells whose content has no break opportunities with
    `class="nowrap"` so the print CSS can keep them on a single line."""
    def repl(m: re.Match) -> str:
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if not _should_nowrap(_cell_plain_text(inner)):
            return m.group(0)
        # Merge into existing class attribute if present, else add one.
        if re.search(r'\bclass\s*=', attrs):
            new_attrs = re.sub(
                r'class\s*=\s*"([^"]*)"',
                lambda c: f'class="{c.group(1)} nowrap"',
                attrs,
                count=1,
            )
            new_attrs = re.sub(
                r"class\s*=\s*'([^']*)'",
                lambda c: f"class='{c.group(1)} nowrap'",
                new_attrs,
                count=1,
            )
        else:
            new_attrs = attrs + ' class="nowrap"'
        return f'<{tag}{new_attrs}>{inner}</{tag}>'

    return _CELL_RE.sub(repl, html)


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


def render_markdown_to_html(md_text: str, base_dir: Path) -> str:
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
    #
    # Note on `re.sub` repl callable: the rendered KaTeX HTML carries literal
    # LaTeX source inside `<annotation encoding="application/x-tex">...`,
    # which contains command backslashes (`\leq`, `\log`, ...). Passing those
    # as a plain string to `re.sub` makes the regex engine try to interpret
    # `\l` etc. as a backreference escape and explode with "bad escape \l".
    # Using a lambda as repl bypasses that template processing entirely.
    for idx, html_math in enumerate(rendered_block):
        marker = _BLOCK_MARK.format(idx=idx)
        wrapped = f'<div class="katex-block">{html_math}</div>'
        # Replace whole paragraph if marker is alone in a <p>.
        html = re.sub(
            rf"<p>\s*{re.escape(marker)}\s*</p>", lambda _m, w=wrapped: w, html
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
            rf"<p>\s*{re.escape(marker)}\s*</p>",
            lambda _m, r=replacement: r,
            html,
        )
        html = html.replace(marker, replacement)

    # Mark short / atomic table cells with class="nowrap".
    html = annotate_short_table_cells(html)

    # Wrap the in-body TOC so `@media print` can hide it from the PDF
    # (PDF viewers get their navigation from WeasyPrint-generated bookmarks
    # instead). No-op for chapters without a `## 目录` section.
    html = wrap_toc_section(html)

    # Inline local images.
    html = inline_images(html, base_dir)
    return html


# Matches the in-body TOC region: an `## 目录` heading, the bullet list that
# follows (markdown-it renders nested ULs without intervening text), and the
# trailing `<hr />` that the `---` rule in the source markdown becomes.
# Stays a `count=1` substitution since chapters only ever have one TOC.
_TOC_SECTION_RE = re.compile(
    r'(<h2[^>]*>\s*目录\s*</h2>\s*<ul\b.*?</ul>\s*<hr\s*/?>)',
    re.DOTALL,
)


def wrap_toc_section(html: str) -> str:
    """Wrap the body-level TOC block in `<div class="toc-section">…</div>` so
    the print stylesheet can hide it. We intentionally keep the wrapping
    tight (h2 + first <ul> + trailing <hr>) rather than greedy-matching all
    the way to the next h2 — that way a chapter that for some reason puts
    real content between its TOC and first section still survives unscathed.

    Returns the HTML unchanged if no TOC is found (e.g. shorter notes that
    skip the convention). The `<hr>` is included so the PDF doesn't end up
    with a dangling horizontal rule above the first body section.
    """
    return _TOC_SECTION_RE.sub(
        lambda m: f'<div class="toc-section">{m.group(1)}</div>',
        html,
        count=1,
    )


# ---------- math consistency verification ----------

# Strip every HTML tag — used to inspect rendered visual spans / MathML for
# residue (literal `\_`, leftover `\,`, etc.).
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Pull the source TeX KaTeX echoes back inside `<annotation>` for every
# rendered formula. Content is HTML-entity encoded. The pattern is
# deliberately tolerant of attribute reordering / quote style so a future
# KaTeX upgrade doesn't silently break the roundtrip check.
_ANNOTATION_RE = re.compile(
    r'<annotation\b[^>]*\bencoding=["\']application/x-tex["\'][^>]*>'
    r'([^<]*)</annotation>'
)
# KaTeX failure signatures in the rendered HTML body:
#  - `katex-error`: emitted when KaTeX gives up entirely (no MathML at all).
#  - `class="math-error"`: our Python-side fallback when render_math.js
#    reports `{"error": "..."}` for a snippet.
#  - `mathcolor="#cc0000"` / `color:#cc0000;`: KaTeX's "silent failure" red
#    used in non-strict mode for unknown commands. The color is hard-coded
#    in KaTeX's source — matching the hex string is the most reliable signal.
# Scoped to the body (see `_html_body_slice`) so the embedded CSS bundles
# (`github-markdown.css` etc.) can't false-positive — and so a future KaTeX
# CSS that legitimately names `.katex-error` won't either.
_KATEX_ERROR_MARKERS = (
    "katex-error",
    'class="math-error"',
    'mathcolor="#cc0000"',
    "color:#cc0000",
)


def _html_body_slice(html: str) -> str:
    """Return just the rendered article portion of the full HTML — everything
    between `<body>` and `</body>`. We use this for the error-marker scan so
    KaTeX class names that happen to live inside embedded CSS rules don't get
    misread as real KaTeX failures."""
    m = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
    return m.group(1) if m else html


class MathInconsistency(RuntimeError):
    """Raised when the cross-stage math consistency check fails. The message
    lists every detected issue so a single run reports as much as possible
    instead of dying on the first one."""


def _normalize_for_compare(tex: str) -> str:
    """KaTeX trims a bit of whitespace before echoing source into the
    `<annotation>` payload, and the markdown extraction also `.strip()`s the
    captured group. Collapse runs of whitespace so the two sides line up even
    when one has `\\;` rendered as a single space and the other kept a tab."""
    return re.sub(r"\s+", " ", tex).strip()


def verify_math_consistency(
    original_md: str,
    katex_md: str,
    html: str,
) -> None:
    """Raise `MathInconsistency` if math regions across the three stages do
    not line up, or if KaTeX produced any output that looks like a rendering
    anomaly.

    Stages:
      1. `original_md` — what the author wrote, may contain markdown-level
         escapes like `\\_` inside `$...$`.
      2. `katex_md` — what we wrote to dist/ after `unescape_math_in_md`.
      3. `html` — what KaTeX emitted, carrying the source TeX it actually
         parsed inside `<annotation encoding="application/x-tex">...</>`.

    Checks:
      A. Region count matches across all three.
      B. For each position, `unescape(original[i])` == `katex_md[i]` (the
         transformation we promised to do is exactly what we did).
      C. For each position, `katex_md[i]` == decoded annotation source from
         `html[i]` (KaTeX parsed exactly what we fed it; nothing dropped).
      D. No leftover `\\_` / `\\*` reached KaTeX (would render as literal
         underscores / asterisks instead of subscripts / emphasis).
      E. No KaTeX error markers in the HTML.
      F. No literal underscore glyph emitted as `<mi mathvariant="normal">_</mi>`
         in the MathML (signature of a lost subscript even when the source
         already had a bare `_`).
    """
    issues: list[str] = []

    orig = extract_math_in_order(original_md, unescape=False)
    katex = extract_math_in_order(katex_md, unescape=False)

    # Annotations come out of the rendered HTML in document order. Decode the
    # HTML entities KaTeX wrote into the payload (`&amp;`, `&lt;`, etc.).
    annotations = [
        html_lib.unescape(s) for s in _ANNOTATION_RE.findall(html)
    ]

    # --- A. region counts ---
    if not (len(orig) == len(katex) == len(annotations)):
        issues.append(
            f"math region count mismatch: original={len(orig)} "
            f"katex_md={len(katex)} html_annotations={len(annotations)}"
        )

    # --- B. original -> katex_md transformation ---
    for i, ((o_tex, o_disp), (k_tex, k_disp)) in enumerate(zip(orig, katex)):
        expected = _unescape_markdown_in_tex(o_tex)
        if expected != k_tex:
            issues.append(
                f"#{i} ({'block' if o_disp else 'inline'}): "
                f"unescape(original) != katex_md\n"
                f"    original: {o_tex!r}\n"
                f"    expected: {expected!r}\n"
                f"    katex_md: {k_tex!r}"
            )
        if o_disp != k_disp:
            issues.append(
                f"#{i}: display-mode flipped between original and katex_md "
                f"(orig={o_disp}, katex={k_disp})"
            )

    # --- C. katex_md -> html annotation roundtrip ---
    for i, ((k_tex, k_disp), a_tex) in enumerate(zip(katex, annotations)):
        if _normalize_for_compare(k_tex) != _normalize_for_compare(a_tex):
            issues.append(
                f"#{i} ({'block' if k_disp else 'inline'}): "
                f"katex_md != html annotation\n"
                f"    katex_md  : {k_tex!r}\n"
                f"    annotation: {a_tex!r}"
            )

    # --- D. leftover markdown escapes inside annotations ---
    for i, a_tex in enumerate(annotations):
        # `\_` and `\*` are never used as real LaTeX commands in this repo's
        # math; their presence here means the unescape pass missed something
        # (or new content was written that needs unescaping).
        leftover = []
        if r"\_" in a_tex:
            leftover.append(r"\_")
        if r"\*" in a_tex:
            leftover.append(r"\*")
        if leftover:
            issues.append(
                f"#{i}: KaTeX received leftover markdown escapes "
                f"{leftover} — subscripts/emphasis will render as literal "
                f"characters.\n    source: {a_tex!r}"
            )

    body = _html_body_slice(html)

    # --- E. KaTeX error spans (body-scoped, see `_html_body_slice` rationale) ---
    for marker in _KATEX_ERROR_MARKERS:
        if marker in body:
            issues.append(
                f"KaTeX error marker {marker!r} present in rendered HTML body "
                "— at least one formula failed to parse."
            )
            break

    # --- F. literal-underscore-as-letter in MathML (lost subscript signature)
    # `\_X` produces `<mi mathvariant="normal">_</mi><mi>X</mi>` in KaTeX's
    # MathML. The visual span has the same signature: an `_` glyph rendered
    # at baseline as an `mord`. We catch it via the MathML tag pattern, which
    # is unambiguous — a real subscript would be `<msub>...</msub>` and the
    # underscore would never appear as `<mi>_</mi>`.
    if re.search(r'<mi[^>]*>_</mi>', body):
        issues.append(
            "MathML contains `<mi>_</mi>` — a literal underscore glyph "
            "where a subscript was almost certainly intended. Check the "
            "source for un-unescaped `\\_` or a typo around `_`."
        )

    if issues:
        raise MathInconsistency(
            "math consistency check failed ({} issue{}):\n  - {}".format(
                len(issues),
                "" if len(issues) == 1 else "s",
                "\n  - ".join(issues),
            )
        )


# ---------- top-level pipeline: md -> {md, html, pdf} triplet ----------


def md_to_dist(md_path: Path, dist_dir: Path) -> dict:
    """Run the full pipeline for one input md and write the three sibling
    artifacts under `dist_dir`. Returns paths keyed by extension.

    Order matters: write the KaTeX-compatible md and the HTML first so they
    are available for inspection (and committed via the verification step)
    before the relatively expensive WeasyPrint PDF pass kicks in. If the
    consistency check fails we skip the PDF entirely — there's no point
    burning ~10 s to render a PDF whose math we've already proven wrong.
    """
    from weasyprint import HTML

    dist_dir.mkdir(parents=True, exist_ok=True)
    stem = md_path.stem

    original_md = md_path.read_text(encoding="utf-8")
    katex_md = unescape_math_in_md(original_md)

    md_out = dist_dir / f"{stem}.md"
    md_out.write_text(katex_md, encoding="utf-8")

    body = render_markdown_to_html(katex_md, md_path.parent)
    full_html = HTML_TEMPLATE.format(
        title=stem,
        github_css=load_text(GITHUB_MARKDOWN_CSS_PATH),
        katex_css=load_katex_css(),
        code_css=CODE_CSS,
        print_css=load_print_css(),
        body=body,
    )
    html_out = dist_dir / f"{stem}.html"
    html_out.write_text(full_html, encoding="utf-8")

    # Cross-stage math consistency. Raises on first batch of issues; we let
    # the exception bubble up to `main`, which logs and sets exit code 1.
    verify_math_consistency(original_md, katex_md, full_html)

    pdf_out = dist_dir / f"{stem}.pdf"
    HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_out))

    return {"md": md_out, "html": html_out, "pdf": pdf_out}


def assert_emoji_fonts_sane() -> None:
    """Fail fast if the bitmap 'Noto Color Emoji' font is selectable.

    WeasyPrint/Pango cannot draw CBDT/CBLC bitmap glyphs: the advance width
    is reserved but nothing is painted, so ✅/❌/⚠ silently vanish from the
    PDF while text extraction still finds them — this shipped broken PDFs
    once and is near-impossible to spot downstream. install.sh applies two
    defenses (apt remove + a fontconfig rejectfont rule); if fc-list still
    sees the font, neither took effect, so abort instead of rendering
    invisible emoji."""
    try:
        out = subprocess.run(
            ["fc-list"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return  # fc-list unavailable — nothing we can verify here
    if re.search(r"noto color emoji", out, re.I):
        sys.exit(
            "[render] ERROR: bitmap font 'Noto Color Emoji' is visible to "
            "fontconfig — WeasyPrint cannot draw its glyphs, so emoji like "
            "✅/❌ would be INVISIBLE in the PDF. Re-run install.sh (it "
            "removes the package and installs a fontconfig rejectfont rule), "
            "then retry."
        )
    if not re.search(r"symbola", out, re.I):
        print(
            "[render] warn: Symbola not installed — ✅/❌ may render as tofu; "
            "run install.sh.",
            file=sys.stderr,
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Render markdown into a {md, html, pdf} triplet under dist/.",
    )
    parser.add_argument("inputs", nargs="+", help="Markdown file(s) or glob patterns.")
    parser.add_argument(
        "--dist", default="dist", help="Output directory (default: ./dist relative to CWD)."
    )
    args = parser.parse_args(argv)

    assert_emoji_fonts_sane()

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
            outs = md_to_dist(path, dist_dir)
        except MathInconsistency as e:
            # The .md and .html were already written; we keep them so the
            # author can inspect what KaTeX received without re-running.
            print(f"[error] {path}: {e}", file=sys.stderr)
            rc = 1
            continue
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            rc = 1
            continue

        def _rel(p: Path) -> str:
            return str(p.relative_to(cwd)) if p.is_relative_to(cwd) else str(p)

        print(
            f"[ok] {path}\n"
            f"       md  -> {_rel(outs['md'])}\n"
            f"       html-> {_rel(outs['html'])}\n"
            f"       pdf -> {_rel(outs['pdf'])}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
