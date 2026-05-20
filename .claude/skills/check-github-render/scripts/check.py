#!/usr/bin/env python3
"""Check that math regions in source markdown will render on github.com.

GitHub renders math in two passes:

  1. CommonMark / GFM parses the markdown. Inside `$...$` and `$$...$$`,
     backslash-escapes for specific punctuation are stripped (`\\_` -> `_`,
     `\\,` -> `,`, etc.) *before* MathJax sees the content.
  2. MathJax 3 receives the post-unescape TeX and renders.

This script replicates that pipeline against `src/<name>.md` files and
reports any of:

  - **MathJax parse errors** (e.g. `\\text{}` with bare `_`, malformed `\\frac`)
  - **Backslash-spacing macros eaten by CommonMark**: `\\,` `\\!` `\\;` `\\>`
    survive into MathJax on our local KaTeX pipeline but get stripped to
    literal punctuation by GitHub's CommonMark pass — silent visual bug
    that MathJax can't tell us about (it just renders the bare comma).
    We grep for these patterns inside math regions and flag them.

Exit code 0 if all files pass, 1 if any issue is reported.

Usage:
    python3 check.py src/<name>.md [more.md ...]
"""
from __future__ import annotations

import argparse
import glob
import html as html_lib
import json
import re
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
NODE_SCRIPT = SKILL_DIR / "scripts" / "mathjax-render.js"


# ---------- math extraction (position-preserving) ----------

# Same regex shapes as the markdown-to-pdf skill so the two checks agree on
# what counts as "a math region". Duplicated rather than imported because
# skills should be self-contained.
_FENCED_RE = re.compile(
    r"(^|\n)(?P<fence>```+|~~~+)[^\n]*\n.*?\n(?P=fence)[ \t]*(?=\n|$)",
    re.DOTALL,
)
_INLINE_CODE_RE = re.compile(r"(`+)(?!`)(.+?)\1", re.DOTALL)
_BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\d)")


def _neutralize_dollars_in_code(text: str) -> str:
    """Return `text` with every `$` *inside* code blocks / inline code
    replaced by a space. Same byte length, same newlines, so any byte
    offset we compute on this string can be reported as a line number in
    the original. Net effect: the math regexes never match `$` inside
    code, but everything outside code keeps its exact position.
    """
    def repl_fenced(m: re.Match) -> str:
        return m.group(0).replace("$", " ")

    text = _FENCED_RE.sub(repl_fenced, text)

    def repl_icode(m: re.Match) -> str:
        return m.group(0).replace("$", " ")

    return _INLINE_CODE_RE.sub(repl_icode, text)


def extract_math_with_positions(md_text: str) -> list[tuple[str, bool, int]]:
    """Return `[(tex, is_display, byte_offset), ...]` in document order.

    `byte_offset` is the position of the opening `$` in the original
    `md_text` — usable directly with `_offset_to_line` for reporting.
    """
    neutral = _neutralize_dollars_in_code(md_text)
    spans: list[tuple[int, int, str, bool]] = []
    block_ranges: list[tuple[int, int]] = []

    # Display math first so its `$$...$$` spans don't get mistaken for
    # two nested inline `$...$` matches.
    for m in _BLOCK_MATH_RE.finditer(neutral):
        spans.append((m.start(), m.end(), m.group(1).strip(), True))
        block_ranges.append((m.start(), m.end()))

    def inside_block(pos: int) -> bool:
        return any(s <= pos < e for s, e in block_ranges)

    for m in _INLINE_MATH_RE.finditer(neutral):
        if not inside_block(m.start()):
            spans.append((m.start(), m.end(), m.group(1).strip(), False))

    spans.sort(key=lambda x: x[0])
    return [(tex, disp, start) for start, _, tex, disp in spans]


def _offset_to_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


# ---------- GitHub pipeline simulation ----------

# Subset of CommonMark's backslash-escape that we know github.com actually
# applies inside math regions. Keep this *conservative* — it must only undo
# escapes that GitHub really undoes. The rest of the static-layer scan
# (`_COMMONMARK_EATS_RE` below) catches the symmetric class: backslash +
# punctuation that gets stripped silently and ruins TeX semantics.
_GITHUB_UNESCAPE_RE = re.compile(r"\\([_*$])")


def simulate_github_unescape(tex: str) -> str:
    """Undo the markdown-level escapes that GitHub strips before MathJax.
    Keeps every other backslash (commands like `\\frac`, `\\\\` line breaks,
    `\\{` literal braces) intact."""
    return _GITHUB_UNESCAPE_RE.sub(r"\1", tex)


# Backslash-escapes that CommonMark *also* eats but whose stripped form is
# semantically wrong. `\,` was a thin space, `,` is just a comma; `\|` was
# a double-bar ‖, `|` is single. MathJax can't tell us about any of this
# because by the time MathJax sees the input the macro is already gone —
# we have to detect it on the source. The leading `(?<!\\)` keeps `\\|`
# (line-break followed by pipe in an array) and `\\,` from false-firing.
_COMMONMARK_EATS_RE = re.compile(r"(?<!\\)\\([,!;>|])")

_BAD_SPACING_HINT = {
    ",": r"`\,` (thin space) — use `\thinspace`",
    "!": r"`\!` (negative thin space) — use `\negthinspace`",
    ";": r"`\;` (thick space) — use `\quad` / `\qquad`",
    ">": r"`\>` (medium space) — use `\quad` / `\qquad`",
    "|": r"`\|` becomes single `|` on GitHub — use `\Vert` for the double-bar ‖",
}


# ---------- markdown-layer checks (rules 1, 2, 9, 10, 11) ----------

# These run on the *source* markdown, not on the extracted math TeX,
# because the bugs they catch happen *before* MathJax sees anything —
# GitHub's CommonMark / GFM tokenizer either fails to recognize the math
# region in the first place, or mangles its contents at the markdown
# parsing layer. MathJax can't report on regions it never received.

# CJK ideograph blocks + CJK punctuation + halfwidth/fullwidth forms.
# Covers the chars that most commonly sit adjacent to `$...$` in this
# repo's notes: Han ideographs, full-width punctuation like `，。：（）「」`,
# and the `　` ideographic space.
_CJK_RE = re.compile(
    "["
    "　-〿"   # CJK symbols and punctuation (incl. ideographic space, 「」 etc.)
    "㐀-䶿"   # CJK unified ideographs ext A
    "一-鿿"   # CJK unified ideographs
    "＀-￯"   # halfwidth/fullwidth forms (incl. ，。：（）！？)
    "]"
)

# Rule 10: `<` / `>` literal inside a subscript/superscript brace group.
# `x_{<t}` (typical NLP "first t-1 tokens" notation) breaks because
# CommonMark treats `<t...` as a candidate HTML tag and confuses the
# parser. The regex looks at `_` or `^`, optional whitespace, an
# *unescaped* `{`, then any non-`}` content, then a literal `<` or `>`.
_SUBSCRIPT_ANGLE_RE = re.compile(r"[_^]\s*(?<!\\)\{[^}]*([<>])")

# Rule 11: `}_<letter>` inside inline math that also contains `[`. The
# combination flips CommonMark into thinking the `_` is an emphasis
# delimiter (right-flanking after `}`, paired with another `_` somewhere
# inside the brackets), and the whole math span fall back to plain text.
# Skip if the `_` is already escaped (`}\_X`) — that's the workaround.
_EMPHASIS_BAIT_RE = re.compile(r"(?<!\\)\}_([A-Za-z])")


def _is_table_row(line: str) -> bool:
    """Heuristic: a GFM table row starts with `|` and has ≥2 unescaped pipes."""
    s = line.strip()
    if not s.startswith("|"):
        return False
    return len(re.findall(r"(?<!\\)\|", s)) >= 2


# ---------- MathJax bridge ----------


def render_via_mathjax(items: list[tuple[str, bool]]) -> list[dict]:
    """Send `[(tex, display), ...]` to mathjax-render.js, one JSON per line.

    Returns parsed responses in the same order — each is either
    `{"ok": True}` or `{"error": "<message>"}`.
    """
    if not items:
        return []
    payload = "\n".join(
        json.dumps({"tex": tex, "display": display}, ensure_ascii=False)
        for tex, display in items
    )
    proc = subprocess.run(
        ["node", str(NODE_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"MathJax renderer failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    out = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(out) != len(items):
        raise RuntimeError(
            f"MathJax renderer returned {len(out)} lines, expected {len(items)}"
        )
    return [json.loads(ln) for ln in out]


# ---------- per-file check ----------

# Recognised `Issue.kind` values:
#   mathjax-error            — MathJax 3 refused to render the post-unescape TeX.
#   commonmark-eats-escape   — `\,` `\!` `\;` `\>` `\|` stripped by CommonMark.
#   block-math-not-isolated  — `$$...$$` not on its own line / no surrounding blanks.
#   cjk-adjacent-to-math     — CJK char or full-width punct touching `$`.
#   pipe-in-table-math       — literal `|` inside math on a GFM table row.
#   angle-in-subscript       — `<` / `>` inside `_{...}` / `^{...}`.
#   emphasis-eats-subscript  — `}_<letter>` inside inline math with `[`.


class Issue:
    __slots__ = ("line", "kind", "snippet", "message")

    def __init__(self, line: int, kind: str, snippet: str, message: str):
        self.line = line
        self.kind = kind
        self.snippet = snippet
        self.message = message

    def format_for(self, path: Path) -> str:
        # `path:line` is recognised by most editors/terminals as a jump target.
        return (
            f"{path}:{self.line}: [{self.kind}] {self.message}\n"
            f"    source: {self.snippet!r}"
        )


def check_file(
    path: Path,
    preview_pdf: Path | None = None,
) -> list[Issue]:
    md_text = path.read_text(encoding="utf-8")
    items = extract_math_with_positions(md_text)
    neutral = _neutralize_dollars_in_code(md_text)
    lines = md_text.splitlines()
    block_ranges = [(m.start(), m.end()) for m in _BLOCK_MATH_RE.finditer(neutral)]
    issues: list[Issue] = []

    def _in_block(pos: int) -> bool:
        return any(s <= pos < e for s, e in block_ranges)

    # ---- static layer: CommonMark-eats backslash-escapes ----
    # `\,` `\!` `\;` `\>` (silently stripped to bare punctuation) and `\|`
    # (silently stripped to single bar, breaking ‖ semantics). Done on the
    # *original* TeX, before our unescape simulation, because these are
    # precisely the patterns whose author-intended form is the backslash
    # version. If we unescape them first there'd be nothing to flag.
    for tex, _disp, offset in items:
        for m in _COMMONMARK_EATS_RE.finditer(tex):
            line = _offset_to_line(md_text, offset)
            hint = _BAD_SPACING_HINT.get(m.group(1), m.group(0))
            issues.append(
                Issue(
                    line=line,
                    kind="commonmark-eats-escape",
                    snippet=tex,
                    message=(
                        f"CommonMark will strip `{m.group(0)}` to `{m.group(1)}` "
                        f"before MathJax sees it. {hint}"
                    ),
                )
            )

    # ---- markdown-layer: `$$` block must stand alone (CLAUDE.md rule 1) ----
    # `$$...$$` only renders as a math block when both delimiters sit on
    # their own line and the block has blank lines around it. Otherwise
    # GitHub may inline it, or worse silently swallow it.
    for m in _BLOCK_MATH_RE.finditer(neutral):
        start, end = m.start(), m.end()
        start_line_idx = md_text.count("\n", 0, start)
        end_line_idx = md_text.count("\n", 0, end - 1)
        line_start = md_text.rfind("\n", 0, start) + 1
        before_text = md_text[line_start:start]
        nl_after_end = md_text.find("\n", end)
        if nl_after_end == -1:
            nl_after_end = len(md_text)
        after_text = md_text[end:nl_after_end]
        snippet = md_text[start:min(end, start + 80)]
        if before_text.strip():
            issues.append(Issue(
                start_line_idx + 1, "block-math-not-isolated", snippet,
                f"text before opening `$$` on same line "
                f"({before_text.strip()[:30]!r}) — `$$` must be on its own line.",
            ))
        elif after_text.strip():
            issues.append(Issue(
                end_line_idx + 1, "block-math-not-isolated", snippet,
                f"text after closing `$$` on same line "
                f"({after_text.strip()[:30]!r}) — `$$` must be on its own line.",
            ))
        else:
            if start_line_idx > 0 and lines[start_line_idx - 1].strip():
                issues.append(Issue(
                    start_line_idx + 1, "block-math-not-isolated", snippet,
                    "missing blank line before opening `$$`.",
                ))
            if end_line_idx < len(lines) - 1 and lines[end_line_idx + 1].strip():
                issues.append(Issue(
                    end_line_idx + 1, "block-math-not-isolated", snippet,
                    "missing blank line after closing `$$`.",
                ))

    # ---- markdown-layer: CJK adjacent to inline `$` (CLAUDE.md rule 2) ----
    # MathJax recognition needs a half-width space between CJK chars/punct
    # and the `$` delimiters. Walking the inline regex over the source
    # (not the extracted TeX) is the only way to see what's outside.
    for m in _INLINE_MATH_RE.finditer(neutral):
        if _in_block(m.start()):
            continue
        before_char = md_text[m.start() - 1] if m.start() > 0 else ""
        after_char = md_text[m.end()] if m.end() < len(md_text) else ""
        line_no = _offset_to_line(md_text, m.start())
        snippet = md_text[m.start():m.end()]
        if before_char and _CJK_RE.match(before_char):
            issues.append(Issue(
                line_no, "cjk-adjacent-to-math", snippet,
                f"CJK char {before_char!r} immediately before opening `$` — "
                f"insert a half-width space so MathJax recognizes the math.",
            ))
        if after_char and _CJK_RE.match(after_char):
            issues.append(Issue(
                line_no, "cjk-adjacent-to-math", snippet,
                f"CJK char {after_char!r} immediately after closing `$` — "
                f"insert a half-width space so MathJax recognizes the math.",
            ))

    # ---- markdown-layer: literal `|` inside math on a table row (rule 9) ----
    # GFM splits the table row at every `|` *before* MathJax sees it, so
    # `$a|b$` becomes two separate cells with broken math fragments.
    for line_idx, line in enumerate(lines):
        if not _is_table_row(line):
            continue
        for m in re.finditer(r"\$[^$\n]+\$", line):
            inner = m.group(0)[1:-1]
            if re.search(r"(?<!\\)\|", inner):
                issues.append(Issue(
                    line_idx + 1, "pipe-in-table-math", m.group(0),
                    "literal `|` in math on a table row — GFM splits the row "
                    "at the pipe before MathJax. Use `\\mid` or move the "
                    "formula to a block `$$…$$` outside the table.",
                ))

    # ---- markdown-layer: rules 10 & 11 on extracted math TeX ----
    for tex, is_display, offset in items:
        line_no = _offset_to_line(md_text, offset)
        # Rule 10: `<` / `>` literal inside `_{...}` / `^{...}`.
        for am in _SUBSCRIPT_ANGLE_RE.finditer(tex):
            sign = am.group(1)
            issues.append(Issue(
                line_no, "angle-in-subscript", tex[:80],
                f"`{sign}` inside subscript/superscript braces — GitHub may "
                f"treat it as the start of an HTML tag and break brace "
                f"matching. Use `\\lt` / `\\gt` (or rewrite e.g. `x_{{1:t-1}}`).",
            ))
        # Rule 11: `}_<letter>` workaround — only when the inline math
        # also contains a `[`, which is the form that actually flips
        # CommonMark into emphasis-parsing mode.
        if not is_display and "[" in tex:
            for em in _EMPHASIS_BAIT_RE.finditer(tex):
                issues.append(Issue(
                    line_no, "emphasis-eats-subscript", tex[:80],
                    f"`}}_{em.group(1)}` inside inline math containing `[…]` — "
                    f"CommonMark may consume the `_` as emphasis and drop "
                    f"all subscripts. Escape this `_` as `\\_` or move to "
                    f"block `$$…$$`.",
                ))

    # ---- render layer: MathJax 3 ----
    payload = [(simulate_github_unescape(tex), disp) for tex, disp, _ in items]
    results = render_via_mathjax(payload)
    for (tex, _disp, offset), res in zip(items, results):
        if "error" in res:
            line = _offset_to_line(md_text, offset)
            issues.append(
                Issue(
                    line=line,
                    kind="mathjax-error",
                    snippet=tex,
                    message=f"MathJax: {res['error']}",
                )
            )

    # ---- visual preview (opt-in) ----
    # Produce a contact-sheet PDF so a multimodal reviewer (or human) can
    # eyeball every formula's actual rendering without visiting github.com.
    # We deliberately render through the same MathJax 3 + AllPackages config
    # that GitHub uses, so the contact sheet is a high-fidelity preview of
    # what readers will see — modulo font rendering, which is glyph-as-path
    # in our SVG output (no font dependency) and TTF on github.com.
    if preview_pdf is not None:
        _write_contact_sheet(path, md_text, items, results, preview_pdf)

    return issues


# ---------- visual contact sheet ----------


_CONTACT_SHEET_CSS = """
@page { size: A4 landscape; margin: 10mm 8mm; }
body { font-family: sans-serif; font-size: 10px; color: #1f2328; }
h1 { font-size: 13px; margin: 0 0 6px 0; }
.subtitle { color: #6e7781; margin: 0 0 10px 0; font-size: 10px; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td { border: 1px solid #d0d7de; padding: 4px 6px; vertical-align: middle; }
th { background: #f6f8fa; text-align: left; font-weight: 600; }
.col-line { width: 6%; color: #6e7781; font-family: monospace; text-align: right; }
.col-src { width: 47%; font-family: monospace; font-size: 9px;
           word-break: break-all; line-height: 1.35; }
.col-rendered { width: 47%; text-align: center; overflow: hidden; }
.col-rendered mjx-container[jax="SVG"] { display: inline-block; }
.col-rendered mjx-container[jax="SVG"][display="true"] {
  display: block; margin: 0.2em auto;
}
/* Long display formulas (`$$...$$`) don't fit the 47% column. Give them the
   full row width — source on top, rendered SVG below — by colspan in the
   markup and these `.display-row` overrides. */
td.display-merged { padding: 6px 8px; }
td.display-merged .display-src {
  font-family: monospace; font-size: 9px;
  word-break: break-all; line-height: 1.35;
  background: #f6f8fa; padding: 4px 6px; border-radius: 3px;
  margin-bottom: 4px;
}
td.display-merged .display-rendered { text-align: center; }
td.display-merged .display-rendered mjx-container[jax="SVG"] {
  display: inline-block; max-width: 100%;
}
tr { page-break-inside: avoid; break-inside: avoid; }
.err { color: #cf222e; background: #ffebe9; }
.err-msg { font-family: monospace; font-size: 9px; color: #cf222e; }
"""


def _write_contact_sheet(
    src_path: Path,
    md_text: str,
    items: list[tuple[str, bool, int]],
    results: list[dict],
    out_pdf: Path,
) -> None:
    """Compose a one-table-per-page PDF showing source TeX vs the MathJax SVG.

    Lazy-imports weasyprint so the skill stays usable in environments that
    only need the text-check half.
    """
    from weasyprint import HTML

    rows: list[str] = []
    for (tex, is_display, offset), res in zip(items, results):
        line = _offset_to_line(md_text, offset)
        delim = "$$" if is_display else "$"
        src_html = (
            f'<code>{html_lib.escape(delim)}'
            f'{html_lib.escape(tex)}'
            f'{html_lib.escape(delim)}</code>'
        )
        if "error" in res:
            # Show the source flagged red and the error message stacked
            # below the (still-rendered) glyph fragment so the reviewer can
            # see what MathJax did with the bad input.
            rendered = (
                (res.get("html", "") or "")
                + f'<div class="err-msg">{html_lib.escape(res["error"])}</div>'
            )
            row_class = ' class="err"'
        else:
            rendered = res.get("html", "")
            row_class = ""

        if is_display:
            # Display formulas (`$$...$$`) often exceed the 47% column —
            # merge cols 2-3 and stack source above rendering so wide
            # formulas have the whole landscape page width to breathe.
            rows.append(
                f"<tr{row_class}>"
                f'<td class="col-line">L{line}</td>'
                f'<td colspan="2" class="display-merged">'
                f'<div class="display-src">{src_html}</div>'
                f'<div class="display-rendered">{rendered}</div>'
                f'</td></tr>'
            )
        else:
            rows.append(
                f"<tr{row_class}>"
                f'<td class="col-line">L{line}</td>'
                f'<td class="col-src">{src_html}</td>'
                f'<td class="col-rendered">{rendered}</td>'
                f"</tr>"
            )

    n_err = sum(1 for r in results if "error" in r)
    subtitle = (
        f"{src_path.name} — {len(items)} formulas rendered via MathJax 3 "
        f"(server-side, AllPackages). "
        f"{'No render errors.' if n_err == 0 else f'{n_err} render error(s) — highlighted red.'}"
    )
    doc = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{html_lib.escape(src_path.name)} — GitHub render preview</title>"
        f"<style>{_CONTACT_SHEET_CSS}</style></head><body>"
        f"<h1>GitHub render preview</h1>"
        f'<p class="subtitle">{html_lib.escape(subtitle)}</p>'
        '<table><thead><tr>'
        '<th class="col-line">Line</th>'
        '<th class="col-src">Source TeX</th>'
        '<th class="col-rendered">Rendered (MathJax 3)</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=doc).write_pdf(str(out_pdf))


# ---------- main ----------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Check that src/*.md math will render on github.com.",
    )
    parser.add_argument("inputs", nargs="+", help="Markdown file(s) or glob patterns.")
    parser.add_argument(
        "--visual",
        action="store_true",
        help=(
            "Also generate a 'contact sheet' PDF per input showing every math "
            "region's source TeX next to its MathJax 3 rendering. Hand the path "
            "to a multimodal reviewer (or open it yourself) to confirm formulas "
            "look right without visiting github.com."
        ),
    )
    parser.add_argument(
        "--preview-dir",
        default="dist/check-github-render",
        help="Output directory for --visual previews (default: dist/check-github-render).",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.inputs:
        if any(ch in pattern for ch in "*?["):
            expanded = sorted(glob.glob(pattern))
            if not expanded:
                print(f"[warn] no files match {pattern}", file=sys.stderr)
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    preview_dir = Path.cwd() / args.preview_dir if args.visual else None

    rc = 0
    total_files = 0
    total_issues = 0
    for path in paths:
        if not path.is_file():
            print(f"[error] not a file: {path}", file=sys.stderr)
            rc = 1
            continue
        total_files += 1
        preview_pdf = (preview_dir / f"{path.stem}.pdf") if preview_dir else None
        try:
            issues = check_file(path, preview_pdf=preview_pdf)
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            rc = 1
            continue

        if issues:
            rc = 1
            total_issues += len(issues)
            for issue in issues:
                print(issue.format_for(path))
        else:
            formula_count = len(extract_math_with_positions(
                path.read_text(encoding="utf-8")
            ))
            print(f"[ok] {path}  ({formula_count} formulas)")
        if preview_pdf is not None:
            print(f"       preview -> {preview_pdf}")

    if total_issues:
        print(f"\n[summary] {total_issues} issue(s) across {total_files} file(s)",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
