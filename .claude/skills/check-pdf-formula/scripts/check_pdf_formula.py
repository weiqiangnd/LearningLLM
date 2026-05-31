#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check-pdf-formula —— 把一章里每条数学公式的「源 TeX」与它在**真实 PDF**
（markdown-to-pdf 的 KaTeX→WeasyPrint 产物）里实际渲染出来的样子配成对，
导出成一份对照表（contact-sheet PDF）+ 机器可读的 mapping.json，供多模态
reviewer 逐条复核「含义一致 + 渲染正确」。

与 check-github-render --visual 的区别：那条把每条公式**孤立**地用 MathJax
重渲一遍（模拟 GitHub）；这条是**从整章 PDF 里按坐标把公式抠出来**——带真实
上下文、真实换行、真实字体回退、真实溢出，能抓到 WeasyPrint 侧才会裂开的坑
（≠ 的 rlap 叠加、下标视觉丢失、display 公式超出页宽等）。

用法见同目录 SKILL.md。核心流程：
  1. 解析章节 → src/<stem>.md、dist/<stem>.html
  2. （默认）若 dist HTML 缺失/比 src 旧，自动调 markdown-to-pdf 重渲
  3. 从 src md 抽公式拿「源行号 + 原始 TeX」（复用 check-github-render 的抽取器）
  4. WeasyPrint 渲染 HTML → 盒模型拿每条 .katex 的页码 + 包围盒，并由**同一次
     渲染**写出 PDF（坐标与盒模型保证一致）
  5. 断言 源公式数 == HTML annotation 数 == 盒子数
  6. 逐条 PyMuPDF 裁剪 + Pillow 自动裁白边 → crops/NNNN.png
  7. 产出 contact-sheet PDF（分片以适配 Read 20 页上限）+ mapping.json
"""
from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ── 依赖（install.sh 负责装）──────────────────────────────────────────────
try:
    import weasyprint
except ImportError:
    sys.exit("[fatal] 缺少 weasyprint，请先跑：bash .claude/skills/check-pdf-formula/scripts/install.sh")
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("[fatal] 缺少 PyMuPDF(fitz)，请先跑：bash .claude/skills/check-pdf-formula/scripts/install.sh")
try:
    from PIL import Image
except ImportError:
    sys.exit("[fatal] 缺少 Pillow，请先跑：bash .claude/skills/check-pdf-formula/scripts/install.sh")

# ── 复用 check-github-render 的公式抽取器（保持抽取正则与全仓一致）────────
SKILLS_DIR = Path(__file__).resolve().parents[2]
_CGR = SKILLS_DIR / "check-github-render" / "scripts"
sys.path.insert(0, str(_CGR))
try:
    from check import extract_math_with_positions, _offset_to_line  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit(f"[fatal] 无法导入 check-github-render 的抽取器（{_CGR}/check.py）：{e}")

# ── 复用 markdown-to-pdf 的渲染函数。md_to_dist(md_path, dist_dir) 支持
#    自定义输出目录——我们把目录指到本 skill 的复核目录，这样重渲只写进
#    dist/pdf-formula-check/<stem>/，**绝不覆盖 dist/ 下的章节三件套**。────
_MD2PDF_DIR = SKILLS_DIR / "markdown-to-pdf" / "scripts"
sys.path.insert(0, str(_MD2PDF_DIR))
try:
    from render import md_to_dist, MathInconsistency  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit(f"[fatal] 无法导入 markdown-to-pdf 的 md_to_dist（{_MD2PDF_DIR}/render.py）：{e}")

REPO = Path.cwd()
SRC_DIR = REPO / "src"
DIST_DIR = REPO / "dist"

# KaTeX 把每条公式的源 TeX 回显在 <annotation encoding="application/x-tex">…</annotation>
_ANNOTATION_RE = re.compile(
    r'<annotation\b[^>]*\bencoding=["\']application/x-tex["\'][^>]*>([^<]*)</annotation>'
)

CSS_PX_TO_PT = 72.0 / 96.0   # WeasyPrint 盒模型是 CSS px(96dpi)，PDF 是 pt(72dpi)
CROP_DPI = 300               # 裁剪渲染 DPI，文字够锐
CROP_ZOOM = CROP_DPI / 72.0
BOX_PAD_PX = 4               # 裁剪前在包围盒四周留的余量（CSS px），给 auto-trim 留白底
TRIM_MARGIN = 6              # auto-trim 后保留的墨迹外白边（裁剪图像素）
# 纵向溢出公式裁图时，把墨迹按行切带：行间空白超过这么多 CSS px 才算"断开"。
# 取值要 > 分式内部缝隙（分子↔分数线↔分母，实测 ~3-4px），又 < 相邻行间距
# （实测表格同列上下两式墨迹间 ~13px），这样只断在行与行之间、不切碎单条分式。
BAND_GAP_CSS = 8.0


# ──────────────────────────────────────────────────────────────────────────
# 章节解析
# ──────────────────────────────────────────────────────────────────────────
def resolve_chapter(arg: str) -> tuple[Path, str]:
    """把用户给的 '04' / 'P03' / '04-Embedding...' / 路径，解析成 (src_md_path, stem)。"""
    p = Path(arg)
    if p.suffix == ".md" and p.exists():
        return p.resolve(), p.stem
    # 当成章节编号 / 文件名前缀，在 src/ 里匹配
    key = p.stem if p.suffix else arg
    cands = sorted(SRC_DIR.glob("*.md"))
    # 1) 完全等于 stem
    for c in cands:
        if c.stem == key:
            return c, c.stem
    # 2) 以「编号-」开头（如 04 → 04-Embedding...；P03 → P03-概率...）
    pref = [c for c in cands if c.stem == key or c.stem.startswith(key + "-")]
    if len(pref) == 1:
        return pref[0], pref[0].stem
    if len(pref) > 1:
        sys.exit(f"[fatal] 章节 '{arg}' 匹配到多个：{[c.name for c in pref]}，请写得更具体。")
    sys.exit(
        f"[fatal] 在 {SRC_DIR} 里找不到章节 '{arg}'。"
        f" 可用：{[c.stem for c in cands]}"
    )


def prepare_source_html(src_md: Path, stem: str, out_dir: Path, do_render: bool) -> Path:
    """拿到用于抠图的 HTML，**永不写 dist/ 主文件**。

    - 默认（do_render=True）：用当前 src **重渲一份到本 skill 的复核目录**
      （out_dir），结果总是反映最新 src，且只落在 dist/pdf-formula-check/ 下。
    - --no-render（do_render=False）：只读复用现成的 dist/<stem>.html（快，
      但不重渲）；缺失则报错。这条路径同样不写任何东西到 dist/。
    """
    if not do_render:
        dist_html = DIST_DIR / f"{stem}.html"
        if not dist_html.exists():
            sys.exit(
                f"[fatal] {dist_html} 不存在，--no-render 下无可复用的 HTML。\n"
                f"        去掉 --no-render 让本 skill 自己渲一份（只写进复核目录，不动 dist 主文件），\n"
                f"        或先用 markdown-to-pdf 正式生成 dist 三件套。"
            )
        print(f"[reuse] --no-render：只读复用 {dist_html.relative_to(REPO)}（不重渲，可能不反映最新 src）")
        return dist_html

    print(f"[render] 用当前 src 重渲到复核目录（不覆盖 dist/ 主文件）：{out_dir.relative_to(REPO)}/ …")
    try:
        outs = md_to_dist(src_md, out_dir)   # 写 out_dir/<stem>.{md,html,pdf}
    except MathInconsistency as e:
        sys.exit(
            f"[fatal] 本章未过 markdown-to-pdf 的数学一致性校验，先修公式再复核：\n{e}"
        )
    return Path(outs["html"])


# ──────────────────────────────────────────────────────────────────────────
# 抽取：源 md 公式（带行号）/ HTML annotation / WeasyPrint 盒模型
# ──────────────────────────────────────────────────────────────────────────
def extract_src_formulas(src_md: Path) -> list[dict]:
    text = src_md.read_text(encoding="utf-8")
    items = extract_math_with_positions(text)  # [(tex, is_display, byte_offset)]
    out = []
    for tex, disp, off in items:
        out.append({
            "tex": tex.strip(),
            "display": bool(disp),
            "line": _offset_to_line(text, off),
        })
    return out


def extract_annotations(html: str) -> list[str]:
    return [html_lib.unescape(m.group(1)).strip() for m in _ANNOTATION_RE.finditer(html)]


def _iter_boxes(box):
    yield box
    for c in getattr(box, "children", []) or []:
        yield from _iter_boxes(c)


def extract_katex_boxes(html_path: Path):
    """返回 (boxes, pdf_bytes)。boxes 按文档序，每个 = dict(page,x,y,w,h,fragmented)（CSS px）。

    关键：盒模型与 PDF 由**同一次** render() 产出，坐标保证一致——所以不去裁
    现成的 dist PDF（可能与本次坐标系存在细微不一致），而是裁这次新渲的 PDF。

    这里只取每条公式最外层 `.katex` 元素的**行盒**几何。行盒高度有时会"夹住"
    溢出内容（行内 `\dfrac`、大根号等在表格单元格里实测只报 ~14px 行高），导致
    单纯按盒裁会切掉分子/分母——这一层由 `crop_formula` 的"触边检测 + 有界扩展"
    兜底，不在这里改盒几何（子树并集被 KaTeX 的 pstrut 撑得忽大忽小、不可靠）。

    分页/折行边界的坑：一条 `.katex` 公式若正好落在 WeasyPrint 的分页或软换行
    处，会被拆成**多个盒片段**（实测 P04 有 3 条 inline 公式跨页各拆成 2 片）。
    若按盒计数，会比"源 md 公式数 / HTML annotation 数"多出来、对不齐。所以这里
    **按元素身份 `id(element)` 归并**：同一公式的多个片段算一条，裁图取**面积最大
    的主片段**（跨页两片无法拼成单一矩形；inline 公式主片通常含绝大部分内容）。
    被拆片的公式标 `fragmented=True`，复核时提示去 rendered.pdf 看完整渲染。
    """
    doc = weasyprint.HTML(filename=str(html_path)).render()
    # 按"首次出现顺序"收集每个 katex 元素的所有盒片段，保住文档序。
    groups: "OrderedDict[int, list[dict]]" = OrderedDict()
    for pno, page in enumerate(doc.pages):
        for b in _iter_boxes(page._page_box):
            el = getattr(b, "element", None)
            if el is None:
                continue
            cls = (el.get("class", "") or "").split()
            # 只取最外层 .katex（每条公式恰一个元素，annotation 挂在它子树里）；
            # .katex-display / .katex-html / .katex-mathml 都是别的 token，不会命中。
            if "katex" in cls:
                groups.setdefault(id(el), []).append({
                    "page": pno,
                    "x": float(b.position_x), "y": float(b.position_y),
                    "w": float(b.width), "h": float(b.height),
                })

    boxes = []
    for frags in groups.values():
        main = max(frags, key=lambda f: f["w"] * f["h"])  # 面积最大的主片段
        main = dict(main)
        main["fragmented"] = len(frags) > 1
        boxes.append(main)

    pdf_bytes = doc.write_pdf()
    return boxes, pdf_bytes


# ──────────────────────────────────────────────────────────────────────────
# 裁剪
# ──────────────────────────────────────────────────────────────────────────
def _ink_row_bands(bw: "Image.Image", gap_px: int) -> list[tuple[int, int]]:
    """把二值墨迹图按行切成若干墨迹带（行间空白 ≤ gap_px 的并到一带）。

    返回 [(top_row, bottom_row), …]（含端，像素行号）。用于在纵向扩大的裁剪窗里
    把"本公式"那一带与混进来的相邻行墨迹 / 表格边框分开。
    """
    h = bw.height
    # 每行是否含墨迹：用 crop 单行的 getbbox 判定（快且不依赖 numpy）
    ink = [bw.crop((0, r, bw.width, r + 1)).getbbox() is not None for r in range(h)]
    bands: list[tuple[int, int]] = []
    r = 0
    while r < h:
        if not ink[r]:
            r += 1
            continue
        t = r
        while r < h and ink[r]:
            r += 1
        b = r - 1
        # 向后看：若与下一段墨迹的空白 ≤ gap_px，则并入同一带
        while r < h:
            gs = r
            while r < h and not ink[r]:
                r += 1
            if r < h and (r - gs) <= gap_px:
                while r < h and ink[r]:
                    r += 1
                b = r - 1
            else:
                r = gs
                break
        bands.append((t, b))
    return bands


def _binary_ink(img: "Image.Image") -> "Image.Image":
    """RGB → 墨迹二值图：墨迹处=255、背景=0（阈值避开抗锯齿淡灰）。"""
    return img.convert("L").point(lambda v: 0 if v > 250 else 255)


def _edge_ink(bw: "Image.Image", edge_px: int) -> tuple[bool, bool]:
    """裁剪图上 / 下边缘 edge_px 行内是否有墨迹 → 判断墨迹有没有被裁剪窗夹断。"""
    h = bw.height
    top = bw.crop((0, 0, bw.width, min(edge_px, h))).getbbox() is not None
    bot = bw.crop((0, max(0, h - edge_px), bw.width, h)).getbbox() is not None
    return top, bot


def crop_formula(pdf: "fitz.Document", box: dict, out_png: Path) -> dict:
    """把一条公式从真实 PDF 里裁成 PNG。

    难点：WeasyPrint 给行内 `.katex` 报的盒高有时只是**行高**，而 `\dfrac` / 大
    根号 / 高上下标的墨迹**溢出**了行盒——只按盒裁会切掉分子/分母，crop 看着像
    渲染坏了（实测 P04 表格内的 `\dfrac` 就是，曾误导复核）。

    两段式裁剪，既补全溢出又不误抓邻行：
      1) 先按行盒（+pad）渲一次，检测墨迹是否触到上/下边（= 被夹断）。
      2) 若触边，朝该方向**有界扩展约一个行高**（够补全 displaystyle 分式，又
         够不到表格同列上下相邻的公式）重渲；再按墨迹带切，只保留覆盖公式纵向
         中心那一带（排除扩展窗里混进来的表格行边框 / 邻行残墨）。
      3) auto-trim 收紧到墨迹 + 留白边。
    普通公式墨迹不触边、走不进第 2 步，裁图与原来逐像素一致——零回归。
    """
    page = pdf[box["page"]]
    pr = page.rect

    def render(x_css, y_css, w_css, h_css):
        x0 = max((x_css - BOX_PAD_PX) * CSS_PX_TO_PT, pr.x0)
        y0 = max((y_css - BOX_PAD_PX) * CSS_PX_TO_PT, pr.y0)
        x1 = min((x_css + w_css + BOX_PAD_PX) * CSS_PX_TO_PT, pr.x1)
        y1 = min((y_css + h_css + BOX_PAD_PX) * CSS_PX_TO_PT, pr.y1)
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(CROP_ZOOM, CROP_ZOOM), clip=clip, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), clip

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    img, clip = render(x, y, w, h)
    bw = _binary_ink(img)

    # 触边 → 墨迹被行盒夹断，朝溢出方向有界扩展重渲
    edge_px = max(2, round(2.0 * CSS_PX_TO_PT * CROP_ZOOM))   # ~2 CSS px 的边缘带
    et, eb = _edge_ink(bw, edge_px)
    expanded = et or eb
    if expanded:
        cap = max(h, 12.0)                       # 扩展上限：约一个行高
        ny = y - (cap if et else 0.0)
        nh = h + (cap if et else 0.0) + (cap if eb else 0.0)
        img, clip = render(x, ny, w, nh)
        bw = _binary_ink(img)
        # 扩展窗里可能混入一条表格行边框 / 邻行残墨：按墨迹带切，把"覆盖公式中心
        # 那一带"以外的区域涂白，只保留本公式（getbbox 再收紧）。
        gap_px = max(1, round(BAND_GAP_CSS * CSS_PX_TO_PT * CROP_ZOOM))
        bands = _ink_row_bands(bw, gap_px)
        if len(bands) > 1:
            cy_img = ((y + h / 2.0) * CSS_PX_TO_PT - clip.y0) * CROP_ZOOM

            def _dist(bd):
                t, b = bd
                return -1.0 if t <= cy_img <= b else min(abs(cy_img - t), abs(cy_img - b))

            bt, bb = min(bands, key=_dist)
            from PIL import ImageDraw
            d = ImageDraw.Draw(bw)
            if bt > 0:
                d.rectangle((0, 0, bw.width, bt - 1), fill=0)
            if bb + 1 < bw.height:
                d.rectangle((0, bb + 1, bw.width, bw.height), fill=0)

    # auto-trim：按（带选后的）墨迹裁紧，display 公式两侧大片留白也一并去掉
    bbox = bw.getbbox()
    if bbox:
        l, t, r, b = bbox
        l = max(0, l - TRIM_MARGIN); t = max(0, t - TRIM_MARGIN)
        r = min(img.width, r + TRIM_MARGIN); b = min(img.height, b + TRIM_MARGIN)
        img = img.crop((l, t, r, b))
    img.save(out_png)
    return {"png": out_png, "w": img.width, "h": img.height, "expanded": expanded}


# ──────────────────────────────────────────────────────────────────────────
# contact-sheet PDF
# ──────────────────────────────────────────────────────────────────────────
def _png_data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


CONTACT_CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", "Liberation Sans", sans-serif; color:#1f2937; }
h1 { font-size: 15pt; margin:0 0 2mm; }
.meta { font-size: 9pt; color:#374151; margin:0 0 4mm; }
table { width:100%; border-collapse: collapse; }
tr { page-break-inside: avoid; }
td { border:0.4pt solid #cbd5e1; padding:3pt 5pt; vertical-align: middle; }
.idx { width: 8%; font-size:9pt; color:#374151; text-align:center; }
.idx b { font-size:11pt; color:#111827; }
.src { width:46%; }
.ren { width:46%; text-align:center; background:#fafafa; }
.disp .src, .disp .ren { display:block; width:100%; }
code { font-family:"DejaVu Sans Mono","Liberation Mono",monospace; font-size:9pt;
       white-space:pre-wrap; word-break:break-word; color:#111827; }
.tag { display:inline-block; font-size:7.5pt; padding:0 4pt; border-radius:3pt;
       border:0.4pt solid; margin-left:4pt; }
.tag-d { color:#1d4ed8; border-color:#1d4ed8; }   /* display */
.tag-i { color:#374151; border-color:#9ca3af; }   /* inline */
.tag-f { color:#b45309; border-color:#b45309; }   /* fragmented across page */
img.f { max-width:100%; height:auto; }
.disprow td { background:#f8fafc; }
.dlabel { font-size:8.5pt; color:#1d4ed8; margin-bottom:2pt; }
"""


def build_contact_html(stem: str, rows: list[dict], part: int, nparts: int,
                       lo: int, hi: int, total: int) -> str:
    head = (f"<h1>公式渲染对照表 · {html_lib.escape(stem)}"
            + (f"（第 {part}/{nparts} 片）" if nparts > 1 else "") + "</h1>")
    meta = (f'<p class="meta">本片公式 {lo}–{hi} / 共 {total} 条。'
            f'左＝源 md 里的 TeX（作者所写，含 <code>\\_</code> 转义）；'
            f'右＝从真实 PDF 里抠出的实际渲染。逐条核对：①含义是否一致 '
            f'②有无渲染缺陷（下标/上标丢失、≠ 裂开、符号叠错、溢出截断）。</p>')
    trs = []
    for r in rows:
        idx = r["index"]; line = r["line"]; tex = html_lib.escape(r["tex"])
        uri = _png_data_uri(r["png_abs"])
        # 跨页/折行被拆片的公式：单图只是主片段，提示去 rendered.pdf 看全貌
        frag = ('<span class="tag tag-f">跨页·图可能不全</span>'
                if r.get("fragmented") else '')
        if r["display"]:
            trs.append(
                f'<tr class="disprow disp"><td colspan="3">'
                f'<div class="dlabel">#{idx} · src L{line} '
                f'<span class="tag tag-d">display $$</span>{frag}</div>'
                f'<div class="src"><code>{tex}</code></div>'
                f'<div class="ren"><img class="f" src="{uri}"></div>'
                f'</td></tr>'
            )
        else:
            trs.append(
                f'<tr>'
                f'<td class="idx"><b>{idx}</b><br>L{line}'
                f'<br><span class="tag tag-i">inline $</span>{frag}</td>'
                f'<td class="src"><code>{tex}</code></td>'
                f'<td class="ren"><img class="f" src="{uri}"></td>'
                f'</tr>'
            )
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CONTACT_CSS}</style></head><body>"
            f"{head}{meta}<table>{''.join(trs)}</table></body></html>")


def render_contact_sheets(stem: str, mapping: list[dict], out_dir: Path,
                          chunk: int) -> list[Path]:
    """把 mapping 切片，每片渲一个 PDF（≤ chunk 条公式），适配 Read 20 页上限。"""
    parts: list[Path] = []
    chunks = [mapping[i:i + chunk] for i in range(0, len(mapping), chunk)]
    nparts = len(chunks)
    for pi, rows in enumerate(chunks, 1):
        lo = rows[0]["index"]; hi = rows[-1]["index"]
        html = build_contact_html(stem, rows, pi, nparts, lo, hi, len(mapping))
        out = out_dir / (f"contact-sheet-{pi}.pdf" if nparts > 1 else "contact-sheet.pdf")
        weasyprint.HTML(string=html, base_url=str(out_dir)).write_pdf(str(out))
        parts.append(out)
    return parts


SHEET_PNG_DPI = 200  # contact-sheet 栅格化 DPI：再高一档，让 scriptscriptstyle 小字
                     # （二级下标如 \text{old} / \theta_\text{old}）缩到列宽后仍清晰，
                     # 避免被压糊成形似 ",," 的假象、引起复核误判


def rasterize_sheets(sheet_pdfs: list[Path]) -> list[Path]:
    """把每张 contact-sheet PDF 的每一页栅格化成 PNG。

    多模态复核统一读这些 PNG —— Read 工具对 PNG 的支持不依赖 poppler，
    比直接 Read PDF（需要 pdftoppm）更稳。PDF 仍保留给人眼看。
    """
    zoom = SHEET_PNG_DPI / 72.0
    pngs: list[Path] = []
    for pdf_path in sheet_pdfs:
        doc = fitz.open(str(pdf_path))
        for pno in range(len(doc)):
            pix = doc[pno].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            out = pdf_path.with_name(f"{pdf_path.stem}-p{pno + 1:02d}.png")
            pix.save(str(out))
            pngs.append(out)
        doc.close()
    return pngs


# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="把一章每条公式的源 TeX 与真实 PDF 渲染配对，导出对照表供多模态复核。")
    ap.add_argument("chapter", help="章节号/文件名/路径，如 04、P03、04-Embedding与位置编码 或 src/04-….md")
    ap.add_argument("--no-render", action="store_true",
                    help="不重渲，只读复用现成的 dist/<stem>.html（快，但可能不反映最新 src）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录，默认 dist/pdf-formula-check/<stem>/")
    ap.add_argument("--chunk", type=int, default=60,
                    help="每个 contact-sheet PDF 最多放多少条公式（默认 60，控制在 Read 20 页内）")
    args = ap.parse_args(argv)

    src_md, stem = resolve_chapter(args.chapter)
    out_dir = Path(args.out_dir) if args.out_dir else (DIST_DIR / "pdf-formula-check" / stem)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 默认重渲到 out_dir（不动 dist 主文件）；--no-render 时只读复用 dist html
    html_path = prepare_source_html(src_md, stem, out_dir, do_render=not args.no_render)

    # 1) 三路抽取
    src_items = extract_src_formulas(src_md)
    html_text = html_path.read_text(encoding="utf-8")
    anns = extract_annotations(html_text)
    boxes, pdf_bytes = extract_katex_boxes(html_path)

    # 2) 对齐校验
    n = len(src_items)
    if not (n == len(anns) == len(boxes)):
        sys.exit(
            "[fatal] 公式数对不上，HTML 可能过期或抽取错位：\n"
            f"        源 md 公式 = {n}\n"
            f"        HTML annotation = {len(anns)}\n"
            f"        WeasyPrint .katex 盒 = {len(boxes)}\n"
            f"        若用了 --no-render，去掉它让本 skill 用最新 src 重渲一份再试。"
        )
    if n == 0:
        print(f"[ok] {stem} 没有数学公式，无需复核。")
        return 0

    # 3) 逐条裁剪
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    rendered_pdf = out_dir / "rendered.pdf"
    rendered_pdf.write_bytes(pdf_bytes)

    mapping = []
    for i, (s, ann, box) in enumerate(zip(src_items, anns, boxes), 1):
        png = crops_dir / f"{i:04d}.png"
        cr = crop_formula(pdf, box, png)
        mapping.append({
            "index": i,
            "line": s["line"],
            "display": s["display"],
            "tex": s["tex"],                 # 源 md 原始 TeX（作者所写）
            "tex_rendered": ann,             # KaTeX 实际收到的 TeX（反转义后）
            "pdf_page": box["page"] + 1,     # 1-based 便于人看
            "fragmented": box.get("fragmented", False),  # 跨页/折行被拆片，单图可能不全
            "expanded": cr.get("expanded", False),       # 墨迹溢出行盒、裁图做过有界纵向扩展
            "bbox_csspx": [round(box["x"], 1), round(box["y"], 1),
                           round(box["w"], 1), round(box["h"], 1)],
            "png": str(png.relative_to(out_dir)),
            "png_abs": png,                  # 仅供内部嵌图，dump 前剔除
            "png_size": [cr["w"], cr["h"]],
        })

    # 4) mapping.json（剔除内部用的 Path 字段）
    map_path = out_dir / "mapping.json"
    formulas_json = [{k: v for k, v in m.items() if k != "png_abs"} for m in mapping]
    map_path.write_text(json.dumps({
        "chapter": stem, "src": str(src_md.relative_to(REPO)),
        "html": str(html_path.relative_to(REPO)),
        "rendered_pdf": str(rendered_pdf.relative_to(REPO)),
        "formula_count": n,
        "formulas": formulas_json,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) contact-sheet PDF（分片）+ 栅格化成 PNG（供多模态复核稳定读取）
    sheets = render_contact_sheets(stem, mapping, out_dir, args.chunk)
    sheet_pngs = rasterize_sheets(sheets)

    # 6) 总结
    rel = lambda p: p.relative_to(REPO)
    n_disp = sum(1 for m in mapping if m["display"])
    n_frag = sum(1 for m in mapping if m.get("fragmented"))
    frag_note = f"，其中 {n_frag} 条跨页拆片（单图可能不全，见 rendered.pdf）" if n_frag else ""
    print(f"\n[ok] {stem}：{n} 条公式（inline {n - n_disp} / display {n_disp}）{frag_note}")
    print(f"     映射文件 : {rel(map_path)}")
    print(f"     单条截图 : {rel(crops_dir)}/0001.png … {n:04d}.png")
    print(f"     真实 PDF : {rel(rendered_pdf)}")
    print(f"     对照表   : {len(sheets)} 份 PDF（给人看）+ {len(sheet_pngs)} 张 PNG（给多模态复核）")
    for s in sheet_pngs:
        print(f"                {rel(s)}")
    print("\n  下一步（多模态复核）：用 Read 工具逐张打开上面的 contact-sheet-*-p*.png，")
    print("  逐条比对左栏源 TeX 与右栏真实渲染，报告含义不一致或渲染缺陷的公式编号。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
