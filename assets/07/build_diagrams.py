"""Generate conceptual diagrams for chapter 07 (Multi-Head Attention 与 MQA / GQA).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
Hand-written SVG (same approach as assets/06/build_diagrams.py) so the layout
can be tuned precisely. Body / sublabel / caption text uses gray-700 (#374151)
or darker per the repo contrast guideline.

Requires the Noto Sans CJK font for clean Chinese glyphs:
    apt-get install -y fonts-noto-cjk

Run from repo root:
    python3 assets/07/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant. Export width per
diagram (default 2400; sparse/few-node -> 1800; many-node/very-wide -> 3000):
    rsvg-convert -w 2400 assets/07/mha-overview.svg  -o /tmp/x.png   # 常规横向流程
    rsvg-convert -w 3000 assets/07/shape-flow.svg    -o /tmp/x.png   # 极宽 7 框横向流程
    rsvg-convert -w 3000 assets/07/mha-mqa-gqa.svg   -o /tmp/x.png   # 30+ 小节点密集图
    pngquant --quality=100 --strip --force --output assets/07/<name>.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("'Noto Sans CJK SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', "
        "'WenQuanYi Zen Hei', sans-serif")
MONO = ("'Noto Sans Mono CJK SC', 'SFMono-Regular', 'Consolas', "
        "'Liberation Mono', monospace")

BG = "#ffffff"
TXT = "#374151"   # gray-700  primary labels
SUB = "#334155"   # slate-700  secondary / sublabel / caption

BLUE_F, BLUE_B = "#dbeafe", "#2563eb"
GREEN_F, GREEN_B = "#dcfce7", "#059669"
ORANGE_F, ORANGE_B = "#ffedd5", "#ea580c"
PURPLE_F, PURPLE_B = "#ede9fe", "#7c3aed"
RED_F, RED_B = "#fee2e2", "#dc2626"
GRAY_F, GRAY_B = "#f3f4f6", "#94a3b8"
TEAL_F, TEAL_B = "#ccfbf1", "#0d9488"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg(path: Path, body: str, viewbox: str):
    style = f"""
  <style>
    text {{ font-family: {FONT}; }}
    .title  {{ font-size: 24px; font-weight: 700; fill: {TXT}; }}
    .lbl    {{ font-size: 16px; font-weight: 600; fill: {TXT}; }}
    .sub    {{ font-size: 14px; fill: {SUB}; }}
    .mono   {{ font-size: 15px; font-family: {MONO}; fill: {TXT}; }}
    .monob  {{ font-size: 17px; font-weight: 700; font-family: {MONO}; fill: {TXT}; }}
    .monos  {{ font-size: 14px; font-family: {MONO}; fill: {TXT}; }}
    .small  {{ font-size: 14px; fill: {SUB}; }}
    .cap    {{ font-size: 14.5px; fill: {SUB}; }}
    .tag    {{ font-size: 13px; font-weight: 600; }}
  </style>
"""
    defs = f"""
  <defs>
    <marker id="aBlue" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{BLUE_B}"/></marker>
    <marker id="aGray" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{GRAY_B}"/></marker>
    <marker id="aOrange" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{ORANGE_B}"/></marker>
    <marker id="aGreen" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{GREEN_B}"/></marker>
    <marker id="aPurple" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{PURPLE_B}"/></marker>
    <marker id="aTeal" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{TEAL_B}"/></marker>
  </defs>
"""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">\n'
        f'  <rect width="100%" height="100%" fill="{BG}"/>\n'
        f'{defs}{style}{body}\n'
        f'</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def node(cx, cy, w, h, label, sub, fill, border, label_cls="lbl"):
    x, y = cx - w / 2, cy - h / 2
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="{fill}" stroke="{border}" stroke-width="2"/>'
    ]
    if sub:
        out.append(f'<text x="{cx}" y="{cy-4}" text-anchor="middle" class="{label_cls}">{esc(label)}</text>')
        out.append(f'<text x="{cx}" y="{cy+16}" text-anchor="middle" class="small">{esc(sub)}</text>')
    else:
        out.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" class="{label_cls}">{esc(label)}</text>')
    return out


def arrow(x1, y1, x2, y2, marker="aBlue", color=BLUE_B, dashed=False, width=2):
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    return [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{dash} marker-end="url(#{marker})"/>']


def _text_width(text, cjk_px, ascii_px):
    """Estimate rendered text width: CJK glyphs (incl. full-width punct, ①②③)
    are ~full em wide in Noto, ASCII letters/digits/spaces are ~half. The old
    flat `len*8` under-counted CJK badly -> tags hugged their box edges."""
    w = 0.0
    for ch in text:
        # CJK ideographs, full-width punctuation, circled digits, CJK symbols
        if ord(ch) > 0x2460:
            w += cjk_px
        else:
            w += ascii_px
    return w


def tag(cx, y, text, fill, border):
    # tag class is 13px; pad generously so CJK labels breathe inside the pill
    w = max(64, _text_width(text, cjk_px=13.5, ascii_px=7.5) + 30)
    return [
        f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="24" rx="7" '
        f'fill="{fill}" stroke="{border}" stroke-width="1.3"/>',
        f'<text x="{cx}" y="{y+16.5}" text-anchor="middle" class="tag" fill="{border}">{esc(text)}</text>'
    ]


# ============================================================
# Diagram 1: Multi-Head Attention 全景
# ============================================================
def diagram_overview():
    W, H = 1860, 760
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'Multi-Head Attention 全景：投影切头 → H 个头并行 attention → 拼接 → 输出投影 W_O</text>']

    cy = 380
    # input X
    xc = 130
    p += node(xc, cy, 150, 100, "X", "[B, L, d_model]", BLUE_F, BLUE_B, label_cls="monob")
    p.append(f'<text x="{xc}" y="{cy+64}" text-anchor="middle" class="small">输入序列</text>')

    # split / projection stage
    sx = 380
    p += arrow(xc + 75, cy, sx - 95, cy)
    p += node(sx, cy, 190, 110, "W_Q / W_K / W_V", "投影 + 切头", PURPLE_F, PURPLE_B, label_cls="mono")
    p.append(f'<text x="{sx}" y="{cy+40}" text-anchor="middle" class="small">→ Q,K,V 各 [B,H,L,d_k]</text>')

    # H parallel heads (stack)
    hx = 760
    head_ys = [cy - 195, cy - 65, cy + 65, cy + 195]
    head_labels = ["head 1", "head 2", "head 3", "head H"]
    for i, (hy, hl) in enumerate(zip(head_ys, head_labels)):
        # arrow from split stage to each head
        p += arrow(sx + 95, cy, hx - 130, hy, marker="aPurple", color=PURPLE_B,
                   width=1.7)
        if i == 3:
            # dots between head 3 and head H
            p.append(f'<text x="{hx}" y="{(head_ys[2]+head_ys[3])/2+6}" '
                     f'text-anchor="middle" class="monob" fill="{SUB}">⋮</text>')
        p += node(hx, hy, 260, 96, hl,
                  "softmax(QₕKₕᵀ/√d_k)·Vₕ", TEAL_F, TEAL_B, label_cls="lbl")
        p.append(f'<text x="{hx}" y="{hy+34}" text-anchor="middle" class="monos" fill="{SUB}">[B, L, d_k]</text>')

    # concat
    cxx = 1230
    for hy in head_ys:
        p += arrow(hx + 130, hy, cxx - 60, cy, marker="aTeal", color=TEAL_B, width=1.7)
    p += node(cxx, cy, 120, 150, "Concat", "拼回 d_model", ORANGE_F, ORANGE_B, label_cls="lbl")
    p.append(f'<text x="{cxx}" y="{cy+44}" text-anchor="middle" class="monos" fill="{SUB}">[B,L,d_model]</text>')

    # W_O
    wox = 1500
    p += arrow(cxx + 60, cy, wox - 75, cy, marker="aOrange", color=ORANGE_B)
    p += node(wox, cy, 150, 100, "W_O", "输出投影 / 融合", GREEN_F, GREEN_B, label_cls="monob")

    # output
    ox = 1760
    p += arrow(wox + 75, cy, ox - 60, cy, marker="aGreen", color=GREEN_B)
    p += node(ox, cy, 120, 100, "O", "[B,L,d_model]", PURPLE_F, PURPLE_B, label_cls="monob")
    p.append(f'<text x="{ox}" y="{cy+64}" text-anchor="middle" class="small">与输入同形</text>')

    # stage tags on top
    p += tag(sx, cy - 110 - 28, "① 切头", PURPLE_F, PURPLE_B)
    p += tag(hx, head_ys[0] - 48 - 28, "② 各头独立 attention", TEAL_F, TEAL_B)
    p += tag(cxx, cy - 75 - 28, "③ 拼接", ORANGE_F, ORANGE_B)
    p += tag(wox, cy - 50 - 28, "④ 输出投影", GREEN_F, GREEN_B)

    # footer
    p.append(f'<rect x="{W/2-560}" y="{H-72}" width="1120" height="46" rx="10" '
             f'fill="{GRAY_F}" stroke="{GRAY_B}" stroke-width="1.4"/>')
    p.append(f'<text x="{W/2}" y="{H-42}" text-anchor="middle" class="mono">'
             f'MultiHead(X) = Concat(head₁, …, head_H) · W_O,    head_h = softmax(QₕKₕᵀ / √d_k) · Vₕ</text>')
    write_svg(ASSETS / "mha-overview.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: 形状变换全过程
# ============================================================
def diagram_shape_flow():
    W, H = 2340, 480
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'多头注意力形状变换全过程：reshape / transpose 切头 → attention → transpose / reshape 合头</text>']

    cy = 205
    bw, bh = 232, 96
    xs = [165, 500, 855, 1210, 1565, 1900, 2175]
    boxes = [
        ("[B, L, d_model]", "输入", BLUE_F, BLUE_B),
        ("[B, L, H, d_k]", "劈开 d_model = H×d_k", PURPLE_F, PURPLE_B),
        ("[B, H, L, d_k]", "头维提前 → 可并行", TEAL_F, TEAL_B),
        ("[B, H, L, d_k]", "各头 attention（形状不变）", GREEN_F, GREEN_B),
        ("[B, L, H, d_k]", "头维换回", TEAL_F, TEAL_B),
        ("[B, L, d_model]", "H×d_k 合回 d_model", PURPLE_F, PURPLE_B),
        ("[B, L, d_model]", "输出（同形）", BLUE_F, BLUE_B),
    ]
    arrow_labels = ["reshape", "transpose(1,2)", "scaled dot-\nproduct attn",
                    "transpose(1,2)", "reshape\n+ contiguous", "W_O"]
    for i, (lab, sub, f, b) in enumerate(boxes):
        w = bw if i not in (0, 6) else 200
        p += node(xs[i], cy, w, bh, lab, sub, f, b, label_cls="mono")
    for i in range(6):
        x1 = xs[i] + (bw if i not in (0,) else 200) / 2
        x2 = xs[i + 1] - (bw if (i + 1) not in (6,) else 200) / 2
        mk, col = ("aGreen", GREEN_B) if i == 2 else ("aBlue", BLUE_B)
        p += arrow(x1, cy, x2, cy, marker=mk, color=col)
        mid = (x1 + x2) / 2
        lbl = arrow_labels[i].split("\n")
        for k, ln in enumerate(lbl):
            yy = cy - 18 - (len(lbl) - 1 - k) * 16
            p.append(f'<text x="{mid}" y="{yy}" text-anchor="middle" class="monos" fill="{SUB}">{esc(ln)}</text>')

    # bracket annotations: split / heads-parallel / merge
    def bracket(x1, x2, y, text, color):
        out = [f'<path d="M {x1} {y} L {x1} {y+12} L {x2} {y+12} L {x2} {y}" '
               f'fill="none" stroke="{color}" stroke-width="1.8"/>']
        out.append(f'<text x="{(x1+x2)/2}" y="{y+32}" text-anchor="middle" class="sub" fill="{color}">{esc(text)}</text>')
        return out
    by = cy + bh / 2 + 26
    p += bracket(xs[0] - 100, xs[2] + bw / 2, by, "切头 split_heads（不改数据，只搬运）", PURPLE_B)
    p += bracket(xs[4] - bw / 2, xs[5] + bw / 2, by, "合头 merge_heads（Concat）", TEAL_B)
    p.append(f'<text x="{xs[3]}" y="{by+32}" text-anchor="middle" class="sub" fill="{GREEN_B}">每个头各算各的</text>')

    p.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" class="cap">'
             f'reshape / transpose 只搬运不计算；唯一改变数据的是各头内部的 attention 与最后的 W_O。transpose 后需 .contiguous() 才能 reshape。</text>')
    write_svg(ASSETS / "shape-flow.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: MHA / GQA / MQA 对比
# ============================================================
def diagram_mha_mqa_gqa():
    W, H = 1820, 640
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'MHA / GQA / MQA：query 头数都是 H，区别只在共享多少份 K / V</text>']

    qn = 8                        # query heads (all three panels)
    qw, qh = 52, 40               # query head box
    kw, kh = 58, 44               # kv head box
    panel_w = 540
    panels = [
        (60,  "MHA", "num_kv_heads = H = 8", 8, BLUE_B, BLUE_F),
        (640, "GQA", "num_kv_heads = G = 2（每 4 个 query 共享）", 2, GREEN_B, GREEN_F),
        (1220, "MQA", "num_kv_heads = 1（全体共享）", 1, ORANGE_B, ORANGE_F),
    ]
    qy = 200
    ky = 430
    for ox, name, sub, g, b, f in panels:
        cx = ox + panel_w / 2
        # panel title
        p.append(f'<text x="{cx}" y="115" text-anchor="middle" class="lbl" fill="{b}">{esc(name)}</text>')
        p.append(f'<text x="{cx}" y="137" text-anchor="middle" class="small">{esc(sub)}</text>')
        # panel frame
        p.append(f'<rect x="{ox}" y="150" width="{panel_w}" height="400" rx="14" '
                 f'fill="none" stroke="{GRAY_B}" stroke-width="1.4" stroke-dasharray="5,4"/>')
        # query head row
        q_gap = (panel_w - 40) / qn
        q_xs = [ox + 20 + q_gap * (i + 0.5) for i in range(qn)]
        for i, qx in enumerate(q_xs):
            p.append(f'<rect x="{qx-qw/2}" y="{qy-qh/2}" width="{qw}" height="{qh}" rx="7" '
                     f'fill="{PURPLE_F}" stroke="{PURPLE_B}" stroke-width="1.6"/>')
            p.append(f'<text x="{qx}" y="{qy+5}" text-anchor="middle" class="monos">Q{i+1}</text>')
        p.append(f'<text x="{ox+12}" y="{qy+5}" text-anchor="end" class="small" fill="{PURPLE_B}">query</text>'
                 if False else "")
        p.append(f'<text x="{cx}" y="{qy-34}" text-anchor="middle" class="small" fill="{PURPLE_B}">8 个 query 头</text>')
        # kv head row
        k_gap = (panel_w - 40) / g
        k_xs = [ox + 20 + k_gap * (i + 0.5) for i in range(g)]
        for i, kx in enumerate(k_xs):
            p.append(f'<rect x="{kx-kw/2}" y="{ky-kh/2}" width="{kw}" height="{kh}" rx="7" '
                     f'fill="{f}" stroke="{b}" stroke-width="1.8"/>')
            p.append(f'<text x="{kx}" y="{ky+5}" text-anchor="middle" class="monos">KV{i+1 if g>1 else ""}</text>')
        p.append(f'<text x="{cx}" y="{ky+kh/2+26}" text-anchor="middle" class="small" fill="{b}">'
                 f'{g} 份 K / V（KV cache ∝ {g}）</text>')
        # connecting lines: each query head -> its kv head
        per = qn // g
        for i, qx in enumerate(q_xs):
            kx = k_xs[i // per]
            p.append(f'<line x1="{qx}" y1="{qy+qh/2}" x2="{kx}" y2="{ky-kh/2}" '
                     f'stroke="{b}" stroke-width="1.5" opacity="0.7"/>')

    # legend / takeaway
    p.append(f'<text x="{W/2}" y="{H-34}" text-anchor="middle" class="cap">'
             f'紫框 = query 头（三者都 8 个）；彩色框 = K / V 头。KV 头越少，KV cache 越省 —— MHA(8) → GQA(2) → MQA(1)，质量略降、显存大省。</text>')
    write_svg(ASSETS / "mha-mqa-gqa.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_overview()
    diagram_shape_flow()
    diagram_mha_mqa_gqa()
    print("wrote 3 SVGs to", ASSETS)
