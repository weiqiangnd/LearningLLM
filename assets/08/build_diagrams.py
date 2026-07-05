"""Generate conceptual diagrams for chapter 08 (FFN、残差连接与归一化 / SwiGLU).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders. Hand-written
SVG (same approach as assets/07/build_diagrams.py) so the layout can be tuned
precisely. Body / sublabel / caption text uses gray-700 (#374151) or darker per
the repo contrast guideline.

Requires the Noto Sans CJK font for clean Chinese glyphs:
    apt-get install -y fonts-noto-cjk

Run from repo root:
    python3 assets/08/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant. All five are moderate
density (not sparse, not a dense matrix) -> default width 2400 (sparse/few-node
-> 1800; many-node/very-wide -> 3000):
    for n in block ffn residual norm swiglu; do
        rsvg-convert -w 2400 assets/08/$n.svg -o /tmp/x.png
        pngquant --quality=100 --strip --force --output assets/08/$n.png /tmp/x.png
    done
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
SLATE6 = "#475569"   # slate-600 — floor for decorative bordered tag text/borders
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
    <marker id="aRed" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{RED_B}"/></marker>
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
    w = 0.0
    for ch in text:
        if ord(ch) > 0x2460:
            w += cjk_px
        else:
            w += ascii_px
    return w


def tag(cx, y, text, fill, border):
    w = max(64, _text_width(text, cjk_px=13.5, ascii_px=7.5) + 30)
    return [
        f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="24" rx="7" '
        f'fill="{fill}" stroke="{border}" stroke-width="1.3"/>',
        f'<text x="{cx}" y="{y+16.5}" text-anchor="middle" class="tag" fill="{border}">{esc(text)}</text>'
    ]


def circle_plus(cx, cy, r=20, color=GREEN_B, fill=GREEN_F):
    return [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="2"/>',
        f'<text x="{cx}" y="{cy+7}" text-anchor="middle" class="lbl" fill="{color}">+</text>'
    ]


# ============================================================
# Diagram 1: Transformer block 全景（Pre-LN + 两残差）
# ============================================================
def diagram_block():
    W, H = 1500, 960
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'Transformer block 全景：两个子层（MHA、FFN），各裹 Norm 与残差旁路（Pre-LN）</text>']

    cx = 560                       # main column x
    res_x = 980                    # residual rail x (right side)
    # vertical positions (top -> bottom)
    y_in = 120
    y_n1, y_attn = 235, 335
    y_add1 = 445
    y_n2, y_ffn = 560, 660
    y_add2 = 770

    # input
    p += node(cx, y_in, 240, 56, "x", "[B, L, d_model]", BLUE_F, BLUE_B, label_cls="monob")

    # ---- sublayer 1: attention ----
    p += node(cx, y_n1, 240, 52, "RMSNorm", "归一化", GRAY_F, GRAY_B)
    p += node(cx, y_attn, 240, 60, "Multi-Head Attention", "跨 token 混合信息", PURPLE_F, PURPLE_B)
    p += circle_plus(cx, y_add1)
    # main path arrows
    p += arrow(cx, y_in + 28, cx, y_n1 - 26)
    p += arrow(cx, y_n1 + 26, cx, y_attn - 30)
    p += arrow(cx, y_attn + 30, cx, y_add1 - 20)
    # residual rail 1: from input split to add1
    p.append(f'<line x1="{cx}" y1="{y_in+28}" x2="{res_x}" y2="{y_in+28}" stroke="{GREEN_B}" stroke-width="2"/>')
    p.append(f'<line x1="{res_x}" y1="{y_in+28}" x2="{res_x}" y2="{y_add1}" stroke="{GREEN_B}" stroke-width="2"/>')
    p += arrow(res_x, y_add1, cx + 20, y_add1, marker="aGreen", color=GREEN_B)
    p.append(f'<text x="{res_x+14}" y="{(y_in+28+y_add1)/2}" class="sub" fill="{GREEN_B}">残差旁路</text>')
    p.append(f'<text x="{res_x+14}" y="{(y_in+28+y_add1)/2+20}" class="sub" fill="{GREEN_B}">（绕过 Norm+子层）</text>')

    # ---- sublayer 2: ffn ----
    p += arrow(cx, y_add1 + 20, cx, y_n2 - 26)
    p += node(cx, y_n2, 240, 52, "RMSNorm", "归一化", GRAY_F, GRAY_B)
    p += node(cx, y_ffn, 240, 60, "FFN (SwiGLU)", "逐 token 非线性深加工", ORANGE_F, ORANGE_B)
    p += circle_plus(cx, y_add2)
    p += arrow(cx, y_n2 + 26, cx, y_ffn - 30)
    p += arrow(cx, y_ffn + 30, cx, y_add2 - 20)
    # residual rail 2: from after add1 to add2
    yb = y_add1 + 20
    p.append(f'<line x1="{cx}" y1="{yb}" x2="{res_x}" y2="{yb}" stroke="{GREEN_B}" stroke-width="2"/>')
    p.append(f'<line x1="{res_x}" y1="{yb}" x2="{res_x}" y2="{y_add2}" stroke="{GREEN_B}" stroke-width="2"/>')
    p += arrow(res_x, y_add2, cx + 20, y_add2, marker="aGreen", color=GREEN_B)

    # output arrow + label (centered below add2 to avoid the residual rail)
    p += arrow(cx, y_add2 + 22, cx, y_add2 + 56, marker="aBlue", color=BLUE_B)
    p.append(f'<text x="{cx}" y="{y_add2+78}" text-anchor="middle" class="mono" fill="{SUB}">out [B, L, d_model]（与 x 同形）</text>')

    # sublayer grouping boxes — each wraps its own [Norm + sub-block], centered on
    # the main column so the dashed frame actually encloses the nodes it labels
    # (the residual add + rail sit just below each box, kept outside to stay clean).
    box_w = 320
    box_x = cx - box_w / 2
    # ① attention sublayer: RMSNorm + MHA
    b1_top, b1_bot = y_n1 - 30, y_attn + 40
    p.append(f'<rect x="{box_x}" y="{b1_top}" width="{box_w}" height="{b1_bot-b1_top}" rx="12" '
             f'fill="none" stroke="{PURPLE_B}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    p += tag(box_x + 66, b1_top - 12, "① 注意力子层", PURPLE_F, PURPLE_B)
    # ② FFN sublayer: RMSNorm + FFN
    b2_top, b2_bot = y_n2 - 30, y_ffn + 40
    p.append(f'<rect x="{box_x}" y="{b2_top}" width="{box_w}" height="{b2_bot-b2_top}" rx="12" '
             f'fill="none" stroke="{ORANGE_B}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    p += tag(box_x + 60, b2_top - 12, "② FFN 子层", ORANGE_F, ORANGE_B)

    # formula footer (two lines so the two equations don't crowd one line)
    p.append(f'<rect x="{W/2-360}" y="{H-78}" width="720" height="64" rx="10" '
             f'fill="{GRAY_F}" stroke="{GRAY_B}" stroke-width="1.4"/>')
    p.append(f'<text x="{W/2}" y="{H-50}" text-anchor="middle" class="mono">'
             f"① x' = x + MHA(Norm(x))</text>")
    p.append(f'<text x="{W/2}" y="{H-24}" text-anchor="middle" class="mono">'
             f"② out = x' + FFN(Norm(x'))</text>")
    write_svg(ASSETS / "block.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: FFN 结构（升维 → 激活 → 降维）
# ============================================================
def diagram_ffn():
    W, H = 1680, 520
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'FFN：升维到 d_ff（≈4×）→ 非线性激活 → 降回 d_model（逐 token、同一套权重）</text>']
    cy = 250
    # input token vector (narrow)
    x0 = 150
    p.append(f'<rect x="{x0-36}" y="{cy-70}" width="72" height="140" rx="9" fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="2"/>')
    p.append(f'<text x="{x0}" y="{cy-86}" text-anchor="middle" class="lbl">x</text>')
    p.append(f'<text x="{x0}" y="{cy+96}" text-anchor="middle" class="small">d_model</text>')

    # W1 (升维)
    w1 = 430
    p += arrow(x0 + 36, cy, w1 - 95, cy, marker="aPurple", color=PURPLE_B)
    p.append(f'<text x="{(x0+36+w1-95)/2}" y="{cy-16}" text-anchor="middle" class="monos" fill="{PURPLE_B}">W₁: d_model→d_ff</text>')
    p += node(w1, cy, 150, 96, "W₁", "升维", PURPLE_F, PURPLE_B, label_cls="monob")

    # wide hidden (升维后)
    h0 = 720
    p += arrow(w1 + 75, cy, h0 - 44, cy, marker="aPurple", color=PURPLE_B)
    p.append(f'<rect x="{h0-44}" y="{cy-115}" width="88" height="230" rx="9" fill="{TEAL_F}" stroke="{TEAL_B}" stroke-width="2"/>')
    p.append(f'<text x="{h0}" y="{cy-131}" text-anchor="middle" class="lbl">h</text>')
    p.append(f'<text x="{h0}" y="{cy+140}" text-anchor="middle" class="small">d_ff（≈4× 宽）</text>')

    # activation
    a0 = 980
    p += arrow(h0 + 44, cy, a0 - 80, cy, marker="aGreen", color=GREEN_B)
    p += node(a0, cy, 160, 96, "激活", "ReLU / GELU / SiLU", GREEN_F, GREEN_B)
    p.append(f'<text x="{a0}" y="{cy-66}" text-anchor="middle" class="small" fill="{GREEN_B}">唯一的非线性来源</text>')
    p.append(f'<text x="{a0}" y="{cy+72}" text-anchor="middle" class="small">逐元素作用，形状不变（仍是 d_ff）</text>')

    # W2 (降维)
    w2 = 1290
    p += arrow(a0 + 80, cy, w2 - 95, cy, marker="aPurple", color=PURPLE_B)
    p.append(f'<text x="{(a0+80+w2-95)/2}" y="{cy-16}" text-anchor="middle" class="monos" fill="{PURPLE_B}">W₂: d_ff→d_model</text>')
    p += node(w2, cy, 150, 96, "W₂", "降维", PURPLE_F, PURPLE_B, label_cls="monob")

    # output (narrow again)
    o0 = 1560
    p += arrow(w2 + 75, cy, o0 - 36, cy, marker="aBlue", color=BLUE_B)
    p.append(f'<rect x="{o0-36}" y="{cy-70}" width="72" height="140" rx="9" fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="2"/>')
    p.append(f'<text x="{o0}" y="{cy-86}" text-anchor="middle" class="lbl">out</text>')
    p.append(f'<text x="{o0}" y="{cy+96}" text-anchor="middle" class="small">d_model（同形）</text>')

    p.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" class="cap">'
             f'FFN(x) = 激活(x·W₁ + b₁)·W₂ + b₂　　中间胖、两头瘦；FFN 占一个 block 约 2/3 的参数。</text>')
    write_svg(ASSETS / "ffn.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: 残差连接 + 梯度高速公路
# ============================================================
def diagram_residual():
    W, H = 1700, 620
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'残差连接：前向 x + F(x)，反向梯度经 +1 直通 —— 深层网络的梯度高速公路</text>']

    # ---- left: single residual block (forward / backward) ----
    p += tag(420, 92, "一个残差块：y = x + F(x)", BLUE_F, BLUE_B)
    cy = 300
    x0 = 150
    p += node(x0, cy, 96, 70, "x", "", BLUE_F, BLUE_B, label_cls="monob")
    fx = 430
    p += arrow(x0 + 48, cy, fx - 95, cy)
    p += node(fx, cy, 150, 84, "F(x)", "子层（Norm+MHA/FFN）", PURPLE_F, PURPLE_B, label_cls="monob")
    addx = 700
    p += arrow(fx + 75, cy, addx - 22, cy)
    p += circle_plus(addx, cy, r=22)
    # residual skip
    p.append(f'<path d="M {x0} {cy-35} Q {x0} {cy-150} {(x0+addx)/2} {cy-150} '
             f'Q {addx} {cy-150} {addx} {cy-24}" fill="none" stroke="{GREEN_B}" stroke-width="2.2" marker-end="url(#aGreen)"/>')
    p.append(f'<text x="{(x0+addx)/2}" y="{cy-160}" text-anchor="middle" class="sub" fill="{GREEN_B}">残差旁路（恒等）</text>')
    p += arrow(addx + 22, cy, addx + 110, cy, marker="aBlue", color=BLUE_B)
    p.append(f'<text x="{addx+130}" y="{cy+5}" class="monob">y</text>')

    # backward gradient annotation
    p.append(f'<text x="420" y="{cy+90}" text-anchor="middle" class="mono" fill="{RED_B}">'
             f'反向：∂y/∂x = 1 + ∂F/∂x</text>')
    p.append(f'<text x="420" y="{cy+118}" text-anchor="middle" class="sub">'
             f'那个 1 让上游梯度【原样直通】，哪怕 ∂F/∂x≈0 也传得回去</text>')

    # divider
    p.append(f'<line x1="900" y1="100" x2="900" y2="540" stroke="{GRAY_B}" stroke-width="1.4" stroke-dasharray="5,5"/>')

    # ---- right: deep stack, with vs without residual ----
    p += tag(1300, 92, "40 层堆叠：底层梯度的命运", GRAY_F, SLATE6)
    # two mini stacks
    def mini_stack(ox, title, color, fill, grad_text, healthy):
        out = [f'<text x="{ox}" y="150" text-anchor="middle" class="lbl" fill="{color}">{esc(title)}</text>']
        # 5 stacked layers as a tower
        ly0, lh, lw = 175, 26, 110
        for i in range(5):
            yy = ly0 + i * (lh + 8)
            out.append(f'<rect x="{ox-lw/2}" y="{yy}" width="{lw}" height="{lh}" rx="5" '
                       f'fill="{fill}" stroke="{color}" stroke-width="1.6"/>')
            if i < 4:
                out.append(f'<line x1="{ox}" y1="{yy+lh}" x2="{ox}" y2="{yy+lh+8}" stroke="{color}" stroke-width="1.6"/>')
        out.append(f'<text x="{ox}" y="{ly0-10}" text-anchor="middle" class="small">第 40 层（顶）</text>')
        out.append(f'<text x="{ox}" y="{ly0+5*(lh+8)+18}" text-anchor="middle" class="small">第 1 层（底）</text>')
        # gradient arrow pointing down (backprop)
        gx = ox + 100
        gcol = GREEN_B if healthy else RED_B
        out += arrow(gx, ly0, gx, ly0 + 5 * (lh + 8), marker=("aGreen" if healthy else "aRed"),
                     color=gcol, width=3 if healthy else 1.2)
        out.append(f'<text x="{gx+12}" y="{ly0+5*(lh+8)/2}" class="sub" fill="{gcol}">梯度反传</text>')
        out.append(f'<text x="{ox}" y="{ly0+5*(lh+8)+44}" text-anchor="middle" class="mono" fill="{gcol}">{esc(grad_text)}</text>')
        return out
    p += mini_stack(1130, "无残差 F(x)", RED_B, RED_F, "底层梯度 ≈ 1e-10（学不动）", False)
    p += mini_stack(1480, "有残差 x+F(x)", GREEN_B, GREEN_F, "底层梯度 健康（可学习）", True)

    p.append(f'<text x="{W/2}" y="{H-24}" text-anchor="middle" class="cap">'
             f'无残差：梯度走「层层连乘」，几十层后指数衰减到几乎为 0；有残差：那条 +1 直通路让梯度健康送回底层。</text>')
    write_svg(ASSETS / "residual.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: LayerNorm vs RMSNorm
# ============================================================
def diagram_norm():
    W, H = 1620, 600
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'LayerNorm vs RMSNorm：都在单个 token 的特征维上算；RMSNorm 省掉「减均值」与 β</text>']

    # left: two grids contrasting the statistic AXIS — LayerNorm horizontal
    # (across one token's features), BatchNorm vertical (across the batch).
    p += tag(440, 92, "归一化「沿哪个方向」统计：LayerNorm 横向（每 token）vs BatchNorm 纵向（跨样本）", BLUE_F, BLUE_B)
    cw, ch, rows, cols = 50, 42, 3, 4
    gy0 = 175

    def grid(gx0, hl_kind, accent_b, accent_f, caption):
        # hl_kind: 'row' -> highlight middle row (LayerNorm); 'col' -> middle col (BatchNorm)
        hr, hc = 1, 1
        for r in range(rows):
            for c in range(cols):
                xx, yy = gx0 + c * cw, gy0 + r * ch
                hl = (r == hr) if hl_kind == 'row' else (c == hc)
                p.append(f'<rect x="{xx}" y="{yy}" width="{cw-8}" height="{ch-8}" rx="5" '
                         f'fill="{accent_f if hl else GRAY_F}" stroke="{accent_b if hl else GRAY_B}" '
                         f'stroke-width="{2 if hl else 1.2}"/>')
        if hl_kind == 'row':
            yy = gy0 + hr * ch
            p.append(f'<rect x="{gx0-6}" y="{yy-6}" width="{cols*cw+4}" height="{ch}" rx="7" '
                     f'fill="none" stroke="{accent_b}" stroke-width="2.6"/>')
        else:
            xx = gx0 + hc * cw
            p.append(f'<rect x="{xx-6}" y="{gy0-6}" width="{cw}" height="{rows*ch+4}" rx="7" '
                     f'fill="none" stroke="{accent_b}" stroke-width="2.6"/>')
        # axis labels around each grid
        p.append(f'<text x="{gx0-8}" y="{gy0-12}" class="small">特征维 →</text>')
        p.append(f'<text x="{gx0-30}" y="{gy0+rows*ch/2}" class="small" '
                 f'transform="rotate(-90 {gx0-30} {gy0+rows*ch/2})" text-anchor="middle">样本 / token ↑</text>')
        p.append(f'<text x="{gx0+cols*cw/2}" y="{gy0+rows*ch+34}" text-anchor="middle" '
                 f'class="sub" fill="{accent_b}">{esc(caption)}</text>')

    grid(150, 'row', BLUE_B, BLUE_F, "LayerNorm：沿一行（一个 token 的 d 个特征）")
    grid(470, 'col', RED_B, RED_F, "BatchNorm：沿一列（同一特征、跨样本）")
    p.append(f'<text x="350" y="{gy0+rows*ch+66}" text-anchor="middle" class="sub">'
             f'Transformer 用左边（与 batch、与别的 token 无关）；CNN 时代的 BatchNorm 是右边</text>')

    # right: two formulas in boxes (fx chosen so right margin ≈ left margin and
    # the mid gap between the left illustration and the formulas isn't too wide)
    fx = 880
    bw, bh = 620, 150
    # LayerNorm box
    ly = 150
    p.append(f'<rect x="{fx}" y="{ly}" width="{bw}" height="{bh}" rx="12" fill="{TEAL_F}" stroke="{TEAL_B}" stroke-width="2"/>')
    p.append(f'<text x="{fx+20}" y="{ly+34}" class="lbl" fill="{TEAL_B}">LayerNorm（原版 Transformer）</text>')
    p.append(f'<text x="{fx+20}" y="{ly+74}" class="mono">x̂ = (x − μ) / √(σ² + ε)</text>')
    p.append(f'<text x="{fx+20}" y="{ly+104}" class="mono">out = γ ⊙ x̂ + β</text>')
    p.append(f'<text x="{fx+20}" y="{ly+134}" class="sub">减均值 μ（中心化）+ 除标准差 σ + 缩放 γ + 平移 β</text>')

    # RMSNorm box
    ry2 = 360
    p.append(f'<rect x="{fx}" y="{ry2}" width="{bw}" height="{bh}" rx="12" fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="2"/>')
    p.append(f'<text x="{fx+20}" y="{ry2+34}" class="lbl" fill="{ORANGE_B}">RMSNorm（LLaMA / Qwen 默认）</text>')
    p.append(f'<text x="{fx+20}" y="{ry2+74}" class="mono">RMS(x) = √(mean(x²) + ε)</text>')
    p.append(f'<text x="{fx+20}" y="{ry2+104}" class="mono">out = γ ⊙ x / RMS(x)</text>')
    p.append(f'<text x="{fx+20}" y="{ry2+134}" class="sub">不减均值、无 β：只「重新缩放」，更省算、更省参数</text>')

    p.append(f'<text x="{W/2}" y="{H-22}" text-anchor="middle" class="cap">'
             f'差别只两处：RMSNorm 不减均值（分母用均方根而非标准差）、没有平移 β。实测质量几乎无损，故成现代 LLM 默认。</text>')
    write_svg(ASSETS / "norm.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 5: SwiGLU
# ============================================================
def diagram_swiglu():
    W, H = 1560, 640
    p = [f'<text x="{W/2}" y="46" text-anchor="middle" class="title">'
         f'SwiGLU：门路径 SiLU(x·W_gate) 与内容路径 x·W_up 逐元素相乘，再过 W_down 降维</text>']

    x0 = 140
    cy = 290
    p += node(x0, cy, 96, 80, "x", "d_model", BLUE_F, BLUE_B, label_cls="monob")

    # gate path (top) and up path (bottom)
    gy, uy = 175, 405
    # W_gate
    p.append(f'<path d="M {x0+48} {cy-12} C 260 {cy-12}, 300 {gy}, 380 {gy}" fill="none" stroke="{PURPLE_B}" stroke-width="2" marker-end="url(#aPurple)"/>')
    p += node(450, gy, 150, 76, "W_gate", "升维", PURPLE_F, PURPLE_B, label_cls="mono")
    p += arrow(525, gy, 640, gy, marker="aGreen", color=GREEN_B)
    p += node(720, gy, 150, 76, "SiLU", "门控（非 0~1）", GREEN_F, GREEN_B)
    p.append(f'<text x="585" y="{gy-44}" text-anchor="middle" class="sub" fill="{GREEN_B}">门路径（gate）</text>')

    # W_up
    p.append(f'<path d="M {x0+48} {cy+12} C 260 {cy+12}, 300 {uy}, 380 {uy}" fill="none" stroke="{PURPLE_B}" stroke-width="2" marker-end="url(#aPurple)"/>')
    p += node(450, uy, 150, 76, "W_up", "升维", PURPLE_F, PURPLE_B, label_cls="mono")
    p.append(f'<text x="585" y="{uy+56}" text-anchor="middle" class="sub" fill="{PURPLE_B}">内容路径（up，不过激活）</text>')

    # elementwise multiply
    mx = 980
    p += arrow(795, gy, mx - 24, cy - 30, marker="aGreen", color=GREEN_B)
    p += arrow(525, uy, mx - 24, cy + 30, marker="aPurple", color=PURPLE_B)
    p.append(f'<circle cx="{mx}" cy="{cy}" r="26" fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="2.2"/>')
    p.append(f'<text x="{mx}" y="{cy+9}" text-anchor="middle" class="title" fill="{ORANGE_B}">⊙</text>')
    p.append(f'<text x="{mx}" y="{cy-40}" text-anchor="middle" class="sub" fill="{ORANGE_B}">逐元素相乘</text>')

    # W_down
    dx = 1200
    p += arrow(mx + 26, cy, dx - 75, cy, marker="aPurple", color=PURPLE_B)
    p += node(dx, cy, 150, 80, "W_down", "降维", PURPLE_F, PURPLE_B, label_cls="mono")
    # output
    ox = 1450
    p += arrow(dx + 75, cy, ox - 40, cy, marker="aBlue", color=BLUE_B)
    p += node(ox, cy, 80, 80, "out", "d_model", BLUE_F, BLUE_B, label_cls="monob")

    p.append(f'<rect x="{W/2-540}" y="{H-104}" width="1080" height="44" rx="10" fill="{GRAY_F}" stroke="{GRAY_B}" stroke-width="1.4"/>')
    p.append(f'<text x="{W/2}" y="{H-75}" text-anchor="middle" class="mono">'
             f'FFN_SwiGLU(x) = ( SiLU(x·W_gate) ⊙ (x·W_up) ) · W_down　　SiLU(x)=x·σ(x)</text>')
    p.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" class="cap">'
             f'三个矩阵（比老式 FFN 多一个）；把 d_ff 收到约 8/3×d_model，总参数量与老式 4× 两矩阵 FFN 持平。</text>')
    write_svg(ASSETS / "swiglu.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 6: GELU 配图 —— 左 Φ(x) 标准正态 CDF，右 GELU(x)=x·Φ(x)
# 这张是函数曲线图（同 P02 的 activations.png 那类），用 matplotlib 直接出
# png+svg，不走上面手写 SVG 那套。坐标轴/标题只用数学记号，无中文，避免字体依赖。
# ============================================================
def diagram_gelu():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import erf, sqrt

    xs = np.linspace(-4.0, 4.0, 600)
    phi = 0.5 * (1.0 + np.vectorize(lambda t: erf(t / sqrt(2.0)))(xs))  # 标准正态 CDF
    gelu = xs * phi                                                     # GELU(x)=x·Φ(x)
    relu = np.maximum(0.0, xs)                                          # 右图作平滑对照

    # GELU 在负区的最低点（解析上 x≈-0.7517、值≈-0.1700），标注用
    dip_x = xs[np.argmin(gelu)]
    dip_y = gelu.min()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # 左：CDF Φ(x)，值域落在 (0,1)，虚线标出上界 1、标出 Φ(0)=0.5
    axL.axhline(0.0, color="#9ca3af", lw=0.8)
    axL.axhline(1.0, color="#9ca3af", lw=0.9, ls="--")
    axL.axvline(0.0, color="#9ca3af", lw=0.8)
    axL.plot(xs, phi, color="#2563eb", lw=2.4)
    axL.plot(0.0, 0.5, "o", color="#2563eb", ms=6)
    axL.annotate(r"$\Phi(0)=0.5$", xy=(0.0, 0.5), xytext=(-3.8, 0.66),
                 fontsize=11, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.0))
    axL.text(2.0, 1.04, "ranges in (0, 1)", fontsize=11, color="#374151")
    axL.set_title(r"$\Phi(x)$: standard normal CDF", fontsize=14)
    axL.set_xlabel("x", fontsize=12)
    axL.set_ylim(-0.12, 1.20)
    axL.grid(alpha=0.25)

    # 右：GELU(x)=x·Φ(x)，叠一条 ReLU 虚线作参照，标出负区小坑
    axR.axhline(0.0, color="#9ca3af", lw=0.8)
    axR.axvline(0.0, color="#9ca3af", lw=0.8)
    axR.plot(xs, relu, color="#9ca3af", lw=1.8, ls="--", label="ReLU (reference)")
    axR.plot(xs, gelu, color="#059669", lw=2.4, label="GELU")
    axR.plot(dip_x, dip_y, "o", color="#059669", ms=6)
    axR.annotate(f"smooth, non-monotonic\nmin $\\approx$ {dip_y:.2f} at x $\\approx$ {dip_x:.2f}",
                 xy=(dip_x, dip_y), xytext=(-3.9, 1.9),
                 fontsize=11, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.0))
    axR.set_title(r"$\mathrm{GELU}(x)=x\cdot\Phi(x)$", fontsize=14)
    axR.set_xlabel("x", fontsize=12)
    axR.legend(loc="lower right", fontsize=11, framealpha=0.9)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ASSETS / "gelu.png", dpi=220, bbox_inches="tight")
    fig.savefig(ASSETS / "gelu.svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Diagram 7: SiLU 配图 —— 左 σ(x) sigmoid，右 SiLU(x)=x·σ(x)
# 同 gelu.png 那类函数曲线图，用 matplotlib 直接出 png+svg。
# 坐标轴/标题只用数学记号，无中文，避免字体依赖。
# ============================================================
def diagram_silu():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(-6.0, 6.0, 600)
    sig = 1.0 / (1.0 + np.exp(-xs))   # sigmoid σ(x)
    silu = xs * sig                   # SiLU(x)=x·σ(x)
    relu = np.maximum(0.0, xs)        # 右图作平滑对照

    # SiLU 在负区的最低点（解析上 x≈-1.2785、值≈-0.2785），标注用
    dip_x = xs[np.argmin(silu)]
    dip_y = silu.min()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # 左：sigmoid σ(x)，值域落在 (0,1)，虚线标出上界 1、标出 σ(0)=0.5
    axL.axhline(0.0, color="#9ca3af", lw=0.8)
    axL.axhline(1.0, color="#9ca3af", lw=0.9, ls="--")
    axL.axvline(0.0, color="#9ca3af", lw=0.8)
    axL.plot(xs, sig, color="#2563eb", lw=2.4)
    axL.plot(0.0, 0.5, "o", color="#2563eb", ms=6)
    axL.annotate(r"$\sigma(0)=0.5$", xy=(0.0, 0.5), xytext=(-5.7, 0.66),
                 fontsize=11, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.0))
    axL.text(3.0, 1.04, "ranges in (0, 1)", fontsize=11, color="#374151")
    axL.set_title(r"$\sigma(x)=1/(1+e^{-x})$: sigmoid", fontsize=14)
    axL.set_xlabel("x", fontsize=12)
    axL.set_ylim(-0.12, 1.20)
    axL.grid(alpha=0.25)

    # 右：SiLU(x)=x·σ(x)，叠一条 ReLU 虚线作参照，标出负区小坑
    axR.axhline(0.0, color="#9ca3af", lw=0.8)
    axR.axvline(0.0, color="#9ca3af", lw=0.8)
    axR.plot(xs, relu, color="#9ca3af", lw=1.8, ls="--", label="ReLU (reference)")
    axR.plot(xs, silu, color="#059669", lw=2.4, label="SiLU")
    axR.plot(dip_x, dip_y, "o", color="#059669", ms=6)
    axR.annotate(f"smooth, non-monotonic\nmin $\\approx$ {dip_y:.2f} at x $\\approx$ {dip_x:.2f}",
                 xy=(dip_x, dip_y), xytext=(-5.8, 2.8),
                 fontsize=11, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.0))
    axR.set_title(r"$\mathrm{SiLU}(x)=x\cdot\sigma(x)$", fontsize=14)
    axR.set_xlabel("x", fontsize=12)
    axR.legend(loc="lower right", fontsize=11, framealpha=0.9)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ASSETS / "silu.png", dpi=220, bbox_inches="tight")
    fig.savefig(ASSETS / "silu.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    diagram_block()
    diagram_ffn()
    diagram_residual()
    diagram_norm()
    diagram_swiglu()
    print("wrote 5 SVGs to", ASSETS)
    diagram_gelu()
    diagram_silu()
    print("wrote silu.png / silu.svg to", ASSETS)
    print("wrote gelu.png / gelu.svg to", ASSETS)
