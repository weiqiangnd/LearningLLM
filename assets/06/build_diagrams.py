"""Generate conceptual diagrams for chapter 06 (Scaled Dot-Product Attention).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
Hand-written SVG (same approach as assets/05/build_diagrams.py) so the layout
can be tuned precisely. Body / sublabel / caption text uses gray-700 (#374151)
or darker per the repo contrast guideline.

Run from repo root:
    python3 assets/06/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant, e.g.
    rsvg-convert -w 1800 assets/06/attention-pipeline.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/06/attention-pipeline.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("'Noto Sans CJK SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Zen Hei', sans-serif")
MONO = "'Noto Sans Mono CJK SC','SFMono-Regular','Consolas','Liberation Mono',monospace"

BG = "#ffffff"
TXT = "#374151"   # gray-700   primary labels
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
    .monos  {{ font-size: 13px; font-family: {MONO}; fill: {TXT}; }}
    .small  {{ font-size: 13px; fill: {SUB}; }}
    .cap    {{ font-size: 14px; fill: {SUB}; }}
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
    <marker id="aRed" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{RED_B}"/></marker>
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


def tag(cx, y, text, fill, border):
    w = max(46, len(text) * 8 + 18)
    return [
        f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="20" rx="6" '
        f'fill="{fill}" stroke="{border}" stroke-width="1.3"/>',
        f'<text x="{cx}" y="{y+14}" text-anchor="middle" class="tag" fill="{border}">{esc(text)}</text>'
    ]


# ============================================================
# Diagram 1: 从输入 X 投影出 Q / K / V
# ============================================================
def diagram_qkv():
    W, H = 1180, 560
    p = [f'<text x="{W/2}" y="44" text-anchor="middle" class="title">'
         f'从输入 X 投影出 Q / K / V —— 同一个序列，三套可学习投影，三个角色</text>']

    # input X on the left
    xc, xcy = 175, 290
    p += node(xc, xcy, 150, 96, "X", "输入序列 [L, d]", BLUE_F, BLUE_B, label_cls="monob")
    p.append(f'<text x="{xc}" y="{xcy+62}" text-anchor="middle" class="small">L 个 token</text>')
    p.append(f'<text x="{xc}" y="{xcy+80}" text-anchor="middle" class="small">每个 d 维（含位置编码）</text>')

    # three projection matrices
    proj_x = 540
    rows = [
        (150, "W_Q", "[d, d_k]", "Q", "[L, d_k]", "查询 Query：我要找什么", PURPLE_F, PURPLE_B, "aPurple"),
        (290, "W_K", "[d, d_k]", "K", "[L, d_k]", "键 Key：我拿什么被匹配", ORANGE_F, ORANGE_B, "aOrange"),
        (430, "W_V", "[d, d_k]", "V", "[L, d_k]", "值 Value：匹配上交出什么", GREEN_F, GREEN_B, "aGreen"),
    ]
    for cy, wname, wsh, oname, osh, desc, fill, border, mk in rows:
        # arrow from X to projection
        p += arrow(xc + 75, xcy, proj_x - 70, cy, marker=mk, color=border)
        # projection matrix box
        p += node(proj_x, cy, 132, 60, wname, wsh, GRAY_F, border, label_cls="mono")
        # arrow to output
        p += arrow(proj_x + 66, cy, proj_x + 168, cy, marker=mk, color=border)
        # output Q/K/V box
        oc = proj_x + 250
        p += node(oc, cy, 120, 60, oname, osh, fill, border, label_cls="monob")
        # description
        p.append(f'<text x="{oc+78}" y="{cy+5}" class="sub">{esc(desc)}</text>')

    # bottom note
    p.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" class="cap">'
             f'Q 与 K 维度必须相同（要做点积）；本章取 d_v = d_k。W_Q ≠ W_K ≠ W_V 各自学一套侧重，是注意力「不对称」的根源。</text>')
    write_svg(ASSETS / "qkv-projection.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: scaled dot-product attention 完整流水线
# ============================================================
def diagram_pipeline():
    W, H = 1900, 560
    p = [f'<text x="{W/2}" y="44" text-anchor="middle" class="title">'
         f'Scaled Dot-Product Attention 流水线：QKᵀ → 缩放 → (causal mask) → softmax → 乘 V</text>']

    cy = 250
    bw, bh = 188, 92
    # stage centers
    xs = [150, 430, 720, 1010, 1300, 1600, 1820]
    # inputs Q,K,V column at far left handled inline

    # Stage boxes
    # 1: Q,K input -> QKᵀ
    p += node(xs[0], cy, 140, bh, "Q, K", "[L, d_k]", BLUE_F, BLUE_B, label_cls="monob")

    p += arrow(xs[0] + 72, cy, xs[1] - bw/2, cy)
    p += node(xs[1], cy, bw, bh, "S = QKᵀ", "点积打分 [L, L]", PURPLE_F, PURPLE_B, label_cls="mono")
    p += tag(xs[1], cy - bh/2 - 28, "① 打分", PURPLE_F, PURPLE_B)

    p += arrow(xs[1] + bw/2, cy, xs[2] - bw/2, cy)
    p += node(xs[2], cy, bw, bh, "S / √d_k", "缩放 [L, L]", TEAL_F, TEAL_B, label_cls="mono")
    p += tag(xs[2], cy - bh/2 - 28, "② 缩放", TEAL_F, TEAL_B)

    p += arrow(xs[2] + bw/2, cy, xs[3] - bw/2, cy)
    p += node(xs[3], cy, bw, bh, "+ mask", "上三角置 -inf", RED_F, RED_B, label_cls="mono")
    p += tag(xs[3], cy - bh/2 - 28, "②.5 可选", RED_F, RED_B)
    p.append(f'<text x="{xs[3]}" y="{cy+bh/2+22}" text-anchor="middle" class="small">仅自回归(causal)时</text>')

    p += arrow(xs[3] + bw/2, cy, xs[4] - bw/2, cy)
    p += node(xs[4], cy, bw, bh, "softmax", "逐行归一 → A [L, L]", ORANGE_F, ORANGE_B, label_cls="mono")
    p += tag(xs[4], cy - bh/2 - 28, "③ 归一", ORANGE_F, ORANGE_B)
    p.append(f'<text x="{xs[4]}" y="{cy+bh/2+22}" text-anchor="middle" class="small">行非负、行和=1</text>')

    p += arrow(xs[4] + bw/2, cy, xs[5] - bw/2, cy)
    p += node(xs[5], cy, bw, bh, "A · V", "加权求和 [L, d_v]", GREEN_F, GREEN_B, label_cls="mono")
    p += tag(xs[5], cy - bh/2 - 28, "④ 求和", GREEN_F, GREEN_B)

    # V feeding into A·V from below
    vy = cy + 150
    p += node(xs[5] - 30, vy, 140, 58, "V", "[L, d_v]", BLUE_F, BLUE_B, label_cls="monob")
    p += arrow(xs[5] - 30, vy - 29, xs[5] - 20, cy + bh/2, marker="aBlue", color=BLUE_B)

    p += arrow(xs[5] + bw/2, cy, xs[6] - 55, cy)
    p += node(xs[6], cy, 130, bh, "O", "输出 [L, d_v]", PURPLE_F, PURPLE_B, label_cls="monob")

    # formula footer
    p.append(f'<rect x="{W/2-470}" y="{H-78}" width="940" height="46" rx="10" '
             f'fill="{GRAY_F}" stroke="{GRAY_B}" stroke-width="1.4"/>')
    p.append(f'<text x="{W/2}" y="{H-48}" text-anchor="middle" class="mono">'
             f'Attention(Q, K, V) = softmax( QKᵀ / √d_k  + mask ) · V</text>')
    write_svg(ASSETS / "attention-pipeline.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: causal mask —— 注意力矩阵变下三角
# ============================================================
def diagram_causal():
    W, H = 1280, 770
    p = [f'<text x="{W/2}" y="44" text-anchor="middle" class="title">'
         f'Causal Mask：softmax 前把上三角（未来）置 -inf，权重矩阵变下三角</text>']

    n = 4
    cell = 78
    gap_label = 30

    def grid(ox, oy, fill_fn, txt_fn, title, sub):
        out = [f'<text x="{ox + n*cell/2}" y="{oy-46}" text-anchor="middle" class="lbl">{esc(title)}</text>']
        out.append(f'<text x="{ox + n*cell/2}" y="{oy-26}" text-anchor="middle" class="small">{esc(sub)}</text>')
        # column header (key j)
        for j in range(n):
            out.append(f'<text x="{ox + j*cell + cell/2}" y="{oy-6}" text-anchor="middle" class="monos" fill="{SUB}">k{j+1}</text>')
        for i in range(n):
            # row header (query i)
            out.append(f'<text x="{ox-14}" y="{oy + i*cell + cell/2 + 4}" text-anchor="end" class="monos" fill="{SUB}">q{i+1}</text>')
            for j in range(n):
                x = ox + j * cell
                y = oy + i * cell
                f, b = fill_fn(i, j)
                out.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" rx="6" '
                           f'fill="{f}" stroke="{b}" stroke-width="1.6"/>')
                t = txt_fn(i, j)
                if t:
                    out.append(f'<text x="{x + (cell-4)/2}" y="{y + (cell-4)/2 + 5}" '
                               f'text-anchor="middle" class="monos">{esc(t)}</text>')
        return out

    # left grid: scores + mask (upper triangle -> -inf)
    ox1, oy1 = 150, 200
    def fill1(i, j):
        return (RED_F, RED_B) if j > i else (BLUE_F, BLUE_B)
    def txt1(i, j):
        return "-inf" if j > i else "s"
    p += grid(ox1, oy1, fill1, txt1, "缩放后分数 + causal mask", "上三角(未来 j>i)填 -inf，下三角保留")

    # arrow between grids
    ax = ox1 + n*cell + 60
    p += arrow(ax, oy1 + n*cell/2, ax + 110, oy1 + n*cell/2, marker="aGray", color=GRAY_B, width=2.4)
    p.append(f'<text x="{ax+55}" y="{oy1 + n*cell/2 - 14}" text-anchor="middle" class="small">逐行</text>')
    p.append(f'<text x="{ax+55}" y="{oy1 + n*cell/2 + 30}" text-anchor="middle" class="small">softmax</text>')

    # right grid: weights -> lower triangular, upper = 0
    ox2 = ax + 150
    def fill2(i, j):
        return (GRAY_F, GRAY_B) if j > i else (GREEN_F, GREEN_B)
    def txt2(i, j):
        return "0" if j > i else "w"
    p += grid(ox2, oy1, fill2, txt2, "注意力权重 A（softmax 后）", "上三角=0（看不到未来），每行和=1")

    # legend
    ly = oy1 + n*cell + 60
    lx = 260
    items = [
        (BLUE_F, BLUE_B, "保留的分数 / 有效 key（j ≤ i，能看见）"),
        (RED_F, RED_B, "被 mask 的未来位置（j > i，填 -inf）"),
        (GREEN_F, GREEN_B, "softmax 后非零权重（注意力真正落点）"),
        (GRAY_F, GRAY_B, "softmax 后权重 = 0（exp(-inf)=0）"),
    ]
    for k, (f, b, t) in enumerate(items):
        yy = ly + k * 30
        p.append(f'<rect x="{lx}" y="{yy}" width="22" height="20" rx="5" fill="{f}" stroke="{b}" stroke-width="1.4"/>')
        p.append(f'<text x="{lx+32}" y="{yy+15}" class="cap">{esc(t)}</text>')

    p.append(f'<text x="{W/2}" y="{H-18}" text-anchor="middle" class="cap">'
             f'每个 query 只能注意到「自己及之前」的 key —— 这正是自回归语言模型「预测下一个 token 时不许偷看答案」的实现。</text>')
    write_svg(ASSETS / "causal-mask.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: 点积打分 S = QKᵀ 的矩阵乘可视化（对应第 3.1 节）
# ============================================================
def diagram_qk_matmul():
    W, H = 1640, 490
    HL_F, HL_B = "#fef9c3", "#ca8a04"   # 高亮黄
    p = [f'<text x="{W/2}" y="44" text-anchor="middle" class="title">'
         f'点积打分：S = QKᵀ 一次矩阵乘，算出全部 L×L 个匹配分（例：L=3, d_k=2）</text>']

    cell = 58
    cy = 270  # 垂直对齐中心

    def matrix(ox, oy, vals, fill, border, hl_rows=(), hl_cols=(), hl_cells=()):
        rows, cols = len(vals), len(vals[0])
        out = []
        for i in range(rows):
            for j in range(cols):
                x, y = ox + j * cell, oy + i * cell
                f, b = fill, border
                if (i, j) in hl_cells or i in hl_rows or j in hl_cols:
                    f, b = HL_F, HL_B
                out.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" rx="6" '
                           f'fill="{f}" stroke="{b}" stroke-width="1.7"/>')
                out.append(f'<text x="{x+(cell-4)/2}" y="{y+(cell-4)/2+5}" text-anchor="middle" '
                           f'class="mono">{esc(str(vals[i][j]))}</text>')
        return out

    Q = [[1, 0], [0, 1], [1, 1]]                 # 行 = q1,q2,q3
    Kt = [[1, 1, 0], [0, 1, 1]]                  # Kᵀ：列 = k1,k2,k3
    S = [[1, 1, 0], [0, 1, 1], [1, 2, 1]]        # S = QKᵀ

    # --- Q [3,2] ---（qx 选定让 Q→Kᵀ→S 整组在画面里水平居中）
    qx, qy = 518, cy - 3 * cell / 2
    p.append(f'<text x="{qx + cell}" y="{qy-30}" text-anchor="middle" class="lbl" fill="{BLUE_B}">Q  [L=3, d_k=2]</text>')
    p += matrix(qx, qy, Q, BLUE_F, BLUE_B, hl_rows=(1,))
    for i, nm in enumerate(["q₁", "q₂", "q₃"]):
        p.append(f'<text x="{qx-12}" y="{qy + i*cell + (cell-4)/2 + 5}" text-anchor="end" class="monos" fill="{SUB}">{nm}</text>')

    # × 号
    mx = qx + 2 * cell + 42
    p.append(f'<text x="{mx}" y="{cy+8}" text-anchor="middle" class="monob" fill="{SUB}">×</text>')

    # --- Kᵀ [2,3] ---
    kx, ky = mx + 42, cy - cell
    p.append(f'<text x="{kx + 3*cell/2}" y="{ky-30}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">Kᵀ  [d_k=2, L=3]</text>')
    p += matrix(kx, ky, Kt, ORANGE_F, ORANGE_B, hl_cols=(2,))
    for j, nm in enumerate(["k₁", "k₂", "k₃"]):
        p.append(f'<text x="{kx + j*cell + (cell-4)/2}" y="{ky + 2*cell + 22}" text-anchor="middle" class="monos" fill="{SUB}">{nm}</text>')

    # = 号
    ex = kx + 3 * cell + 42
    p.append(f'<text x="{ex}" y="{cy+8}" text-anchor="middle" class="monob" fill="{SUB}">=</text>')

    # --- S [3,3] ---
    sx, sy = ex + 42, cy - 3 * cell / 2
    p.append(f'<text x="{sx + 3*cell/2}" y="{sy-30}" text-anchor="middle" class="lbl" fill="{PURPLE_B}">S = QKᵀ  [L=3, L=3]</text>')
    p += matrix(sx, sy, S, PURPLE_F, PURPLE_B, hl_cells=((1, 2),))
    for i, nm in enumerate(["q₁", "q₂", "q₃"]):
        p.append(f'<text x="{sx-12}" y="{sy + i*cell + (cell-4)/2 + 5}" text-anchor="end" class="monos" fill="{SUB}">{nm}</text>')
    for j, nm in enumerate(["k₁", "k₂", "k₃"]):
        p.append(f'<text x="{sx + j*cell + (cell-4)/2}" y="{sy - 6}" text-anchor="middle" class="monos" fill="{SUB}">{nm}</text>')

    # 高亮单元格的算式注解
    p.append(f'<rect x="{W/2 - 300}" y="{H-110}" width="600" height="50" rx="10" '
             f'fill="{HL_F}" stroke="{HL_B}" stroke-width="1.6"/>')
    p.append(f'<text x="{W/2}" y="{H-78}" text-anchor="middle" class="mono">'
             f'高亮格 S₂₃ = q₂ · k₃ = (0, 1) · (0, 1) = 0×0 + 1×1 = 1</text>')
    p.append(f'<text x="{W/2}" y="{H-32}" text-anchor="middle" class="cap">'
             f'S 的第 i 行 j 列 = Q 第 i 行（qᵢ）与 Kᵀ 第 j 列（kⱼ）的点积 —— 一个矩阵乘把 L×L 个点积全算完。</text>')
    write_svg(ASSETS / "qk-score-matmul.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_qkv()
    diagram_pipeline()
    diagram_causal()
    diagram_qk_matmul()
    print("wrote 4 SVGs to", ASSETS)
