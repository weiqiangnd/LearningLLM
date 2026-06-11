"""Generate conceptual diagrams for chapter 05 (从 RNN 到 attention).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
Hand-written SVG (same approach as assets/04/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses
gray-700 (#374151) or darker per the repo contrast guideline.

Run from repo root:
    python3 assets/05/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant, e.g.
    rsvg-convert -w 1440 assets/05/seq2seq-bottleneck.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/05/seq2seq-bottleneck.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Zen Hei', sans-serif")
MONO = "'SFMono-Regular','Consolas','Liberation Mono',monospace"

BG = "#ffffff"
TXT = "#1e293b"   # slate-800  primary labels
SUB = "#334155"   # slate-700  secondary / sublabel / caption (>= gray-700)

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
    .title  {{ font-size: 23px; font-weight: 700; fill: {TXT}; }}
    .lbl    {{ font-size: 15px; font-weight: 600; fill: {TXT}; }}
    .sub    {{ font-size: 13px; fill: {SUB}; }}
    .mono   {{ font-size: 14px; font-family: {MONO}; fill: {TXT}; }}
    .monos  {{ font-size: 12px; font-family: {MONO}; fill: {TXT}; }}
    .small  {{ font-size: 12.5px; fill: {SUB}; }}
    .cap    {{ font-size: 13.5px; fill: {SUB}; }}
    .tag    {{ font-size: 12px; font-weight: 600; }}
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


# ============================================================
# Diagram 1: seq2seq 的定长 context 瓶颈（为什么 attention 会出现）
# ============================================================
def diagram_bottleneck():
    W, H = 1280, 600
    p = [f'<text x="{W/2}" y="42" text-anchor="middle" class="title">'
         f'seq2seq 的瓶颈：整句话被压进一个定长向量 c —— 句子越长越记不住</text>']

    cw, ch = 92, 58           # RNN cell 尺寸
    ey = 200                  # encoder 行 y 中心
    dy = 482                  # decoder 行 y 中心
    xs = [185, 325, 465, 605] # encoder 列 x
    src = ["x₁", "x₂", "x₃", "x₄"]

    # --- encoder header ---
    p.append(f'<text x="395" y="78" text-anchor="middle" class="lbl" fill="{BLUE_B}">编码器 Encoder：RNN 顺序读源序列</text>')
    # encoder input tokens + cells + recurrence arrows
    for i, x in enumerate(xs):
        p += node(x, ey - 90, cw - 6, 36, src[i], "", GRAY_F, GRAY_B, label_cls="mono")
        p += arrow(x, ey - 72, x, ey - 30, marker="aGray", color=GRAY_B)
        p += node(x, ey, cw, ch, "RNN", f"h{chr(0x2080+i+1)}", BLUE_F, BLUE_B)
        if i < len(xs) - 1:
            p += arrow(x + cw/2, ey, xs[i+1] - cw/2, ey, marker="aBlue", color=BLUE_B)
    p.append(f'<text x="{xs[0]-cw/2-6}" y="{ey+5}" text-anchor="end" class="small">h₀</text>')

    # --- context vector (the bottleneck) ---
    ctxx = 760
    p += arrow(xs[-1] + cw/2, ey, ctxx - 70, ey, marker="aRed", color=RED_B, width=2.6)
    p.append(f'<rect x="{ctxx-65}" y="{ey-78}" width="130" height="156" rx="13" '
             f'fill="{RED_F}" stroke="{RED_B}" stroke-width="2.6"/>')
    p.append(f'<text x="{ctxx}" y="{ey-44}" text-anchor="middle" class="lbl" fill="{RED_B}">context</text>')
    p.append(f'<text x="{ctxx}" y="{ey-20}" text-anchor="middle" class="mono" fill="{RED_B}">c = h₄</text>')
    # small fixed-length vector cells inside
    for r in range(4):
        yy = ey + 2 + r * 17
        p.append(f'<rect x="{ctxx-22}" y="{yy}" width="44" height="14" fill="#ffffff" stroke="{RED_B}" stroke-width="1.2"/>')
    p.append(f'<text x="{ctxx}" y="{ey+92}" text-anchor="middle" class="small" fill="{RED_B}">定长！</text>')

    # --- decoder ---
    p.append(f'<text x="395" y="372" text-anchor="middle" class="lbl" fill="{GREEN_B}">解码器 Decoder：只拿着这一个 c 往外吐</text>')
    dxs = [185, 325, 465, 605]
    tgt = ["y₁", "y₂", "y₃", "y₄"]
    for i, x in enumerate(dxs):
        p += node(x, dy, cw, ch, "RNN", f"s{chr(0x2080+i+1)}", GREEN_F, GREEN_B)
        p += arrow(x, dy - ch/2, x, dy - ch/2 - 30, marker="aGreen", color=GREEN_B)
        p += node(x, dy - ch/2 - 48, cw - 6, 34, tgt[i], "", GRAY_F, GRAY_B, label_cls="mono")
        if i < len(dxs) - 1:
            p += arrow(x + cw/2, dy, dxs[i+1] - cw/2, dy, marker="aGreen", color=GREEN_B)
    # context feeds the decoder start —— 走左侧空带绕开 token / cell，从 s₁ 左边进入
    s1x = dxs[0] - cw / 2
    p.append(f'<path d="M {ctxx} {ey+78} C 520 315, 200 315, 110 360 C 110 430, 110 {dy}, {s1x} {dy}" '
             f'fill="none" stroke="{RED_B}" stroke-width="2.4" marker-end="url(#aRed)"/>')
    p.append(f'<text x="125" y="350" class="small" fill="{RED_B}">c 是 decoder 唯一的「源信息」</text>')

    # --- bottleneck annotation band ---
    by = 540
    p.append(f'<rect x="60" y="{by}" width="{W-120}" height="52" rx="11" '
             f'fill="{RED_F}" stroke="{RED_B}" stroke-width="1.8"/>')
    p.append(f'<text x="{W/2}" y="{by+32}" text-anchor="middle" class="lbl" fill="{RED_B}">'
             f'瓶颈：无论源句多长，全部信息都得挤进这一个固定大小的 c —— 长句的开头很容易被「冲掉」</text>')

    # right-side: parameter sharing note
    p.append(f'<text x="{ctxx+120}" y="{ey-30}" class="small">同一套 RNN 权重</text>')
    p.append(f'<text x="{ctxx+120}" y="{ey-10}" class="small">在每一步反复使用</text>')

    write_svg(ASSETS / "seq2seq-bottleneck.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: Bahdanau attention 三步（打分 → softmax → 加权求和）
# ============================================================
def diagram_attention():
    W, H = 1280, 600
    p = [f'<text x="{W/2}" y="42" text-anchor="middle" class="title">'
         f'Attention：解码每一步「回看」编码器全部隐藏态，动态算一个 cₜ</text>']

    # encoder hidden states (left column)
    hx = 150
    hy0, dh = 120, 92
    hs = ["h₁", "h₂", "h₃", "h₄"]
    p.append(f'<text x="{hx}" y="{hy0-34}" text-anchor="middle" class="lbl" fill="{BLUE_B}">编码器隐藏态</text>')
    hcy = []
    for i, lab in enumerate(hs):
        cy = hy0 + i * dh
        hcy.append(cy)
        p += node(hx, cy, 108, 52, lab, f"源位置 {i+1}", BLUE_F, BLUE_B)

    # decoder hidden s_{t-1} (query) —— 放在打分列正下方，单根 dashed 箭头表示「参与每个打分」
    qx, qy = 440, 510
    p += node(qx, qy, 188, 58, "sₜ₋₁", "解码器上一步隐藏 = query", GREEN_F, GREEN_B)
    p.append(f'<text x="{qx}" y="{qy-40}" text-anchor="middle" class="small" fill="{SUB}">「我现在想生成什么」</text>')

    # step 1: score column
    scx = 440
    p.append(f'<text x="{scx}" y="{hy0-34}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">① 打分 eₜⱼ</text>')
    p.append(f'<text x="{scx}" y="{hy0-14}" text-anchor="middle" class="small">align(sₜ₋₁, hⱼ)</text>')
    raw = ["0.4", "2.1", "0.7", "-0.3"]
    for i, cy in enumerate(hcy):
        p += node(scx, cy, 92, 46, raw[i], "", ORANGE_F, ORANGE_B, label_cls="mono")
        p += arrow(hx + 54, cy, scx - 46, cy, marker="aBlue", color=BLUE_B)
    # query 参与每个打分：从 s 往上一根 dashed 箭头指向打分列
    p.append(f'<path d="M {qx} {qy-29} L {qx} {hcy[-1]+27}" '
             f'fill="none" stroke="{GREEN_B}" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#aGreen)"/>')
    p.append(f'<text x="{qx+14}" y="{(qy+hcy[-1])/2}" class="small" fill="{GREEN_B}">query 和每个 hⱼ 配对打分</text>')

    # step 2: softmax -> weights
    wx = 660
    p.append(f'<text x="{wx}" y="{hy0-34}" text-anchor="middle" class="lbl" fill="{PURPLE_B}">② softmax</text>')
    p.append(f'<text x="{wx}" y="{hy0-14}" text-anchor="middle" class="small">αₜⱼ（和为 1）</text>')
    weights = [0.12, 0.66, 0.16, 0.06]   # = softmax([0.4, 2.1, 0.7, -0.3])，与上面的 raw 分一致
    for i, cy in enumerate(hcy):
        p += arrow(scx + 46, cy, wx - 70, cy, marker="aOrange", color=ORANGE_B)
        # weight bar
        bw = 8 + weights[i] * 150
        p.append(f'<rect x="{wx-70}" y="{cy-13}" width="{bw}" height="26" rx="5" fill="{PURPLE_F}" stroke="{PURPLE_B}" stroke-width="1.6"/>')
        p.append(f'<text x="{wx-70+bw+8}" y="{cy+5}" class="monos" fill="{PURPLE_B}">{weights[i]:.2f}</text>')

    # step 3: weighted sum -> context c_t
    cx, cy_ = 980, hy0 + 1.5 * dh
    p.append(f'<text x="{cx}" y="{hy0-34}" text-anchor="middle" class="lbl" fill="{RED_B}">③ 加权求和</text>')
    p.append(f'<text x="{cx}" y="{hy0-14}" text-anchor="middle" class="small">cₜ = Σ αₜⱼ hⱼ</text>')
    for i, cy in enumerate(hcy):
        p += arrow(wx + 95, cy, cx - 58, cy_, marker="aPurple", color=PURPLE_B)
    p += node(cx, cy_, 116, 64, "cₜ", "上下文向量", RED_F, RED_B)

    # context c_t -> 生成本步输出 y_t（cₜ 与 sₜ₋₁ 一起）
    p += node(cx, cy_ + 150, 150, 56, "yₜ", "本步输出 token", GREEN_F, GREEN_B)
    p += arrow(cx, cy_ + 32, cx, cy_ + 150 - 28, marker="aRed", color=RED_B, width=2.2)
    p.append(f'<text x="{cx+86}" y="{cy_+95}" text-anchor="middle" class="small" fill="{SUB}">cₜ 与 sₜ₋₁</text>')
    p.append(f'<text x="{cx+86}" y="{cy_+113}" text-anchor="middle" class="small" fill="{SUB}">一起生成 yₜ</text>')

    # bottom band
    by = 548
    p.append(f'<rect x="60" y="{by}" width="{W-120}" height="46" rx="11" '
             f'fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.6"/>')
    p.append(f'<text x="{W/2}" y="{by+29}" text-anchor="middle" class="lbl" fill="{BLUE_B}">'
             f'关键：cₜ 每一步都不一样 —— 该看源句哪里，由当前 query 和各 hⱼ 的匹配度（注意力权重）动态决定</text>')

    write_svg(ASSETS / "attention-mechanism.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: 从「挂在 RNN 上的 attention」到 self-attention（承上启下）
# ============================================================
def diagram_bridge():
    W, H = 1280, 560
    p = [f'<text x="{W/2}" y="42" text-anchor="middle" class="title">'
         f'抽掉 RNN、只留 attention，并让序列对「自己」做注意力 → self-attention</text>']

    # left panel: RNN + attention (this chapter)
    lx, lw = 60, 540
    p.append(f'<rect x="{lx}" y="80" width="{lw}" height="400" rx="14" fill="#f8fafc" stroke="{BLUE_B}" stroke-width="2"/>')
    p.append(f'<text x="{lx+lw/2}" y="112" text-anchor="middle" class="lbl" fill="{BLUE_B}">本章：RNN + attention</text>')
    # encoder states row
    exs = [lx+110, lx+210, lx+310, lx+410]
    for i, x in enumerate(exs):
        p += node(x, 180, 78, 44, f"h{chr(0x2080+i+1)}", "", BLUE_F, BLUE_B, label_cls="mono")
        if i < 3:
            p += arrow(x+39, 180, exs[i+1]-39, 180, marker="aBlue", color=BLUE_B)
    p.append(f'<text x="{lx+lw/2}" y="150" text-anchor="middle" class="small">encoder 隐藏态（仍由 RNN 顺序产生）</text>')
    # decoder query attends to all
    p += node(lx+lw/2, 380, 150, 54, "decoder sₜ", "query", GREEN_F, GREEN_B)
    for x in exs:
        p.append(f'<path d="M {lx+lw/2} 353 C {lx+lw/2} 300, {x} 320, {x} 204" '
                 f'fill="none" stroke="{ORANGE_B}" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#aOrange)"/>')
    p.append(f'<text x="{lx+lw/2}" y="430" text-anchor="middle" class="small" fill="{SUB}">attention 只是「外挂」在 RNN 上；token 仍要顺序算</text>')

    # middle arrow
    mx = 620
    p += arrow(mx, 280, mx+58, 280, marker="aRed", color=RED_B, width=3)
    p.append(f'<text x="{mx+29}" y="262" text-anchor="middle" class="tag" fill="{RED_B}">去掉 RNN</text>')

    # right panel: self-attention (chapter 6)
    rx, rw = 700, 520
    p.append(f'<rect x="{rx}" y="80" width="{rw}" height="400" rx="14" fill="#f8fafc" stroke="{PURPLE_B}" stroke-width="2"/>')
    p.append(f'<text x="{rx+rw/2}" y="112" text-anchor="middle" class="lbl" fill="{PURPLE_B}">第 6 章：self-attention</text>')
    txs = [rx+110, rx+230, rx+350, rx+440]
    toks = ["x₁", "x₂", "x₃", "x₄"]
    ty = 300
    for i, x in enumerate(txs):
        p += node(x, ty, 70, 46, toks[i], "", PURPLE_F, PURPLE_B, label_cls="mono")
    # every token attends to every token (draw a few arcs from x3)
    src_x = txs[2]
    for x in txs:
        if x == src_x:
            continue
        p.append(f'<path d="M {src_x} {ty-23} C {src_x} {ty-90}, {x} {ty-90}, {x} {ty-23}" '
                 f'fill="none" stroke="{PURPLE_B}" stroke-width="1.5" marker-end="url(#aPurple)"/>')
    p.append(f'<text x="{rx+rw/2}" y="155" text-anchor="middle" class="small">每个 token 直接对所有 token 做注意力（这里画的是 x₃ 看全体）</text>')
    p.append(f'<text x="{rx+rw/2}" y="370" text-anchor="middle" class="sub">每个 token 既出 query，又出 key / value</text>')
    p.append(f'<text x="{rx+rw/2}" y="394" text-anchor="middle" class="small" fill="{SUB}">没有循环，全部位置并行 —— 这就是 Transformer 的内核</text>')

    # bottom band
    by = 506
    p.append(f'<rect x="60" y="{by}" width="{W-120}" height="44" rx="11" '
             f'fill="{PURPLE_F}" stroke="{PURPLE_B}" stroke-width="1.6"/>')
    p.append(f'<text x="{W/2}" y="{by+28}" text-anchor="middle" class="lbl" fill="{PURPLE_B}">'
             f'attention 从 RNN 的「配角」变成唯一主角；代价是丢了顺序，于是需要第 4 章的位置编码补回来</text>')

    write_svg(ASSETS / "rnn-to-selfattention.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: 对齐矩阵示例（真实英→法翻译，Bahdanau 2014 Fig.3 片段）
# ============================================================
def diagram_alignment():
    # 源句（列）：英语；译文（行）：法语
    cols = ["the", "European", "Economic", "Area"]
    rows = ["la", "zone", "économique", "européenne"]
    # 每行 ≈ 一个 softmax 分布（和 ≈ 1）：注意力权重 αₜⱼ
    A = [
        [0.90, 0.03, 0.03, 0.04],   # la         → the
        [0.05, 0.05, 0.05, 0.85],   # zone       → Area
        [0.02, 0.08, 0.88, 0.02],   # économique → Economic
        [0.02, 0.92, 0.04, 0.02],   # européenne → European
    ]

    cell = 76
    gx, gy = 300, 178                       # 网格左上角
    nC, nR = len(cols), len(rows)
    gw, gh = nC * cell, nR * cell
    W, H = 940, 682

    p = [f'<text x="{W/2}" y="44" text-anchor="middle" class="title">'
         f'对齐矩阵示例：英 → 法翻译里 attention「看」哪个源词</text>']
    p.append(f'<text x="{W/2}" y="70" text-anchor="middle" class="sub">'
             f'取自 Bahdanau 2014 论文 Fig.3 的片段 “… the European Economic Area …” → “… la zone économique européenne …”</text>')

    # 轴名
    p.append(f'<text x="{gx+gw/2}" y="{gy-58}" text-anchor="middle" class="lbl" fill="{BLUE_B}">源句（英语原文）→ 列 j</text>')
    p.append(f'<text x="{gx-150}" y="{gy+gh/2}" text-anchor="middle" class="lbl" fill="{GREEN_B}" '
             f'transform="rotate(-90 {gx-150} {gy+gh/2})">译文（法语，逐词生成）→ 行 t</text>')

    # 列头（英语源词，斜排避免拥挤）
    for c, w in enumerate(cols):
        cxh = gx + c * cell + cell / 2
        p.append(f'<text x="{cxh}" y="{gy-12}" text-anchor="start" class="mono" '
                 f'transform="rotate(-38 {cxh} {gy-12})">{esc(w)}</text>')
    # 行头（法语译词）
    for r, w in enumerate(rows):
        cyh = gy + r * cell + cell / 2
        p.append(f'<text x="{gx-12}" y="{cyh+5}" text-anchor="end" class="mono">{esc(w)}</text>')

    # 热力格子：蓝色，越深 = 权重越大（fill-opacity ∝ α）
    for r in range(nR):
        for c in range(nC):
            a = A[r][c]
            x, y = gx + c * cell, gy + r * cell
            op = 0.06 + 0.94 * a              # 留一点底色，0 也不至于纯白
            p.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                     f'fill="{BLUE_B}" fill-opacity="{op:.3f}" stroke="#ffffff" stroke-width="3"/>')
            if a >= 0.10:                      # 只在显著格子里标数值
                tcol = "#ffffff" if a >= 0.5 else TXT
                p.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" '
                         f'class="monos" fill="{tcol}">{a:.2f}</text>')

    # 高亮「语序翻转」的 3×3 反对角线块（European/Economic/Area × zone/économique/européenne）
    bx, by = gx + cell, gy + cell
    p.append(f'<rect x="{bx}" y="{by}" width="{3*cell}" height="{3*cell}" rx="6" '
             f'fill="none" stroke="{ORANGE_B}" stroke-width="2.6" stroke-dasharray="7,5"/>')
    p.append(f'<text x="{bx+3*cell+14}" y="{by+3*cell/2-8}" class="lbl" fill="{ORANGE_B}">反对角线</text>')
    p.append(f'<text x="{bx+3*cell+14}" y="{by+3*cell/2+12}" class="small" fill="{SUB}">形-名语序</text>')
    p.append(f'<text x="{bx+3*cell+14}" y="{by+3*cell/2+30}" class="small" fill="{SUB}">英法相反</text>')

    # 颜色图例（横向渐变条）
    lgx, lgy, lgw = gx, gy + gh + 42, gw
    p.append(f'<defs><linearGradient id="alnGrad" x1="0" y1="0" x2="1" y2="0">'
             f'<stop offset="0" stop-color="{BLUE_B}" stop-opacity="0.06"/>'
             f'<stop offset="1" stop-color="{BLUE_B}" stop-opacity="1"/></linearGradient></defs>')
    p.append(f'<rect x="{lgx}" y="{lgy}" width="{lgw}" height="16" rx="3" '
             f'fill="url(#alnGrad)" stroke="{SUB}" stroke-width="0.8"/>')
    p.append(f'<text x="{lgx}" y="{lgy+34}" text-anchor="start" class="small">α≈0（几乎不看）</text>')
    p.append(f'<text x="{lgx+lgw}" y="{lgy+34}" text-anchor="end" class="small">α≈1（强对齐）</text>')
    p.append(f'<text x="{lgx+lgw/2}" y="{lgy-6}" text-anchor="middle" class="small" fill="{SUB}">'
             f'格子颜色 = 注意力权重 αₜⱼ（每一行是一个 softmax 分布、横向和为 1）</text>')

    # 底部说明带（两行，避免单行溢出框宽）
    cy_band = lgy + 56
    p.append(f'<rect x="48" y="{cy_band}" width="{W-96}" height="70" rx="11" '
             f'fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="1.6"/>')
    p.append(f'<text x="{W/2}" y="{cy_band+28}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">'
             f'“la” 顺位对齐 “the” —— 对角线起步正常；</text>')
    p.append(f'<text x="{W/2}" y="{cy_band+52}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">'
             f'但 “European Economic Area” → “zone économique européenne” 形-名语序翻转，对角线在此「拐弯」成反对角线</text>')

    write_svg(ASSETS / "alignment-matrix.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 5: 时机之差 —— attention 算在 RNN 单元之前(Bahdanau) vs 之后(Luong)
# ============================================================
def stage_box(cx, cy, w, h, title, subs, fill, border):
    """一个阶段框：粗体标题 + 最多两行小字 sub。"""
    x, y = cx - w / 2, cy - h / 2
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
           f'fill="{fill}" stroke="{border}" stroke-width="2"/>']
    if subs:
        out.append(f'<text x="{cx}" y="{cy-12}" text-anchor="middle" class="lbl">{esc(title)}</text>')
        for i, s in enumerate(subs):
            out.append(f'<text x="{cx}" y="{cy+8+i*18}" text-anchor="middle" class="small">{esc(s)}</text>')
    else:
        out.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" class="lbl">{esc(title)}</text>')
    return out


def diagram_timing():
    W, H = 1280, 624
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'时机之差：attention 算在 RNN 单元「之前」(Bahdanau) 还是「之后」(Luong)</text>']

    bw, bh = 250, 74          # 阶段框尺寸
    hw, hh = 96, 104          # 编码器 H 框尺寸
    sy = [270, 372, 474]      # 三个阶段的 y 中心
    chip_y = 185              # 「上一步状态」chip 行

    def panel(px, accent, ptitle, psub, order, h_cx, h_stage_idx):
        pw, ph = 556, 456
        cx = px + pw / 2
        out = [f'<rect x="{px}" y="92" width="{pw}" height="{ph}" rx="14" '
               f'fill="#f8fafc" stroke="{accent}" stroke-width="2"/>']
        out.append(f'<text x="{cx}" y="124" text-anchor="middle" class="lbl" fill="{accent}">{esc(ptitle)}</text>')
        out.append(f'<text x="{cx}" y="148" text-anchor="middle" class="sub">{esc(psub)}</text>')

        # 上一步状态 chip
        out += stage_box(cx, chip_y, bw, 40, "上一步状态  yₜ₋₁ , sₜ₋₁", [], GRAY_F, GRAY_B)
        out += arrow(cx, chip_y + 20, cx, sy[0] - bh / 2, marker="aGray", color=GRAY_B)

        # 三个阶段框（order 给出每个阶段的 kind / 文案）
        for i, (kind, title, subs) in enumerate(order):
            fill, border = (ORANGE_F, ORANGE_B) if kind == "attn" else \
                           (BLUE_F, BLUE_B) if kind == "rnn" else (GREEN_F, GREEN_B)
            out += stage_box(cx, sy[i], bw, bh, title, subs, fill, border)

        # 阶段间主箭头（携带产出量的标注）
        flows = ["cₜ", "sₜ"] if order[0][0] == "attn" else ["sₜ", "cₜ"]
        for i in range(2):
            mk, col = ("aOrange", ORANGE_B) if flows[i] == "cₜ" else ("aBlue", BLUE_B)
            y1, y2 = sy[i] + bh / 2, sy[i + 1] - bh / 2
            out += arrow(cx, y1, cx, y2, marker=mk, color=col, width=2.2)
            out.append(f'<text x="{cx+14}" y="{(y1+y2)/2+5}" class="mono" fill="{col}">{flows[i]}</text>')

        # 编码器 H 框 + 虚线喂给 attention 阶段
        a_cy = sy[h_stage_idx]
        out.append(f'<rect x="{h_cx-hw/2}" y="{a_cy-hh/2}" width="{hw}" height="{hh}" rx="11" '
                   f'fill="{TEAL_F}" stroke="{TEAL_B}" stroke-width="2"/>')
        out.append(f'<text x="{h_cx}" y="{a_cy-22}" text-anchor="middle" class="lbl" fill="{TEAL_B}">H</text>')
        out.append(f'<text x="{h_cx}" y="{a_cy-2}" text-anchor="middle" class="small">编码器</text>')
        out.append(f'<text x="{h_cx}" y="{a_cy+16}" text-anchor="middle" class="small">全部</text>')
        out.append(f'<text x="{h_cx}" y="{a_cy+34}" text-anchor="middle" class="small">隐藏态</text>')
        # H 在外侧：左面板 H 在 attention 左边，右面板 H 在 attention 右边
        if h_cx < cx:
            out += arrow(h_cx + hw/2, a_cy, cx - bw/2, a_cy, marker="aTeal", color=TEAL_B, dashed=True, width=1.8)
        else:
            out += arrow(h_cx - hw/2, a_cy, cx + bw/2, a_cy, marker="aTeal", color=TEAL_B, dashed=True, width=1.8)
        return out

    # 左：Bahdanau —— attention 在 RNN 之前；query = sₜ₋₁
    p += panel(
        56, ORANGE_B, "Bahdanau（加性，2014）", "attention 在 RNN 单元【之前】",
        order=[
            ("attn", "① Attention", ["query = sₜ₋₁ → 算 cₜ"]),
            ("rnn",  "② RNN 单元", ["输入 [yₜ₋₁ ; cₜ]", "更新出 sₜ"]),
            ("out",  "③ 预测 yₜ", ["由 sₜ（+cₜ）投影"]),
        ],
        h_cx=130, h_stage_idx=0)

    # 右：Luong —— attention 在 RNN 之后；query = sₜ
    p += panel(
        668, BLUE_B, "Luong（乘性，2015）", "attention 在 RNN 单元【之后】",
        order=[
            ("rnn",  "① RNN 单元", ["输入 yₜ₋₁ ，由 sₜ₋₁", "更新出 sₜ"]),
            ("attn", "② Attention", ["query = sₜ → 算 cₜ"]),
            ("out",  "③ 预测 yₜ", ["由 [sₜ ; cₜ] 投影"]),
        ],
        h_cx=1160, h_stage_idx=1)

    # 底部说明带
    by = 562
    p.append(f'<rect x="56" y="{by}" width="{W-112}" height="46" rx="11" '
             f'fill="{GRAY_F}" stroke="{SUB}" stroke-width="1.6"/>')
    p.append(f'<text x="{W/2}" y="{by+29}" text-anchor="middle" class="lbl" fill="{TXT}">'
             f'三步套路（打分 → softmax → 加权求和）完全相同 —— 只差 attention 摆在 RNN 单元前/后，query 相应地取 sₜ₋₁ / sₜ</text>')

    write_svg(ASSETS / "attention-timing.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_bottleneck()
    diagram_attention()
    diagram_bridge()
    diagram_alignment()
    diagram_timing()
    print("wrote 5 SVGs to", ASSETS)
