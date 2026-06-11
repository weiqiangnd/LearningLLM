"""Generate conceptual diagrams for chapter 04 (Embedding 与位置编码).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
Hand-written SVG (same approach as assets/03/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses
gray-700 (#374151) or darker per the repo contrast guideline.

Run from repo root:
    python3 assets/04/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant, e.g.
    rsvg-convert -w 1400 assets/04/embedding-lookup.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/04/embedding-lookup.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Zen Hei', sans-serif")
MONO = "'SFMono-Regular','Consolas','Liberation Mono',monospace"

# ---------- shared palette (Flat Icon) ----------
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
# Diagram 1: token embedding lookup (one-hot x E = pick a row)
# ============================================================
def diagram_embedding_lookup():
    W, H = 1240, 560
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'Token Embedding：查表的本质 —— one-hot × 矩阵 E = 抽出对应那一行</text>']

    # left: input id
    p += node(120, 175, 150, 60, "token id", "t = 2", GRAY_F, GRAY_B)
    p += arrow(196, 175, 286, 175, marker="aGray", color=GRAY_B)

    # one-hot vector (vertical 5-cell column)
    ohx, ohy, cell = 300, 95, 32
    p.append(f'<text x="{ohx+cell/2}" y="{ohy-12}" text-anchor="middle" class="sub">one-hot (V=5)</text>')
    onehot = [0, 0, 1, 0, 0]
    for r, v in enumerate(onehot):
        fill = ORANGE_F if v == 1 else "#ffffff"
        bd = ORANGE_B if v == 1 else GRAY_B
        yy = ohy + r * cell
        p.append(f'<rect x="{ohx}" y="{yy}" width="{cell}" height="{cell}" '
                 f'fill="{fill}" stroke="{bd}" stroke-width="1.6"/>')
        p.append(f'<text x="{ohx+cell/2}" y="{yy+21}" text-anchor="middle" class="monos">{v}</text>')
        p.append(f'<text x="{ohx-12}" y="{yy+21}" text-anchor="end" class="monos" fill="{SUB}">id {r}</text>')

    # multiply sign
    p.append(f'<text x="{ohx+cell+34}" y="180" text-anchor="middle" class="title">×</text>')

    # E matrix [V=5, d=4]
    ex, ey, ew, eh = ohx + cell + 70, ohy, 4 * 40, 5 * cell
    p.append(f'<text x="{ex+ew/2}" y="{ey-12}" text-anchor="middle" class="sub">embedding 矩阵 E   [V=5, d=4]</text>')
    vals = [
        [".11", "-.3", ".02", ".5"],
        ["-.2", ".4", ".1", "-.1"],
        [".9", "-.6", ".3", ".7"],   # row 2 -> highlighted
        [".0", ".2", "-.4", ".3"],
        ["-.5", ".1", ".6", "-.2"],
    ]
    for r in range(5):
        for c in range(4):
            xx = ex + c * 40
            yy = ey + r * cell
            hl = (r == 2)
            fill = ORANGE_F if hl else "#ffffff"
            bd = ORANGE_B if hl else GRAY_B
            p.append(f'<rect x="{xx}" y="{yy}" width="40" height="{cell}" '
                     f'fill="{fill}" stroke="{bd}" stroke-width="{1.8 if hl else 1.2}"/>')
            p.append(f'<text x="{xx+20}" y="{yy+21}" text-anchor="middle" class="monos">{vals[r][c]}</text>')
    # row-2 callout
    p.append(f'<text x="{ex+ew+14}" y="{ey+2*cell+21}" class="small" fill="{ORANGE_B}">← 第 2 行</text>')

    # equals -> output vector
    p.append(f'<text x="{ex+ew+96}" y="180" text-anchor="middle" class="title">=</text>')
    rx = ex + ew + 130
    p.append(f'<text x="{rx+80}" y="{ey-12}" text-anchor="middle" class="sub">embedding 向量 e_t   [d=4]</text>')
    for c in range(4):
        xx = rx + c * 40
        p.append(f'<rect x="{xx}" y="{ey+2*cell}" width="40" height="{cell}" '
                 f'fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="1.8"/>')
        p.append(f'<text x="{xx+20}" y="{ey+2*cell+21}" text-anchor="middle" class="monos">{vals[2][c]}</text>')
    p.append(f'<text x="{rx+80}" y="{ey+3*cell+24}" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">就是 E 的第 2 行</text>')

    # middle emphasis band
    my = 360
    p.append(f'<rect x="60" y="{my}" width="{W-120}" height="50" rx="11" '
             f'fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.8"/>')
    p.append(f'<text x="{W/2}" y="{my+31}" text-anchor="middle" class="lbl" fill="{BLUE_B}">'
             f'one-hot 乘矩阵 = 抽出对应行 → 实现上直接「按行号索引（查表）」即可，不必真做大矩阵乘法</text>')

    # bottom: weight tying note
    ty = 445
    p.append(f'<rect x="60" y="{ty}" width="{W-120}" height="92" rx="11" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,3"/>')
    p.append(f'<text x="80" y="{ty+28}" class="lbl">weight tying（权重共享）</text>')
    p.append(f'<text x="80" y="{ty+54}" class="sub">输入端 embedding 矩阵 E 形状 [V, d]；输出端 lm_head 形状 [d, V] —— 二者互为转置。</text>')
    p.append(f'<text x="80" y="{ty+76}" class="sub">很多模型令 W_out = E^T，让「id→向量」与「向量→词表分数」共享同一套语义，省一份参数。</text>')

    write_svg(ASSETS / "embedding-lookup.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: self-attention is permutation-equivariant
# ============================================================
def diagram_why_position():
    W, H = 1180, 560
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'自注意力看不见顺序：打乱输入，token a 的注意力分数集合不变</text>']

    def token_row(y, order, title, color_b):
        p.append(f'<text x="60" y="{y-44}" class="lbl">{esc(title)}</text>')
        labels = order
        cx0 = 90
        centers = {}
        for idx, t in enumerate(labels):
            cx = cx0 + idx * 95
            fill = {"a": BLUE_F, "b": GREEN_F, "c": PURPLE_F}[t]
            bd = {"a": BLUE_B, "b": GREEN_B, "c": PURPLE_B}[t]
            p.append(f'<rect x="{cx}" y="{y-26}" width="62" height="52" rx="10" '
                     f'fill="{fill}" stroke="{bd}" stroke-width="2"/>')
            p.append(f'<text x="{cx+31}" y="{y+7}" text-anchor="middle" class="lbl">{esc(t)}</text>')
            centers[t] = cx + 31
        return centers

    # row 1: original order a b c
    y1 = 150
    c1 = token_row(y1, ["a", "b", "c"], "原始顺序 (a, b, c)", BLUE_B)
    # token a's scores
    p.append(f'<text x="370" y="{y1-8}" class="sub">token a 算出的分数集合：</text>')
    p.append(f'<rect x="370" y="{y1+4}" width="360" height="34" rx="8" fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.6"/>')
    p.append(f'<text x="386" y="{y1+26}" class="mono">{{ a·a , a·b , a·c }}</text>')

    # row 2: shuffled order c b a
    y2 = 300
    c2 = token_row(y2, ["c", "b", "a"], "打乱顺序 (c, b, a)", PURPLE_B)
    p.append(f'<text x="370" y="{y2-8}" class="sub">token a 算出的分数集合：</text>')
    p.append(f'<rect x="370" y="{y2+4}" width="360" height="34" rx="8" fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.6"/>')
    p.append(f'<text x="386" y="{y2+26}" class="mono">{{ a·a , a·c , a·b }}</text>')

    # equal sign between the two score sets
    p.append(f'<text x="765" y="{(y1+y2)/2+24}" text-anchor="middle" class="title" fill="{GREEN_B}">=</text>')
    p.append(f'<text x="820" y="{(y1+y2)/2+6}" class="lbl" fill="{GREEN_B}">同一个集合</text>')
    p.append(f'<text x="820" y="{(y1+y2)/2+30}" class="small">只是排列变了，</text>')
    p.append(f'<text x="820" y="{(y1+y2)/2+50}" class="small">softmax 聚合结果一样</text>')

    # conclusion band
    cy = 420
    p.append(f'<rect x="60" y="{cy}" width="{W-120}" height="64" rx="12" '
             f'fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="2"/>')
    p.append(f'<text x="{W/2}" y="{cy+27}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">'
             f'注意力分数 = 向量点积，只看「是哪两个向量」，不看「排第几位」</text>')
    p.append(f'<text x="{W/2}" y="{cy+50}" text-anchor="middle" class="sub">'
             f'⇒ 自注意力把序列当成无序集合（置换等变）→ 必须额外注入位置编码</text>')

    p.append(f'<text x="{W/2}" y="{H-22}" text-anchor="middle" class="cap">'
             f'embedding 查表丢了顺序、点积算分又不看位置 —— 对 attention 而言「狗咬人」和「人咬狗」没区别，所以位置编码是刚需。</text>')

    write_svg(ASSETS / "why-position.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: the four PE methods — two injection routes
# ============================================================
def diagram_pe_landscape():
    W, H = 1280, 620
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'四种位置编码：注入的位置不同 —— 改输入（绝对）vs 改注意力（相对）</text>']

    # pipeline across the top
    cy, h = 130, 64
    nodes = [
        (140, 150, "input_ids", "[B, L]", GRAY_F, GRAY_B),
        (360, 160, "token embedding", "查表 → [B, L, d]", GREEN_F, GREEN_B),
        (610, 180, "+ 位置（路线 A）", "加到输入向量上", ORANGE_F, ORANGE_B),
        (890, 190, "自注意力（路线 B）", "算分时注入相对位置", BLUE_F, BLUE_B),
        (1140, 150, "logits", "→ 下一个 token", PURPLE_F, PURPLE_B),
    ]
    centers = []
    for cx, w, lbl, sub, f, b in nodes:
        p += node(cx, cy, w, h, lbl, sub, f, b)
        centers.append((cx, w))
    for i in range(len(centers) - 1):
        cx1, w1 = centers[i]
        cx2, w2 = centers[i + 1]
        p += arrow(cx1 + w1 / 2 + 2, cy, cx2 - w2 / 2 - 4, cy)

    # divider
    p.append(f'<line x1="60" y1="205" x2="{W-60}" y2="205" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="4,4"/>')

    # two route columns of cards
    def card(x, y, w, hh, tag, tag_b, name, route, abs_rel, traits, fill, bd):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="12" '
                 f'fill="{fill}" stroke="{bd}" stroke-width="2"/>')
        p.append(f'<rect x="{x+16}" y="{y+16}" width="118" height="24" rx="6" '
                 f'fill="#ffffff" stroke="{tag_b}" stroke-width="1.4"/>')
        p.append(f'<text x="{x+75}" y="{y+33}" text-anchor="middle" class="tag" fill="{tag_b}">{esc(tag)}</text>')
        p.append(f'<text x="{x+150}" y="{y+34}" class="lbl">{esc(name)}</text>')
        p.append(f'<text x="{x+16}" y="{y+62}" class="small">{esc(route)}　{esc(abs_rel)}</text>')
        for k, t in enumerate(traits):
            p.append(f'<text x="{x+16}" y="{y+86+k*22}" class="sub">{esc(t)}</text>')

    cw, chh = 560, 150
    # Route A header
    p.append(f'<text x="{60+cw/2}" y="245" text-anchor="middle" class="lbl" fill="{ORANGE_B}">路线 A：加到输入 embedding（绝对位置）</text>')
    card(60, 262, cw, chh, "sinusoidal", ORANGE_B, "正弦位置编码", "公式算 · 不占参数", "（绝对）",
         ["• 固定 sin/cos 公式给每个位置算向量", "• 可外推但效果一般；原始 Transformer 用"], ORANGE_F, ORANGE_B)
    card(60, 262 + chh + 18, cw, chh, "learned", ORANGE_B, "可学习位置编码", "一张位置表 · 占参数", "（绝对）",
         ["• 再来一张 nn.Embedding 学位置向量", "• 灵活但上下文长度被限死为 max_len；GPT-2/BERT 用"], ORANGE_F, ORANGE_B)

    # Route B header
    bx = 60 + cw + 40
    p.append(f'<text x="{bx+cw/2}" y="245" text-anchor="middle" class="lbl" fill="{BLUE_B}">路线 B：改注意力算分（相对位置）</text>')
    card(bx, 262, cw, chh, "RoPE", BLUE_B, "旋转位置编码", "旋转 q/k · 不占参数", "（相对）",
         ["• 按位置旋转 q/k，点积只依赖相对距离", "• 外推友好，当下主流：Llama/Qwen/Mistral"], BLUE_F, BLUE_B)
    card(bx, 262 + chh + 18, cw, chh, "ALiBi", BLUE_B, "线性偏置注意力", "加偏置 · 不占参数", "（相对）",
         ["• 给分数按距离减线性惩罚（每头一斜率）", "• 外推极好但先验偏强；BLOOM/MPT 用"], BLUE_F, BLUE_B)

    write_svg(ASSETS / "pe-landscape.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: RoPE — rotate q/k, dot product depends on m - n
# ============================================================
def diagram_rope():
    W, H = 1200, 580
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'RoPE：把位置编码成旋转角度 —— 旋转后点积只依赖相对位置 (n − m)</text>']

    # left panel: a unit-circle showing q rotated by m*theta, k by n*theta
    cx, cyc, r = 290, 285, 150
    p.append(f'<circle cx="{cx}" cy="{cyc}" r="{r}" fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1.5"/>')
    # axes
    p.append(f'<line x1="{cx-r-20}" y1="{cyc}" x2="{cx+r+20}" y2="{cyc}" stroke="{GRAY_B}" stroke-width="1"/>')
    p.append(f'<line x1="{cx}" y1="{cyc+r+20}" x2="{cx}" y2="{cyc-r-20}" stroke="{GRAY_B}" stroke-width="1"/>')

    import math
    def vec(angle_deg, color, label, lr=1.0):
        a = math.radians(angle_deg)
        ex = cx + lr * r * math.cos(a)
        ey = cyc - lr * r * math.sin(a)
        p.append(f'<line x1="{cx}" y1="{cyc}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" '
                 f'stroke-width="3" marker-end="url(#a{label[0]})"/>')
        return ex, ey

    # q rotated by m*theta (e.g. 25deg), k rotated by n*theta (e.g. 70deg)
    qx, qy = vec(25, BLUE_B, "Blue", 0.92)
    kx, ky = vec(70, PURPLE_B, "Purple", 0.92)
    p.append(f'<text x="{qx+14}" y="{qy+6}" class="lbl" fill="{BLUE_B}">R(mθ)·q</text>')
    p.append(f'<text x="{kx-6}" y="{ky-12}" class="lbl" fill="{PURPLE_B}">R(nθ)·k</text>')
    # arc of relative angle between them
    p.append(f'<path d="M {cx+70*math.cos(math.radians(25)):.1f},{cyc-70*math.sin(math.radians(25)):.1f} '
             f'A 70 70 0 0 0 {cx+70*math.cos(math.radians(70)):.1f},{cyc-70*math.sin(math.radians(70)):.1f}" '
             f'fill="none" stroke="{ORANGE_B}" stroke-width="2.5"/>')
    p.append(f'<text x="{cx+30}" y="{cyc-72}" class="lbl" fill="{ORANGE_B}">(n−m)θ</text>')
    p.append(f'<text x="{cx}" y="{cyc+r+48}" text-anchor="middle" class="small">'
             f'旋转不改长度，只改方向；点积 = 长度 × 夹角余弦</text>')

    # right panel: the key identity + worked numbers
    rx = 600
    p.append(f'<rect x="{rx}" y="100" width="{W-rx-60}" height="180" rx="12" '
             f'fill="{TEAL_F}" stroke="{TEAL_B}" stroke-width="2"/>')
    p.append(f'<text x="{rx+20}" y="132" class="lbl" fill="{TEAL_B}">核心恒等式（旋转矩阵性质）</text>')
    p.append(f'<text x="{rx+20}" y="170" class="mono">⟨ R(mθ)q , R(nθ)k ⟩</text>')
    p.append(f'<text x="{rx+20}" y="202" class="mono">  = qᵀ R((n−m)θ) k</text>')
    p.append(f'<text x="{rx+20}" y="240" class="sub">绝对的 m、n 在点积里抵消，</text>')
    p.append(f'<text x="{rx+20}" y="262" class="sub">只剩相对位置 (n − m)。</text>')

    # numeric example
    p.append(f'<rect x="{rx}" y="300" width="{W-rx-60}" height="190" rx="12" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1.2" stroke-dasharray="5,3"/>')
    p.append(f'<text x="{rx+20}" y="330" class="lbl">最小例子：q = k = [1, 0]，θ = 1</text>')
    rows = [
        ("(m, n) = (0, 1)", "距离 1", "cos(1) ≈ 0.540", GREEN_B),
        ("(m, n) = (5, 6)", "距离 1", "cos(1) ≈ 0.540", GREEN_B),
        ("(m, n) = (0, 2)", "距离 2", "cos(2) ≈ −0.416", PURPLE_B),
    ]
    for k, (lhs, dist, rhs, col) in enumerate(rows):
        yy = 360 + k * 40
        p.append(f'<text x="{rx+20}" y="{yy}" class="mono">{esc(lhs)}</text>')
        p.append(f'<text x="{rx+200}" y="{yy}" class="small">{esc(dist)}</text>')
        p.append(f'<text x="{rx+275}" y="{yy}" class="mono" fill="{col}">{esc(rhs)}</text>')
    p.append(f'<text x="{rx+20}" y="{360+3*40+6}" class="sub" font-weight="600">'
             f'相对距离相同 → 分数相同；距离变了 → 分数才变。</text>')

    p.append(f'<text x="{W/2}" y="{H-22}" text-anchor="middle" class="cap">'
             f'd 维时把向量切成 d/2 个二维对，各用角速度 θᵢ = base^(−2i/d) 独立旋转；base（rope_theta）越大，可分辨的最大距离越远（扩上下文的旋钮）。</text>')

    write_svg(ASSETS / "rope.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 5: ALiBi — linear distance penalty on attention scores
# ============================================================
def diagram_alibi():
    W, H = 1180, 540
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'ALiBi：在注意力分数上按距离减线性惩罚（q/k 不动）</text>']

    # the formula band
    p.append(f'<rect x="60" y="80" width="{W-120}" height="56" rx="11" '
             f'fill="{TEAL_F}" stroke="{TEAL_B}" stroke-width="2"/>')
    p.append(f'<text x="{W/2}" y="116" text-anchor="middle" class="mono" fill="{TEAL_B}">'
             f'score(i, j) = qᵢ · kⱼ  −  m · (i − j)      （因果：j ≤ i，距离 i−j ≥ 0，斜率 m &gt; 0）</text>')

    # left: one query row, bars shrinking with distance
    bx, by = 90, 200
    p.append(f'<text x="{bx}" y="{by-14}" class="lbl">同一个 query i 看前面各 key j：离得越远，惩罚越大、权重越小</text>')
    keys = ["j=i", "j=i−1", "j=i−2", "j=i−3", "j=i−4"]
    pens = ["−0", "−m", "−2m", "−3m", "−4m"]
    bar_h = [120, 96, 74, 54, 36]
    for k in range(5):
        x = bx + k * 130
        h = bar_h[k]
        yb = by + 150 - h
        p.append(f'<rect x="{x}" y="{yb}" width="86" height="{h}" rx="8" '
                 f'fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.8"/>')
        p.append(f'<text x="{x+43}" y="{by+170}" text-anchor="middle" class="small">{esc(keys[k])}</text>')
        p.append(f'<text x="{x+43}" y="{yb-8}" text-anchor="middle" class="tag" fill="{TEAL_B}">{esc(pens[k])}</text>')
    p.append(f'<text x="{bx}" y="{by+200}" class="sub">柱高 ≈ 注意力权重（softmax 后）：距离每 +1，分数固定再减 m。</text>')

    # right: per-head slopes
    rx = 790
    p.append(f'<rect x="{rx}" y="178" width="{W-rx-60}" height="190" rx="12" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1.2" stroke-dasharray="5,3"/>')
    p.append(f'<text x="{rx+20}" y="208" class="lbl">每个注意力头用不同斜率 m</text>')
    slopes = [("head 1", "m = 1/2", "目光短浅，盯眼前", RED_B),
              ("head 2", "m = 1/4", "看中等距离", ORANGE_B),
              ("head 3", "m = 1/8", "看得更远", GREEN_B)]
    for k, (hd, mm, note, col) in enumerate(slopes):
        yy = 238 + k * 40
        p.append(f'<text x="{rx+20}" y="{yy}" class="tag" fill="{col}">{esc(hd)}</text>')
        p.append(f'<text x="{rx+92}" y="{yy}" class="mono">{esc(mm)}</text>')
        p.append(f'<text x="{rx+186}" y="{yy}" class="small">{esc(note)}</text>')
    p.append(f'<text x="{rx+20}" y="{238+3*40+4}" class="sub">斜率取等比数列，</text>')
    p.append(f'<text x="{rx+20}" y="{238+3*40+26}" class="sub">合起来兼顾局部与全局。</text>')

    p.append(f'<text x="{W/2}" y="{H-22}" text-anchor="middle" class="cap">'
             f'ALiBi 的惩罚是按距离外推的纯公式，所以外推极好；代价是「远处不重要」写成了死规则，对需要回看远处的任务偏保守。BLOOM、MPT 用过。</text>')

    write_svg(ASSETS / "alibi.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 6: where lm_head sits — hidden vector h -> logits
# ============================================================
# ============================================================
# Diagram 7: word2vec — CBOW vs skip-gram
# ============================================================
def diagram_word2vec():
    W, H = 1500, 640
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'word2vec：用「预测上下文」学词向量 —— CBOW 与 skip-gram</text>']

    # ---- top: a sentence with a sliding window ----
    sy = 100
    p.append(f'<text x="60" y="{sy-12}" class="lbl">在句子上开一个滑动窗口（中心词 + 周围词）：</text>')
    words = ["the", "cat", "sat", "on", "the", "mat"]
    # center = "sat" (idx 2); context (window=1) = cat, on
    roles = {2: ("center", PURPLE_F, PURPLE_B), 1: ("ctx", BLUE_F, BLUE_B),
             3: ("ctx", BLUE_F, BLUE_B)}
    wx = 90
    cx_of = {}
    for i, w in enumerate(words):
        role = roles.get(i)
        fill, bd = ("#ffffff", GRAY_B) if role is None else (role[1] and role[1], role[2])
        if role is None:
            fill, bd = "#ffffff", GRAY_B
        else:
            fill, bd = role[1], role[2]
        bw = 78
        p.append(f'<rect x="{wx}" y="{sy}" width="{bw}" height="40" rx="8" '
                 f'fill="{fill}" stroke="{bd}" stroke-width="{2 if role else 1.3}"/>')
        p.append(f'<text x="{wx+bw/2}" y="{sy+26}" text-anchor="middle" class="mono">{esc(w)}</text>')
        cx_of[i] = wx + bw / 2
        wx += bw + 14
    # window bracket over cat sat on
    p.append(f'<rect x="{cx_of[1]-45}" y="{sy-6}" width="{cx_of[3]-cx_of[1]+90}" height="52" rx="10" '
             f'fill="none" stroke="{ORANGE_B}" stroke-width="2" stroke-dasharray="6,4"/>')
    p.append(f'<text x="{cx_of[2]}" y="{sy+62}" text-anchor="middle" class="small" fill="{PURPLE_B}">中心词 sat</text>')
    p.append(f'<text x="{cx_of[1]}" y="{sy+62}" text-anchor="middle" class="small" fill="{BLUE_B}">上下文</text>')
    p.append(f'<text x="{cx_of[3]}" y="{sy+62}" text-anchor="middle" class="small" fill="{BLUE_B}">上下文</text>')

    # ---- two panels ----
    pty = 200
    pw, ph = 660, 280
    # CBOW (left): context -> center
    lx = 60
    p.append(f'<rect x="{lx}" y="{pty}" width="{pw}" height="{ph}" rx="12" '
             f'fill="#fbfdff" stroke="{BLUE_B}" stroke-width="2"/>')
    p.append(f'<text x="{lx+24}" y="{pty+34}" class="lbl" fill="{BLUE_B}">CBOW：用上下文预测中心词</text>')
    # context boxes
    p += node(lx+110, pty+110, 120, 48, "cat", "上下文", BLUE_F, BLUE_B)
    p += node(lx+110, pty+185, 120, 48, "on", "上下文", BLUE_F, BLUE_B)
    # projection
    p += node(lx+330, pty+147, 150, 64, "投影层", "查向量 + 求和", GRAY_F, GRAY_B)
    # output
    p += node(lx+540, pty+147, 130, 64, "softmax", "→ 预测 sat", PURPLE_F, PURPLE_B)
    p += arrow(lx+170, pty+110, lx+253, pty+135, marker="aBlue")
    p += arrow(lx+170, pty+185, lx+253, pty+160, marker="aBlue")
    p += arrow(lx+406, pty+147, lx+473, pty+147, marker="aPurple", color=PURPLE_B)

    # skip-gram (right): center -> context
    rx = 780
    p.append(f'<rect x="{rx}" y="{pty}" width="{pw}" height="{ph}" rx="12" '
             f'fill="#fbfdff" stroke="{PURPLE_B}" stroke-width="2"/>')
    p.append(f'<text x="{rx+24}" y="{pty+34}" class="lbl" fill="{PURPLE_B}">skip-gram：用中心词预测上下文</text>')
    p += node(rx+100, pty+147, 120, 56, "sat", "中心词", PURPLE_F, PURPLE_B)
    p += node(rx+320, pty+147, 150, 64, "投影层", "查向量", GRAY_F, GRAY_B)
    p += node(rx+540, pty+110, 130, 48, "softmax→cat", "", BLUE_F, BLUE_B)
    p += node(rx+540, pty+185, 130, 48, "softmax→on", "", BLUE_F, BLUE_B)
    p += arrow(rx+160, pty+147, rx+243, pty+147, marker="aGray", color=GRAY_B)
    p += arrow(rx+396, pty+135, rx+473, pty+110, marker="aBlue")
    p += arrow(rx+396, pty+160, rx+473, pty+185, marker="aBlue")

    # ---- bottom band ----
    by = 510
    p.append(f'<rect x="60" y="{by}" width="{W-120}" height="100" rx="12" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,3"/>')
    p.append(f'<text x="80" y="{by+28}" class="lbl" fill="{GREEN_B}">关键产物</text>')
    p.append(f'<text x="80" y="{by+52}" class="sub">训练完，「投影层」那张表——每个词一个向量——就是我们要的词向量；语义相近的词，向量自然也相近。</text>')
    p.append(f'<text x="80" y="{by+82}" class="sub">负采样（negative sampling）：词表几十万、softmax 分母太贵 → 改成二分类——真实(中心,上下文)判正 + 随机抽 k 个词判负，便宜得多。</text>')

    write_svg(ASSETS / "word2vec.svg", "\n".join(p), f"0 0 {W} {H}")


def diagram_lm_head_pipeline():
    W, H = 1480, 470
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'lm_head 在流水线的哪一步：隐藏向量 h（每位置 d 维）→ 词表分数 logits</text>']

    cy, h = 215, 80
    # (cx, w, label, sub, fill, border, highlight)
    nodes = [
        (115,  130, "input_ids",       "[B, L]",            GRAY_F,   GRAY_B,   False),
        (340,  185, "token embedding",  "查表(+位置)→[B,L,d]", GREEN_F,  GREEN_B,  False),
        (575,  170, "Transformer × N",  "注意力 + FFN 堆叠",  BLUE_F,   BLUE_B,   False),
        (820,  180, "隐藏向量 h",        "[B, L, d]",         TEAL_F,   TEAL_B,   True),
        (1075, 195, "lm_head（线性）",   "[d, V] · 本节主角",  PURPLE_F, PURPLE_B, True),
        (1330, 160, "logits",           "[B, L, V]",         ORANGE_F, ORANGE_B, False),
    ]
    centers = []
    for cx, w, lbl, sub, f, b, hl in nodes:
        sw = 3 if hl else 2
        x, y = cx - w / 2, cy - h / 2
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
                 f'fill="{f}" stroke="{b}" stroke-width="{sw}"/>')
        p.append(f'<text x="{cx}" y="{cy-4}" text-anchor="middle" class="lbl">{esc(lbl)}</text>')
        p.append(f'<text x="{cx}" y="{cy+17}" text-anchor="middle" class="small">{esc(sub)}</text>')
        centers.append((cx, w))
    for i in range(len(centers) - 1):
        cx1, w1 = centers[i]
        cx2, w2 = centers[i + 1]
        p += arrow(cx1 + w1 / 2 + 2, cy, cx2 - w2 / 2 - 4, cy)

    # trailing note after logits
    p.append(f'<text x="{1330}" y="{cy+h/2+30}" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">→ softmax / 采样</text>')
    p.append(f'<text x="{1330}" y="{cy+h/2+50}" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">→ 下一个 token</text>')

    # weight-tying arc: embedding (n1) -> lm_head (n4) over the top
    sx = 340; ex = 1075; topy = 100
    p.append(f'<path d="M {sx},{cy-h/2} C {sx},{topy} {ex},{topy} {ex},{cy-h/2}" '
             f'stroke="{PURPLE_B}" stroke-width="2" fill="none" stroke-dasharray="6,4" '
             f'marker-end="url(#aPurple)"/>')
    p.append(f'<text x="{(sx+ex)/2}" y="{topy-8}" text-anchor="middle" class="small" '
             f'fill="{PURPLE_B}">weight tying：lm_head 权重 = embedding 矩阵的转置 E^T（可共享一份）</text>')

    # bottom explainer band (what is h / where the step happens)
    by = 320
    p.append(f'<rect x="60" y="{by}" width="{W-120}" height="118" rx="12" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,3"/>')
    p.append(f'<text x="80" y="{by+28}" class="lbl" fill="{TEAL_B}">h 是什么？</text>')
    p.append(f'<text x="80" y="{by+52}" class="sub">最后一层 Transformer 输出的、每个位置一个的隐藏向量，维度 d（和 embedding 同维）。长度 L 的序列就有 L 个 h。</text>')
    p.append(f'<text x="80" y="{by+82}" class="lbl" fill="{PURPLE_B}">这一步发生在哪？</text>')
    p.append(f'<text x="80" y="{by+106}" class="sub">就在最后：lm_head 把每个 h（d 维）线性映射成词表上的 V 个分数 logits——本节的 weight tying 说的就是它的权重可与输入 embedding 共享。</text>')

    write_svg(ASSETS / "lm-head-pipeline.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_embedding_lookup()
    diagram_why_position()
    diagram_pe_landscape()
    diagram_rope()
    diagram_alibi()
    diagram_lm_head_pipeline()
    diagram_word2vec()
    print("SVGs generated under", ASSETS)
