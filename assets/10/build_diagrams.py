"""Generate conceptual diagrams for chapter 10 (自回归语言建模目标).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders, Noto Sans
CJK SC. Hand-written SVG (same approach as assets/09/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses gray-700
(#374151) or darker per the repo contrast guideline.

Diagrams
  1. ce-loss-pipeline.svg          —— cross-entropy loss 计算流程：3 列网格布局
                                      （列1 步骤①②、列2 步骤③④、列3 步骤⑤居中），
                                      每步一个块框、块内序列0/序列1 上下堆叠。用具体
                                      张量小方块（示例 B=2, L=3, V=4）演示 input_ids
                                      → logits → shift → reshape → cross_entropy →
                                      标量 loss，同色串起「位置 t 的 logits 行 ↔ 它
                                      预测的下一个 token 标签」。用 -w 3000。
  2. teacher-forcing-vs-inference.svg —— teacher forcing 并行训练 vs 自回归串行
                                      推理：同一个分布 Pθ(·|前文) 的写与读。

Run from repo root:
    python3 assets/10/build_diagrams.py
Then export each SVG to PNG（ce-loss-pipeline 很宽，用 -w 3000；另一张 -w 2400）:
    rsvg-convert -w 3000 assets/10/ce-loss-pipeline.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/10/ce-loss-pipeline.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("'Noto Sans CJK SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', "
        "'WenQuanYi Zen Hei', sans-serif")
MONO = ("'Noto Sans Mono CJK SC', 'SFMono-Regular', 'Consolas', "
        "'Liberation Mono', monospace")

# ---------- shared palette (Flat Icon) ----------
BG = "#ffffff"
TXT = "#374151"   # gray-700  primary labels
SUB = "#334155"   # slate-700  secondary / sublabel / caption (>= gray-700)
FAINT = "#475569" # slate-600  only for genuinely de-emphasized notes (boxed)

BLUE_F, BLUE_B = "#dbeafe", "#2563eb"
GREEN_F, GREEN_B = "#dcfce7", "#059669"
ORANGE_F, ORANGE_B = "#ffedd5", "#ea580c"
PURPLE_F, PURPLE_B = "#ede9fe", "#7c3aed"
RED_F, RED_B = "#fee2e2", "#dc2626"
PINK_F, PINK_B = "#fce7f3", "#db2777"
TEAL_F, TEAL_B = "#cffafe", "#0891b2"
AMBER_F, AMBER_B = "#fef3c7", "#b45309"
GRAY_F, GRAY_B = "#f3f4f6", "#94a3b8"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg(path: Path, body: str, viewbox: str):
    style = f"""
  <style>
    text {{ font-family: {FONT}; }}
    .title  {{ font-size: 25px; font-weight: 700; fill: {TXT}; }}
    .h2     {{ font-size: 18px; font-weight: 700; fill: {TXT}; }}
    .lbl    {{ font-size: 16px; font-weight: 600; fill: {TXT}; }}
    .sub    {{ font-size: 14px; fill: {SUB}; }}
    .mono   {{ font-size: 14px; font-family: {MONO}; fill: {TXT}; }}
    .monob  {{ font-size: 15px; font-weight: 600; font-family: {MONO}; fill: {TXT}; }}
    .small  {{ font-size: 13px; fill: {SUB}; }}
    .cap    {{ font-size: 15px; fill: {SUB}; }}
    .tag    {{ font-size: 13px; font-weight: 700; }}
  </style>
"""
    defs = f"""
  <defs>
    <marker id="aBlue" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{BLUE_B}"/></marker>
    <marker id="aSlate" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{SUB}"/></marker>
    <marker id="aGreen" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{GREEN_B}"/></marker>
    <marker id="aRed" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{RED_B}"/></marker>
    <marker id="aOrange" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{ORANGE_B}"/></marker>
    <marker id="aPurple" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{PURPLE_B}"/></marker>
  </defs>
"""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">\n'
        f'  <rect width="100%" height="100%" fill="{BG}"/>\n'
        f'{defs}{style}{body}\n'
        f'</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def rrect(x, y, w, h, fill, border, rx=11, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{border}" stroke-width="{sw}"{d}/>')


def txt(x, y, s, cls="sub", anchor="middle", extra=""):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{extra}>{esc(s)}</text>'


def arrow(x1, y1, x2, y2, marker="aSlate", color=SUB, dashed=False, width=2.2):
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{dash} marker-end="url(#{marker})"/>')


# ============================================================
# Diagram 1: cross-entropy loss pipeline (concrete-tensor grids)
# ------------------------------------------------------------
# 3 列网格布局：列1 步骤①②、列2 步骤③④、列3 步骤⑤居中；每个步骤一个块框，
# 块内序列0 / 序列1 上下堆叠。用具体张量小方块（示例 B=2, L=3, V=4）演示
# input_ids → logits → shift → reshape → cross_entropy → 标量 loss，同色串起
# 「位置 t 的 logits 行 ↔ 它预测的下一个 token 标签」，reshape 阶段黑框标出标签
# 指向的列。宽高比均衡（既不像竖排太长、也不像横排太扁）。手写网格、不走
# write_svg 的共享 style，确保逐格上色版式逐像素一致；用 -w 3000 导出。
# ============================================================
def ce_loss_pipeline():
    INK, INK_SOFT = "#374151", "#475569"
    DROP_F, DROP_S = "#f1f5f9", "#cbd5e1"
    BOX_F, BOX_S = "#fcfdff", "#cbd5e1"
    COLORS = {                                  # fill, stroke, deep(label/picked)
        "A": ("#dbeafe", "#2563eb", "#93c5fd"),  # blue   序列0 位置0
        "B": ("#dcfce7", "#16a34a", "#86efac"),  # green  序列0 位置1
        "C": ("#fef3c7", "#d97706", "#fcd34d"),  # amber  序列1 位置0
        "D": ("#ede9fe", "#7c3aed", "#c4b5fd"),  # purple 序列1 位置1
    }
    LOSS_F, LOSS_S = "#fde68a", "#b45309"
    CELL, CGAP, SEQGAP = 54, 6, 36
    sv = []

    def rect(x, y, w, h, fill, stroke, rx=7, sw=2.2):
        sv.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                  f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def T(x, y, s, size=18, fill=INK, weight="400", anchor="middle", font=FONT):
        sv.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
                  f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')

    def bw_(c): return c * CELL + (c - 1) * CGAP
    def bh_(r): return r * CELL + (r - 1) * CGAP

    def grid(x, y, spec, values=None, picked=None):
        rows, cols = len(spec), len(spec[0])
        for r in range(rows):
            for c in range(cols):
                cx, cy = x + c * (CELL + CGAP), y + r * (CELL + CGAP)
                rect(cx, cy, CELL, CELL, spec[r][c][0], spec[r][c][1])
                if values and values[r][c] is not None:
                    T(cx + CELL / 2, cy + CELL / 2 + 8, values[r][c], 22, INK, "600", font=MONO)
            if picked is not None and picked[r] is not None:
                c = picked[r]; cx, cy = x + c * (CELL + CGAP), y + r * (CELL + CGAP)
                rect(cx - 2, cy - 2, CELL + 4, CELL + 4, "none", "#111827", rx=8, sw=3.2)

    def vcols(x, y, labs):
        for c, l in enumerate(labs):
            T(x + c * (CELL + CGAP) + CELL / 2, y, l, 15, INK_SOFT, "600", font=MONO)

    def elbow(pts, l1=None, l2=None, lx=None, ly=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        sv.append(f'<polyline points="{p}" fill="none" stroke="#475569" stroke-width="3" '
                  f'marker-end="url(#ah)"/>')
        if l1 and lx is not None:
            T(lx, ly, l1, 15, INK, "600")
            if l2:
                T(lx, ly + 19, l2, 13, INK_SOFT, "400")

    def box(bx, by, bw0, bh0, title, shape):
        rect(bx, by, bw0, bh0, BOX_F, BOX_S, rx=14, sw=2)
        T(bx + 18, by + 28, title, 18, INK, "700", anchor="start")
        if shape:
            T(bx + bw0 - 16, by + 28, shape, 14, INK_SOFT, "400", anchor="end", font=MONO)

    ids = [[2, 0, 3], [1, 3, 2]]
    rowcol = [["A", "B", None], ["C", "D", None]]
    order = ["A", "B", "C", "D"]
    labels_1d = [0, 3, 3, 2]

    LM, GUT, RGAP, RM = 60, 190, 118, 60
    PAD, HH, SL = 22, 40, 30

    box1w = SL + bw_(3) + 2 * PAD
    box2w = SL + bw_(4) + 24 + 2 * PAD
    box3w = SL + bw_(4) + 14 + bw_(1) + 2 * PAD
    box4w = 72 + bw_(4) + 40 + bw_(1) + 2 * PAD
    box5w = bw_(1) + 118 + 196 + 2 * PAD
    box1h = HH + 20 + bh_(1) + SEQGAP + bh_(1) + PAD
    box2h = HH + 20 + bh_(3) + SEQGAP + bh_(3) + PAD
    box3h = HH + 20 + bh_(2) + SEQGAP + bh_(2) + PAD
    box4h = HH + 20 + bh_(4) + PAD
    box5h = HH + bh_(4) + PAD

    col1w = max(box1w, box2w); col2w = max(box3w, box4w); col3w = box5w
    CX1 = LM + col1w / 2
    CX2 = LM + col1w + GUT + col2w / 2
    CX3 = LM + col1w + GUT + col2w + GUT + col3w / 2
    W = LM + col1w + GUT + col2w + GUT + col3w + RM

    rowAh = max(box1h, box3h)
    RYA = 130; RYB = RYA + rowAh + RGAP
    B2_UP = 90   # ② 区块比行顶上移一点，缩短 ①→② 的长箭头

    b1x = CX1 - box1w / 2; b1y = RYA
    b2x = CX1 - box2w / 2; b2y = RYB - B2_UP
    b3x = CX2 - box3w / 2; b3y = RYA
    b4x = CX2 - box4w / 2; b4y = RYB
    H_rows_bottom = max(b2y + box2h, b4y + box4h)
    b5x = CX3 - box5w / 2; b5y = (RYA + H_rows_bottom) / 2 - box5h / 2

    # ① input_ids
    box(b1x, b1y, box1w, box1h, "① input_ids", "[2,3]")
    gx = b1x + PAD + SL; cy0 = b1y + HH + 20
    vcols(gx, b1y + HH + 6, ["t0", "t1", "t2"])
    for si, yy in enumerate([cy0, cy0 + bh_(1) + SEQGAP]):
        T(gx - 12, yy + CELL / 2 + 7, f"序{si}", 15, INK_SOFT, "600", anchor="end")
        grid(gx, yy, [[("#eef2f7", "#94a3b8")] * 3], [[str(v) for v in ids[si]]])

    # ② logits
    box(b2x, b2y, box2w, box2h, "② logits", "[2,3,4]")
    gx = b2x + PAD + SL; cy0 = b2y + HH + 20
    vcols(gx, b2y + HH + 6, ["v0", "v1", "v2", "v3"])
    for si, yy in enumerate([cy0, cy0 + bh_(3) + SEQGAP]):
        T(gx - 12, yy + bh_(3) / 2 + 7, f"序{si}", 15, INK_SOFT, "600", anchor="end")
        spec = []
        for r in range(3):
            k = rowcol[si][r]
            spec.append([(DROP_F, DROP_S)] * 4 if k is None
                        else [(COLORS[k][0], COLORS[k][1])] * 4)
        grid(gx, yy, spec)
        for r in range(3):
            T(gx + bw_(4) + 16, yy + r * (CELL + CGAP) + CELL / 2 + 6, f"t{r}",
              13, INK_SOFT, "600", anchor="start", font=MONO)

    # ③ shift
    box(b3x, b3y, box3w, box3h, "③ shift（行↔下一个 token）", "")
    gx = b3x + PAD + SL; cy0 = b3y + HH + 20
    vcols(gx, b3y + HH + 6, ["v0", "v1", "v2", "v3"])
    T(gx + bw_(4) + 14 + CELL / 2, b3y + HH + 6, "label", 14, INK_SOFT, "600", font=MONO)
    keys = [["A", "B"], ["C", "D"]]
    labv = [[ids[0][1], ids[0][2]], [ids[1][1], ids[1][2]]]
    for si, yy in enumerate([cy0, cy0 + bh_(2) + SEQGAP]):
        T(gx - 12, yy + bh_(2) / 2 + 7, f"序{si}", 15, INK_SOFT, "600", anchor="end")
        grid(gx, yy, [[(COLORS[keys[si][r]][0], COLORS[keys[si][r]][1])] * 4 for r in range(2)])
        lx = gx + bw_(4) + 14
        grid(lx, yy, [[(COLORS[keys[si][r]][2], COLORS[keys[si][r]][1])] for r in range(2)],
             [[str(labv[si][r])] for r in range(2)])

    # ④ reshape
    box(b4x, b4y, box4w, box4h, "④ reshape（黑框=标签列）", "")
    gx = b4x + PAD + 72; cy0 = b4y + HH + 20
    vcols(gx, b4y + HH + 6, ["v0", "v1", "v2", "v3"])
    T(gx + bw_(4) + 40 + CELL / 2, b4y + HH + 6, "label", 14, INK_SOFT, "600", font=MONO)
    grid(gx, cy0, [[(COLORS[k][0], COLORS[k][1])] * 4 for k in order], picked=labels_1d[:])
    rowtag = ["序0·t0", "序0·t1", "序1·t0", "序1·t1"]
    for i, k in enumerate(order):
        T(gx - 10, cy0 + i * (CELL + CGAP) + CELL / 2 + 6, rowtag[i], 13, COLORS[k][1], "600", anchor="end")
    grid(gx + bw_(4) + 40, cy0, [[(COLORS[k][2], COLORS[k][1])] for k in order],
         [[str(v)] for v in labels_1d])

    # ⑤ ℓ → mean → loss
    box(b5x, b5y, box5w, box5h, "⑤ ℓ → loss", "")
    gx = b5x + PAD; cy0 = b5y + HH
    ells = ["ℓ₀", "ℓ₁", "ℓ₂", "ℓ₃"]
    grid(gx, cy0, [[(COLORS[k][0], COLORS[k][1])] for k in order], [[e] for e in ells])
    midy = cy0 + bh_(4) / 2
    mx1 = gx + CELL + 16; mx2 = mx1 + 86
    sv.append(f'<line x1="{mx1:.1f}" y1="{midy:.1f}" x2="{mx2:.1f}" y2="{midy:.1f}" '
              f'stroke="#475569" stroke-width="3" marker-end="url(#ah)"/>')
    T((mx1 + mx2) / 2, midy - 12, "mean", 15, INK, "600")
    lossx = mx2 + 18
    rect(lossx, midy - 44, 196, 88, LOSS_F, LOSS_S, rx=12, sw=2.8)
    T(lossx + 98, midy - 6, "loss", 22, INK, "700")
    T(lossx + 98, midy + 22, "[ ] 标量", 15, INK, "400", font=MONO)

    # 步骤间箭头
    elbow([(CX1, b1y + box1h), (CX1, b2y)], "模型前向", "每位置出 V 维打分",
          CX1 + 120, (b1y + box1h + b2y) / 2 - 2)
    elbow([(CX2, b3y + box3h), (CX2, b4y)], "reshape", "拍平成 4 行",
          CX2 + 96, (b3y + box3h + b4y) / 2 - 2)
    vx1 = b2x + box2w + 34
    y2 = b2y + box2h / 2; y3 = b3y + box3h / 2
    elbow([(b2x + box2w, y2), (vx1, y2), (vx1, y3), (b3x, y3)],
          "shift 错开一位", "logits 去末·labels 去首", (vx1 + b3x) / 2, (y2 + y3) / 2 - 8)
    vx2 = b4x + box4w + 34
    y4 = b4y + box4h / 2; y5 = b5y + box5h / 2
    elbow([(b4x + box4w, y4), (vx2, y4), (vx2, y5), (b5x, y5)],
          "F.cross_entropy", "每行 −log p[label]", (vx2 + b5x) / 2, (y4 + y5) / 2 - 8)

    # 标题 / 底注
    T(W / 2, 52, "Cross-Entropy Loss：[ B, L ] 的 token id 一步步变成一个标量 loss", 30, INK, "700")
    T(W / 2, 82, "具体示例：B = 2（两条序列）, L = 3（每条 3 个 token）, V = 4（词表大小为 4）",
      17, INK_SOFT, "400")
    botY = H_rows_bottom + 52
    T(W / 2, botY,
      "shift 让「位置 t 的预测」对准「位置 t+1 的真实 token」——这正是自回归「预测下一个」目标的工程落地。",
      18, INK, "600")
    H = botY + 34

    out = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.0f} {H:.0f}">'
           f'<defs><marker id="ah" markerWidth="11" markerHeight="11" refX="8" refY="4" '
           f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#475569"/></marker></defs>'
           f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="{BG}"/>'
           + "".join(sv) + "</svg>\n")
    (ASSETS / "ce-loss-pipeline.svg").write_text(out, encoding="utf-8")




# ============================================================
# Diagram 2: teacher forcing (parallel) vs autoregressive (serial)
# ============================================================
def teacher_forcing_vs_inference():
    b = []
    W, H = 1420, 760
    b.append(txt(W / 2, 46, "Teacher Forcing 并行训练  vs  自回归串行推理", cls="title"))
    b.append(txt(W / 2, 72, "同一个分布 Pθ(·│前文) 的两面：训练塑造它，推理从它采样",
                 cls="cap"))

    def tokbox(cx, y, label, fill, border, w=66, h=42):
        return [rrect(cx - w / 2, y, w, h, fill, border, rx=9, sw=1.8),
                txt(cx, y + 27, label, cls="monob")]

    # ---------------- LEFT panel: training ----------------
    lx, lw = 40, 660
    b.append(rrect(lx, 96, lw, 600, "#fcfdff", BLUE_B, rx=16, sw=2))
    b.append(txt(lx + lw / 2, 132, "训练：teacher forcing（一次并行）", cls="h2"))

    cols = [lx + 150, lx + 268, lx + 386, lx + 504, lx + 622]  # 5 positions x0..x4
    # row: real sequence (ground-truth tokens)
    b.append(txt(lx + 86, 192, "真实序列", cls="lbl", anchor="end"))
    for i, c in enumerate(cols):
        b += tokbox(c, 172, f"x{i}", GREEN_F, GREEN_B)
    # band: causal-masked single forward
    by = 246
    b.append(rrect(lx + 60, by, lw - 120, 50, AMBER_F, AMBER_B, rx=10, sw=1.8))
    b.append(txt(lx + lw / 2, by + 31,
                 "因果掩码 + 一次前向（每个位置前文 = 真值、彼此独立）", cls="sub"))
    # feed arrows from real tokens down into band
    for c in cols[:4]:
        b.append(arrow(c, 214, c, by - 4, marker="aSlate", width=2))
    # row: predictions ŷ1..ŷ4 (only positions 0..3 have a target)
    py = 330
    b.append(txt(lx + 86, py + 27, "模型预测", cls="lbl", anchor="end"))
    for i, c in enumerate(cols[:4]):
        b.append(arrow(c, by + 50, c, py - 4, marker="aSlate", width=2))
        b += tokbox(c, py, f"ŷ{i+1}", BLUE_F, BLUE_B)
    # row: compare against ground-truth x1..x4
    gy = 430
    b.append(txt(lx + 86, gy + 27, "对照真值", cls="lbl", anchor="end"))
    for i, c in enumerate(cols[:4]):
        b.append(arrow(c, py + 42, c, gy - 4, marker="aRed", color=RED_B,
                       dashed=True, width=1.8))
        b += tokbox(c, gy, f"x{i+1}", GREEN_F, GREEN_B)
    # converge to loss
    ly = 528
    b.append(rrect(lx + 130, ly, lw - 260, 52, RED_F, RED_B, rx=10, sw=1.8))
    b.append(txt(lx + lw / 2, ly + 32, "cross-entropy 求平均 → loss（标量）", cls="lbl"))
    for c in cols[:4]:
        b.append(arrow(c, gy + 42, c, ly - 4, marker="aRed", color=RED_B, width=1.8))
    b.append(txt(lx + lw / 2, 626,
                 "全部 L−1 个位置同时算 loss——整列一次前向，不偷看未来。", cls="small"))
    b.append(txt(lx + lw / 2, 664,
                 "前文来自真实语料 → 高效、稳定，但埋下 exposure bias。", cls="small"))

    # ---------------- RIGHT panel: inference ----------------
    rx, rw = 720, 660
    b.append(rrect(rx, 96, rw, 600, "#fffdfb", ORANGE_B, rx=16, sw=2))
    b.append(txt(rx + rw / 2, 132, "推理：自回归（串行）", cls="h2"))

    midc = rx + rw / 2 - 40
    # prompt
    b += [rrect(midc - 150, 162, 300, 48, PURPLE_F, PURPLE_B, rx=10, sw=1.8),
          txt(midc, 191, "prompt（前文 token 序列）", cls="sub")]
    # three loop steps
    steps = [
        (236, "① 模型前向", BLUE_F, BLUE_B),
        (308, "② 取最后位置 logits → 采样 ŷ", TEAL_F, TEAL_B),
        (380, "③ 把 ŷ 拼回前文末尾", GREEN_F, GREEN_B),
    ]
    sx, sw2 = midc - 175, 350
    prev_y = 210
    for y, label, f, bd in steps:
        b.append(arrow(midc, prev_y, midc, y - 4, marker="aSlate", width=2.2))
        b.append(rrect(sx, y, sw2, 46, f, bd, rx=10, sw=1.8))
        b.append(txt(midc, y + 29, label, cls="sub"))
        prev_y = y + 46
    # loop-back arrow on the right side: step3 -> step1
    rxe = sx + sw2
    loop = (f'<path d="M {rxe} {380+23} H {rxe+46} V {236+23} H {rxe+4}" '
            f'fill="none" stroke="{ORANGE_B}" stroke-width="2.2" '
            f'marker-end="url(#aOrange)"/>')
    b.append(loop)
    b.append(txt(rxe + 54, (236 + 380) / 2 + 28, "下一步", cls="small", anchor="start"))
    # exit to <eos>
    b.append(arrow(midc, 426, midc, 470 - 4, marker="aRed", color=RED_B, width=2.2))
    b.append(txt(midc + 14, 452, "<eos> 或到上限", cls="small", anchor="start"))
    b += [rrect(midc - 150, 470, 300, 46, RED_F, RED_B, rx=10, sw=1.8),
          txt(midc, 498, "停止生成，输出整段文本", cls="sub")]
    b.append(txt(rx + rw / 2, 560,
                 "每步只取最后一个位置的 logits 采样，再拼回——串行，", cls="small"))
    b.append(txt(rx + rw / 2, 582,
                 "一次只产出一个 token（第 14 章 KV cache 为它提速）。", cls="small"))
    b.append(txt(rx + rw / 2, 626,
                 "前文来自模型自己生成的 ŷ——训练时从没见过这种前文。", cls="small"))
    b.append(txt(rx + rw / 2, 664,
                 "采样策略（greedy / temperature / top-p，第 2 章）都在这一步做文章。",
                 cls="small"))

    b.append(txt(W / 2, H - 18,
                 "训练（写）用 cross-entropy 把分布塑造对；推理（读）从同一个分布采样。区别全在「前文从哪来」与「能不能并行」。",
                 cls="cap"))
    write_svg(ASSETS / "teacher-forcing-vs-inference.svg", "\n".join(b), f"0 0 {W} {H}")


if __name__ == "__main__":
    ce_loss_pipeline()
    teacher_forcing_vs_inference()
    print("wrote ce-loss-pipeline.svg, teacher-forcing-vs-inference.svg")
