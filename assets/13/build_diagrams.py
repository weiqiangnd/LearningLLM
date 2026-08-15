"""Generate conceptual diagrams for chapter 13 (从零实现 mini-GPT).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders, Noto Sans
CJK SC. Hand-written SVG (same approach as assets/09/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses gray-700
(#374151) or darker per the repo contrast guideline.

Diagrams
  1. loop.svg          —— 「训练 → 生成」两条闭环共用同一套权重：上排是训练闭环
                          （语料 → tokenizer → 长 id 序列 → 取批 → 模型 → loss →
                          反向更新），下排是生成闭环（起始 token → 模型 → 末位
                          logits → 采样 → 接回序列末尾）。
  2. batching.svg      —— 从一条长 id 序列滑窗取样本：x / y 错一位对齐，一条样本
                          就是 L 条监督，再堆成 [B, L] 的一批。
  3. generate-cost.svg —— 自回归生成每一步都把整个前缀重算一遍：逐步增长的前向
                          面积里，只有最后一格是新的，其余全是重复计算。

Run from repo root:
    python3 assets/13/build_diagrams.py
Then export each SVG to PNG (default -w 2400):
    rsvg-convert -w 2400 assets/13/loop.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/13/loop.png /tmp/x.png
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
SUB = "#334155"   # slate-700  secondary / sublabel / caption
FAINT = "#475569" # slate-600  only for boxed, de-emphasized notes

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


def write_svg(path: Path, body: str, viewbox: str, bump: int = 0):
    """bump: 整体字号加几 px。节点框宽是按基准字号定死的，所以只给版面本来
    就宽松、放大后不会撑框的图开（当前只有 batching），其余保持基准字号。"""
    b = bump
    style = f"""
  <style>
    text {{ font-family: {FONT}; }}
    .title  {{ font-size: {25 + b}px; font-weight: 700; fill: {TXT}; }}
    .h2     {{ font-size: {18 + b}px; font-weight: 700; fill: {TXT}; }}
    .lbl    {{ font-size: {16 + b}px; font-weight: 600; fill: {TXT}; }}
    .sub    {{ font-size: {14 + b}px; fill: {SUB}; }}
    .mono   {{ font-size: {14 + b}px; font-family: {MONO}; fill: {TXT}; }}
    .monob  {{ font-size: {14 + b}px; font-weight: 600; font-family: {MONO}; fill: {TXT}; }}
    .monos  {{ font-size: {13 + b}px; font-family: {MONO}; fill: {SUB}; }}
    .small  {{ font-size: {13 + b}px; fill: {SUB}; }}
    .cap    {{ font-size: {15 + b}px; fill: {SUB}; }}
    .tag    {{ font-size: {13 + b}px; font-weight: 700; }}
  </style>
"""
    defs = f"""
  <defs>
    <marker id="aGray" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{GRAY_B}"/></marker>
    <marker id="aSlate" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{SUB}"/></marker>
    <marker id="aBlue" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{BLUE_B}"/></marker>
    <marker id="aOrange" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{ORANGE_B}"/></marker>
    <marker id="aGreen" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{GREEN_B}"/></marker>
    <marker id="aRed" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{RED_B}"/></marker>
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


def path_arrow(d, marker="aSlate", color=SUB, dashed=False, width=2.2):
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
            f'{dash} marker-end="url(#{marker})"/>')


def node(cx, cy, w, h, label, fill, border, sub1=None, sub2=None):
    """A rounded box with a bold label plus up to two smaller lines under it."""
    out = [rrect(cx - w / 2, cy - h / 2, w, h, fill, border, rx=12, sw=2)]
    if sub1 is None and sub2 is None:
        out.append(txt(cx, cy + 6, label, cls="lbl"))
    elif sub2 is None:
        out.append(txt(cx, cy - 5, label, cls="lbl"))
        out.append(txt(cx, cy + 18, sub1, cls="monos"))
    else:
        out.append(txt(cx, cy - 14, label, cls="lbl"))
        out.append(txt(cx, cy + 8, sub1, cls="monos"))
        out.append(txt(cx, cy + 28, sub2, cls="small"))
    return out


# ============================================================
# Diagram 1: 训练闭环 + 生成闭环，共用同一套权重
# ============================================================
def loop():
    W, H = 1560, 764
    p = []
    p.append(txt(W / 2, 46, "mini-GPT 的两条闭环：训练塑造分布，生成从分布里采样", cls="title"))
    p.append(txt(W / 2, 76,
                 "上排训练闭环把语料变成权重，下排生成闭环把权重变回文本 —— "
                 "中间那个模型是同一个，只是一个在更新它、一个在读它", cls="cap"))

    # ---------------- training loop ----------------
    top, hgt = 108, 272
    p.append(rrect(40, top, W - 80, hgt, "#fbfdff", BLUE_B, rx=18, sw=1.4, dash="7,5"))
    p.append(txt(64, top + 30, "训练闭环（Cell 2 → Cell 8）", cls="h2", anchor="start"))

    cy = top + 132
    xs = [190, 420, 650, 880, 1130, 1390]
    p += node(xs[0], cy, 190, 92, "tiny-shakespeare",
              GRAY_F, GRAY_B, "1,115,394 字符", "一份 1.1 MB 的纯文本")
    p += node(xs[1], cy, 190, 92, "字符级 tokenizer",
              TEAL_F, TEAL_B, "V = 65", "每个字符一个 id")
    p += node(xs[2], cy, 190, 92, "一条长 id 序列",
              PURPLE_F, PURPLE_B, "[1115394]", "前 90% 训练 / 后 10% 验证")
    p += node(xs[3], cy, 190, 92, "滑窗取一批",
              AMBER_F, AMBER_B, "x, y: [32, 128]", "y 是 x 右移一位")
    p += node(xs[4], cy, 220, 92, "mini-GPT",
              ORANGE_F, ORANGE_B, "logits: [32, 128, 65]", "4.76 M 参数 · 6 层")
    p += node(xs[5], cy, 190, 92, "cross-entropy",
              RED_F, RED_B, "一个标量 loss", "4096 个位置取平均")

    for a, b, wa, wb in [(0, 1, 95, 95), (1, 2, 95, 95), (2, 3, 95, 95),
                         (3, 4, 95, 110), (4, 5, 110, 95)]:
        p.append(arrow(xs[a] + wa, cy, xs[b] - wb - 6, cy, marker="aGray", color=GRAY_B))

    # backward pass: loss -> model
    p.append(path_arrow(f"M {xs[5]} {cy + 46} L {xs[5]} {cy + 96} "
                        f"L {xs[4]} {cy + 96} L {xs[4]} {cy + 52}",
                        marker="aRed", color=RED_B, width=2.4))
    p.append(txt((xs[4] + xs[5]) / 2, cy + 116,
                 "反向传播 → 梯度裁剪 → AdamW 更新参数，回到取下一批", cls="small"))

    # ---------------- generation loop ----------------
    top2, hgt2 = 416, 276
    p.append(rrect(40, top2, W - 80, hgt2, "#fffdfa", ORANGE_B, rx=18, sw=1.4, dash="7,5"))
    p.append(txt(64, top2 + 30, "生成闭环（Cell 7 / Cell 10）", cls="h2", anchor="start"))

    cy2 = top2 + 132
    gx = [190, 480, 780, 1080, 1390]
    p += node(gx[0], cy2, 210, 92, "起始上下文",
              GRAY_F, GRAY_B, "ids: [1, L₀]", "一个换行符，或一段 prompt")
    p += node(gx[1], cy2, 220, 92, "mini-GPT",
              ORANGE_F, ORANGE_B, "只喂最近 128 个", "同一套权重，dropout 关掉")
    p += node(gx[2], cy2, 220, 92, "取最后一个位置",
              BLUE_F, BLUE_B, "logits[:, -1, :] → [1, 65]", "前面位置的预测用不上")
    p += node(gx[3], cy2, 220, 92, "温度 / top-k 采样",
              GREEN_F, GREEN_B, "得到 1 个新 token", "旋钮见第 2 章")
    p += node(gx[4], cy2, 200, 92, "输出文本",
              PINK_F, PINK_B, "decode(ids)", "一次一个字符生成出来")

    for a, b, wa, wb in [(0, 1, 105, 110), (1, 2, 110, 110), (2, 3, 110, 110),
                         (3, 4, 110, 100)]:
        p.append(arrow(gx[a] + wa, cy2, gx[b] - wb - 6, cy2, marker="aGray", color=GRAY_B))

    p.append(path_arrow(f"M {gx[3]} {cy2 + 46} L {gx[3]} {cy2 + 100} "
                        f"L {gx[0]} {cy2 + 100} L {gx[0]} {cy2 + 52}",
                        marker="aOrange", color=ORANGE_B, width=2.4))
    p.append(txt((gx[0] + gx[3]) / 2, cy2 + 120,
                 "把新 token 接到序列末尾，再走一遍 —— 每多写一个字符就重跑一次前向", cls="small"))

    # ---------------- bottom note ----------------
    p.append(rrect(40, H - 62, W - 80, 44, "#f8fafc", GRAY_B, rx=10, sw=1.2))
    p.append(txt(W / 2, H - 34,
                 "同一个概率分布 P(下一个 token | 前文) 的两面：训练用真实语料把它拟合出来，"
                 "生成按它一个 token 一个 token 地采样",
                 cls="cap"))
    write_svg(ASSETS / "loop.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: 滑窗取批 —— x / y 错一位，一条样本 = L 条监督
# ============================================================
def batching():
    W, H = 1520, 796
    p = []
    p.append(txt(W / 2, 46, "从一条长 id 序列到一批训练样本", cls="title"))
    p.append(txt(W / 2, 76,
                 "随机取一个起点 i，切 L+1 个 token：前 L 个当输入 x，后 L 个当标签 y —— "
                 "y 就是 x 右移一位", cls="cap"))

    # ---- long id strip ----
    sx, sy, cw, ch = 90, 156, 26, 40
    n_cells = 46
    p.append(txt(sx, sy - 22, "train_data（整份语料编码成的一条长序列，1,003,854 个 token）",
                 cls="h2", anchor="start"))
    for k in range(n_cells):
        inside = 12 <= k <= 24
        fill = AMBER_F if inside else "#f8fafc"
        border = AMBER_B if inside else GRAY_B
        p.append(rrect(sx + k * cw, sy, cw - 3, ch, fill, border, rx=4, sw=1.2))
    p.append(txt(sx + n_cells * cw + 16, sy + 26, "…", cls="lbl", anchor="start"))
    # window annotation
    wx0 = sx + 12 * cw
    wx1 = sx + 25 * cw - 3
    p.append(arrow(wx0 + 6, sy + ch + 20, wx1 - 6, sy + ch + 20, marker="aSlate", color=SUB, width=1.8))
    p.append(arrow(wx1 - 6, sy + ch + 20, wx0 + 6, sy + ch + 20, marker="aSlate", color=SUB, width=1.8))
    p.append(txt((wx0 + wx1) / 2, sy + ch + 44, "随机起点 i，切出 L+1 = 129 个 token",
                 cls="small", anchor="middle"))

    # ---- x / y rows ----
    rx0, ry = 260, 302
    p.append(txt(150, ry + 28, "x（输入）", cls="lbl", anchor="start"))
    p.append(txt(150, ry + 100, "y（标签）", cls="lbl", anchor="start"))
    demo = ["F", "i", "r", "s", "t", " ", "C", "i", "t"]
    for k, chx in enumerate(demo):
        p.append(rrect(rx0 + k * 74, ry, 66, 46, BLUE_F, BLUE_B, rx=8, sw=1.6))
        p.append(txt(rx0 + k * 74 + 33, ry + 30, "␣" if chx == " " else chx, cls="monob"))
    for k, chx in enumerate(demo[1:] + ["i"]):
        p.append(rrect(rx0 + k * 74, ry + 72, 66, 46, GREEN_F, GREEN_B, rx=8, sw=1.6))
        p.append(txt(rx0 + k * 74 + 33, ry + 102, "␣" if chx == " " else chx, cls="monob"))
    p.append(txt(rx0 + 9 * 74 + 10, ry + 30, "…  共 128 个", cls="small", anchor="start"))
    p.append(txt(rx0 + 9 * 74 + 10, ry + 102, "…  共 128 个", cls="small", anchor="start"))
    for k in range(1, 9):          # x[k] 就是 y[k-1]：斜箭头把这层对应关系画出来
        p.append(arrow(rx0 + k * 74 + 33, ry + 50, rx0 + (k - 1) * 74 + 40,
                       ry + 68, marker="aGray", color=GRAY_B, width=1.5))
    p.append(txt(rx0 + 4 * 74, ry + 150,
                 "同一个窗口错开一位：位置 t 的输入是 x[t]，它要预测的答案就是 y[t] = x[t+1]",
                 cls="cap"))

    # ---- supervision list ----
    lx, ly = 90, 512
    p.append(rrect(lx, ly, 700, 250, "#fbfdff", BLUE_B, rx=14, sw=1.4))
    p.append(txt(lx + 20, ly + 32, "一条样本 = 128 条监督（列出前 5 条）", cls="h2", anchor="start"))
    rows = [("'F'", "'i'"), ("'Fi'", "'r'"), ("'Fir'", "'s'"),
            ("'Firs'", "'t'"), ("'First'", "' '")]
    for k, (ctx, tgt) in enumerate(rows):
        yy = ly + 66 + k * 34
        p.append(txt(lx + 30, yy, f"看到 {ctx}", cls="mono", anchor="start"))
        p.append(txt(lx + 250, yy, "→", cls="mono", anchor="start"))
        p.append(txt(lx + 290, yy, f"该预测 {tgt}", cls="mono", anchor="start"))
    p.append(txt(lx + 20, ly + 232,
                 "因果掩码保证第 t 个位置只看得到 x[0..t]，所以这 128 条能一次并行算完",
                 cls="small", anchor="start"))

    # ---- batch stack ----
    bx, by = 900, 512
    p.append(rrect(bx, by, 540, 250, "#fffdfa", ORANGE_B, rx=14, sw=1.4))
    p.append(txt(bx + 20, by + 32, "堆成一批：32 个互不相干的窗口", cls="h2", anchor="start"))
    for k in range(5):
        yy = by + 58 + k * 26
        p.append(rrect(bx + 34 + k * 7, yy, 300, 20, AMBER_F, AMBER_B, rx=4, sw=1.2))
        if k == 4:
            p.append(txt(bx + 190, yy + 15, "…", cls="small"))
    p.append(txt(bx + 370, by + 92, "x: [32, 128]", cls="monob", anchor="start"))
    p.append(txt(bx + 370, by + 118, "y: [32, 128]", cls="monob", anchor="start"))
    p.append(txt(bx + 20, by + 208,
                 "一次前向 = 32 × 128 = 4096 条监督；起点随机，", cls="small", anchor="start"))
    p.append(txt(bx + 20, by + 230,
                 "所以不必把语料整整齐齐切成互不重叠的段落", cls="small", anchor="start"))

    write_svg(ASSETS / "batching.svg", "\n".join(p), f"0 0 {W} {H}", bump=2)


# ============================================================
# Diagram 3: 生成的重复计算 —— 每步都把整个前缀重算一遍
# ============================================================
def generate_cost():
    W, H = 1420, 606
    p = []
    p.append(txt(W / 2, 46, "自回归生成的开销：每写一个字符，都把整个前缀重算一遍", cls="title"))
    p.append(txt(W / 2, 76,
                 "第 n 步要对长度 n 的前缀做一次完整前向，但真正新增的只有最后那一格 —— "
                 "其余全是上一步算过的", cls="cap"))

    cw, chh = 54, 42
    x0, y0 = 150, 130
    steps = [1, 2, 3, 4, 5]
    for r, n in enumerate(steps):
        yy = y0 + r * 62
        p.append(txt(x0 - 22, yy + 28, f"第 {n} 步", cls="lbl", anchor="end"))
        for k in range(n):
            new = (k == n - 1)
            fill = GREEN_F if new else GRAY_F
            border = GREEN_B if new else GRAY_B
            p.append(rrect(x0 + k * cw, yy, cw - 6, chh, fill, border, rx=6, sw=1.6))
        p.append(txt(x0 + n * cw + 18, yy + 28,
                     f"前向 {n} 个位置，其中 {n - 1} 个是重算的",
                     cls="small", anchor="start"))
    p.append(txt(x0 - 22, y0 + 5 * 62 + 24, "…", cls="lbl", anchor="end"))

    # legend
    ly = y0 + 5 * 62 + 46
    p.append(rrect(x0, ly, 22, 22, GREEN_F, GREEN_B, rx=5, sw=1.5))
    p.append(txt(x0 + 32, ly + 17, "这一步真正新增的位置", cls="small", anchor="start"))
    p.append(rrect(x0 + 240, ly, 22, 22, GRAY_F, GRAY_B, rx=5, sw=1.5))
    p.append(txt(x0 + 272, ly + 17, "上一步已经算过、这一步又算一遍的位置", cls="small", anchor="start"))

    # right panel: the arithmetic
    px, py, pw, ph = 830, 130, 520, 250
    p.append(rrect(px, py, pw, ph, "#fbfdff", BLUE_B, rx=14, sw=1.4))
    p.append(txt(px + 20, py + 34, "生成 128 个 token 的总开销", cls="h2", anchor="start"))
    lines = [
        ("累计前向的位置数", "1 + 2 + … + 128 = 8,256"),
        ("其中真正新出现的", "128"),
        ("重复计算占比", "≈ 98.4%"),
    ]
    for k, (a, b) in enumerate(lines):
        yy = py + 76 + k * 40
        p.append(txt(px + 30, yy, a, cls="sub", anchor="start"))
        p.append(txt(px + pw - 30, yy, b, cls="monob", anchor="end"))
    p.append(txt(px + 20, py + 210,
                 "序列越长，浪费的比例越高 —— 这是推理慢的主因之一", cls="small", anchor="start"))

    # bottom: pointer to KV cache
    bx, by, bw, bh = 830, 410, 520, 128
    p.append(rrect(bx, by, bw, bh, "#f0fdf4", GREEN_B, rx=14, sw=1.6))
    p.append(txt(bx + 20, by + 34, "为什么可以省掉这些重复", cls="h2", anchor="start"))
    p.append(txt(bx + 20, by + 64,
                 "因果掩码下，前面 token 的 K / V 不受后面影响，", cls="small", anchor="start"))
    p.append(txt(bx + 20, by + 88,
                 "算过一次就永远有效 —— 缓存起来即可，这就是 KV cache", cls="small", anchor="start"))
    p.append(txt(bx + 20, by + 112, "（第 14 章的主题）", cls="small", anchor="start"))

    p.append(txt(W / 2, H - 26,
                 "本章这个 generate 循环是最朴素的写法：正确、好懂，但每一步都在做重复功",
                 cls="cap"))
    write_svg(ASSETS / "generate-cost.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    loop()
    batching()
    generate_cost()
    print("wrote:", ", ".join(sorted(pth.name for pth in ASSETS.glob("*.svg"))))
