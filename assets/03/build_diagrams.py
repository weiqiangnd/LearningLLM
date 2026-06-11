"""Generate conceptual diagrams for chapter 03 (Tokenizer).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
Hand-written SVG (same approach as assets/P05/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses
gray-700 (#374151) or darker per the repo contrast guideline.

Run from repo root:
    python3 assets/03/build_diagrams.py
Then export each SVG to PNG with rsvg-convert + pngquant, e.g.
    rsvg-convert -w 1400 assets/03/pipeline.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/03/pipeline.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Zen Hei', sans-serif")

# ---------- shared palette (Flat Icon) ----------
BG = "#ffffff"
TXT = "#1e293b"   # slate-800  primary labels
SUB = "#334155"   # slate-700  secondary / sublabel / caption (>= gray-700)
FAINT = "#64748b" # slate-500  only for genuinely de-emphasized notes

BLUE_F, BLUE_B = "#dbeafe", "#2563eb"
GREEN_F, GREEN_B = "#dcfce7", "#059669"
ORANGE_F, ORANGE_B = "#ffedd5", "#ea580c"
PURPLE_F, PURPLE_B = "#ede9fe", "#7c3aed"
RED_F, RED_B = "#fee2e2", "#dc2626"
GRAY_F, GRAY_B = "#f3f4f6", "#94a3b8"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg(path: Path, body: str, viewbox: str):
    style = f"""
  <style>
    text {{ font-family: {FONT}; }}
    .title  {{ font-size: 23px; font-weight: 700; fill: {TXT}; }}
    .lbl    {{ font-size: 15px; font-weight: 600; fill: {TXT}; }}
    .sub    {{ font-size: 13px; fill: {SUB}; }}
    .mono   {{ font-size: 14px; font-family: 'SFMono-Regular','Consolas','Liberation Mono',monospace; fill: {TXT}; }}
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
# Diagram 1: where the tokenizer sits in the LLM pipeline
# ============================================================
def diagram_pipeline():
    W, H = 1410, 450
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'Tokenizer 在 LLM 流水线中的位置：文本世界 ↔ id 世界的翻译官</text>']

    cy, h = 240, 86
    # nodes: (cx, w, label, sub, fill, border)
    nodes = [
        (95,   125, "输入文本",        "你好，世界",        GRAY_F,   GRAY_B),
        (285,  150, "Tokenizer",       "切分→查表→id",      ORANGE_F, ORANGE_B),
        (500,  155, "token ids",       "[108386, 99489…]",  BLUE_F,   BLUE_B),
        (715,  155, "Embedding",       "id→向量(查表)",     GREEN_F,  GREEN_B),
        (930,  165, "Transformer × N", "注意力 + FFN 堆叠", BLUE_F,   BLUE_B),
        (1150, 160, "logits → 采样",   "得到下一个 id",     PURPLE_F, PURPLE_B),
        (1345, 120, "Tokenizer",       "id→文本",           ORANGE_F, ORANGE_B),
    ]

    # translucent band behind the "all-numbers" middle (ids .. logits)
    band_x1 = 500 - 155 / 2 - 12
    band_x2 = 1150 + 160 / 2 + 12
    p.append(f'<rect x="{band_x1}" y="158" width="{band_x2-band_x1}" height="168" rx="14" '
             f'fill="#f1f5f9" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,4"/>')
    p.append(f'<text x="{(band_x1+band_x2)/2}" y="178" text-anchor="middle" class="sub" '
             f'font-weight="600">模型内部：从头到尾只有数字（id / 向量 / logits），看不见一个汉字或字母</text>')

    centers = []
    for cx, w, lbl, sub, f, b in nodes:
        p += node(cx, cy, w, h, lbl, sub, f, b)
        centers.append((cx, w))

    # left→right arrows
    for i in range(len(centers) - 1):
        cx1, w1 = centers[i]
        cx2, w2 = centers[i + 1]
        p += arrow(cx1 + w1 / 2 + 2, cy, cx2 - w2 / 2 - 4, cy)

    # arrow labels: encode at front, decode at back
    p.append(f'<text x="{(285+150/2+500-155/2)/2}" y="{cy-58}" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">encode</text>')
    p.append(f'<text x="{(1150+160/2+1345-120/2)/2}" y="{cy-50}" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">decode</text>')

    # autoregressive loop-back: logits -> embedding (over the top)
    sx, sy = 1150, cy - h / 2
    ex, ey = 715, cy - h / 2
    midy = 132
    p.append(f'<path d="M {sx},{sy} C {sx},{midy} {ex},{midy} {ex},{ey}" '
             f'stroke="{PURPLE_B}" stroke-width="2" fill="none" stroke-dasharray="6,4" '
             f'marker-end="url(#aPurple)"/>')
    p.append(f'<text x="{(sx+ex)/2}" y="{midy-8}" text-anchor="middle" class="small" '
             f'fill="{PURPLE_B}">自回归：新 id 拼回输入末尾，再算下一个（详见第 2 章）</text>')

    p.append(f'<text x="{W/2}" y="410" text-anchor="middle" class="cap">'
             f'模型看不懂文字、只认识整数 id；tokenizer 是两端唯一的翻译工序（橙色）——本章只讲这道工序，中间的 Embedding / Transformer 留给后续章节。</text>')

    write_svg(ASSETS / "pipeline.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: char vs word vs subword granularity
# ============================================================
def diagram_granularity():
    W, H = 1200, 540
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'三种切分粒度：为什么 LLM 选「子词」</text>']

    rows = [
        # (y, name, name_sub, color_f, color_b, en_tokens, zh_tokens, trade_line1, trade_line2)
        (118, "按字符 char", "拆到单个字符", BLUE_F, BLUE_B,
         list("tokenization"), ["机", "器", "学", "习"],
         "词表极小（几十～几千）；", "但序列超长、单个 token 几乎没有语义"),
        (262, "按词 word", "整词一个 token", RED_F, RED_B,
         ["tokenization"], ["机器学习"],
         "序列最短；", "但词表爆炸（百万级），没见过的词直接变 UNK（OOV）"),
        (406, "子词 subword（BPE）", "高频片段成块", GREEN_F, GREEN_B,
         ["token", "ization"], ["机器", "学习"],
         "折中：词表可控（几万）、几乎无 OOV、", "常见词整体成 token —— LLM 的主流选择"),
    ]

    box_left = 60
    tokens_x = 250          # where token boxes start
    trade_x = 880           # where trade-off text starts

    def draw_tokens(x, y, toks, border, fill):
        cur = x
        ph = 30
        for t in toks:
            w = max(26, 11 * len(t) + 16)
            p.append(f'<rect x="{cur}" y="{y-ph/2}" width="{w}" height="{ph}" rx="6" '
                     f'fill="{fill}" stroke="{border}" stroke-width="1.6"/>')
            p.append(f'<text x="{cur+w/2}" y="{y+5}" text-anchor="middle" class="mono">{esc(t)}</text>')
            cur += w + 7
        return cur

    for (y, name, nsub, cf, cb, en, zh, trade1, trade2) in rows:
        # row container
        p.append(f'<rect x="{box_left}" y="{y-58}" width="{W-2*box_left}" height="116" rx="12" '
                 f'fill="#fbfdff" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,4"/>')
        # left name tag
        p.append(f'<rect x="{box_left+14}" y="{y-26}" width="170" height="52" rx="9" '
                 f'fill="{cf}" stroke="{cb}" stroke-width="2"/>')
        p.append(f'<text x="{box_left+99}" y="{y-3}" text-anchor="middle" class="lbl">{esc(name)}</text>')
        p.append(f'<text x="{box_left+99}" y="{y+17}" text-anchor="middle" class="small">{esc(nsub)}</text>')
        # english tokens (upper line) + chinese tokens (lower line)
        end_en = draw_tokens(tokens_x, y - 16, en, cb, cf)
        end_zh = draw_tokens(tokens_x, y + 22, zh, cb, cf)
        # count tag
        cnt = f"{len(en)} + {len(zh)} 个 token"
        p.append(f'<text x="{trade_x-20}" y="{y-30}" text-anchor="end" class="small" '
                 f'fill="{cb}">{esc(cnt)}</text>')
        # trade-off text (two explicit lines)
        p.append(f'<text x="{trade_x}" y="{y-2}" class="sub">{esc(trade1)}</text>')
        p.append(f'<text x="{trade_x}" y="{y+20}" class="sub">{esc(trade2)}</text>')

    p.append(f'<text x="{W/2}" y="500" text-anchor="middle" class="cap">'
             f'同一段「tokenization / 机器学习」在三种粒度下的切法：char 丢语义、word 易 OOV，subword 在词表大小与序列长度之间取平衡。</text>')

    write_svg(ASSETS / "granularity.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: BPE training — the merge loop + worked example
# ============================================================
def diagram_bpe_merge():
    W, H = 1220, 650
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'BPE 训练：反复「找最高频相邻对 → 合并」，直到词表达到目标大小</text>']

    # ---- top: the loop (3 steps) ----
    ly = 130
    steps = [
        (210, "① 统计相邻对", "数所有相邻符号对出现的频次", BLUE_F, BLUE_B),
        (610, "② 选最高频对", "频次最大的那一对", PURPLE_F, PURPLE_B),
        (1010, "③ 合并 + 记录", "拼成新符号，加进词表，记下规则", GREEN_F, GREEN_B),
    ]
    cw, ch = 300, 70
    cs = []
    for cx, lbl, sub, f, b in steps:
        p += node(cx, ly, cw, ch, lbl, sub, f, b)
        cs.append(cx)
    p += arrow(cs[0] + cw / 2 + 2, ly, cs[1] - cw / 2 - 4, ly)
    p += arrow(cs[1] + cw / 2 + 2, ly, cs[2] - cw / 2 - 4, ly)
    # loop back arc step3 -> step1 over the top
    p.append(f'<path d="M {cs[2]},{ly-ch/2} C {cs[2]},70 {cs[0]},70 {cs[0]},{ly-ch/2}" '
             f'stroke="{ORANGE_B}" stroke-width="2" fill="none" stroke-dasharray="6,4" '
             f'marker-end="url(#aOrange)"/>')
    p.append(f'<text x="{(cs[0]+cs[2])/2}" y="62" text-anchor="middle" class="small" '
             f'fill="{ORANGE_B}">重复，每轮词表 +1，直到达到目标大小</text>')

    # divider
    p.append(f'<line x1="60" y1="200" x2="{W-60}" y2="200" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="4,4"/>')
    p.append(f'<text x="60" y="232" class="lbl">一个最小可手算的例子</text>')

    # ---- left: corpus + base vocab ----
    cx0 = 60
    p.append(f'<rect x="{cx0}" y="250" width="360" height="180" rx="11" '
             f'fill="{GRAY_F}" stroke="{GRAY_B}" stroke-width="1.5"/>')
    p.append(f'<text x="{cx0+18}" y="276" class="lbl">语料（词 × 频次）</text>')
    corpus = ["hug × 10", "pug × 5", "pun × 12", "bun × 4", "hugs × 5"]
    for i, c in enumerate(corpus):
        p.append(f'<text x="{cx0+24}" y="{303+i*22}" class="mono">{esc(c)}</text>')
    p.append(f'<text x="{cx0+200}" y="303" class="sub">初始按字符拆：</text>')
    p.append(f'<text x="{cx0+200}" y="325" class="mono">h u g</text>')
    p.append(f'<text x="{cx0+200}" y="347" class="mono">p u n …</text>')
    p.append(f'<text x="{cx0+200}" y="378" class="sub">基础词表 (7)：</text>')
    p.append(f'<text x="{cx0+200}" y="400" class="mono">b g h n p s u</text>')

    # ---- middle: the 3 learned merges ----
    mx = 470
    p.append(f'<text x="{mx}" y="276" class="lbl">学到的 3 条合并规则（按顺序）</text>')
    merges = [
        ("merge 1", "(u, g) → ug", "频次 20，最高", PURPLE_B),
        ("merge 2", "(u, n) → un", "频次 16", PURPLE_B),
        ("merge 3", "(h, ug) → hug", "频次 15", PURPLE_B),
    ]
    for i, (tag, rule, note, b) in enumerate(merges):
        yy = 296 + i * 46
        p.append(f'<rect x="{mx}" y="{yy}" width="330" height="38" rx="8" '
                 f'fill="{PURPLE_F}" stroke="{b}" stroke-width="1.6"/>')
        p.append(f'<text x="{mx+14}" y="{yy+24}" class="tag" fill="{b}">{esc(tag)}</text>')
        p.append(f'<text x="{mx+90}" y="{yy+24}" class="mono">{esc(rule)}</text>')
        p.append(f'<text x="{mx+232}" y="{yy+24}" class="small">{esc(note)}</text>')
    p.append(f'<text x="{mx}" y="458" class="sub">词表：7 → 10（每条规则 +1 个新符号）</text>')

    # ---- right: how "hugs" evolves ----
    rx = 850
    p.append(f'<rect x="{rx}" y="250" width="310" height="180" rx="11" '
             f'fill="{GREEN_F}" stroke="{GREEN_B}" stroke-width="1.5"/>')
    p.append(f'<text x="{rx+18}" y="276" class="lbl">编码 "hugs" 时按规则依次套用</text>')
    evo = [
        ("初始", "h · u · g · s"),
        ("套 merge1 (ug)", "h · ug · s"),
        ("套 merge3 (hug)", "hug · s"),
    ]
    for i, (tag, state) in enumerate(evo):
        yy = 308 + i * 38
        p.append(f'<text x="{rx+20}" y="{yy}" class="small">{esc(tag)}</text>')
        p.append(f'<text x="{rx+170}" y="{yy}" class="mono">{esc(state)}</text>')
    p.append(f'<text x="{rx+20}" y="420" class="sub" font-weight="600">结果：[hug, s]（2 个 token）</text>')

    # ---- bottom caption ----
    p.append(f'<rect x="60" y="468" width="{W-120}" height="118" rx="11" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,3"/>')
    p.append(f'<text x="80" y="496" class="lbl">几个要点</text>')
    p.append(f'<text x="80" y="522" class="sub">• 训练产物 = 一份「词表」+ 一串「有序的合并规则」；vocab_size 越大，合并轮数越多、token 越长。</text>')
    p.append(f'<text x="80" y="546" class="sub">• 编码新词时，把它拆成字符后，按训练时学到的「先后顺序」逐条套用合并规则——顺序很关键。</text>')
    p.append(f'<text x="80" y="570" class="sub">• 没在语料里出现过的词（如 "bug"）也能切：拆成 [b, ug] 这种已知片段，不会报错。</text>')

    write_svg(ASSETS / "bpe-merge.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: byte-level BPE never goes OOV
# ============================================================
def diagram_bbpe():
    W, H = 1120, 540
    p = [f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
         f'Byte-level BPE（BBPE）：先转 UTF-8 字节，从根上消灭 OOV</text>']

    # ---- top row: char-level BPE, rare char -> UNK ----
    ry1 = 120
    p.append(f'<rect x="60" y="{ry1-44}" width="{W-120}" height="120" rx="12" '
             f'fill="{RED_F}" stroke="{RED_B}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    p.append(f'<text x="80" y="{ry1-20}" class="lbl" fill="{RED_B}">字符级 BPE：词表里没有的字 → UNK（信息丢失）</text>')
    # flow
    p += node(165, ry1+30, 130, 56, "中", "（假设训练没见过）", "#ffffff", RED_B)
    p += arrow(232, ry1+30, 322, ry1+30, marker="aRed", color=RED_B)
    p += node(415, ry1+30, 150, 56, "查字符词表", "找不到这个字", "#ffffff", RED_B)
    p += arrow(492, ry1+30, 582, ry1+30, marker="aRed", color=RED_B)
    p += node(660, ry1+30, 120, 56, "[UNK]", "未知符号", RED_F, RED_B)
    p += arrow(722, ry1+30, 812, ry1+30, marker="aRed", color=RED_B)
    p += node(910, ry1+30, 150, 56, "decode 还原", "→ �  还原失败", "#ffffff", RED_B)

    # ---- middle emphasis ----
    my = 275
    p.append(f'<rect x="300" y="{my-26}" width="520" height="52" rx="11" '
             f'fill="{ORANGE_F}" stroke="{ORANGE_B}" stroke-width="2"/>')
    p.append(f'<text x="560" y="{my-2}" text-anchor="middle" class="lbl" fill="{ORANGE_B}">'
             f'基础词表 = 全部 256 个字节</text>')
    p.append(f'<text x="560" y="{my+17}" text-anchor="middle" class="small">'
             f'任何 Unicode 字符都能拆成字节，且字节只有 256 种 → 永远不会 OOV</text>')

    # ---- bottom row: byte-level BPE ----
    ry2 = 400
    p.append(f'<rect x="60" y="{ry2-44}" width="{W-120}" height="120" rx="12" '
             f'fill="{GREEN_F}" stroke="{GREEN_B}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    p.append(f'<text x="80" y="{ry2-20}" class="lbl" fill="{GREEN_B}">字节级 BPE：先 UTF-8 编码成字节，每个字节都在 256 基础词表里</text>')
    p += node(165, ry2+30, 130, 56, "中", "同一个字", "#ffffff", GREEN_B)
    p += arrow(232, ry2+30, 322, ry2+30, marker="aGreen", color=GREEN_B)
    p += node(420, ry2+30, 160, 56, "UTF-8 编码", "E4 B8 AD（3 字节）", "#ffffff", GREEN_B)
    p += arrow(502, ry2+30, 592, ry2+30, marker="aGreen", color=GREEN_B)
    p += node(680, ry2+30, 150, 56, "字节 token", "(可再被合并)", GREEN_F, GREEN_B)
    p += arrow(757, ry2+30, 847, ry2+30, marker="aGreen", color=GREEN_B)
    p += node(940, ry2+30, 140, 56, "decode 还原", "→ 中  ✓", "#ffffff", GREEN_B)

    p.append(f'<text x="{W/2}" y="510" text-anchor="middle" class="cap">'
             f'GPT-2、Qwen、Llama 等都用 byte-level BPE：多语言 / emoji / 生僻字一律先转字节再合并，根除 UNK。代价是没被合并的字符要占多个 token（一个汉字最多 3 个）。</text>')

    write_svg(ASSETS / "bbpe.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 5: stage-2 architecture map (Attention-Is-All-You-Need style)
#   A vertical decoder-only LLM pipeline covering every component touched
#   across stage-2 chapters 3-13. Visual language follows AIAYN fig.1:
#   pastel sublayers, yellow Add & Norm bars, orange attention, blue FFN,
#   the N× block bracket and the ⊕ positional-encoding add. Each component
#   box carries a chapter chip; the bottom panel lists what every chapter
#   (3-13) contributes. `highlight` (a set of box keys) can ring a box in
#   red for a "you are here" variant — chapter 3 passes an empty set so
#   nothing is highlighted (the figure is a neutral stage-2 overview).
# ============================================================
def architecture_map(highlight, out_path, hl_caption):
    W, H = 1200, 992
    CX = 340
    BW = 300
    HLC = "#dc2626"
    ATTN_F, ATTN_B = "#fbe1c4", "#d9760f"
    NORM_F, NORM_B = "#fdeeb3", "#b8920a"
    FFN_F,  FFN_B  = "#cfe0fb", "#2f6fd0"
    EMB_F,  EMB_B  = "#f8d3e4", "#c43f86"
    PE_F,   PE_B   = "#e7dcfb", "#7a4fd0"
    LIN_F,  LIN_B  = "#cfe0fb", "#2f6fd0"
    SM_F,   SM_B   = "#cdebd0", "#2f9d52"
    TOK_F,  TOK_B  = "#cdeaf0", "#1f93b0"
    TRM_F,  TRM_B  = "#eef1f4", "#7c8896"

    p = []
    p.append(f'<text x="{W/2}" y="40" text-anchor="middle" class="title">'
             f'阶段 2 全景：一个 LLM 从文本到下一个 token 的完整流水线</text>')
    p.append(f'<text x="{W/2}" y="64" text-anchor="middle" class="sub" '
             f'font-size="13.5">{esc(hl_caption)}</text>')

    # ---- left half: the pipeline ----
    # transformer block container + N× curly brace
    p.append(f'<rect x="175" y="356" width="330" height="292" rx="14" '
             f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1.2" stroke-dasharray="6,4"/>')

    def lbrace(xR, T, B, depth, r=9):
        mid = (T + B) / 2
        qx = xR - depth
        cx = qx - depth
        return (f"M {xR},{T} Q {qx},{T} {qx},{T+r} L {qx},{mid-r} "
                f"Q {qx},{mid} {cx},{mid} Q {qx},{mid} {qx},{mid+r} "
                f"L {qx},{B-r} Q {qx},{B} {xR},{B}")
    p.append(f'<path d="{lbrace(172, 360, 644, 10)}" fill="none" '
             f'stroke="{SUB}" stroke-width="2"/>')
    p.append(f'<text x="143" y="497" text-anchor="end" class="title" font-size="19">N×</text>')
    p.append(f'<text x="143" y="519" text-anchor="end" class="small">解码器层</text>')

    # vertical connectors (data path, arrows pointing up)
    conns = [(905, 882), (826, 804), (748, 717), (683, 633), (563, 550),
             (506, 488), (432, 414), (370, 329), (275, 259), (205, 189), (135, 111)]
    for y_low, y_high in conns:
        p.append(f'<line x1="{CX}" y1="{y_low}" x2="{CX}" y2="{y_high}" '
                 f'stroke="{GRAY_B}" stroke-width="2" marker-end="url(#aGray)"/>')
    # positional-encoding side arrow into the ⊕
    p.append(f'<line x1="300" y1="700" x2="{CX-19}" y2="700" '
             f'stroke="{GRAY_B}" stroke-width="2" marker-end="url(#aGray)"/>')

    def chip(box_right, cy, text, accent, on):
        w = 11 * len(text) + 18
        x = box_right + 12
        fill = HLC if on else "#ffffff"
        stroke = HLC if on else accent
        tcol = "#ffffff" if on else "#374151"
        return [f'<rect x="{x}" y="{cy-12}" width="{w}" height="24" rx="7" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
                f'<text x="{x+w/2}" y="{cy+5}" text-anchor="middle" class="tag" '
                f'fill="{tcol}">{esc(text)}</text>']

    def abox(cy, h, label, sub, sub2, fill, border, key, chap):
        on = key in highlight
        x, y = CX - BW/2, cy - h/2
        out = []
        if on:
            out.append(f'<rect x="{x-7}" y="{y-7}" width="{BW+14}" height="{h+14}" '
                       f'rx="15" fill="#fff5f5"/>')
        bw = 3.2 if on else 1.8
        bcol = HLC if on else border
        out.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{h}" rx="11" '
                   f'fill="{fill}" stroke="{bcol}" stroke-width="{bw}"/>')
        if sub2:
            out.append(f'<text x="{CX}" y="{cy-10}" text-anchor="middle" class="lbl">{esc(label)}</text>')
            out.append(f'<text x="{CX}" y="{cy+9}" text-anchor="middle" class="small">{esc(sub)}</text>')
            out.append(f'<text x="{CX}" y="{cy+27}" text-anchor="middle" class="small">{esc(sub2)}</text>')
        elif sub:
            out.append(f'<text x="{CX}" y="{cy-3}" text-anchor="middle" class="lbl">{esc(label)}</text>')
            out.append(f'<text x="{CX}" y="{cy+16}" text-anchor="middle" class="small">{esc(sub)}</text>')
        else:
            out.append(f'<text x="{CX}" y="{cy+5}" text-anchor="middle" class="lbl">{esc(label)}</text>')
        if chap:
            out += chip(x + BW, cy, chap, border, on)
        return out

    # boxes, top -> bottom (output at top, input at bottom — AIAYN orientation)
    p += abox(92, 38, "采样 → 下一个 token", "", "", TRM_F, TRM_B, "out", "第 2 · 10 章")
    p += abox(162, 54, "Softmax", "logits → 概率分布", "", SM_F, SM_B, "sm", "第 10 章")
    p += abox(232, 54, "Linear（lm_head）", "隐藏向量 → 词表 logits", "", LIN_F, LIN_B, "lin", "第 4 章")
    p += abox(302, 54, "Final Norm", "末层归一化（RMSNorm）", "", NORM_F, NORM_B, "fn", "第 8 章")
    p += abox(392, 44, "Add & Norm", "", "", NORM_F, NORM_B, "an2", "第 8 章")
    p += abox(460, 56, "Feed Forward（FFN）", "SwiGLU / GeLU", "", FFN_F, FFN_B, "ffn", "第 8 章")
    p += abox(528, 44, "Add & Norm", "", "", NORM_F, NORM_B, "an1", "第 8 章")
    p += abox(598, 70, "Masked Multi-Head Self-Attention",
              "Q / K / V · 缩放点积 · 因果掩码", "多头拼接 · MQA / GQA",
              ATTN_F, ATTN_B, "attn", "第 6 · 7 章")
    p += abox(776, 56, "Token Embedding", "id → 稠密向量（查表 / weight tying）", "",
              EMB_F, EMB_B, "emb", "第 4 章")
    p += abox(854, 56, "Tokenizer", "文本 ↔ id（BPE / BBPE / 词表）", "",
              TOK_F, TOK_B, "tok", "第 3 章")
    p += abox(924, 38, "输入文本 Input Text", "", "", TRM_F, TRM_B, "in", "")

    # ⊕ positional-encoding add — moved up to sit midway between the
    # transformer block (above) and the embedding (below)
    p.append(f'<circle cx="{CX}" cy="700" r="17" fill="#ffffff" stroke="{SUB}" stroke-width="2"/>')
    p.append(f'<text x="{CX}" y="707" text-anchor="middle" class="title" font-size="22">+</text>')

    # positional-encoding side box feeding the ⊕; "第 4 章" tag to its LEFT
    pe_on = "pe" in highlight
    pex, pey, pew, peh = 108, 671, 192, 58
    pe_stroke = HLC if pe_on else PE_B
    pe_bw = 3.2 if pe_on else 1.8
    pe_chip_fill = HLC if pe_on else "#ffffff"
    pe_chip_txt = "#ffffff" if pe_on else "#374151"
    if pe_on:
        p.append(f'<rect x="{pex-6}" y="{pey-6}" width="{pew+12}" height="{peh+12}" rx="14" fill="#fff5f5"/>')
    p.append(f'<rect x="{pex}" y="{pey}" width="{pew}" height="{peh}" rx="11" '
             f'fill="{PE_F}" stroke="{pe_stroke}" stroke-width="{pe_bw}"/>')
    p.append(f'<text x="{pex+pew/2}" y="{pey+24}" text-anchor="middle" class="lbl">Positional Encoding</text>')
    p.append(f'<text x="{pex+pew/2}" y="{pey+43}" text-anchor="middle" class="small">位置编码 · RoPE / ALiBi</text>')
    p.append(f'<rect x="44" y="{pey+peh/2-11}" width="58" height="22" rx="7" '
             f'fill="{pe_chip_fill}" stroke="{pe_stroke}" stroke-width="1.4"/>')
    p.append(f'<text x="73" y="{pey+peh/2+5}" text-anchor="middle" class="tag" fill="{pe_chip_txt}">第 4 章</text>')

    # ---- right half: per-chapter directory (text panel) ----
    # left = the pipeline picture; right = what every stage-2 chapter does.
    p.append(f'<line x1="618" y1="96" x2="618" y2="900" stroke="{GRAY_B}" '
             f'stroke-width="1" stroke-dasharray="5,5"/>')
    p.append(f'<text x="650" y="120" class="title" font-size="18">阶段 2 各章分工</text>')
    p.append(f'<text x="650" y="142" class="sub">第 3–13 章——左图每个组件框右侧的章节角标，对应下面这一条：</text>')
    entries = [
        ("第 3 章 · Tokenizer", "文本 ↔ id：BPE / BBPE、词表、特殊 token", TOK_B),
        ("第 4 章 · Embedding 与位置编码", "id → 向量、weight tying；RoPE / 正弦 / ALiBi", EMB_B),
        ("第 5 章 · 从 RNN 到 attention", "讲清注意力「为什么会出现」", "#64748b"),
        ("第 6 章 · Scaled Dot-Product Attention", "Q / K / V、softmax、因果掩码", ATTN_B),
        ("第 7 章 · Multi-Head Attention 与 MQA / GQA", "分头、拼接、形状变换全过程", ATTN_B),
        ("第 8 章 · FFN、残差连接、归一化", "LayerNorm / RMSNorm / SwiGLU", FFN_B),
        ("第 9 章 · Transformer 整体架构", "Decoder-only vs Encoder-Decoder、Pre / Post-LN", "#64748b"),
        ("第 10 章 · 自回归语言建模目标", "交叉熵 loss、teacher forcing 与采样的关系", SM_B),
        ("第 11 章 · 论文精读", "Attention Is All You Need 逐节解读", "#64748b"),
        ("第 12 章 · 论文串读", "GPT / LLaMA / Qwen 系列的关键改动", "#64748b"),
        ("第 13 章 · 从零实现 mini-GPT", "把以上所有组件拼起来，训练 → 生成闭环", "#64748b"),
    ]
    y0, step = 184, 64
    for i, (head, role, col) in enumerate(entries):
        ey = y0 + i * step
        p.append(f'<rect x="650" y="{ey-13}" width="4" height="40" rx="2" fill="{col}"/>')
        p.append(f'<text x="666" y="{ey+2}" class="lbl">{esc(head)}</text>')
        p.append(f'<text x="666" y="{ey+22}" class="sub">{esc(role)}</text>')

    write_svg(out_path, "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_pipeline()
    diagram_granularity()
    diagram_bpe_merge()
    diagram_bbpe()
    architecture_map(set(), ASSETS / "architecture-map.svg",
                     "左侧是流水线（自下而上：文本从底部进入、下一个 token 从顶部产出），"
                     "右侧逐条列出阶段 2 每一章的分工")
    print("SVGs generated under", ASSETS)
