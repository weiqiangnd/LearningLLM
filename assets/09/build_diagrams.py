"""Generate conceptual diagrams for chapter 09 (Transformer 整体架构).

Style: Flat Icon (style 1) — white bg, soft fills, colored borders, Noto Sans
CJK SC. Hand-written SVG (same approach as assets/03/build_diagrams.py) so the
layout can be tuned precisely. Body / sublabel / caption text uses gray-700
(#374151) or darker per the repo contrast guideline.

Diagrams
  1. panorama.svg       —— 技术视角全景：从 input_ids 到 logits，标注每个组件的
                           形状变换与参数量（以 Qwen3-8B 为标尺），并放大一个
                           Decoder Block 展示它怎么由 RMSNorm+GQA+SwiGLU 拼装。
  2. encoder-decoder.svg —— 原始 Transformer 两栈结构：encoder（双向）/ decoder
                           （masked + cross-attention）/ 三种注意力的接线。
  3. pre-post-ln.svg    —— Pre-LN vs Post-LN 的数据流与残差高速公路对比。
  4. residual-stream.svg —— 残差流视角：一条贯穿所有层的「主干总线」，每个子层只
                           往上面加一个增量 Δ。

Run from repo root:
    python3 assets/09/build_diagrams.py
Then export each SVG to PNG (default -w 2400, dense ones -w 3000):
    rsvg-convert -w 3000 assets/09/panorama.svg -o /tmp/x.png
    pngquant --quality=100 --strip --force --output assets/09/panorama.png /tmp/x.png
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

# AIAYN-flavoured component colours (match assets/03 architecture-map)
ATTN_F, ATTN_B = "#fbe1c4", "#d9760f"
NORM_F, NORM_B = "#fdeeb3", "#b8920a"
FFN_F,  FFN_B  = "#cfe0fb", "#2f6fd0"
EMB_F,  EMB_B  = "#f8d3e4", "#c43f86"
PE_F,   PE_B   = "#e7dcfb", "#7a4fd0"
SM_F,   SM_B   = "#cdebd0", "#2f9d52"
TOK_F,  TOK_B  = "#cdeaf0", "#1f93b0"
TRM_F,  TRM_B  = "#eef1f4", "#7c8896"


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
    .monob  {{ font-size: 14px; font-weight: 600; font-family: {MONO}; fill: {TXT}; }}
    .small  {{ font-size: 13px; fill: {SUB}; }}
    .cap    {{ font-size: 15px; fill: {SUB}; }}
    .tag    {{ font-size: 13px; font-weight: 700; }}
  </style>
"""
    defs = f"""
  <defs>
    <marker id="aBlue" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{BLUE_B}"/></marker>
    <marker id="aGray" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{GRAY_B}"/></marker>
    <marker id="aSlate" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 11 4, 0 8" fill="{SUB}"/></marker>
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


def chip(cx, cy, text, fill, border, tcol="#ffffff", w=None):
    if w is None:
        w = 12 * len(text) + 20
    return [rrect(cx - w / 2, cy - 14, w, 28, fill, border, rx=8, sw=1.6),
            txt(cx, cy + 5, text, cls="tag", extra=f' fill="{tcol}"')]


# ============================================================
# Diagram 1: technical panorama — shapes + param counts (Qwen3-8B)
# ============================================================
def panorama():
    W, H = 1440, 1044
    p = []
    p.append(txt(W / 2, 48, "Transformer 整体架构全景：从 input_ids 到 logits", cls="title"))
    p.append(txt(W / 2, 76,
                 "以 Qwen3-8B 为标尺 —— 每格标注张量形状的变换与该组件的参数量；"
                 "B = batch，L = 序列长度，d = 4096", cls="cap"))

    # =================== left column: macro data path ===================
    CXL = 250
    BW = 320

    def shapebox(cy, h, label, shape, fill, border, param=None, plabel=""):
        x, y = CXL - BW / 2, cy - h / 2
        out = [rrect(x, y, BW, h, fill, border, rx=12, sw=2)]
        out.append(txt(CXL, cy - 6, label, cls="lbl"))
        out.append(txt(CXL, cy + 17, shape, cls="monob"))
        if param is not None:
            # param chip hanging off the right edge
            cw = 11 * len(param) + 20
            cx = x + BW + 14 + cw / 2
            out += chip(cx, cy - 8, param, "#ffffff", border, tcol=TXT, w=cw)
            if plabel:
                out.append(txt(cx, cy + 23, plabel, cls="small"))
        return out

    # boxes top -> bottom (input at top, logits at bottom)
    p += shapebox(150, 58, "input_ids", "[B, L]  (int64)", TRM_F, TRM_B)
    p += shapebox(254, 64, "Token Embedding", "[B, L] → [B, L, 4096]", EMB_F, EMB_B,
                  param="622 M", plabel="151936×4096")
    # decoder block stack container (tall)
    cont_y, cont_h = 430, 250
    p.append(rrect(CXL - BW / 2 - 24, cont_y - cont_h / 2, BW + 48, cont_h,
                   "#f8fafc", GRAY_B, rx=16, sw=1.3, dash="7,5"))
    p.append(txt(CXL - BW / 2 - 6, cont_y - cont_h / 2 + 22, "Decoder Block × 36",
                 cls="h2", anchor="start"))
    p += shapebox(370, 60, "① 注意力子层 (GQA)", "[B,L,4096] → [B,L,4096]", ATTN_F, ATTN_B,
                  param="≈42 M", plabel="每层")
    p.append(txt(CXL, 446, "⊕ 残差 · RMSNorm 裹在外面", cls="small"))
    p += shapebox(488, 60, "② FFN 子层 (SwiGLU)", "[B,L,4096] → [B,L,4096]", FFN_F, FFN_B,
                  param="≈151 M", plabel="每层")
    p.append(txt(CXL, 534, "形状守恒 → 可堆叠 36 层", cls="small"))
    # final norm / lm_head / logits
    p += shapebox(608, 58, "Final RMSNorm", "[B, L, 4096]", NORM_F, NORM_B,
                  param="~0", plabel="4096")
    p += shapebox(710, 64, "lm_head (Linear)", "[B,L,4096] → [B,L,151936]", FFN_F, FFN_B,
                  param="622 M", plabel="4096×151936")
    p += shapebox(822, 58, "logits → softmax / 采样", "[B, L, 151936]", SM_F, SM_B,
                  param=None)
    p.append(txt(CXL, 872, "（采样 = 第 2 章 / 训练 = 第 10 章）", cls="small"))

    # vertical connectors (downward data flow)
    flow = [(150 + 29, 254 - 32), (254 + 32, cont_y - cont_h / 2 - 0),
            (cont_y + cont_h / 2, 608 - 29), (608 + 29, 710 - 32),
            (710 + 32, 822 - 29)]
    for y1, y2 in flow:
        p.append(arrow(CXL, y1, CXL, y2, marker="aGray", color=GRAY_B))
    # inside-container small arrow between the two sublayers
    p.append(arrow(CXL, 370 + 30, CXL, 458 - 4, marker="aGray", color=GRAY_B))

    # =================== right: exploded single block ===================
    EX = 1090            # exploded column centre
    EBW = 470            # inner op-box width
    ex_x = EX - EBW / 2 - 70
    ex_w = EBW + 140
    ex_top, ex_bot = 112, 818
    p.append(rrect(ex_x, ex_top, ex_w, ex_bot - ex_top, "#fcfcfd", PURPLE_B,
                   rx=18, sw=1.6, dash="2,0"))
    p.append(txt(EX, ex_top + 30, "放大一个 Decoder Block", cls="h2"))
    p.append(txt(EX, ex_top + 54, "Pre-LN · RMSNorm · GQA · SwiGLU（Qwen3 每一层都长这样）",
                 cls="sub"))

    # dashed "zoom lens" from the macro stack to the exploded panel
    p.append(f'<line x1="{CXL + BW / 2 + 24}" y1="{cont_y - 70}" x2="{ex_x}" y2="{ex_top + 8}" '
             f'stroke="{PURPLE_B}" stroke-width="1.6" stroke-dasharray="4,4"/>')
    p.append(f'<line x1="{CXL + BW / 2 + 24}" y1="{cont_y + 70}" x2="{ex_x}" y2="{ex_bot - 8}" '
             f'stroke="{PURPLE_B}" stroke-width="1.6" stroke-dasharray="4,4"/>')

    def opbox(cy, h, label, shape, param, fill, border, w=EBW):
        x, y = EX - w / 2, cy - h / 2
        out = [rrect(x, y, w, h, fill, border, rx=10, sw=2)]
        if shape:
            out.append(txt(EX, cy - 8, label, cls="lbl"))
            out.append(txt(EX, cy + 15, shape, cls="mono"))
        else:
            out.append(txt(EX, cy + 5, label, cls="lbl"))
        if param:
            pw = 11 * len(param) + 18
            out += chip(x + w - pw / 2 - 10, y + h / 2, param, "#ffffff", border,
                        tcol=TXT, w=pw)
        return out

    # x_in
    p.append(txt(EX, ex_top + 86, "输入  x : [B, L, 4096]", cls="monob"))
    y = ex_top + 104

    # ---- attention sublayer group ----
    grp1_top = y + 4
    # rmsnorm1
    p += opbox(y + 36, 46, "RMSNorm₁", "", "4096", NORM_F, NORM_B)
    # GQA detail box (taller, multi-line)
    gqa_cy = y + 150
    gx, gw, gh = EX - EBW / 2, EBW, 156
    p.append(rrect(gx, gqa_cy - gh / 2, gw, gh, ATTN_F, ATTN_B, rx=10, sw=2))
    p.append(txt(EX, gqa_cy - gh / 2 + 26, "Grouped-Query Attention", cls="lbl"))
    p.append(txt(EX, gqa_cy - gh / 2 + 48, "32 Q头 · 8 KV头 · d_k=128 · 含 RoPE", cls="sub"))
    p.append(txt(EX, gqa_cy - gh / 2 + 74,
                 "q:4096→4096  k,v:4096→1024  o:4096→4096", cls="mono"))
    p.append(txt(EX, gqa_cy - gh / 2 + 98,
                 "[B,L,4096]→[B,32,L,128]→softmax(QKᵀ/√d_k)·V", cls="mono"))
    p.append(txt(EX, gqa_cy - gh / 2 + 120, "→ 合并 → o_proj → [B,L,4096]", cls="mono"))
    p += chip(gx + gw - 44, gqa_cy + gh / 2 - 16, "≈42 M", "#ffffff", ATTN_B, tcol=TXT, w=70)
    grp1_bot = gqa_cy + gh / 2
    # residual add 1
    add1_y = grp1_bot + 34
    p.append(f'<circle cx="{EX}" cy="{add1_y}" r="17" fill="#ffffff" stroke="{SUB}" stroke-width="2"/>')
    p.append(txt(EX, add1_y + 7, "+", cls="title", extra=' font-size="22"'))
    p.append(txt(EX + 26, add1_y + 5, "残差", cls="small", anchor="start"))

    # arrows within group 1
    p.append(arrow(EX, ex_top + 96, EX, y + 36 - 23, marker="aGray", color=GRAY_B))
    p.append(arrow(EX, y + 36 + 23, EX, gqa_cy - gh / 2 - 2, marker="aGray", color=GRAY_B))
    p.append(arrow(EX, grp1_bot, EX, add1_y - 17, marker="aGray", color=GRAY_B))

    # ---- FFN sublayer group ----
    rms2_y = add1_y + 52
    p += opbox(rms2_y, 46, "RMSNorm₂", "", "4096", NORM_F, NORM_B)
    ffn_cy = rms2_y + 116
    fx, fw, fh = EX - EBW / 2, EBW, 134
    p.append(rrect(fx, ffn_cy - fh / 2, fw, fh, FFN_F, FFN_B, rx=10, sw=2))
    p.append(txt(EX, ffn_cy - fh / 2 + 26, "SwiGLU FFN", cls="lbl"))
    p.append(txt(EX, ffn_cy - fh / 2 + 48, "SiLU(gate) ⊙ up，再降维", cls="sub"))
    p.append(txt(EX, ffn_cy - fh / 2 + 74, "gate,up:4096→12288  down:12288→4096", cls="mono"))
    p.append(txt(EX, ffn_cy - fh / 2 + 98, "[B,L,4096]→[B,L,12288]→[B,L,4096]", cls="mono"))
    p += chip(fx + fw - 44, ffn_cy + fh / 2 - 16, "≈151 M", "#ffffff", FFN_B, tcol=TXT, w=80)
    add2_y = ffn_cy + fh / 2 + 34
    p.append(f'<circle cx="{EX}" cy="{add2_y}" r="17" fill="#ffffff" stroke="{SUB}" stroke-width="2"/>')
    p.append(txt(EX, add2_y + 7, "+", cls="title", extra=' font-size="22"'))
    p.append(txt(EX + 26, add2_y + 5, "残差", cls="small", anchor="start"))

    p.append(arrow(EX, add1_y + 17, EX, rms2_y - 23, marker="aGray", color=GRAY_B))
    p.append(arrow(EX, rms2_y + 23, EX, ffn_cy - fh / 2 - 2, marker="aGray", color=GRAY_B))
    p.append(arrow(EX, ffn_cy + fh / 2, EX, add2_y - 17, marker="aGray", color=GRAY_B))

    # out
    out_y = add2_y + 40
    p.append(txt(EX, out_y, "输出  out : [B, L, 4096]   —— 与输入同形，可堆叠", cls="monob"))

    # residual bypass arcs (left side of exploded column)
    bx = EX - EBW / 2 - 40
    # bypass 1: from x_in level down around attn group to add1
    p.append(f'<path d="M {EX - 110} {ex_top + 96} L {bx} {ex_top + 96} '
             f'L {bx} {add1_y} L {EX - 17} {add1_y}" fill="none" '
             f'stroke="{GREEN_B}" stroke-width="2.4" marker-end="url(#aGreen)"/>')
    p.append(txt(bx - 6, (ex_top + 96 + add1_y) / 2, "残差旁路", cls="small",
                 anchor="end", extra=f' fill="{GREEN_B}"'))
    # bypass 2
    p.append(f'<path d="M {EX - 110} {add1_y + 17} L {bx} {add1_y + 17} '
             f'L {bx} {add2_y} L {EX - 17} {add2_y}" fill="none" '
             f'stroke="{GREEN_B}" stroke-width="2.4" marker-end="url(#aGreen)"/>')

    # =================== bottom: parameter ledger ===================
    ly = 904
    p.append(txt(CXL, ly, "参数账（Qwen3-8B，未绑定权重）", cls="h2", anchor="middle"))
    ledger = [
        ("Token Embedding", "622 M", EMB_B),
        ("36 × Decoder Block", "6.95 B", FFN_B),
        ("Final RMSNorm", "~0", NORM_B),
        ("lm_head", "622 M", FFN_B),
    ]
    lx0 = 70
    cw = 188
    gap = 26
    cx = lx0
    for i, (name, val, col) in enumerate(ledger):
        p.append(rrect(cx, ly + 22, cw, 70, "#ffffff", col, rx=10, sw=1.8))
        p.append(txt(cx + cw / 2, ly + 47, name, cls="sub"))
        p.append(txt(cx + cw / 2, ly + 76, val, cls="lbl"))
        if i < len(ledger) - 1:
            p.append(txt(cx + cw + gap / 2, ly + 62, "+", cls="title", extra=' font-size="22"'))
        cx += cw + gap
    p.append(txt(cx + 28, ly + 62, "=", cls="title", anchor="middle", extra=' font-size="22"'))
    p += chip(cx + 152, ly + 57, "≈ 8.2 B 参数", GREEN_F, GREEN_B, tcol=TXT, w=160)
    p.append(txt(lx0 + 4, ly + 116,
                 "一个 block 里 FFN 约占 3/4 参数（151M / 193M）；整模型里 36 层的 block 堆叠占了大头。",
                 cls="small", anchor="start"))

    write_svg(ASSETS / "panorama.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: Encoder-Decoder two-stack structure (deeper than ch2)
# ============================================================
def encoder_decoder():
    W, H = 1480, 960
    p = []
    p.append(txt(W / 2, 46, "Encoder-Decoder：原始 Transformer 的两栈结构（Vaswani 2017）", cls="title"))
    p.append(txt(W / 2, 74,
                 "三种注意力各司其职：encoder 双向自注意力 · decoder 掩码自注意力 · "
                 "cross-attention 把两栈接起来", cls="cap"))

    ENC, DEC = 340, 1000
    BW = 360

    def stage(cx, cy, h, label, sub, fill, border):
        out = [rrect(cx - BW / 2, cy - h / 2, BW, h, fill, border, rx=11, sw=2)]
        if sub:
            out.append(txt(cx, cy - 5, label, cls="lbl"))
            out.append(txt(cx, cy + 17, sub, cls="sub"))
        else:
            out.append(txt(cx, cy + 6, label, cls="lbl"))
        return out

    def updown(cx, y_low, y_high):
        return arrow(cx, y_low, cx, y_high, marker="aGray", color=GRAY_B)

    # ---------- encoder column (bottom -> top) ----------
    p += stage(ENC, 812, 50, "源序列  x₁ … xₙ", "（待翻译 / 待理解的输入）", TRM_F, TRM_B)
    p += stage(ENC, 720, 56, "Input Embedding + 位置编码", "[B, Lₛ, d]", EMB_F, EMB_B)
    # encoder block container
    ec_y, ec_h = 478, 226
    p.append(rrect(ENC - BW / 2 - 22, ec_y - ec_h / 2, BW + 44, ec_h,
                   "#f8fafc", GRAY_B, rx=15, sw=1.3, dash="7,5"))
    p.append(txt(ENC - BW / 2 - 34, ec_y, "N×", cls="title", anchor="end", extra=' font-size="20"'))
    p += stage(ENC, ec_y + 54, 62, "双向 Self-Attention", "每个 token 看全句（无掩码）", ATTN_F, ATTN_B)
    p += stage(ENC, ec_y - 54, 58, "Feed-Forward (FFN)", "Add & Norm 裹在每个子层外", FFN_F, FFN_B)
    p.append(updown(ENC, ec_y + 54 - 31, ec_y - 54 + 29))
    p += stage(ENC, 250, 56, "Encoder 输出（memory）", "[B, Lₛ, d] —— 提供 K, V", GREEN_F, GREEN_B)
    p.append(txt(ENC, 160, "encoder：把整句源文编码成一排上下文向量", cls="small"))
    # encoder up arrows
    p.append(updown(ENC, 812 - 25, 720 + 28))
    p.append(updown(ENC, 720 - 28, ec_y + ec_h / 2))
    p.append(updown(ENC, ec_y - ec_h / 2, 250 + 28))

    # ---------- decoder column (bottom -> top) ----------
    p += stage(DEC, 812, 50, "目标序列（右移）", "<bos> y₁ … y_{t-1}", TRM_F, TRM_B)
    p += stage(DEC, 720, 56, "Output Embedding + 位置编码", "[B, Lₜ, d]", EMB_F, EMB_B)
    dc_y, dc_h = 468, 326
    p.append(rrect(DEC - BW / 2 - 22, dc_y - dc_h / 2, BW + 44, dc_h,
                   "#f8fafc", GRAY_B, rx=15, sw=1.3, dash="7,5"))
    p.append(txt(DEC + BW / 2 + 34, dc_y, "N×", cls="title", anchor="start", extra=' font-size="20"'))
    p += stage(DEC, dc_y + 106, 60, "Masked Self-Attention", "因果掩码：只看已生成的左侧", ATTN_F, ATTN_B)
    p += stage(DEC, dc_y, 60, "Cross-Attention", "Q 来自 decoder，K/V 来自 encoder", RED_F, RED_B)
    p += stage(DEC, dc_y - 106, 58, "Feed-Forward (FFN)", "Add & Norm 裹在每个子层外", FFN_F, FFN_B)
    p.append(updown(DEC, dc_y + 106 - 30, dc_y + 30))
    p.append(updown(DEC, dc_y - 30, dc_y - 106 + 29))
    p += stage(DEC, 250, 50, "Linear", "d → 词表 V", FFN_F, FFN_B)
    p += stage(DEC, 168, 50, "Softmax → 输出概率", "预测下一个目标 token", SM_F, SM_B)
    p.append(updown(DEC, 812 - 25, 720 + 28))
    p.append(updown(DEC, 720 - 28, dc_y + dc_h / 2))
    p.append(updown(DEC, dc_y - dc_h / 2, 250 + 25))
    p.append(updown(DEC, 250 - 25, 168 + 25))

    # ---------- cross-attention wiring (encoder memory -> every decoder block) ----------
    sx = ENC + BW / 2
    ty = dc_y
    tx = DEC - BW / 2
    midx = (sx + tx) / 2
    p.append(f'<path d="M {sx} 250 C {midx} 250, {midx} {ty}, {tx - 4} {ty}" '
             f'fill="none" stroke="{RED_B}" stroke-width="2.6" stroke-dasharray="2,0" '
             f'marker-end="url(#aRed)"/>')
    p.append(txt(midx, 250 - 16, "K, V（来自 encoder）", cls="small",
                 extra=f' fill="{RED_B}" font-weight="600"'))
    p.append(txt(midx, ty + 62, "cross-attention：decoder 每个位置都能看 encoder 的全部输出",
                 cls="small", extra=f' fill="{RED_B}"'))

    # ---------- bottom legend: three attention types ----------
    ly = 880
    p.append(rrect(70, ly, W - 140, 56, "#fbfbfc", GRAY_B, rx=12, sw=1.2))
    items = [
        ("双向自注意力", "encoder 内，token 互看全句", ATTN_B),
        ("掩码自注意力", "decoder 内，只看左侧（因果）", ATTN_B),
        ("cross-attention", "decoder 看 encoder：Q↓ / K,V←", RED_B),
    ]
    seg = (W - 200) / 3
    for i, (h, d, col) in enumerate(items):
        x = 110 + i * seg
        p.append(f'<rect x="{x}" y="{ly + 18}" width="18" height="18" rx="4" '
                 f'fill="#ffffff" stroke="{col}" stroke-width="2.4"/>')
        p.append(txt(x + 28, ly + 26, h, cls="lbl", anchor="start"))
        p.append(txt(x + 28, ly + 46, d, cls="small", anchor="start"))

    write_svg(ASSETS / "encoder-decoder.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 3: Pre-LN vs Post-LN
# ============================================================
def pre_post_ln():
    W, H = 1420, 820
    p = []
    p.append(txt(W / 2, 46, "Pre-LN vs Post-LN：归一化放在子层「前」还是「后」", cls="title"))
    p.append(txt(W / 2, 74,
                 "差别只挪了一个 Norm，却决定了深层 Transformer 训不训得稳", cls="cap"))

    def flow_block(cx, title, tcol, steps, formula, notes, hi_highway):
        # steps: list of (label, sub, fill, border); drawn top->bottom
        out = []
        out.append(rrect(cx - 300, 96, 600, 640, "#fcfcfd", tcol, rx=16, sw=1.8))
        out.append(txt(cx, 130, title, cls="h2", extra=f' fill="{tcol}"'))
        cy = 196
        h = 56
        gap = 40
        centers = []
        for (lab, sub, f, b) in steps:
            if lab == "ADD":
                out.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#ffffff" '
                           f'stroke="{SUB}" stroke-width="2.2"/>')
                out.append(txt(cx, cy + 8, "+", cls="title", extra=' font-size="24"'))
                centers.append((cy, 20))
                cy += 2 * 20 + gap
            else:
                out.append(rrect(cx - 150, cy - h / 2, 300, h, f, b, rx=10, sw=2))
                out.append(txt(cx, cy - 4 if sub else cy + 6, lab, cls="lbl"))
                if sub:
                    out.append(txt(cx, cy + 17, sub, cls="small"))
                centers.append((cy, h / 2))
                cy += h + gap
        # main trunk arrows
        for i in range(len(centers) - 1):
            y1 = centers[i][0] + centers[i][1]
            y2 = centers[i + 1][0] - centers[i + 1][1]
            out.append(arrow(cx, y1, cx, y2 - 2, marker="aGray", color=GRAY_B))
        # residual highway (from input down to the ADD)
        add_idx = [i for i, s in enumerate(steps) if s[0] == "ADD"][0]
        top_y = centers[0][0]
        add_y = centers[add_idx][0]
        hx = cx - 230 if hi_highway else cx + 230
        col = GREEN_B if hi_highway else SUB
        wd = 3.2 if hi_highway else 2.2
        # 从「输入 x」框的侧边引出，绕过中间子层，箭头收进 ADD 的侧面
        start_x = cx - 150 if hi_highway else cx + 150
        end_x = cx - 20 if hi_highway else cx + 20
        out.append(f'<path d="M {start_x} {top_y} L {hx} {top_y} L {hx} {add_y} L {end_x} {add_y}" '
                   f'fill="none" stroke="{col}" stroke-width="{wd}" '
                   f'marker-end="url(#{"aGreen" if hi_highway else "aSlate"})"/>')
        lab = "干净的恒等\n残差高速公路" if hi_highway else "残差先汇合\n再被 Norm 重整"
        out.append(txt(hx + (-8 if hi_highway else 8), (top_y + add_y) / 2 - 8,
                       lab.split("\n")[0], cls="small",
                       anchor="end" if hi_highway else "start",
                       extra=f' fill="{col}" font-weight="600"'))
        out.append(txt(hx + (-8 if hi_highway else 8), (top_y + add_y) / 2 + 10,
                       lab.split("\n")[1], cls="small",
                       anchor="end" if hi_highway else "start",
                       extra=f' fill="{col}" font-weight="600"'))
        # formula + notes
        out.append(rrect(cx - 270, 612, 540, 40, "#ffffff", tcol, rx=8, sw=1.4))
        out.append(txt(cx, 638, formula, cls="monob"))
        ny = 676
        for n in notes:
            out.append(txt(cx, ny, n, cls="sub"))
            ny += 24
        return out

    # Post-LN (original)
    post_steps = [
        ("输入 x", "[B, L, d]", TRM_F, TRM_B),
        ("Sublayer", "Attention / FFN", ATTN_F, ATTN_B),
        ("ADD", "", "", ""),
        ("LayerNorm", "在主干上归一化", NORM_F, NORM_B),
        ("输出", "送入下一层", TRM_F, TRM_B),
    ]
    p += flow_block(360, "Post-LN（原版 2017）", ORANGE_B, post_steps,
                    "out = LayerNorm( x + Sublayer(x) )",
                    ["Norm 压在残差汇合之后的主干上，",
                     "残差幅度被反复重整 → 深层梯度不稳，",
                     "必须靠 warmup 小步起步、细心调。"],
                    hi_highway=False)

    # Pre-LN (modern)
    pre_steps = [
        ("输入 x", "[B, L, d]", TRM_F, TRM_B),
        ("LayerNorm / RMSNorm", "只归一化「读进子层的副本」", NORM_F, NORM_B),
        ("Sublayer", "Attention / FFN", ATTN_F, ATTN_B),
        ("ADD", "", "", ""),
        ("输出", "送入下一层", TRM_F, TRM_B),
    ]
    p += flow_block(1060, "Pre-LN（现代默认）", GREEN_B, pre_steps,
                    "out = x + Sublayer( Norm(x) )",
                    ["残差是一条没有 Norm 挡路的恒等高速公路，",
                     "梯度从顶层直达底层、幅度稳，",
                     "训练稳得多，warmup 可短可免。"],
                    hi_highway=True)

    # bottom takeaway
    p.append(txt(W / 2, 778,
                 "一句话：Pre-LN 把残差留成干净直通路，所以成了现代 LLM（GPT-2 起 / LLaMA / Qwen）的默认；"
                 "Post-LN 表达力略强但难训，要靠 DeepNorm、warmup 等手段稳住。",
                 cls="cap"))

    write_svg(ASSETS / "pre-post-ln.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: residual stream — the additive backbone bus
# ============================================================
def residual_stream():
    W, H = 1260, 1000
    p = []
    p.append(txt(W / 2, 46, "残差流（residual stream）视角：贯穿所有层的主干总线", cls="title"))
    p.append(txt(W / 2, 74,
                 "每个子层只「读取流的当前态 → 算一个增量 Δ → 加回去」，流本身从不被改写、只被累加",
                 cls="cap"))

    SX = 440          # stream x
    top_y, bot_y = 150, 902   # embedding 写流点下移，给最底层那个 ⊕ 让出空间
    # the thick stream
    p.append(f'<line x1="{SX}" y1="{bot_y}" x2="{SX}" y2="{top_y}" '
             f'stroke="{BLUE_B}" stroke-width="14" stroke-linecap="round" opacity="0.30"/>')
    p.append(f'<line x1="{SX}" y1="{bot_y}" x2="{SX}" y2="{top_y - 6}" '
             f'stroke="{BLUE_B}" stroke-width="2.5" marker-end="url(#aBlue)"/>')
    lbly = 300
    p.append(txt(SX - 28, lbly, "残差流  [B, L, d=4096]", cls="lbl",
                 anchor="middle", extra=f' transform="rotate(-90 {SX - 28} {lbly})" fill="{BLUE_B}"'))

    def writenode(cy, label, sub, fill, border, delta_lab):
        """A sublayer that branches off the stream, computes, and adds back."""
        bx = SX + 210          # sublayer box centre (well clear of the stream)
        bw, bh = 290, 62
        bl = bx - bw / 2
        out = []
        # read-off tap (from stream to box) — enters at cy-17
        out.append(arrow(SX + 8, cy - 17, bl - 2, cy - 17, marker="aGray", color=GRAY_B))
        # the sublayer box
        out.append(rrect(bl, cy - bh / 2, bw, bh, fill, border, rx=10, sw=2))
        out.append(txt(bx, cy - 4, label, cls="lbl"))
        out.append(txt(bx, cy + 16, sub, cls="small"))
        # add-back (box to stream ⊕) — returns at cy+17
        out.append(f'<circle cx="{SX}" cy="{cy + 17}" r="15" fill="#ffffff" stroke="{SUB}" stroke-width="2"/>')
        out.append(txt(SX, cy + 23, "+", cls="title", extra=' font-size="19"'))
        out.append(f'<path d="M {bl - 2} {cy + 17} L {SX + 15} {cy + 17}" '
                   f'fill="none" stroke="{GREEN_B}" stroke-width="2.4" marker-end="url(#aGreen)"/>')
        out.append(txt((bl + SX + 15) / 2, cy + 5, delta_lab, cls="small",
                       anchor="middle", extra=f' fill="{GREEN_B}" font-weight="600"'))
        return out

    # bottom: embedding writes initial state
    p.append(rrect(SX - 150, bot_y - 4, 300, 52, EMB_F, EMB_B, rx=10, sw=2))
    p.append(txt(SX, bot_y + 18, "Token Embedding", cls="lbl"))
    p.append(txt(SX, bot_y + 38, "把初始向量写上流", cls="small"))

    p.append(txt(SX + 150, 250, "↑ 每层把自己的 Δ 累加到流上，逐层精炼", cls="sub", anchor="middle"))
    # representative layers — within a layer attention is the lower (earlier) sublayer
    layers = [(770, "层 1"), (560, "层 2")]
    for base, name in layers:
        p.append(txt(SX - 168, base, name, cls="h2", anchor="end"))
        p += writenode(base + 45, "RMSNorm → Attention", "跨 token 混合信息", ATTN_F, ATTN_B, "Δ_attn")
        p += writenode(base - 45, "RMSNorm → FFN", "逐 token 非线性深加工", FFN_F, FFN_B, "Δ_ffn")
    # ellipsis for the remaining layers
    p.append(txt(SX, 458, "⋮", cls="title", extra=' font-size="34"'))
    p.append(txt(SX + 210, 458, "（第 3 … 36 层同理：读流 → 算 Δ → 加回）", cls="sub", anchor="middle"))

    # top: final norm + lm_head read the final state
    p.append(arrow(SX, 430, SX, top_y + 64, marker="aBlue", color=BLUE_B))
    p.append(rrect(SX - 175, top_y + 4, 350, 54, NORM_F, NORM_B, rx=10, sw=2))
    p.append(txt(SX, top_y + 26, "Final RMSNorm → lm_head", cls="lbl"))
    p.append(txt(SX, top_y + 46, "读取流的末态 → 词表 logits", cls="small"))

    # right-side explanatory panel
    px = 900
    p.append(rrect(px, 150, 330, 760, "#fbfbfc", PURPLE_B, rx=14, sw=1.5))
    p.append(txt(px + 165, 190, "为什么这个视角好用", cls="h2"))
    bullets = [
        ("流 = 模型的「工作内存」",
         ["所有层共享同一条 d 维总线，", "信息在上面一路传递、逐步精炼。"]),
        ("子层只做加法",
         ["Attention / FFN 算出的不是成品，", "而是一个增量 Δ，加到流上即可。"]),
        ("梯度直通",
         ["因为是「加」，反传时残差那条", "路导数恒为 1 —— 第 8 章的高速公路。"]),
        ("读写都过 Norm",
         ["Pre-LN 下，子层读流时先 Norm，", "但写回流的是未归一化的 Δ。"]),
        ("末端一次读出",
         ["Final Norm + lm_head 只在最后", "读流的末态，映射到词表。"]),
    ]
    by = 244
    for head, lines in bullets:
        p.append(f'<circle cx="{px + 24}" cy="{by - 4}" r="5" fill="{PURPLE_B}"/>')
        p.append(txt(px + 40, by, head, cls="lbl", anchor="start"))
        yy = by + 23
        for ln in lines:
            p.append(txt(px + 40, yy, ln, cls="small", anchor="start"))
            yy += 22
        by += 138

    write_svg(ASSETS / "residual-stream.svg", "\n".join(p), f"0 0 {W} {H}")


# ============================================================
# Diagram 5: language-model evolution timeline (n-gram -> Transformer -> LLM)
# ============================================================
def timeline():
    W, H = 1720, 600
    GRY_F, GRY_B = "#eef0f3", "#7c8896"
    p = []
    p.append(txt(W / 2, 44, "语言模型演变时间线：从 n-gram 一步步走到 Transformer，再到大模型", cls="title"))
    p.append(txt(W / 2, 72,
                 "每一步都在补上一步的短板——Transformer（2017）是这条线的集大成者，也是本章的主角",
                 cls="cap"))

    axis_y = 320
    x0, x1 = 80, 1648
    p.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" '
             f'stroke="{GRAY_B}" stroke-width="3" marker-end="url(#aGray)"/>')
    p.append(txt(x1 + 8, axis_y + 5, "时间", cls="small", anchor="start"))

    # (x, year, name, d1, d2, fill, border, above, highlight)
    ms = [
        (152, "1990s 前后", "n-gram 统计 LM", "数频次估概率、只看前几个词",
         "数据稀疏、几乎不懂语义", GRY_F, GRY_B, True, False),
        (354, "2003", "神经语言模型", "神经网络 + 词向量替代查表",
         "语义相近的词共享统计强度", GREEN_F, GREEN_B, False, False),
        (556, "2013", "word2vec", "词向量学得又快又好（第 4 章）",
         "但它是静态的、不随上下文变", GREEN_F, GREEN_B, True, False),
        (758, "2014–2015", "RNN seq2seq + attention", "循环网络把历史编进状态（第 5 章）",
         "attention 打破定长瓶颈", ORANGE_F, ORANGE_B, False, False),
        (960, "2017", "Transformer", "扔掉 RNN、纯 self-attention",
         "并行 + 任意两 token 长程直连", PINK_F, RED_B, True, True),
        (1162, "2018", "GPT / BERT", "Transformer + 大规模预训练",
         "开启 decoder / encoder 两条路线", BLUE_F, BLUE_B, False, False),
        (1364, "2020", "GPT-3", "175B 参数、纯 decoder-only",
         "in-context learning 涌现", BLUE_F, BLUE_B, True, False),
        (1566, "2023+", "LLaMA / Qwen …", "开源大模型成为主流",
         "现代配方的集大成者（正是本章）", PURPLE_F, PURPLE_B, False, False),
    ]
    cw, ch = 206, 128
    for (x, yr, name, d1, d2, f, b, above, hl) in ms:
        r = 11 if hl else 8
        if above:
            cy_top = axis_y - 44 - ch
            p.append(f'<line x1="{x}" y1="{cy_top + ch}" x2="{x}" y2="{axis_y - r}" '
                     f'stroke="{b}" stroke-width="2"/>')
        else:
            cy_top = axis_y + 44
            p.append(f'<line x1="{x}" y1="{axis_y + r}" x2="{x}" y2="{cy_top}" '
                     f'stroke="{b}" stroke-width="2"/>')
        if hl:
            p += chip(x, cy_top - 26, "本章主角", RED_F, RED_B, tcol=RED_B, w=88)
        p.append(rrect(x - cw / 2, cy_top, cw, ch, f, b, rx=12, sw=3 if hl else 2))
        yy = cy_top + 28
        p.append(txt(x, yy, yr, cls="lbl", extra=f' fill="{b}"'))
        p.append(txt(x, yy + 27, name + ("  ★" if hl else ""), cls="lbl"))
        p.append(txt(x, yy + 51, d1, cls="small"))
        p.append(txt(x, yy + 73, d2, cls="small"))
        # the axis dot drawn last so it sits above the connector
        p.append(f'<circle cx="{x}" cy="{axis_y}" r="{r}" fill="{b}" '
                 f'stroke="#ffffff" stroke-width="2.5"/>')

    p.append(txt(W / 2, H - 26,
                 "一条主线：n-gram 不懂语义 → 词向量给语义 → RNN 上下文化 → attention 破瓶颈 "
                 "→ Transformer 并行 + 长程直连 → 预训练 + 规模",
                 cls="cap"))
    write_svg(ASSETS / "timeline.svg", "\n".join(p), f"0 0 {W} {H}")


if __name__ == "__main__":
    panorama()
    encoder_decoder()
    pre_post_ln()
    residual_stream()
    timeline()
    print("all diagrams written")
