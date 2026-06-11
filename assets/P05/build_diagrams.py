"""Generate four conceptual diagrams for P05.

Style: Flat Icon (style 1) — white bg, soft fills, colored borders.
All matplotlib-style labels are kept ASCII to avoid CJK font issues
in the SVG → PNG pipeline; the chapter body explains the meaning.

Run from repo root:
    python3 assets/P05/build_diagrams.py
Then convert each generated SVG with rsvg-convert + pngquant, e.g.
    rsvg-convert -w 1440 assets/P05/v-q-tree.svg -o /tmp/x.png
    pngquant --quality=85-100 --strip --force --output assets/P05/v-q-tree.png /tmp/x.png
"""
from pathlib import Path

ASSETS = Path(__file__).parent

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif"

# ---------- shared palette (Flat Icon) ----------
BG = "#ffffff"
TXT = "#1e293b"
SUB = "#475569"
MUTED = "#94a3b8"

BLUE_F, BLUE_B = "#dbeafe", "#2563eb"
GREEN_F, GREEN_B = "#dcfce7", "#059669"
ORANGE_F, ORANGE_B = "#ffedd5", "#ea580c"
PURPLE_F, PURPLE_B = "#ede9fe", "#7c3aed"
RED_F, RED_B = "#fee2e2", "#dc2626"
GRAY_F, GRAY_B = "#f3f4f6", "#94a3b8"


def write_svg(path: Path, body: str, viewbox: str):
    style = f"""
  <style>
    text {{ font-family: {FONT}; }}
    .title {{ font-size: 22px; font-weight: 700; fill: {TXT}; }}
    .sub   {{ font-size: 13px; fill: {SUB}; }}
    .lbl   {{ font-size: 14px; font-weight: 600; fill: {TXT}; }}
    .small {{ font-size: 12px; fill: {SUB}; }}
    .formula {{ font-size: 14px; fill: {TXT}; font-style: italic; }}
    .muted {{ fill: {MUTED}; }}
  </style>
"""
    defs = """
  <defs>
    <marker id="aBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/>
    </marker>
    <marker id="aGray" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 9 3, 0 6" fill="#94a3b8"/>
    </marker>
    <marker id="aOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#ea580c"/>
    </marker>
    <marker id="aPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#7c3aed"/>
    </marker>
  </defs>
"""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">\n'
        f'  <rect width="100%" height="100%" fill="{BG}"/>\n'
        f'{defs}{style}{body}\n'
        f'</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


# ============================================================
# Diagram 1: V vs Q  ("分支树")
# ============================================================
def diagram_v_q_tree():
    parts = []
    W, H = 960, 560
    parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" class="title">V(s) vs Q(s,a)：两者都从 s 出发，差别在第一步动作怎么决定</text>')

    # ----- Left panel: V(s) -----
    def panel(cx, label_top, frame_color, mode):
        """mode = 'V' or 'Q'."""
        # frame
        parts.append(
            f'<rect x="{cx-220}" y="80" width="440" height="430" rx="14" '
            f'fill="#ffffff" stroke="{frame_color}" stroke-width="1.5" stroke-dasharray="6,4"/>'
        )
        parts.append(f'<text x="{cx}" y="110" text-anchor="middle" class="lbl" fill="{frame_color}">{label_top}</text>')

        # root state s
        sx, sy = cx, 150
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="26" fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="2"/>')
        parts.append(f'<text x="{sx}" y="{sy+5}" text-anchor="middle" class="lbl">s</text>')

        # three actions
        actions = [
            ("a₁", cx - 140, 250, "π(a₁|s)=0.5", "Q=8", GREEN_F, GREEN_B),
            ("a₂", cx,        250, "π(a₂|s)=0.3", "Q=5", GREEN_F, GREEN_B),
            ("a₃", cx + 140, 250, "π(a₃|s)=0.2", "Q=3", GREEN_F, GREEN_B),
        ]
        chosen_idx = 1 if mode == 'Q' else None  # Q panel: a2 锁定

        # Stagger weight labels vertically to avoid overlap
        # left branch label high, center middle, right low
        # All three labels sit in the upper half between s and the a-nodes,
        # offset to the appropriate side of their edge so they don't overlap.
        label_y_offsets = {0: -18, 1: -8, 2: -18}
        label_anchors = {0: "end", 1: "start", 2: "start"}
        label_dx = {0: -8, 1: 10, 2: 8}
        for i, (name, ax, ay, weight, qv, fill, bd) in enumerate(actions):
            faded = (mode == 'Q' and i != chosen_idx)
            line_color = MUTED if faded else SUB
            dash = ' stroke-dasharray="4,3"' if faded else ''
            # edge s -> a
            parts.append(
                f'<line x1="{sx}" y1="{sy+26}" x2="{ax}" y2="{ay-26}" '
                f'stroke="{line_color}" stroke-width="1.6"{dash}/>'
            )
            # weight label on edge — offset to side & stagger vertically
            mx = (sx + ax) / 2 + label_dx[i]
            my = (sy + ay) / 2 + label_y_offsets[i]
            wcolor = MUTED if faded else SUB
            parts.append(
                f'<text x="{mx}" y="{my}" text-anchor="{label_anchors[i]}" '
                f'class="small" fill="{wcolor}">{weight}</text>'
            )
            # node a
            node_fill = "#ffffff" if faded else fill
            node_border = MUTED if faded else bd
            parts.append(
                f'<circle cx="{ax}" cy="{ay}" r="26" fill="{node_fill}" '
                f'stroke="{node_border}" stroke-width="2"/>'
            )
            text_fill = MUTED if faded else TXT
            parts.append(f'<text x="{ax}" y="{ay+5}" text-anchor="middle" class="lbl" fill="{text_fill}">{name}</text>')

            # downstream "按 π 走" label
            yd = ay + 50
            txt = "按 π 继续走"
            sub_color = MUTED if faded else SUB
            parts.append(
                f'<line x1="{ax}" y1="{ay+26}" x2="{ax}" y2="{yd}" '
                f'stroke="{line_color}" stroke-width="1.4"{dash}/>'
            )
            parts.append(f'<text x="{ax}" y="{yd+14}" text-anchor="middle" class="small" fill="{sub_color}">{txt}</text>')
            # Q value
            parts.append(f'<text x="{ax}" y="{yd+32}" text-anchor="middle" class="small" fill="{sub_color}">{qv}</text>')

        # bottom formula box
        by = 450
        if mode == 'V':
            parts.append(
                f'<rect x="{cx-200}" y="{by-30}" width="400" height="48" rx="8" '
                f'fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{cx}" y="{by-8}" text-anchor="middle" class="formula">'
                f'V(s) = Σ π(a|s)·Q(s,a)</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{by+12}" text-anchor="middle" class="small">'
                f'= 0.5·8 + 0.3·5 + 0.2·3 = 6.1</text>'
            )
        else:
            parts.append(
                f'<rect x="{cx-200}" y="{by-30}" width="400" height="48" rx="8" '
                f'fill="{GREEN_F}" stroke="{GREEN_B}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{cx}" y="{by-8}" text-anchor="middle" class="formula">'
                f'Q(s,a₂) = 5</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{by+12}" text-anchor="middle" class="small">'
                f'第一步动作锁定为 a₂，之后仍按 π 行动</text>'
            )

    panel(240, "V(s)：第一步动作也由 π 决定", BLUE_B, 'V')
    panel(720, "Q(s, a₂)：第一步锁定 a₂，其余按 π", GREEN_B, 'Q')

    # bottom legend
    ly = 535
    parts.append(f'<text x="{W/2}" y="{ly}" text-anchor="middle" class="small">'
                 f'相同点：都是「从 s 起，按 π 行动」的累计折扣回报的期望　|　差别：V 对所有动作加权平均，Q 把第一步定住</text>')

    write_svg(ASSETS / "v-q-tree.svg", "\n".join(parts), f"0 0 {W} {H}")


# ============================================================
# Diagram 2: Advantage A = Q - V （零均值化柱状图）
# ============================================================
def diagram_advantage_bars():
    parts = []
    W, H = 760, 500
    parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" class="title">优势 A(s,a) = Q(s,a) − V(s)：按 π 加权后均值为 0</text>')

    # data: Q values and π weights, V = sum(pi*Q)
    actions = [
        ("a₁", 8.0, 0.40),
        ("a₂", 6.5, 0.30),
        ("a₃", 4.0, 0.20),
        ("a₄", 2.5, 0.10),
    ]
    V = sum(q * w for _, q, w in actions)  # = 8*0.4 + 6.5*0.3 + 4*0.2 + 2.5*0.1 = 3.2+1.95+0.8+0.25 = 6.2

    # plot box
    ox, oy = 110, 90        # origin (top-left of plot)
    pw, ph = 560, 320
    bottom = oy + ph
    parts.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="#fbfdff" stroke="{GRAY_B}" stroke-width="1"/>')

    qmax = 10.0
    def y_of(q):
        return bottom - (q / qmax) * ph

    # gridlines
    for q in range(0, 11, 2):
        y = y_of(q)
        parts.append(f'<line x1="{ox}" y1="{y}" x2="{ox+pw}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{ox-8}" y="{y+4}" text-anchor="end" class="small">{q}</text>')

    # bars
    n = len(actions)
    bar_w = 70
    slot = pw / n
    bar_positions = []
    for i, (name, q, w) in enumerate(actions):
        cx = ox + slot * (i + 0.5)
        bx = cx - bar_w / 2
        by_ = y_of(q)
        # Q bar
        parts.append(
            f'<rect x="{bx}" y="{by_}" width="{bar_w}" height="{bottom - by_}" '
            f'fill="{BLUE_F}" stroke="{BLUE_B}" stroke-width="1.5" rx="3"/>'
        )
        # Q value label on top
        parts.append(f'<text x="{cx}" y="{by_-8}" text-anchor="middle" class="small">Q={q:.1f}</text>')

        # advantage segment (Q − V): draw above or below the V line on its own
        a = q - V
        v_y = y_of(V)
        if a > 0:
            # upward arrow from V to Q
            parts.append(
                f'<line x1="{cx + bar_w/2 + 8}" y1="{v_y}" x2="{cx + bar_w/2 + 8}" y2="{by_+2}" '
                f'stroke="{GREEN_B}" stroke-width="2" marker-end="url(#aGreen)" />'
            )
        else:
            parts.append(
                f'<line x1="{cx + bar_w/2 + 8}" y1="{v_y}" x2="{cx + bar_w/2 + 8}" y2="{by_-2}" '
                f'stroke="{RED_B}" stroke-width="2" marker-end="url(#aRed)" />'
            )
        # A label
        sign = "+" if a > 0 else ""
        a_color = GREEN_B if a > 0 else RED_B
        # place A label to the right of the bar, midway between V and Q
        ay_label = (v_y + by_) / 2
        parts.append(
            f'<text x="{cx + bar_w/2 + 22}" y="{ay_label+4}" class="small" fill="{a_color}">'
            f'A={sign}{a:.1f}</text>'
        )

        # x-axis labels: action + weight
        parts.append(f'<text x="{cx}" y="{bottom+22}" text-anchor="middle" class="lbl">{name}</text>')
        parts.append(f'<text x="{cx}" y="{bottom+40}" text-anchor="middle" class="small">π={w:.2f}</text>')

        bar_positions.append((cx, by_, q, w, a))

    # V line
    v_y = y_of(V)
    parts.append(
        f'<line x1="{ox}" y1="{v_y}" x2="{ox+pw}" y2="{v_y}" '
        f'stroke="{ORANGE_B}" stroke-width="2" stroke-dasharray="6,4"/>'
    )
    parts.append(
        f'<text x="{ox+pw+6}" y="{v_y+4}" class="small" fill="{ORANGE_B}">V(s)={V:.1f}</text>'
    )

    # extra arrow markers (green / red)
    extra_defs = (
        '<marker id="aGreen" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">'
        f'<polygon points="0 0, 9 3, 0 6" fill="{GREEN_B}"/></marker>'
        '<marker id="aRed" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">'
        f'<polygon points="0 0, 9 3, 0 6" fill="{RED_B}"/></marker>'
    )

    # caption — weighted sum
    parts.append(
        f'<text x="{W/2}" y="465" text-anchor="middle" class="formula">'
        f'Σ π(a|s)·A(s,a) = 0.40·(+1.8) + 0.30·(+0.3) + 0.20·(−2.2) + 0.10·(−3.7) = 0</text>'
    )
    parts.append(
        f'<text x="{W/2}" y="485" text-anchor="middle" class="small">'
        f'高于平均的动作（绿）与低于平均的动作（红）按概率加权正好抵消</text>'
    )

    body = extra_defs + "\n" + "\n".join(parts)
    write_svg(ASSETS / "advantage-bars.svg", body, f"0 0 {W} {H}")


# ============================================================
# Diagram 3: Baseline variance reduction （散点对比）
# ============================================================
def diagram_baseline_variance():
    import math, random
    random.seed(7)

    parts = []
    W, H = 960, 520
    parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" class="title">Baseline 降方差：期望不变，但散点更窄、梯度更稳</text>')

    # Simulate state-dependent baseline: V(s_t) ≈ true mean return at that state.
    # G_t = V(s_t) + noise.  Subtracting V(s_t) removes the "between-states"
    # variance and leaves only the "within-state" noise.
    N = 30
    state_vals = [random.uniform(60, 320) for _ in range(N)]  # V(s_t)
    Gs = [v + random.gauss(0, 30) for v in state_vals]         # G_t = V + small noise
    advs = [g - v for g, v in zip(Gs, state_vals)]

    def panel(x0, y0, pw, ph, ys, ymin, ymax, title, fill, border, mean_color, mean_val):
        # title
        parts.append(f'<text x="{x0 + pw/2}" y="{y0-12}" text-anchor="middle" class="lbl" fill="{border}">{title}</text>')
        # box
        parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fbfdff" stroke="{GRAY_B}" stroke-width="1"/>')
        # y gridlines & ticks (5)
        step = (ymax - ymin) / 4
        for i in range(5):
            yv = ymin + step * i
            yp = y0 + ph - (yv - ymin) / (ymax - ymin) * ph
            parts.append(f'<line x1="{x0}" y1="{yp}" x2="{x0+pw}" y2="{yp}" stroke="#e5e7eb" stroke-width="1"/>')
            parts.append(f'<text x="{x0-6}" y="{yp+4}" text-anchor="end" class="small">{yv:.0f}</text>')
        # zero / mean line
        ym = y0 + ph - (mean_val - ymin) / (ymax - ymin) * ph
        parts.append(f'<line x1="{x0}" y1="{ym}" x2="{x0+pw}" y2="{ym}" stroke="{mean_color}" stroke-width="2" stroke-dasharray="6,4"/>')
        label = f"mean ≈ {mean_val:.0f}"
        parts.append(f'<text x="{x0+pw-6}" y="{ym-6}" text-anchor="end" class="small" fill="{mean_color}">{label}</text>')
        # scatter
        for i, y in enumerate(ys):
            xp = x0 + 12 + (pw - 24) * i / (N - 1)
            yp = y0 + ph - (y - ymin) / (ymax - ymin) * ph
            parts.append(f'<circle cx="{xp}" cy="{yp}" r="4.5" fill="{fill}" stroke="{border}" stroke-width="1.4"/>')
        # x label
        parts.append(f'<text x="{x0 + pw/2}" y="{y0+ph+22}" text-anchor="middle" class="small">episode index (1..30)</text>')

        # std annotation
        mu = sum(ys) / len(ys)
        std = math.sqrt(sum((y - mu)**2 for y in ys) / len(ys))
        parts.append(f'<text x="{x0+10}" y="{y0+18}" class="small" fill="{border}">σ ≈ {std:.0f}</text>')

    # Left panel: G_t
    panel(80, 90, 380, 320, Gs, 0, 400, "G_t（直接当权重）", BLUE_F, BLUE_B, ORANGE_B, sum(Gs)/N)
    # Right panel: G_t - V
    panel(540, 90, 380, 320, advs, -200, 200, "G_t − V(s_t)（减去状态相关 baseline）", GREEN_F, GREEN_B, ORANGE_B, 0)

    # bottom caption
    parts.append(
        f'<text x="{W/2}" y="450" text-anchor="middle" class="formula">'
        f'E[ ∇log π · (G − b) ] = E[ ∇log π · G ]　（b 不依赖 a → 期望不变）</text>'
    )
    parts.append(
        f'<text x="{W/2}" y="472" text-anchor="middle" class="small">'
        f'但 Var[ G − b ] 比 Var[ G ] 小很多，梯度估计噪声更低，收敛更稳。最优 b 是 V^π(s)（actor-critic 的来历）。</text>'
    )

    write_svg(ASSETS / "baseline-variance.svg", "\n".join(parts), f"0 0 {W} {H}")


# ============================================================
# Diagram 4: Policy Gradient training pipeline
# ============================================================
def diagram_pg_pipeline():
    parts = []
    W, H = 1200, 520
    parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" class="title">Policy Gradient 训练流水线：θ → rollout → 算梯度 → 更新 θ</text>')

    # node positions (cx, cy, w, h, label, sub, fill, border)
    nodes = [
        (110, 230, 130, 70, "参数 θ",          "策略网络 π_θ 的权重",       PURPLE_F, PURPLE_B),
        (300, 230, 150, 70, "采样 N 条 rollout", "用 π_θ 与环境交互",        BLUE_F,   BLUE_B),
        (500, 230, 160, 70, "算每步 G_t",       "G_t = r_t + γG_{t+1}",     BLUE_F,   BLUE_B),
        (710, 230, 170, 70, "算 ∇log π · G_t",  "对每条轨迹的每一步",       GREEN_F,  GREEN_B),
        (930, 230, 150, 70, "N 条平均",         "得到 ∇_θ J̃（MC 估计）",   GREEN_F,  GREEN_B),
        (1110,230, 70,  70, "θ ← θ + η·∇J̃",   "",                          ORANGE_F, ORANGE_B),
    ]
    # actually 1110 makes node go past 1200; recompute layout
    # let's use 6 columns evenly across width 1200 margins 60
    cols_x = [110, 290, 480, 680, 890, 1080]
    widths = [120, 140, 150, 170, 140, 120]
    labels = [
        ("参数 θ",          "策略网络 π_θ 的权重",     PURPLE_F, PURPLE_B),
        ("① 采样 N 条 rollout", "用 π_θ 与环境交互",      BLUE_F,   BLUE_B),
        ("② 算每步 G_t",       "G_t = r_t + γG_{t+1}",  BLUE_F,   BLUE_B),
        ("③ 算 ∇log π · G_t",  "对每条轨迹每一步",       GREEN_F,  GREEN_B),
        ("④ N 条平均",         "得 ∇_θ J̃（MC 估计）",  GREEN_F,  GREEN_B),
        ("⑤ θ ← θ + η·∇J̃",   "梯度上升一步",           ORANGE_F, ORANGE_B),
    ]
    parts_nodes = []
    centers = []
    for cx, w, (lbl, sub, fill, border) in zip(cols_x, widths, labels):
        cy = 230
        h = 80
        x = cx - w/2
        y = cy - h/2
        parts_nodes.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
            f'fill="{fill}" stroke="{border}" stroke-width="2"/>'
        )
        parts_nodes.append(f'<text x="{cx}" y="{cy-4}" text-anchor="middle" class="lbl">{lbl}</text>')
        if sub:
            parts_nodes.append(f'<text x="{cx}" y="{cy+18}" text-anchor="middle" class="small">{sub}</text>')
        centers.append((cx, cy, w, h))

    # arrows between adjacent nodes
    for i in range(len(centers) - 1):
        cx1, cy1, w1, _ = centers[i]
        cx2, cy2, w2, _ = centers[i+1]
        x1 = cx1 + w1/2
        x2 = cx2 - w2/2 - 4
        parts_nodes.append(
            f'<line x1="{x1}" y1="{cy1}" x2="{x2}" y2="{cy2}" '
            f'stroke="{BLUE_B}" stroke-width="2" marker-end="url(#aBlue)"/>'
        )

    # feedback loop arrow: from last node back to first node θ, arc above
    cx_last, cy_last, w_last, h_last = centers[-1]
    cx_first, cy_first, w_first, h_first = centers[0]
    sx = cx_last
    sy = cy_last - h_last/2
    ex = cx_first
    ey = cy_first - h_first/2
    midy = 110
    path = f'M {sx},{sy} C {sx},{midy} {ex},{midy} {ex},{ey}'
    parts_nodes.append(
        f'<path d="{path}" stroke="{ORANGE_B}" stroke-width="2" fill="none" '
        f'stroke-dasharray="6,4" marker-end="url(#aOrange)"/>'
    )
    parts_nodes.append(
        f'<text x="{(sx+ex)/2}" y="{midy-10}" text-anchor="middle" class="small" fill="{ORANGE_B}">'
        f'更新后的 θ 回到下一轮 rollout</text>'
    )

    # bottom annotation block — what's MC and what's autograd
    by = 350
    parts_nodes.append(
        f'<rect x="80" y="{by}" width="1040" height="120" rx="10" '
        f'fill="#f8fafc" stroke="{GRAY_B}" stroke-width="1" stroke-dasharray="5,3"/>'
    )
    parts_nodes.append(f'<text x="100" y="{by+24}" class="lbl">理论 ↔ 工程的对应</text>')
    parts_nodes.append(
        f'<text x="100" y="{by+48}" class="small">'
        f'• 理论目标：∇_θ J(θ) = 𝔼_{{τ~π_θ}}[ Σ_t ∇log π_θ(a_t|s_t) · G_t ]　——一个对 trajectory 分布的期望，没法直接算。</text>'
    )
    parts_nodes.append(
        f'<text x="100" y="{by+70}" class="small">'
        f'• 蒙特卡洛估计（步骤 ①②③④）：从 π_θ 采 N 条 trajectory，平均括号里的量，得到无偏估计 ∇_θ J̃。</text>'
    )
    parts_nodes.append(
        f'<text x="100" y="{by+92}" class="small">'
        f'• 梯度上升（步骤 ⑤）：θ ← θ + η · ∇_θ J̃。注意是 +，因为要"最大化" J；PyTorch 里把 loss 写成 −J 后照常 .backward()。</text>'
    )

    write_svg(ASSETS / "pg-pipeline.svg", "\n".join(parts_nodes), f"0 0 {W} {H}")


if __name__ == "__main__":
    diagram_v_q_tree()
    diagram_advantage_bars()
    diagram_baseline_variance()
    diagram_pg_pipeline()
    print("SVGs generated under", ASSETS)
