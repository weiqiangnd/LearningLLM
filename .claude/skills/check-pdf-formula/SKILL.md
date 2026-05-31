---
name: check-pdf-formula
description: >-
  For ONE chapter, pair every math formula's source TeX (from `src/<name>.md`)
  with how it ACTUALLY renders in the real PDF (`markdown-to-pdf`'s
  KaTeX→WeasyPrint output), so a multimodal reviewer can eyeball each
  formula for meaning-consistency AND rendering defects. Unlike
  check-github-render --visual (which re-renders each formula in isolation
  with MathJax to simulate github.com), this one CROPS each formula out of
  the actual chapter PDF by its WeasyPrint box coordinates — catching
  PDF-side breakage that isolated rendering can't see (≠ rlap split,
  subscripts visually dropped, display equations overflowing the page,
  font fallback). Emits per-formula PNG crops + a mapping.json + a
  contact-sheet (PDF for humans, PNG pages for the multimodal review step).
  MANUAL trigger only — run it when explicitly asked. Trigger on:
  "检查 pdf 公式" "pdf 公式渲染复核" "pdf 公式对照" "校验 pdf 里的公式"
  "check pdf formula" "verify pdf math rendering" "公式截图对照".
---

# Check PDF Formula Rendering（真实 PDF 公式对照复核）

针对**单独一章**的一个**渲染层复核**，回答的问题是：**这一章的每条数学公式，在我们最终导出的 PDF 里，渲染出来的样子跟源 md 里写的 TeX 含义一致吗？有没有渲染裂开？** 做法是把每条公式从**真实 PDF**（`markdown-to-pdf` 的 KaTeX→WeasyPrint 产物）里按坐标抠出来，和源 TeX 摆在一起，交给多模态 reviewer（就是你，Claude，用 Read 工具看图）逐条核对。

## 跟另外两条公式 skill 的分工

仓库里现在有三条碰公式的 skill，**互补不重复**：

| skill | 引擎 / 目标 | 看到的是什么 | 抓什么坑 |
|---|---|---|---|
| `markdown-to-pdf`（一致性校验） | KaTeX 文本层 | 三个表示层的 TeX 字符串比对 | 公式被吞/漏转义/KaTeX 报错（**不看视觉**） |
| `check-github-render --visual` | MathJax 3（模拟 github.com） | 每条公式**孤立重渲**的对照表 | GitHub 侧 MathJax 语法 / markdown 解析坑 |
| **`check-pdf-formula`（本条）** | **真实 PDF**（KaTeX→WeasyPrint） | 从**整章 PDF** 里按坐标**抠出**的实际渲染 | **WeasyPrint 侧才裂开的视觉坑**：≠ 的 `\rlap` 斜杠叠错、下标/上标视觉丢失、display 公式溢出页宽被截、字体回退异常、上下文里的换行 |

一句话：前两条比的是「TeX 字符串 / 孤立渲染」，本条比的是「公式在**最终 PDF 成品**里长什么样」——这是其它两条都看不到的最后一层。

## 输出契约

- 输入：一个章节标识——章节号（`04`、`P03`）、文件名（`04-Embedding与位置编码`）或路径（`src/04-….md`）。**一次只处理一章**。
- 渲染来源（**绝不覆盖 `dist/` 的章节三件套**）：
  - **默认**：用当前 `src/<stem>.md` 调 `markdown-to-pdf` 的 `md_to_dist(src, out_dir)` **重渲一份到复核目录** `dist/pdf-formula-check/<stem>/`，结果总是反映最新 src。`dist/<stem>.{md,html,pdf}` 一个字节都不会动。
  - **`--no-render`**：跳过重渲，**只读复用**现成的 `dist/<stem>.html`（快；但若 dist 比 src 旧，复核的就是旧渲染）。该路径同样不写任何东西进 `dist/`。缺 `dist/<stem>.html` 时报错。
- 输出（默认落在 `dist/pdf-formula-check/<stem>/`，已 gitignore）：
  - **`mapping.json`** —— 机器可读映射。每条公式一项：`index` / `line`（源 md 行号）/ `display`（是否 `$$` 块级）/ `tex`（源 md 原始 TeX，含作者写的 `\_` 转义）/ `tex_rendered`（KaTeX 实际收到的反转义 TeX）/ `pdf_page` / `bbox_csspx` / `png`（裁剪图相对路径）。
  - **`crops/NNNN.png`** —— 每条公式从真实 PDF 里单独抠出的截图（auto-trim 裁紧白边）。需要放大看某一条时单独 Read 它。
  - **`rendered.pdf`** —— 本次渲染出的整章 PDF（坐标与裁剪同源，便于核对/排查）。
  - **`contact-sheet*.pdf`** —— 对照表，给**人眼**看：左栏「源行号 + 源 TeX（等宽）」，右栏「真实 PDF 截图」；inline 三栏排，display 跨整行（源在上、渲染在下，不被栏宽切）。按 `--chunk`（默认 60 条/份）分片。
  - **`contact-sheet*-pNN.png`** —— 上面每张 PDF 逐页栅格化的 PNG，给**多模态复核**用。**复核统一读这些 PNG**：Read 对 PNG 的支持不依赖 `poppler/pdftoppm`，比直接 Read PDF 稳。

## 用法

```bash
# 1. 首次使用先装依赖（幂等；会先把 markdown-to-pdf 的依赖装齐，再补 PyMuPDF/Pillow）
bash .claude/skills/check-pdf-formula/scripts/install.sh

# 2. 跑一章（默认用最新 src 重渲到复核目录，不动 dist 主文件）
python3 .claude/skills/check-pdf-formula/scripts/check_pdf_formula.py 04
python3 .claude/skills/check-pdf-formula/scripts/check_pdf_formula.py P03

# 3. 确信 dist 已是最新、想省掉重渲（只读复用 dist/<stem>.html）
python3 .claude/skills/check-pdf-formula/scripts/check_pdf_formula.py 04 --no-render

# 4. 自定义每份对照表的公式数 / 输出目录
python3 .claude/skills/check-pdf-formula/scripts/check_pdf_formula.py 04 --chunk 40 --out-dir /tmp/check04
```

跑完后脚本会打印所有 `contact-sheet-*-pNN.png` 的路径，并提示下一步。

## 复核流程（脚本产出后，Claude 要做的事）

脚本本身**只产出对照素材**，真正的「多模态视觉复核」由你（Claude）执行：

1. 按脚本打印的清单，用 **Read 工具逐张打开 `contact-sheet-*-pNN.png`**（每张 ≤ 一页 A4，一次一张）。
2. 对每一行，比对**左栏源 TeX** 与**右栏真实渲染**，重点看两类问题：
   - **含义一致性**：渲染出来的式子和源 TeX 表达的是不是同一个数学对象（下标/上标/分式/求和上下限/向量加粗/转置符号有没有错位、丢失、串行）。
   - **渲染缺陷**（WeasyPrint 侧特有）：`≠` 是否裂成斜杠叠 `=` 错位、下标整体没渲出来、display 公式右侧是否被页边截断、符号叠加（`\rlap` / `\overline` / 根号）是否漏出、字体回退导致的缺字豆腐。
3. 把有问题的公式按 `#编号 / 源行号` 列出来，附一句「源 TeX 是 X，但 PDF 里渲成了 Y」，必要时回到 `mapping.json` 拿 `line` 定位源 md，或单独 Read `crops/NNNN.png` 放大确认。
4. 全部正常时明确说一句「N 条公式逐条核对无渲染异常」。

> 提示：若某章公式特别多（对照表分了好几片），逐片 Read 完再汇总，不要只看第一片就下结论。

## 实现要点（出问题时排查）

- **为什么能精确定位每条公式**：用 `markdown-to-pdf` 同一套 HTML 喂给 `weasyprint.HTML(...).render()`，遍历**盒模型**（`page._page_box` 的后代），取所有 `class` 含 `katex` token 的盒子（`.katex-display` / `.katex-html` / `.katex-mathml` 是别的 token，不会命中）。盒子给出页码 + CSS px 包围盒。
- **分页/折行拆片的归并**（曾踩坑，P04 复现）：一条 `.katex` 公式若正好落在 WeasyPrint 的**分页或软换行**处，会被拆成**多个盒片段**——直接按盒计数会比"源 md 公式数 / annotation 数"多出来、三路对不齐（P04 实测多 3 个）。所以脚本**按元素身份 `id(element)` 归并**：同一公式的所有片段算一条，裁图取**面积最大的主片段**（跨页两片拼不成单矩形；inline 主片通常含绝大部分内容）。被拆片的公式在 `mapping.json` 标 `fragmented:true`、对照表打「跨页·图可能不全」橙标，提示去 `rendered.pdf` 看完整渲染。
- **为什么裁的是「新渲的 PDF」而不是 `dist/<stem>.pdf`**：盒模型坐标和 PDF 必须来自**同一次** `render()` 才能保证逐 px 对齐——所以脚本对要复核的那份 HTML 调 `weasyprint.HTML(...).render()` 拿盒模型、再 `doc.write_pdf()` 现写一份 `rendered.pdf` 去裁它（写到复核目录）。这份 PDF 与 `dist/<stem>.pdf` 同源同 HTML、视觉等价，可放心当「真实 PDF」。**全过程不写 `dist/` 主文件**：默认模式下连那份 HTML 都是 `md_to_dist(src, 复核目录)` 新渲到复核目录的，`--no-render` 模式则只读 `dist/<stem>.html`。
- **坐标换算**：WeasyPrint 盒模型是 CSS px（96 dpi），PDF 是 pt（72 dpi），换算系数 `72/96 = 0.75`。裁剪用 PyMuPDF `get_pixmap(clip=…)`，`CROP_DPI=300` 保证文字锐利。
- **auto-trim**：display 公式因 KaTeX 把内层 `.katex` 设成块级居中，包围盒是整行宽、两侧大片留白。裁剪后用 Pillow 按墨迹 `getbbox()` 裁紧，保留 `TRIM_MARGIN` 的白边。
- **纵向溢出兜底「触边检测 + 有界扩展 + 墨迹带切」**（曾踩坑：P04 表格内行内 `\dfrac` 的分母被切，crop 看着像渲染坏，引起复核误判）：WeasyPrint 给行内 `.katex` 报的盒高有时只是**行高**，而 `\dfrac` / 大根号 / 高上下标的墨迹**溢出**了行盒，只按盒裁会切掉分子/分母。脚本先按行盒（+`BOX_PAD_PX`）渲一次，检测墨迹是否触到上/下边；触边就朝该方向**有界扩展约一个行高**（够补全 displaystyle 分式，又够不到表格同列上下相邻的公式）重渲，再用 `_ink_row_bands` 把墨迹按行切带、只保留**覆盖公式纵向中心那一带**（排除扩展窗里混进来的表格行边框 / 邻行残墨，`BAND_GAP_CSS` 控制断带阈值），最后才 auto-trim。**没触边的普通公式走不进这条分支，裁图与改动前逐像素一致——零回归**；做过扩展的公式在 `mapping.json` 标 `expanded:true`。注意这一路是**纯像素分析**，不依赖 KaTeX 的 `pstrut` 子树并集（实测被 strut 撑得忽大忽小、无法区分 `\dfrac` 与普通 `\frac`，不可靠）。
- **contact-sheet 栅格化 DPI = 200**（`SHEET_PNG_DPI`）：曾踩坑——150 dpi 下二级下标 scriptscriptstyle 的小字（如 `\pi_{\theta_\text{old}}` 的 `old`）缩到列宽后会被压糊成形似 `,,` 的假象、引起复核误判。提到 200 后这类小字清晰可读。单条 crop 本身仍是 300 dpi，最清楚的复核办法是单独 Read `crops/NNNN.png`。
- **三路对齐校验**：`源 md 公式数 == HTML <annotation> 数 == WeasyPrint .katex 盒数`，三者必须相等，否则判定 HTML 过期/抽取错位并报错（多见于 `--no-render` 复用了旧 dist html；去掉它让脚本用最新 src 重渲即可）。源 md 抽取**复用 `check-github-render` 的 `extract_math_with_positions`**（保持抽取正则全仓一致，并直接拿到源行号）。
- **源 TeX 两份**：`tex` 是源 md 里作者写的原文（带 `\_` 等 markdown 级转义）；`tex_rendered` 是 KaTeX 实际收到的（反转义后），即 HTML `<annotation>` 里的回声。对照表展示 `tex`（作者视角），排查反转义问题时看 `tex_rendered`。

## 已知限制

- **依赖 `markdown-to-pdf` 与 `check-github-render` 两条 skill 共存**：`install.sh` 复用前者的依赖与渲染脚本，运行时 import 后者的抽取器。单独搬走本目录会缺依赖。
- **只复核「视觉渲染」，不复核数学正确性本身**：脚本判断不了「这条公式的数学推导对不对」，只把源 TeX 与渲染摆在一起给你看;含义层判断仍靠多模态 reviewer。
- **盒模型是 WeasyPrint 私有 API**（`page._page_box` / `box.position_x`）：跨大版本可能变动。当前固定在 `install.sh` 装的 WeasyPrint 版本上验证过；升级后若定位异常，先核对盒模型遍历那段。
- **跨页/折行被拆片的公式只裁主片段**：这类公式现在能被正确**计数**（按元素身份归并，三路对齐不再误报），但裁图只取面积最大的主片段——跨页的另一半截在别的页上、拼不进单张图。脚本会把它标 `fragmented` 并在对照表打橙标提示，复核这几条时去 `rendered.pdf` 看完整渲染即可。
- **纵向有界扩展的极端边界**：触边扩展每侧上限约**一个行高**（足够 displaystyle 分式 / 单层根号）。若真有**叠得极高**的行内构造（如行内嵌套双层分式、`\sum` 带巨大上下限挤在正文行里）溢出超过一个行高，crop 仍可能差一点没补全——但这类写法本身就是排版反例，正文里几乎不会出现；真遇到时仍可去 `rendered.pdf` 看全貌。带选阈值 `BAND_GAP_CSS` 是按"分式内部缝隙 ~3-4px ＜ 相邻行间距 ~13px"定的经验值，极端紧排的表格若行距小于这个量级，理论上仍可能把相邻行并进同一带（实测本仓库各章都正常）。
