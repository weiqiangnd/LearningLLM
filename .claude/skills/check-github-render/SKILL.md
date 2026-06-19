---
name: check-github-render
description: >-
  Verify that math formulas in source markdown (`src/<name>.md`) will render
  correctly on **github.com**. Replicates GitHub's two-pass pipeline
  (CommonMark backslash-unescape inside math regions, then MathJax 3) to
  catch failures the local `markdown-to-pdf` pipeline can't see — GitHub
  uses MathJax while our PDF uses KaTeX, and the two engines have subtly
  different leniency. Also folds in a few non-formula markdown pitfalls
  (bold flanking against full-width punctuation, stray `~` strikethrough,
  fenced code blocks indented inside list items). Pass `--visual` to
  additionally generate a contact-
  sheet PDF showing every formula's actual MathJax 3 rendering side-by-
  side with its source TeX — let a multimodal reviewer eyeball it instead
  of manually checking github.com. Trigger on: "check github render"
  "检查 github 渲染" "校验 github 数学公式" "github mathjax check"
  "verify github math" "github 渲染预览" "看看 github 怎么渲染".
---

# Check GitHub Math Rendering

针对 `src/*.md` 的一个**渲染层校验**，回答一个问题：**这份 md 推到 github.com 上之后能正确渲染么？** 主体是数学公式（MathJax 3 那套），外加 CLAUDE.md 记录的几条**非公式 markdown 坑**（粗体 flanking、裸 `~` 删除线、list 内三反引号围栏）——这些与数学无关、却同样把 GitHub 渲染搞坏，本仓库又没别的工具自动跑它们，所以一并收进来。

跟 `markdown-to-pdf` skill 的校验**不重复**——那条管线用 KaTeX、保证 PDF 三件套对；这条用 MathJax 3、模拟 github.com 的实际渲染路径。两个引擎在「看到的输入」「能接受的语法」上都有差异，所以两套校验**互补不替代**。

## 输出契约

- 输入：`src/<name>.md` 一份或多份。
- 输出：
  - **stdout**：每个 issue 一行，格式 `文件:行号 [类型] 消息` + 一行 `source: <TeX>` 紧跟。全部通过时 exit code 0，发现任何 issue exit code 1。
  - **`--visual` 模式**（可选）：除上面文字报告之外，每个输入还生成一份 **contact-sheet PDF**（默认 `dist/check-github-render/<stem>.pdf`），列出每段数学公式的「行号 / 源 TeX / MathJax 3 实际渲染」三栏。给多模态 reviewer 看一眼或自己点开核对，**避免人肉去 github.com 复查**。Display 公式 (`$$...$$`) 自动跨列、源在上、渲染在下，避免长公式被栏宽切掉。

## 用法

```bash
# 1. 首次使用先装依赖（幂等，已装会跳过）
bash .claude/skills/check-github-render/scripts/install.sh

# 2. 文字校验一篇章节（快，1-3 秒）
python3 .claude/skills/check-github-render/scripts/check.py src/P05-强化学习够用版.md

# 3. 文字校验 + 视觉 PDF（多花约 5-10 秒，多 1 个 PDF）
python3 .claude/skills/check-github-render/scripts/check.py src/P05-强化学习够用版.md --visual

# 4. 批量
python3 .claude/skills/check-github-render/scripts/check.py src/*.md --visual

# 自定义 PDF 输出目录
python3 .claude/skills/check-github-render/scripts/check.py src/P05-*.md --visual --preview-dir /tmp/preview
```

## 模拟 GitHub 渲染的两段管线

github.com 把 md 里的数学公式渲染出来，要经过两道处理：

```
源 md  ──(CommonMark 解析 + 抽数学)──>  数学内容  ──(CommonMark 反转义)──>  MathJax  ──>  视觉
                                                       │
                                       这一步把 \_ → _、\$ → $、\* → *
                                       —— 还把 \, → , 、\! → ! 等吃掉
```

这条 skill 走五段（前四段管公式，第五段管非公式的 markdown 坑）：

1. **抽数学区段**：用与 `markdown-to-pdf` 一致的正则（已经验证过对 src/*.md 全集 0 误抓），按文档顺序拿到所有 `$...$` / `$$...$$`，并记下每段在源文件里的 byte offset → 行号。
2. **模拟反转义**：对每段数学内容，**只反转义 GitHub 实际在 `$...$` 内会还原的那些**（`\_` `\*` `\$` —— 实测 github.com 上的工作流），保留 `\\` `\{` `\}` 这类 TeX 命令字符。
3. **MathJax 渲染**：把反转义后的 TeX 喂给 mathjax-full（服务端，Node + liteAdaptor），逐条检查输出里是不是含 `<merror>` 元素或 `data-mjx-error` 属性。
4. **静态层补一脚**：MathJax 3 看不到下面这些"在 markdown 解析阶段就坏了"的形态——它们到 MathJax 时要么已经被还原成裸标点（`\,` → `,`、`\|` → `|`），要么整个数学段根本没被识别（CJK 紧贴、表格列分隔吃 `|` 等），MathJax 自然无法报错。脚本对源 md 直接 grep 这些模式补报：
   - `\,` `\!` `\;` `\>` `\|` 这五个 CommonMark 会无声吃掉的反斜杠转义
   - `\\` 行分隔符（matrix / cases / aligned 的换行）—— GitHub 把 `$$` 内的 `\\` 反转义成单个 `\`，整块塌成一行（实测 `pmatrix` 回来只剩一个 `<mtr>`、两行被并进相邻单元格）。报 `backslash-rowbreak-eaten`，提示改用 `\cr`（`\\\\` 能救 GitHub 但会让 KaTeX/PDF 多出一空行；`\cr` 两边都对）
   - **多行环境放进了行内 `$...$`**（`pmatrix` / `cases` / `aligned` / `array`，或裸 `\cr` / `\\` 行分隔符）——GitHub 行内数学是 inline/text style，多行矩阵在里头会塌行、错位或干脆不堆叠；**同样的 TeX 放进 `$$` 块级就正常**。报 `multirow-in-inline-math`，提示挪进独占一行的 `$$` 块。**只查行内 `$...$`，`$$` 块级豁免**。注意服务端 contact-sheet 强制 display style 渲染，**看不出**这个坑（页面上是对的、GitHub 上却塌），所以专门加这条静态规则补。
   - `$$...$$` 块级公式没独占一行 / 缺前后空行
   - CJK 字符或全角标点紧贴 `$`（如 `中文$x$中文`、`$y$。`）——CJK 集合除了汉字 / 全角标点，**也含 General Punctuation 区里当中文标点用的破折号 `——`、省略号 `…`、弯引号 `“” ‘’`**（`$\exp(x)$——把` 这种照样报），但**不含**当范围号用的 en dash `–`。
   - 表格行（`| ... |`）内数学段里出现裸 `|`
   - `_{<...}` / `^{<...}` 这种下标里塞了 `<` `>` 字面字符的写法
   - `}_<char>` 高危下标（`_` 未转义）的两种形态：(a) 单段 inline math 同时含 `[` 与 `}_<char>`；(b) 同一行 ≥2 段独立 inline `$...$` 各带一个 `}_<char>`、跨段 emphasis 配对（无需方括号）
   - **emphasis 吃掉行内 math（`emphasis-eats-math`，render-based）**：上面 (a)/(b) 只建模 `}_`-型开启符配 `}_`-型 bait，漏掉**闭合侧是 right-flanking `_` 但前面不是 `}`** 的情形——典型如 `$\text{head}_h$ … $d_{\text{model}}$`，`}_h` 开启 emphasis、`d_{` 的 `_`（前接字母、后接 `{`）闭合，把两段 `$…$` 一起吃掉（GitHub 渲成 `head<em>h$ … $d</em>{model}`）。这条不靠静态正则，而是把整行**渲染两遍**——原样一遍、把每个 `$…$` 替换成中性占位符再渲一遍（模拟 GitHub 对 math 的原子抽取），**只有原样渲染多出 emphasis 标签时才报**，从而把这个 bug 跟极常见、完全正常的 `**含 $math$ 的加粗**`（emphasis 来自字面 `**`、两遍都在）区分开。仅在**一行 ≥2 段 inline math**（真正会跨段配对的场景）时启用；单段交给 GitHub 原子保护、且场景 (a) 已被 11a 覆盖
   - 非规范不等号 `\char` / `\not=` / 字面 `≠` / `\unicode{x2260}`——报 `ne-non-canonical`。本仓库的源码规范是**一律写 `\ne`**：GitHub MathJax 直接渲染 `\ne` 没问题；`markdown-to-pdf` 在送进 KaTeX 之前自动把 `\ne`/`\neq` 改写成 `\mathrel{\char"2260}`，绕开 KaTeX 那个会让 WeasyPrint 裂开的 rlap 斜杠叠加，从而拿到一个干净的 ≠ 字形。这条规则**反过来**——它揪的是源码里**不该出现**的几种非规范写法：`\char` 在 MathJax 上不支持（GitHub 渲成红字 `\char"2260`）；`\not=` PDF 改写器不覆盖（KaTeX 的 rlap 照样裂）；字面 `≠` 和 `\unicode{x2260}` 在 KaTeX 里 "No character metrics"（PDF 里直接看不见）。提示统一回 `\ne`。**收进这条 GitHub-render skill 而非塞进 markdown-to-pdf，是因为本仓库需要一个「提交前一次性扫公式坑」的统一入口**。
5. **非公式 markdown 层**：CLAUDE.md「非公式（markdown 层）的 GitHub 渲染避坑」一节那三条坑，与数学无关、但同样把 GitHub 渲染搞坏，而且本仓库没有别的工具自动跑它们，所以一并折进来（实现对齐 CLAUDE.md 里那几段手动 grep/python 片段，跳过代码块）：
   - **粗体 flanking 失效**：`**bold**` 的闭合 `**` 卡在全角标点 + CJK 之间会漏出字面 `**`。用 markdown-it-py 的 CommonMark 把整行渲成 inline，看 `<code>` 之外是否残留 `**` → 报 `emphasis-flanking`。
   - **裸 `~` 删除线误配对**：同一作用域（表格行收窄到单元格、否则整行）≥2 个裸 `~` → 报 `stray-tilde`；先剔除真正的 `~~删除线~~`，孤立单个 `~`（如 `~16 GB`）放行。
   - **list 里直接套三反引号围栏**：带前导空格的 ```` ``` ```` 围栏（list 内嵌的典型形态）→ 报 `fenced-code-in-list`；列 0 围栏和推荐的 6 空格缩进 code block 都不报。

## 这条 skill 能抓到的、抓不到的

| CLAUDE.md 规则 | 抓得到？ | 怎么抓的 |
|---|---|---|
| 1 `$$` 块级独占一行 + 前后空行 | ✅ | 静态层扫源 md，发现 `$$...$$` 同行有其他文字、或前后非空行就报 `block-math-not-isolated` |
| 2 `$..$` 与 CJK 字符/全角标点邻接 | ✅ | 静态层扫每段 inline math 外侧字符，CJK / 全角标点贴边就报 `cjk-adjacent-to-math` |
| 3 `\left\{ ... \right\}` | ❌（无需抓） | 实测 MathJax 3 已经支持（早期 MathJax 2 才报错）。CLAUDE.md 那条**过时了**——可以放心写 `\left\{`。 |
| 4 `\text{}` 内裸 `_` | ✅ | MathJax 3 报 `'_' allowed only in math mode` |
| 5 `{+1.23}` 包数字 | ❌ | TeX 语义层（两个引擎一致），渲不出错；只是视觉上前后多一截空白 |
| 6 `\,` `\!` `\;` `\>` 间距宏 | ✅ | 静态层 grep |
| 7 `\|` 应改 `\Vert` | ✅ | 同上静态层，并入 `commonmark-eats-escape`——`\|` 在 GitHub 上渲染成单竖线，与 KaTeX 管线不一致 |
| 8 `[ ]` 内侧加 `\thinspace` | ❌ | 纯视觉风格选择，不会渲染失败 |
| 9 表格单元格内裸 `\|` | ✅ | 静态层识别 `\| ... \|` 表格行，扫该行的 `$...$` 内是否含未转义 `\|`，命中报 `pipe-in-table-math` |
| 10 下标里 `<` / `>` | ✅ | 静态层在每段抽出的 TeX 里搜 `[_^]\s*\{[^}]*[<>]`，命中报 `angle-in-subscript` |
| 11 `}_<char>` 下标被 emphasis 吃掉 | ✅ | 静态层两路报 `emphasis-eats-subscript`：(a) 单段 inline math 同时含 `[` 与 `}_<char>`；(b) 同一行 ≥2 段独立 inline `$...$` 各带一个 `}_<char>`（跨段配对、无需方括号）。`}\_V` 的修正形态不报 |
| 11′ emphasis 吃掉行内 math（闭合侧非 `}_`） | ✅ | 上面 (a)/(b) 的盲区补充：闭合 `_` 是 right-flanking 但前面不是 `}`（如 `$\text{head}_h$ … $d_{\text{model}}$` 里的 `d_`）时，render-based 的 `emphasis-eats-math` 抓——原样 vs math-masked 各渲一遍，原样多出 emphasis 标签才报；仅对 ≥2 段 inline math 的行启用，正常的 `**含 $math$ 的加粗**` 不误报 |
| 非规范不等号 ≠ 写法 | ✅ | 源码规范是 `\ne`（markdown-to-pdf 内部自动改写为 `\mathrel{\char"2260}`，GitHub 直接渲 `\ne`）。这条静态层扫的是源码里不该出现的非规范写法：`\char`（MathJax 不支持，GitHub 红字）、`\not=`（PDF 改写器不覆盖、KaTeX rlap 仍裂）、字面 `≠` 和 `\unicode{x2260}`（KaTeX 无 metrics、PDF 看不见）。命中报 `ne-non-canonical`，提示改写成 `\ne` |
| 多行矩阵 / cases 放进行内 `$...$` | ✅ | 静态层扫行内 `$...$` 内是否含 `\begin{pmatrix\|cases\|aligned\|array}` 或裸 `\cr` / `\\`，命中报 `multirow-in-inline-math`，提示挪进 `$$` 块。GitHub 行内 inline style 下多行矩阵会塌行 / 错位，而服务端 contact-sheet 强制 display style **看不出**这个坑，故专设静态规则；`$$` 块级豁免 |
| 任意 TeX 语法错（`Misplaced &`、`Missing argument for \frac` 等） | ✅ | MathJax 自报 |
| `\mathcal{L}\_V` 这种 workaround 写法 | ✅ 不报（这是正常写法） | 反转义后 `\mathcal{L}_V`，渲染正常 |

CLAUDE.md「**非公式（markdown 层）**的 GitHub 渲染避坑」一节（与数学无关，第五段管）：

| CLAUDE.md 规则 | 抓得到？ | 怎么抓的 |
|---|---|---|
| 粗体/斜体紧贴全角标点 flanking 失效 | ✅（仅 `**`） | markdown-it-py CommonMark 渲整行 inline，`<code>` 外残留 `**` 就报 `emphasis-flanking`。**只查 `**`（粗体）**，与 CLAUDE.md 那段排查 python 一致；裸 `*` 斜体误报率高，未纳入 |
| 同一作用域 ≥2 个裸 `~`（删除线误配对） | ✅ | 剔除 `~~..~~` 后，表格行按单元格、否则按整行数裸 `~`，≥2 报 `stray-tilde`；孤立单 `~` 放行 |
| list 里直接套三反引号围栏 | ✅（启发式） | 带前导空格的 ```` ``` ```` 围栏 opener 报 `fenced-code-in-list`；列 0 围栏与 6 空格缩进 code block 不报 |

简单说：CLAUDE.md「公式的 GitHub 渲染避坑」一节里 1/2/4/6/7/9/10/11 全部静态可查（部分靠 MathJax 自报，部分靠脚本 grep 源 md），剩下 5/8 是视觉风格选择（不破渲染、两个引擎一致）、3 已过时；「非公式」一节那三条也已并入第五段。**理论上推一次 commit 之前跑一次 `check.py src/*.md` 应该够覆盖公式 + 非公式两类坑**——再加上 `--visual` 出一份 contact-sheet 给多模态 reviewer 复核，可以彻底不用去 github.com 肉眼检查。

> `emphasis-flanking` 检查需要 `markdown-it-py`（`install.sh` 会装）。环境里没有时该项**自动跳过**并在 stderr 打一行 `[warn]`，其余检查照常跑。

## `--visual` 模式做了什么

文字校验通过后，contact-sheet PDF 的生成大致是：

1. 每段数学公式都把源 TeX 经过 GitHub 风格反转义后送进 mathjax-full，拿回 `<mjx-container><svg>...</svg></mjx-container>` 的完整渲染。
2. 把所有公式按文档顺序排进一张表格——inline 公式用三栏（行号 / 源 / 渲染），display 公式跨第 2-3 栏，源放上面、渲染居中放下面（不会被栏宽切）。
3. MathJax 报错的格子整行染红、错误消息列在渲染那一格的下方，肉眼一眼就能找到。
4. 整张表用 weasyprint 转成 A4 landscape PDF；每一行内禁止跨页（`page-break-inside: avoid`），所以不会一条公式被切两半。
5. 输出尺寸参考：~500 公式的章节约 19 页 PDF，正好在 Read 工具的 20 页处理上限内，多模态 reviewer 一次能看完整章。

**为什么不直接拉 github.com 渲染**：

- github.com 上的数学是客户端 MathJax 渲的，需要 headless 浏览器（chromium / playwright）才能截图——这套依赖比 mathjax-full 重得多。
- github.com 用的 MathJax 3 配置和我们这套 mathjax-full + AllPackages 在渲染层面**完全等价**（同一引擎、同一全功能包），只在字体上不同（github.com 用 STIX TTF，我们用 SVG glyph-as-path）——对"形状是否正确"的复核没影响。
- 不需要先 push 到远端再检查；本地 commit 之前就能跑。

如果将来真有「字体看着是否一致」的需求，再加一条 headless-browser 截图分支不迟。

## 实现要点

- **MathJax 3 vs MathJax 2 leniency**：MathJax 3 比 MathJax 2 宽容很多——`\left\{`、未知宏、缺命令包都不一定报错。我们用 MathJax 3（github.com 也用的版本），所以校验结果跟实际线上一致；这也意味着 CLAUDE.md 个别条目记录的"会报错"现在不会报，得相应订正。
- **CommonMark 反转义的覆盖**：CommonMark spec 上 ASCII 标点都能被 `\` 转义，但实测 GitHub 在 `$...$` 内只反转义少数几个（至少 `\_`、`\$`、`\*` 是确认的，`\,` `\!` `\;` `\>` 也会被吃；`\{` `\}` 保留）。**`\\` 也会被吃**——`\\` → 单个 `\`，导致 matrix/cases 换行失效（早期版本误以为 `\\` 会保留，已订正）。本 skill 的反转义子集 (`\_` `\*` `\$`) 仍是**保守的"肯定会被吃"集合**，但 `simulate_github_unescape` **不**主动把 `\\` 折成 `\`（塌成一行不是 MathJax 报错、折了也无从报），改由静态层 `backslash-rowbreak-eaten` 规则直接扫源 md 里的 `\\` 来补报。
- **位置追踪**：抽数学时不直接 mask 代码块（会改变 byte 位置），而是把代码块里的 `$` 字符替换成空格（保留 `\n` 和长度），这样 math 正则跳过代码区、行号计算照旧。
- **MathJax 启动成本**：Node + mathjax-full 启动要 1–2 s。校验单文件时单进程一次性处理整篇所有公式（一行一条 JSON 走 stdin），批量校验时每个文件起一次 Node。
- **错误检测**：MathJax 3 出错时不抛异常，而是在 SVG 输出里嵌一个 `<merror>` / `data-mjx-error="..."` 元素。我们扫这两个标记拿到错误消息回报。

## 已知限制

- **markdown 解析层是用启发式 grep 模拟的，不是真正的 GFM tokenizer**：rule 1/2/9/10/11 走的是源 md 上的正则扫描，覆盖到 CLAUDE.md 列举过的高频形态，但理论上还可能有别的 markdown 解析路径会出 bug 而脚本看不见。要 100% 复刻需要直接跑 GitHub `/markdown` API 或本地接入完整的 GFM 解析器，本 skill 没做。
- **不验证视觉布局**：公式宽度超出 GitHub 网页的容器、字体回退等视觉问题，render 层都看不见——`--visual` 出的 contact-sheet 是用 mathjax-full 的 SVG-as-path 输出，能体现"形状/语法"是否对，但跟 github.com 上 STIX TTF 渲染的字体粗细可能略有差异。
- **静态层假设**：
  - `\,` `\!` `\;` `\>` `\|` 一律被 CommonMark 吃，不区分上下文。如果某些位置上 GitHub 没吃（理论上不应该），会有假阳性——但 CLAUDE.md 的经验观察确实是无差别吃。前缀加了 `(?<!\\)` 让 `\\,` `\\|` 这种 array 换行 + 列分隔的形态不会被误报。
  - rule 9 表格识别走"行首是 `|` 且 ≥2 个未转义 `|`"的简易启发，对非标准表格（如 setext / pipe-table 之外的形态）可能漏报。
  - rule 11 两路触发：(a) 单段 inline math 同时含 `[` 与 `}_<char>`；(b) 同一行 ≥2 段独立 inline `$...$` 各带一个 `}_<char>`（跨段 emphasis 配对、无需方括号）。**单段、且整行只此一段带 `}_` 下标且不含 `[`** 的形态（如 `$\mathbf{x}_i^\top \mathbf{x}_j$` 单独成行）不报——GitHub 把整段 `$...$` 当原子、内部 `_` 安全。场景 (b) 的判定按"同一行有几段 inline span 带 bait"计，不按 bait 总数，以免误伤单段多下标。
  - rule 11′（`emphasis-eats-math`，render-based）补 11(a)/(b) 的盲区：闭合侧 `_` 是 right-flanking 但**前面不是 `}`**（如 `d_{`、`x_$`），静态 `}_` 正则抓不到，靠"原样 vs math-masked 各渲一遍、原样多出 emphasis 标签才报"来抓。它与静态 11 是**互补**关系：实测纯 markdown-it-py 的 CommonMark **不复现**场景 (b) 那种两个 `}_` 开启符（都是 left-flanking、互不配对），所以 (b) 仍靠静态层；而 11′ 专抓"一个开启符 + 一个 right-flanking 闭合符跨段配对"。11′ 只对**一行 ≥2 段 inline math**启用，回避单段里 opener+closer 自配（GitHub 原子抽取下其实正常渲染）可能造成的假阳性；依赖 `markdown-it-py`，缺它时与 `emphasis-flanking` 一同跳过。
- **非公式 markdown 层假设**：
  - `emphasis-flanking` **只查 `**` 粗体**，不查 `*` 斜体——后者裸 `*`（乘号、glob、prose 里的星号）误报率太高，CLAUDE.md 那段排查 python 也只查 `**`。逐行 `renderInline` 是对整篇 CommonMark 的近似（不跨行、不建块级上下文），对本仓库的行内强调足够。
  - `stray-tilde` / `fenced-code-in-list` 的代码块跟踪用"遇到围栏行就翻转"的简易开关（和 CLAUDE.md 片段同款），嵌套围栏 / 围栏内出现 ```` ``` ```` 当样例文本的极端情形可能误判。
  - `fenced-code-in-list` 靠"围栏 opener 带前导空格"启发——理论上 list 外缩进的围栏（罕见）也会被报；推荐的 6 空格缩进 code block（无 ```` ``` ````）和列 0 围栏都不会误报。
