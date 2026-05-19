---
name: check-github-render
description: >-
  Verify that math formulas in source markdown (`src/<name>.md`) will render
  correctly on **github.com**. Replicates GitHub's two-pass pipeline
  (CommonMark backslash-unescape inside math regions, then MathJax 3) to
  catch failures the local `markdown-to-pdf` pipeline can't see — GitHub
  uses MathJax while our PDF uses KaTeX, and the two engines have subtly
  different leniency. Pass `--visual` to additionally generate a contact-
  sheet PDF showing every formula's actual MathJax 3 rendering side-by-
  side with its source TeX — let a multimodal reviewer eyeball it instead
  of manually checking github.com. Trigger on: "check github render"
  "检查 github 渲染" "校验 github 数学公式" "github mathjax check"
  "verify github math" "github 渲染预览" "看看 github 怎么渲染".
---

# Check GitHub Math Rendering

针对 `src/*.md` 的一个**渲染层校验**，专门回答一个问题：**这份 md 推到 github.com 上之后，里面的数学公式能正确渲染么？**

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

这条 skill 同样走两段：

1. **抽数学区段**：用与 `markdown-to-pdf` 一致的正则（已经验证过对 src/*.md 全集 0 误抓），按文档顺序拿到所有 `$...$` / `$$...$$`，并记下每段在源文件里的 byte offset → 行号。
2. **模拟反转义**：对每段数学内容，**只反转义 GitHub 实际在 `$...$` 内会还原的那些**（`\_` `\*` `\$` —— 实测 github.com 上的工作流），保留 `\\` `\{` `\}` 这类 TeX 命令字符。
3. **MathJax 渲染**：把反转义后的 TeX 喂给 mathjax-full（服务端，Node + liteAdaptor），逐条检查输出里是不是含 `<merror>` 元素或 `data-mjx-error` 属性。
4. **静态层补一脚**：MathJax 3 看不到 `\,` `\!` `\;` `\>` 这类被 CommonMark 吃掉的间距宏（它们到 MathJax 时已经变成裸标点，没法报错），单独 grep 源里的这几个模式，在 mathjax-full 之外补报。

## 这条 skill 能抓到的、抓不到的

| CLAUDE.md 规则 | 抓得到？ | 怎么抓的 |
|---|---|---|
| 3 `\left\{ ... \right\}` | ❌ | 实测 MathJax 3 已经支持（早期 MathJax 2 才报错）。CLAUDE.md 那条**过时了**——可以放心写 `\left\{`。 |
| 4 `\text{}` 内裸 `_` | ✅ | MathJax 3 报 `'_' allowed only in math mode` |
| 6 `\,` `\!` `\;` `\>` 间距宏 | ✅ | 静态层 grep |
| 9 `}_V` workaround 必要性 | ❌ | 这条是 CommonMark **emphasis 分词**层的，不是 MathJax 层；要抓需要跑 GitHub `/markdown` API 或本地复刻 GFM tokenizer |
| 1 `$$` 块级空行 | ❌ | markdown 解析层 |
| 2 `$..$` 与 CJK 邻接 | ❌ | markdown 解析层 |
| 5 `{+1.23}` 包数字 | ❌ | TeX 语义层（两个引擎一致），渲不出错 |
| 7 表格里裸 `\|` | ❌ | markdown 表格 parser 层 |
| 8 下标里 `<` | ❌ | markdown HTML-标签解析层 |
| 任意 TeX 语法错（`Misplaced &`、`Missing argument for \frac` 等） | ✅ | MathJax 自报 |
| `\mathcal{L}\_V` 这种 workaround 写法 | ✅ 不报（这是正常写法） | 反转义后 `\mathcal{L}_V`，渲染正常 |

简单说：**MathJax 层的渲染错误抓得到；CommonMark / markdown 解析层的怪事抓不到**——那些需要真跑 GitHub `/markdown` API 才能 100% 复刻。CLAUDE.md 的 1/2/7/8 几条仍然要靠人在 github.com 上肉眼复查。

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
- **CommonMark 反转义的覆盖**：CommonMark spec 上 ASCII 标点都能被 `\` 转义，但实测 GitHub 在 `$...$` 内只反转义少数几个（至少 `\_`、`\$`、`\*` 是确认的，`\,` `\!` `\;` `\>` 也会被吃，`\\` `\{` `\}` 保留）。本 skill 反转义的子集 (`\_` `\*` `\$`) 是**保守的"肯定会被吃"集合**，覆盖不全的会被静态层那一脚补上。
- **位置追踪**：抽数学时不直接 mask 代码块（会改变 byte 位置），而是把代码块里的 `$` 字符替换成空格（保留 `\n` 和长度），这样 math 正则跳过代码区、行号计算照旧。
- **MathJax 启动成本**：Node + mathjax-full 启动要 1–2 s。校验单文件时单进程一次性处理整篇所有公式（一行一条 JSON 走 stdin），批量校验时每个文件起一次 Node。
- **错误检测**：MathJax 3 出错时不抛异常，而是在 SVG 输出里嵌一个 `<merror>` / `data-mjx-error="..."` 元素。我们扫这两个标记拿到错误消息回报。

## 已知限制

- **不验证 markdown 解析层**：rule 1/2/7/8 这种"GitHub 在解析阶段就没把这段识别成 math"的情况抓不到。最严重的实例（rule 9 那种 emphasis 吃下标）需要真跑 GFM tokenizer 才能复刻，本 skill 没做。
- **不验证视觉布局**：公式宽度超出 GitHub 网页的容器、字体回退等视觉问题，render 层都看不见。
- **静态层假设**：`\,` `\!` `\;` `\>` 一律被 CommonMark 吃，不区分上下文。如果某些位置上 GitHub 没吃（理论上不应该），会有假阳性——但 CLAUDE.md 的 rule 6 经验观察确实是无差别吃。
