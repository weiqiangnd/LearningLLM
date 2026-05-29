---
name: markdown-to-pdf
description: >-
  Render a project Markdown file (e.g. `src/P01-PyTorch与张量.md`,
  `src/02-生成参数与采样策略.md`) into a {md, html, pdf} triplet under
  `dist/`. The pipeline first unescapes math-level `\_` etc. into a
  KaTeX-compatible `.md`, renders it to a self-contained `.html` with
  server-side KaTeX, runs a cross-stage math consistency check, and only
  then produces the `.pdf`. All three files share the same stem.
  Trigger on: "生成 pdf" "导出 pdf" "markdown 转 pdf" "渲染 pdf" "出 PDF"
  "md to pdf" "render markdown to pdf" "export chapter pdf".
---

# Markdown → {md, html, pdf} triplet

把仓库里的一个 `.md` 章节按 **GitHub 渲染样式** 导出成三件套（KaTeX 兼容的 md、自包含 html、PDF），统一放到 `dist/` 目录。

## 输出契约

- 输入：仓库 `src/` 目录下任意 `<name>.md`（含中文文件名）。
- 输出（同 stem，仅扩展名不同，统一落在仓库根的 `dist/` 下）：
  - `dist/<name>.md` —— **KaTeX 兼容的 markdown**。除了 `$...$` / `$$...$$` 内的 `\_` `\*` `\$` 被反转义成 `_` `*` `$` 之外，与源文件按字节一致。任何 KaTeX 前端（VS Code 的 Markdown All in One 等）都能直接渲染。**注意**：这份 md 在 GitHub 上不再渲染数学（因为 GitHub 走 markdown → MathJax 两段管线，少了反转义步骤后下划线会被 emphasis 吃掉）——这是有意为之，因为这份文件的目标是 KaTeX 工具链。
  - `dist/<name>.html` —— **自包含 HTML**。KaTeX 服务端渲染（`htmlAndMathml`），所有图片 base64 内嵌，KaTeX 字体内嵌，github-markdown.css + print.css 嵌入。打开浏览器直接看，不需要网络。
  - `dist/<name>.pdf` —— 上面那份 HTML 经 WeasyPrint 出的 PDF。
- 渲染规则：
  - **GFM**：标题、表格、任务列表、围栏代码块（Pygments 着色）、链接、引用块。
  - **数学公式**：行内 `$...$` 与块级 `$$...$$` 走 KaTeX 服务端渲染（不依赖运行时 JS）。
  - **图片**：相对路径（章节 md 在 `src/`，配图在仓库根 `assets/`，所以写 `../assets/P01/foo.png`）一律读取本地文件并嵌入为 base64 data URI；HTTP/HTTPS 图片也会**尝试下载并 base64 内嵌**（带 10 秒超时和 4 MB 上限），网络失败时保留原 `src`。已知的「Open in Colab」徽章直接走 skill 自带的本地 SVG（见 `assets/colab-badge.svg`），即便环境出网受限也能渲染出来。
  - **字体**：
    - 正文（中英文）：**LXGW WenKai**（霞鹜文楷），由 `install.sh` 通过 `apt install fonts-lxgw-wenkai` 安到系统字体。这是一份双语字体——Latin 字母按楷体风格画、与汉字同 x-height，混排时高度齐平。找不到时回退到 Noto Sans CJK SC。
    - 代码：`Liberation Mono` / `DejaVu Sans Mono`（系统自带）；含中文的代码段落回退到 `LXGW WenKai Mono`。
  - **样式**：套用 `github-markdown.css`，整体观感对齐 GitHub.com 网页。
  - **页脚**：每页底部统一渲染 `GitHub: https://github.com/weiqiangnd/LearningLLM`（左）/ `Author: WeiQiang with Claude`（中）/ `当前页 / 总页数`（右），由 `assets/print.css` 里的 `@page` margin box 控制——改署名 / 仓库地址直接编辑该文件。

## 用法

```bash
# 1. 首次使用先装依赖（幂等，已装会跳过）
bash .claude/skills/markdown-to-pdf/scripts/install.sh

# 2. 渲染一个章节
python3 .claude/skills/markdown-to-pdf/scripts/render.py src/P01-PyTorch与张量.md

# 输出（三件套）：
#   dist/P01-PyTorch与张量.md
#   dist/P01-PyTorch与张量.html
#   dist/P01-PyTorch与张量.pdf
```

也支持批量：

```bash
python3 .claude/skills/markdown-to-pdf/scripts/render.py src/P01-*.md src/02-*.md
```

## 流水线分步骤

`render.py` 的处理顺序，每一步都对应 dist/ 里能看见的中间产物或一个失败时的明确告警：

1. **读源 md** （`src/<name>.md`）。
2. **生成 KaTeX 兼容 md** —— `unescape_math_in_md()` 只反转义 `$...$` / `$$...$$` 内的 `\_` `\*` `\$`；代码块、行内代码、正文一字不动。结果写入 `dist/<name>.md`。
3. **渲染 HTML** —— `render_markdown_to_html()` 走 markdown-it-py + Pygments + KaTeX (`htmlAndMathml`)，图片 base64 内嵌，最终拼接 `HTML_TEMPLATE` 写入 `dist/<name>.html`。
4. **数学一致性校验** —— `verify_math_consistency()` 把三份产物里的数学公式按文档顺序对齐，逐条比对（细节见下一节）。一旦发现异常，**抛 `MathInconsistency`、跳过 PDF 生成**；`dist/` 里的 `.md` / `.html` 保留以便排查。
5. **生成 PDF** —— WeasyPrint 把同一份 HTML 渲成 `dist/<name>.pdf`。

## 数学一致性校验做了哪些事

校验跨越三个表示层比对，任何一层不一致都终止整条流水线（不生成 PDF）：

| 层 | 比对内容 | 触发的典型 bug |
|---|---|---|
| A. 区段计数 | 原始 md / KaTeX md / HTML annotation 三者数学公式数目相等 | 某段公式被吞掉、被多算 |
| B. 反转义闭合 | `_unescape_markdown_in_tex(原始)` == `KaTeX md` | unescape 改动了不该动的内容、或者漏掉某段 |
| C. KaTeX 输入回声 | `KaTeX md` == HTML 中 `<annotation encoding="application/x-tex">` 解码后的内容 | KaTeX 没拿到我们以为给它的 TeX |
| D. 残留转义 | annotation 内不出现 `\_` / `\*` | 反转义漏网，下标会变字面下划线 |
| E. KaTeX 错误标记 | HTML 中不出现 `katex-error` / `class="math-error"` / `mathcolor="#cc0000"` / `color:#cc0000` | 未知宏、语法错误（KaTeX `strict:"ignore"` 会静默把它涂成 `#cc0000` 红） |
| F. MathML 下标丢失签名 | HTML 中不出现 `<mi ...>_</mi>` | 字面下划线被当作 `<mord>` 渲染（不是 `<msub>`），下标视觉缺失 |

C 层的存在让"静态扫描"那种漏检方案显得脆弱——它不是去匹配 md 源文件里的可疑模式，而是直接对比 **KaTeX 实际接收到的 TeX** 和我们写入 `dist/<name>.md` 的 TeX。两边一致就说明渲染管线没有偷偷加塞或漏掉东西。

## 实现要点（出问题时排查）

- **数学公式**：`scripts/render_math.js` 接收 stdin 上一行一条的 JSON（`{"tex": "...", "display": true/false}`），逐条用 KaTeX 渲染回 HTML 字符串，输出模式是 `htmlAndMathml`——同时拿到视觉 HTML 与 `<annotation encoding="application/x-tex">` 里的源 TeX 回声，后者是校验步骤的锚点。Python 端通过单次 Node 子进程一次性处理整篇文章里的全部公式，避免反复启动 Node。
- **数学定界提取**：先把 fenced code block / inline code 用占位符遮起来，再用正则提取 `$$...$$` / `$...$`，最后恢复 code block——避免错误地把代码里的 `$` 当成公式。`_mask_code()` / `_restore_code()` 这对函数被 unescape / 校验 / HTML 渲染三处共用。
- **TeX 反转义 `\_` / `\*` / `\$` → `_` / `*` / `$`**：本仓库 md 源里为了避开 markdown emphasis 经常写 `\mathbb{E}\_{...}` 这种「反斜杠 + 下划线」。GitHub 走 markdown → MathJax 是两段管线，markdown 先把 `\_` 还原成 `_` 再交给 MathJax，所以网页上下标正常。我们的管线是「先把数学整段拽出来再交给 KaTeX」，反斜杠会原样进 KaTeX，结果 `\_` 被当成字面下划线、整段下标变成一坨 `E_τ∼π_θ`。`render.py` 的 `unescape_math_in_md()` 在**生成 `dist/<name>.md` 这一步**统一把这三种 markdown 级转义还原；之后无论是 HTML 渲染还是 PDF 渲染都走这份反转义过的 md。`_unescape_markdown_in_tex()` 在底层 `mask_code_and_math()` 里保留为幂等的"二道防线"。
- **`re.sub` 跟 KaTeX HTML 的兼容**：KaTeX `htmlAndMathml` 输出里嵌着大段源 TeX（`\leq`、`\log` 等带反斜杠的命令）。把这些 HTML 当 `re.sub` 的字面 replacement 字符串传进去会触发 "bad escape \l" 报错（正则会把 `\l` 当模板转义解释）。所有把 KaTeX 输出回填到占位符的 `re.sub` 都用 `lambda _m, w=wrapped: w` 形式的可调用 repl，跳过模板处理。
- **字体内嵌**：`render.py` 的 `inline_font_urls()` 通用函数会把 CSS 里所有 `url(fonts/*.woff2)` 改写成 base64 data URI。目前只有 `assets/katex.min.css`（KaTeX 数学字体）走这条路径——正文 LXGW 文楷由系统 fontconfig 提供、不走 CSS @font-face。`install.sh` 负责从 `node_modules/katex/dist/` 复制 KaTeX 资源到 `assets/`。
- **图片嵌入**：扫描 `<img src="...">`，本地路径全部转 base64；HTTP/HTTPS 图片走 `urllib` 抓回来再内嵌（带 10s 超时和 4 MB 上限，失败时保留原 `src`、只打 warn）；`data:` URI 原样透传。少数已知图（目前只有 Colab 徽章）通过 `_REMOTE_IMAGE_LOCAL_MAP` 直接映射到 `assets/colab-badge.svg`（来源是 GitHub Camo 缓存的官方原图），保证哪怕网络拒绝出站也能渲染出徽章。
- **TOC 锚点**：md 里 `[xxx](#锚点)` 形式的目录链接保留，但 WeasyPrint 输出的 PDF 内部跳转依赖 `id` 锚点——`markdown-it-py` 的 `anchor` 插件按 GitHub slug 规则生成 `id`，行为应与 GitHub 一致。
- **页脚**：`assets/print.css` 的 `@page` 用 `@bottom-left` / `@bottom-center` / `@bottom-right` 三个 margin box 渲染「仓库链接 · 作者 · 页码」，8 pt 灰字、`white-space: nowrap`、`padding-top: 6mm` 把文字压在底边距下方。要改署名、加页眉或换页码格式，直接动这段 `@page` 即可。

## 已知踩坑

- **不要在 `.markdown-body` 的 `font-family` 里挂任何 emoji 字体**（`"Apple Color Emoji"` / `"Noto Color Emoji"` 都不行）。WeasyPrint+Pango 在 font-family 末尾出现 emoji 字体时会把数字与 CJK 边界的 advance width 算错——`"0 维"` / `"99%"` 这种位置会被撑出近 4 个字符宽的空格，看着像被 justify 过。emoji 字形（☑ ✅ 等）会通过 fontconfig 的系统级 fallback 命中，**不需要**显式列在 CSS 字体栈里。
- **不要给 `.markdown-body` 加 `overflow-wrap: anywhere`**——会在 Latin 单词中间字符级折断（"Ran|k"）。用 `overflow-wrap: break-word` 已足够把超长 URL 折下来，又不会蹂躏短词。
- **不要给 `<p>` 加 `text-align: justify`**：CJK 段尾的尾字会把整行空白吃掉，效果反而难看。WeasyPrint 默认 `start` 就够。
- **嵌套 `<ul ul>` 别让默认 `list-style: circle` 留着**——`◦` (U+25E6) 在 WeasyPrint+Pango 的字体 fallback 链里可能落到 color emoji 字体上，TOC 二级条目（2.1、2.2…）前面会冒出一个像 GitHub octocat 的彩色图标。强制把所有层级的 `<ul>` 设成 `list-style: disc`，统一用 `•`。

## 已知限制

- 不依赖浏览器，所以**纯 JS 驱动的内容渲染不到** PDF——但本仓库目前没有这类内容。
- 极长行的代码块可能横向溢出页面：必要时在源 md 中手动换行。
