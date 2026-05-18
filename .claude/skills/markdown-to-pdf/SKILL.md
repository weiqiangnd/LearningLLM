---
name: markdown-to-pdf
description: >-
  Render a project Markdown file (e.g. `src/P01-PyTorch与张量.md`,
  `src/02-生成参数与采样策略.md`) to a GitHub-style PDF under `dist/`. The PDF
  preserves GFM rendering, KaTeX math (`$...$` / `$$...$$`), embeds local
  images (e.g. `../assets/P01/*.png`) as base64, and keeps the original
  filename with extension swapped to `.pdf`. Trigger on: "生成 pdf"
  "导出 pdf" "markdown 转 pdf" "渲染 pdf" "出 PDF" "md to pdf"
  "render markdown to pdf" "export chapter pdf".
---

# Markdown → GitHub-style PDF

把仓库里的一个 `.md` 章节按 **GitHub 渲染样式** 导出成 PDF，放到 `dist/` 目录。

## 输出契约

- 输入：仓库 `src/` 目录下任意 `<name>.md`（含中文文件名）。
- 输出：`dist/<name>.pdf`（同名，仅替换扩展名；统一落在仓库根的 `dist/` 下）。
- 渲染规则：
  - **GFM**：标题、表格、任务列表、围栏代码块（Pygments 着色）、链接、引用块。
  - **数学公式**：行内 `$...$` 与块级 `$$...$$` 走 KaTeX 服务端渲染（不依赖运行时 JS）。
  - **图片**：相对路径（章节 md 在 `src/`，配图在仓库根 `assets/`，所以写 `../assets/P01/foo.png`）一律读取本地文件并嵌入为 base64 data URI；HTTP/HTTPS 图片也会**尝试下载并 base64 内嵌**（带 10 秒超时和 4 MB 上限），网络失败时保留原 `src`。已知的「Open in Colab」徽章直接走 skill 自带的本地 SVG（见 `assets/colab-badge.svg`），即便环境出网受限也能渲染出来。
  - **字体**：
    - 正文（中英文）：**LXGW WenKai**（霞鹜文楷），由 `install.sh` 通过 `apt install fonts-lxgw-wenkai` 安到系统字体。这是一份双语字体——Latin 字母按楷体风格画、与汉字同 x-height，混排时高度齐平。找不到时回退到 Noto Sans CJK SC。
    - 代码：`Liberation Mono` / `DejaVu Sans Mono`（系统自带）；含中文的代码段落回退到 `LXGW WenKai Mono`。
  - **样式**：套用 `github-markdown.css`，整体观感对齐 GitHub.com 网页。
  - **页脚**：每页底部统一渲染 `GitHub: https://github.com/weiqiangnd/LearningLLM`（左）/ `Author: WeiQiang`（中）/ `当前页 / 总页数`（右），由 `assets/print.css` 里的 `@page` margin box 控制——改署名 / 仓库地址直接编辑该文件。

## 用法

```bash
# 1. 首次使用先装依赖（幂等，已装会跳过）
bash .claude/skills/markdown-to-pdf/scripts/install.sh

# 2. 渲染一个章节（章节 md 现在统一在 src/ 下）
python3 .claude/skills/markdown-to-pdf/scripts/render.py src/P01-PyTorch与张量.md

# 输出：dist/P01-PyTorch与张量.pdf
```

也支持批量：

```bash
python3 .claude/skills/markdown-to-pdf/scripts/render.py src/P01-*.md src/02-*.md
```

## 实现要点（出问题时排查）

- **数学公式**：`scripts/render_math.js` 接收 stdin 上一行一条的 JSON（`{"tex": "...", "display": true/false}`），逐条用 KaTeX 渲染回 HTML 字符串。Python 端通过单次 Node 子进程一次性处理整篇文章里的全部公式，避免反复启动 Node。
- **数学定界提取**：先把 fenced code block / inline code 用占位符遮起来，再用正则提取 `$$...$$` / `$...$`，最后恢复 code block——避免错误地把代码里的 `$` 当成公式。
- **TeX 反转义 `\_` / `\*` / `\$` → `_` / `*` / `$`**：本仓库 md 源里为了避开 markdown emphasis 经常写 `\mathbb{E}\_{...}` 这种「反斜杠 + 下划线」。GitHub 走 markdown → MathJax 是两段管线，markdown 先把 `\_` 还原成 `_` 再交给 MathJax，所以网页上下标正常。我们的管线是「先把数学整段拽出来再交给 KaTeX」，反斜杠会原样进 KaTeX，结果 `\_` 被当成字面下划线、整段下标变成一坨 `E_τ∼π_θ`。`render.py` 的 `_unescape_markdown_in_tex()` 在数学提取阶段统一把这三种 markdown 级转义还原。
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
