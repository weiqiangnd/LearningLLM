# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

这是一份**个人 LLM 学习笔记**，不是软件项目。目标读者是希望系统理解大模型概念、使用、训练、微调方法的中文学习者。每一章是一个独立的学习单元，**没有跨章节的代码依赖、构建系统、测试套件**——不要寻找或假设它们存在。

## 章节结构约定

每一章由两份**互为对照**的文件组成（主线章节用 `NN` 两位数字编号；阶段 0 预备知识章节用 `P0N` 编号，与主线解耦，避免后续插入新预备知识内容时打乱主线编号）：

- `NN-标题.md` / `P0N-标题.md` —— 概念讲解、原理推导、对比表格、踩坑记录。回答「**为什么这样做**」。
- `NN.ipynb` / `P0N.ipynb` —— 可在 Colab 上直接跑通的代码示例。回答「**代码每一行做什么**」，逐行注释非常密集。

下文若无特别说明，`NN` 同时涵盖主线和预备知识两种编号。

写作或修改章节时务必同时更新两份文件，保持术语、代码片段、命名一致。**在 markdown 文档里给出代码片段时，确保它与 NN.ipynb 里的对应 cell 逐字一致（包括注释）**——这是本仓库刻意的内容分工。

**每一章必须自包含**——`.md` 与 `.ipynb` 都不允许把内容外包给"另一章"。具体三条：

- **`.md` 自包含**：所有代码片段、表格、示意图都在 `.md` 里直接给出，**不要出现「详见 `NN.ipynb` 的 Cell X」这类把内容外包给 ipynb 的句式**。读者只看 `.md` 就能获取完整信息；ipynb 是同步的可运行版本，不是 `.md` 的"代码附录"。文中需要引用某段实战代码时，用「实战中的 Cell N」这种章节内坐标，不要写成「`NN.ipynb` 的 Cell N」。**例外**：每章顶部的「Open in Colab」直链是导航元素，URL 里包含 `NN.ipynb` 是必要的，不算违反这条约定。
- **`.ipynb` 自包含**：每章 ipynb 都要从硬件自检 → 装依赖 → 加载模型这套样板从头跑起，**不允许出现「（与第 NN 章一致）」「沿用上一章环境」之类把 cell 内容外包给其他章节的注释**——读者从任意一章 Run All 都应能跑通。`.md` 中介绍铺垫 cell 时，可点明"和第 NN 章相同"以避免重复阅读，但代码与逐行注释仍要在本章 ipynb 中完整列出，不省略。
- **预备知识 P0N 章节不引用主线 NN 章节**：阶段 0 的内容与主线**完全解耦**，是任何 LLM 学习者都能独立读完的通用 PyTorch / 概率 / 优化 / RL 入门——`.md` 与 `.ipynb` **均不要出现「主线第 NN 章」「后续 NN 章会用到」「这就是 NN 章 attention 里的形状变换」之类把预备知识内容钉死到本仓库主线编号上的句式**。需要给"这个工具后面会怎么用"一点上下文时，用通用的领域语言（如「Transformer 的 multi-head attention 里」「LLM 训练中的下一个 token 预测」「写训练脚本时几乎都会遇到」）替代具体章节号。理由：预备知识编号 P0N 与主线 NN 解耦的初衷就是允许主线随时插入新章节、调整顺序而不打乱预备知识；预备知识文档里钉死主线章节号会让这种重排自动产生大量待修订的"幽灵引用"。主线 NN 章节之间倒可以彼此引用具体章节号——它们本就是顺序读物。

文件命名上 `.md` 文件名带描述（如 `01-IPython-Jupyter-Colab入门.md`），`.ipynb` 只用编号（`01.ipynb`）。新增章节请沿用此命名。

每一份 `NN-标题.md` 顶部需要包含三个固定元素：
- 一句「Open in Colab」直链（指向同章 `NN.ipynb`），格式：`https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/NN.ipynb`
- 一份**硬件门槛**标注，明确本章 ipynb 在哪种 Colab 运行时上能跑通（如「T4（15 GB）✅」/「需要 L4 或 A100，T4 显存不足」/「概念章，CPU 即可」）。读者不切到对应运行时就贸然 Run All 会浪费时间。
- 一份目录（TOC），列出本章二级标题及主要三级标题，方便长文档导航

`README.md` 顶部维护一份「学习路径」与「已完成章节」清单，每完成新章节需要追加更新；「已完成章节」中每一章要附 Open in Colab 徽章直链。

学习路径中**阶段 0 的 P0N 预备知识章节不要求读者一次读完**——改在主线**首次实际用到**该项背景知识的那一章下方，用一行 `〔预备知识〕首次用到 ... ——若不熟悉，建议先读 P0N。` 内联标注，已熟悉者可直接跳过。新增主线章节或新增 P0N 时同步维护这些"首次出现点"，不要把所有预备知识标注堆在某一章。

## 运行环境

- **默认运行平台是 Google Colab 免费版（T4，15 GB 显存）**——本地 macOS 上不会跑通需要 GPU + bitsandbytes 的代码。涉及大模型加载/推理的示例**默认基于 4-bit 量化**，确保 T4 能跑。
- **进阶章节允许使用 Colab Pro 的 L4（24 GB）或 A100（40 GB）**。例如 SFT/RM/RLHF/DPO 全流程、多模态训练、长上下文训练这些 T4 显存吃不下的章节，可显式声明所需 GPU 并以此为基线设计代码（Pro 用户能直接 Run All）。声明硬件需求时优先级是 **T4 > L4 > A100**——能在 T4 上跑就不要无故升档；确实跑不了再升 L4；只有 L4 也吃力时才升 A100。
- **每章必须在 `.md` 顶部和 `ipynb` 第一个 cell 显式标注硬件门槛**（具体格式见「章节结构约定」）。这样读者切到错误运行时不会浪费 5 分钟下载权重才发现 OOM。
- **默认示范模型是 `Qwen/Qwen3-8B`**（chat 模型，仓库名不带 `-Instruct` 后缀），用 bitsandbytes 4-bit NF4 + 双量化加载，权重约 5.5 GB。Qwen3 系列要求 `transformers>=4.51`，需在安装 cell 中显式锁定版本下界。某些 T4 上无法 LoRA 微调的章节可降档到 `Qwen/Qwen3-1.7B` 等小模型。
- **T4 是 Turing 架构，不原生支持 BF16**——若示例要用半精度（不量化），必须写 `torch_dtype=torch.float16`；想用 BF16 至少要 L4/A100（Ampere 及以上）。声明 L4/A100 的章节默认用 BF16。
- Notebook 中安装依赖统一用 `%%capture` + `!pip install -q -U ...`，且 `%%capture` **必须严格在 cell 第一行**（VS Code 的 Colab 扩展对此严格，浏览器版宽松）。
- 每一章 ipynb 的**第一个 cell 固定是硬件自检**：打印 `torch.cuda.is_available()`、GPU 名称、VRAM；若章节要求 L4/A100，在自检中加一行断言（如检测到 `Tesla T4` 直接 `raise RuntimeError("本章需要 L4 或 A100，请切换 Colab 运行时")`）让读者第一时间发现运行时不对。
- 不要假设有 `requirements.txt` 或虚拟环境配置——示例代码自带 `!pip install`，让读者在 Colab 里点一下就能跑。
- **概念性章节**（论文精读、scaling law、推理框架对比、分布式训练总览等）允许 ipynb 不依赖 GPU——可以只跑可视化、API 调用、小规模 demo；这种章节硬件门槛标为「概念章，CPU 即可」。

## 写作风格

- **中文为主**，技术名词保留英文（如 Transformer、tokenizer、attention），常用括号给中英对照。
- 章节按「**是什么 → 架构图/关系 → 表格对比 → 实战示例 → 关键概念回顾 → 本节小结 → 预告下一章**」的节奏组织。
- Notebook 行内 `#` 注释**故意写得啰嗦**——解释参数选择、踩坑提示，不要为求「整洁」精简。
- **概念讲解不跳步**。读者基线是「具有基础线性代数知识」；其余术语（softmax、交叉熵、layer norm、KV cache、RoPE 等）**首次出现必须先一两句话定义再使用**。
- **配示例与公式辅助理解**：
  - 公式用 LaTeX 语法（`$...$` 行内、`$$...$$` 块级），关键变量在公式后用一句话注明形状或含义。
  - 抽象原理尽量配最小可手算的数值示例（2×2 矩阵、长度为 3 的序列），让读者能在草稿纸上跟着算。
  - 形状变换（reshape、transpose、broadcasting）每一步标注张量形状，如 `(B, L, D) → (B, L, H, D/H) → (B, H, L, D/H)`。

### 架构图 / 原理图

仓库引入 [`fireworks-tech-graph`](https://github.com/yizhiyanhua-ai/fireworks-tech-graph) skill 用来生成 SVG 架构图、流程图、原理图，已安装在 `.claude/skills/fireworks-tech-graph/`，依赖系统包 `librsvg2-bin`（提供 `rsvg-convert`）。绘图时遵循以下约定：

- **统一使用默认 style 1（Flat Icon）**——白底、淡色填充、彩色边框、Helvetica/PingFang 字体；除非有特别理由，否则不要切换到 Dark Terminal、Blueprint 等其他 style，保持全仓库视觉一致。
- **生成方式**：调 `python3 .claude/skills/fireworks-tech-graph/scripts/generate-from-template.py <template-type> <output.svg> '<json>'`，再 `rsvg-convert -w <宽度> <svg> -o <tmp.png> && pngquant --quality=85-100 --strip --force --output <png> <tmp.png>` 导出 PNG（系统依赖 `librsvg2-bin` + `pngquant`）。**优先保证清晰度**，不再卡 PNG 体积上限。
- **导出宽度按信息密度选**：默认 `-w 720` 适用于节点 ≤ 6 / 文本稀疏的图；信息密度高（多容器、并排矩阵、长 sublabel、>10 节点）的图升到 `-w 1200`，避免缩放后字糊成一片。中间档 `-w 960` 用得少。
- **画面留白要"刚好"**：SVG 设计时让内容在 viewBox 里**四周留一圈适度留白**——边距太小（接近 0）会让框线压在边缘看着压抑；留白太多则正文显得空旷。一个简单参考：标题与最近的内容上沿留 ~20 px，最外层节点离左右边界 ~40 px、距底部 ~30 px；JSON 里的 `width`/`height` 跟着内容实际占用范围调整，不要无脑沿用 960×600 默认值。
- **生成后必须人眼复核 PNG**：`rsvg-convert` 不会检测越界——viewBox 太小、legend 装不下、节点重叠都不会报错，但 PNG 里会出现"被裁掉一半的字"。每张图渲染完都要用 Read 工具看一眼，确认（a）底部 legend 整条都在画面内（b）节点之间无重叠（c）容器内的标签没有被里面的子节点压住（d）主图与 legend、注解之间没有大段空白。
- **几个生成器常见的"看不见的坑"**（都来自 fireworks-tech-graph 的 `generate-from-template.py`）：
    - **legend 是纵向布局**，每条占 22 px。viewBox 高度必须 ≥ `legend_y + 22 × N + 20`，否则末尾几条会被静默裁掉（最容易踩）。
    - **container 的 `subtitle` 字段会与内部节点重叠**——容器副标题画在节点位置上方一点，节点稍矮就盖住它。要么不写 `subtitle`、要么把 `container.height` 拉大 ~30 px 给副标题让位。
    - **container 内节点的 `type_label` 也是上压**，节点本身要至少 60 px 高才装得下「TYPE_LABEL + label」两行；放 50 px 高的节点会被 type_label 遮住。
    - **手写 SVG（不走模板）时**底部 legend / caption 别紧贴 H 边——`pngquant` 偶尔会让边缘 1–2 px 颜色塌陷，看起来像被裁。统一留 ≥ 18 px。
- **文件存放在仓库根目录的 `assets/`**：每张图同时提交 `.svg`（源文件，便于以后微调）和 `.png`（实际引用的位图）。命名 `NN-用途.svg`/`NN-用途.png`，`NN` 与所属章节编号对齐（如 `01-stack.png`、`02-generate-pipeline.png`）；预备知识章节用 `P0N-用途.png`。
- **在 `.md` 中通过相对路径引用 PNG**：`![alt 文本](./assets/NN-用途.png)`。GitHub 上 PNG 渲染最稳定；`.svg` 在 GitHub README 里的 inline 渲染对外部字体不友好，所以默认引用 PNG。
- **不替换原有 ASCII 框图**：现有 `.md` 里的 ASCII 流程图保留（适合监控终端、纯文本 diff、复制粘贴），新生成的 SVG/PNG 作为视觉辅助插在 ASCII 之后，让两种形态各自发挥所长。
- **新增/修改章节时主动配图**：每章在「架构总览」「流水线/工作流」「关键算法步骤」这类抽象概念处至少配 1 张图；超长章节（>500 行）建议 3–5 张分散在不同小节。

### 公式的 GitHub 渲染避坑

GitHub 用 MathJax 渲染数学公式，但 markdown 解析先于 MathJax，下列写法已踩过坑：

- **`$$` 块级公式独占一行，前后各留空行**——否则部分 GitHub 渲染路径会吞掉它。
- **`$...$` 行内公式与中文之间留半角空格**（如「其中 $\ell_i$ 是 logit」），紧贴中文易识别失败。**这条对中文标点同样适用**：`，。；：）（——「」` 等紧贴 `$` 也会让 MathJax 识别失败。例如 `差 1，$\exp$ 之后` 的逗号紧贴 `$`，会让 `$\exp$` 渲染成字面字符；正确写法是 `差 1， $\exp$ 之后` 或干脆把逗号挪开。同理 `（temperature）$T$ 后是` → `（temperature） $T$ 后是`，`$\exp(\ell_i)$——把` → `$\exp(\ell_i)$ ——把`。规则要在 `$` **两侧**都成立。
- **不要写 `\left\{ ... \right\}`**：会报 `Missing or unrecognized delimiter for \left`。改用普通 `\{ ... \}`；需变长用 `\left\lbrace ... \right\rbrace`。
- **`\text{}` 内不要放下划线（无论 `_` 还是 `\_` 都不行）**——GitHub markdown 在 `$$` 块内仍会把 `\_` 还原为 `_`，而 MathJax 的 `\text{}` 不接受裸 `_`，会报 `'_' allowed only in math mode`。下划线分隔的标识符改用连字符（`\text{next-token}`）或拆开成 `\text{next}\_\text{token}` 让 `_` 变成下标。
- **数组/列表里的负数包成 `{-1.23}`**——`,\ -1.23` 这种写法 MathJax 会把 `-` 当二元减号，渲染成 `, − 1.23,`（前后自动加空格、看起来像断成两段）。用 `{}` 包起来让 `-` 变一元负号即可：`,\ {-1.23},`。
- **不要用 `\,` 控制间距**——`\` 后跟 ASCII 标点是 CommonMark 的 backslash-escape，`\,` 会被 markdown 还原成 `,`（实测在 `$$` 块内也会发生），结果 `[\,0.42` 渲染成 `[, 0.42`。要细空格用 `\thinspace`，或者干脆删掉、依赖 MathJax 默认间距。
- **表格单元格内的公式不能含裸 `|`**——会被当列分隔符吞掉，改用 `\mid` 或挪出表格放块级。
- **下标 / 上标里出现 `<` 或 `>` 要写成 `\lt` / `\gt`**——`x_{<t}` 这种写法（NLP 里"前 t 个 token"的标准记号）GitHub 渲染会报 `Extra open brace or missing close brace`。原因是 markdown 解析先于 MathJax，`<t}` 被当作疑似 HTML 标签处理，干扰了花括号配对。改写成 `x_{\lt t}` / `x_{\gt t}` 即可；另一种写法是直接换成 `x_{1:t-1}` 完全避开 `<`。

### matplotlib 图标题/坐标轴避坑：一律用英文

Colab 默认运行时的 matplotlib 字体是 `DejaVu Sans`，**不包含中文字形**。在 ipynb 里写 `plt.title('训练 loss 曲线')` 这种含中文的字符串，PNG 输出会把每个汉字渲染成豆腐块（□），同时控制台刷一堆 `UserWarning: Glyph XXXXX missing from font(s) DejaVu Sans` 警告。

约束：

- **所有 matplotlib 字符串参数（`title`/`xlabel`/`ylabel`/`suptitle`/`label`/`text`/`annotate` 等）一律用英文**。技术名词本来就英文（loss、accuracy、epoch、warmup、lr...），改起来通常一句直译就够。
- 这条**只针对 matplotlib 渲染出的字符**——`print(...)` 输出、cell 注释、markdown 文档都不受限，照常用中文。
- 不要靠 "在每个 ipynb 顶部装 `fonts-noto-cjk` + 设 `rcParams`" 来绕过：会让每章 ipynb 多一个 `apt-get install` cell（违背 "Run All 即可" 的体验），且字体安装失败时 fallback 还是豆腐块。**统一用英文是最稳的做法**。
- 新增/修改章节后，扫一眼有没有违规：`grep -nP "(plt\.title|set_title|plt\.xlabel|plt\.ylabel|set_xlabel|set_ylabel|suptitle|axvline.*label|axhline.*label).*[\x{4e00}-\x{9fff}]" *.ipynb` 应当无输出。

## Git 工作流

本仓库**不走功能分支 + PR 流程**。所有变更直接 commit 到 `main` 并 push 到 `origin/main`，不要新建 `claude/...` 之类的工作分支。如果 session 启动时被分配到了非 main 分支，先切回 main 再开始改动。

Claude 代为提交时的 commit 规范：
- **Author** 设为 `weiqiang <weiqiangnd@gmail.com>`（用 `git commit --author=...`，不要修改全局 git config）
- commit message 末尾加一行 `Co-Authored-By: Claude <模型名> <noreply@anthropic.com>`，**模型名填当前 session 实际运行的 Claude 版本**（例：`Opus 4.7 (1M context)`、`Sonnet 4.6`），不要照抄旧 commit 的字符串
- 不要在 message 里贴 `https://claude.ai/code/session_...` 链接
