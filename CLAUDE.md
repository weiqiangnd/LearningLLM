# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

这是一份**个人 LLM 学习笔记**，不是软件项目。目标读者是希望系统理解大模型概念、使用、训练、微调方法的中文学习者。每一章是一个独立的学习单元，**没有跨章节的代码依赖、构建系统、测试套件**——不要寻找或假设它们存在。

## 章节结构约定

每一章由两份**互为对照**的文件组成（主线章节用 `NN` 两位数字编号；阶段 0 预备知识章节用 `P0N` 编号，与主线解耦，避免后续插入新预备知识内容时打乱主线编号）。**所有章节文件统一放在仓库 `src/` 目录下**，配图资源仍在仓库根目录的 `assets/` 下（与 `src/` 平级，章节 md 里通过 `../assets/<NN>/...` 引用）：

- `src/NN-标题.md` / `src/P0N-标题.md` —— 概念讲解、原理推导、对比表格、踩坑记录。回答「**为什么这样做**」。
- `src/NN.ipynb` / `src/P0N.ipynb` —— 可在 Colab 上直接跑通的代码示例。回答「**代码每一行做什么**」，逐行注释非常密集。

下文若无特别说明，`NN` 同时涵盖主线和预备知识两种编号。

写作或修改章节时务必同时更新两份文件，保持术语、代码片段、命名一致。**在 markdown 文档里给出代码片段时，确保它与 NN.ipynb 里的对应 cell 逐字一致（包括注释）**——这是本仓库刻意的内容分工。

**每一章必须自包含**——`.md` 与 `.ipynb` 都不允许把内容外包给"另一章"。具体三条：

- **`.md` 自包含**：所有代码片段、表格、示意图都在 `.md` 里直接给出，**不要出现「详见 `NN.ipynb` 的 Cell X」这类把内容外包给 ipynb 的句式**。读者只看 `.md` 就能获取完整信息；ipynb 是同步的可运行版本，不是 `.md` 的"代码附录"。文中需要引用某段实战代码时，用「实战中的 Cell N」这种章节内坐标，不要写成「`NN.ipynb` 的 Cell N」。**例外**：每章顶部的「Open in Colab」直链是导航元素，URL 里包含 `NN.ipynb` 是必要的，不算违反这条约定。
- **`.ipynb` 自包含**：每章 ipynb 都要从硬件自检 → 装依赖 → 加载模型这套样板从头跑起，**不允许出现「（与第 NN 章一致）」「沿用上一章环境」之类把 cell 内容外包给其他章节的注释**——读者从任意一章 Run All 都应能跑通。`.md` 中介绍铺垫 cell 时，可点明"和第 NN 章相同"以避免重复阅读，但代码与逐行注释仍要在本章 ipynb 中完整列出，不省略。
- **预备知识 P0N 章节不引用主线 NN 章节**：阶段 0 的内容与主线**完全解耦**，是任何 LLM 学习者都能独立读完的通用 PyTorch / 概率 / 优化 / RL 入门——`.md` 与 `.ipynb` **均不要出现「主线第 NN 章」「后续 NN 章会用到」「这就是 NN 章 attention 里的形状变换」之类把预备知识内容钉死到本仓库主线编号上的句式**。需要给"这个工具后面会怎么用"一点上下文时，用通用的领域语言（如「Transformer 的 multi-head attention 里」「LLM 训练中的下一个 token 预测」「写训练脚本时几乎都会遇到」）替代具体章节号。理由：预备知识编号 P0N 与主线 NN 解耦的初衷就是允许主线随时插入新章节、调整顺序而不打乱预备知识；预备知识文档里钉死主线章节号会让这种重排自动产生大量待修订的"幽灵引用"。主线 NN 章节之间倒可以彼此引用具体章节号——它们本就是顺序读物。**P0N 之间同理也允许互引**（P01–P05 自身就是一条顺序学习路径），但**只允许后面的章节回引前面的章节**：正文里「P02 已经把训练循环立起来了」「P03 里讲过 KL」这类回引是刻意设计，不算违规；**前面的章节不要把读者支到还没读到的后面章节去**（如在 P02 正文里写「P03 的实战里有专门的反例演示」）——按 P01→P05 顺序读的读者此刻还没见过那一章。**唯一例外是预告式指针**：每章末尾的「预告 P0(N+1)」、正文里「细节留到 P0M 展开」「下一章 P0M 会再讲一遍」这类只说"后面会讲"、不要求读者现在跳转的句式不受限。解耦条款只针对 P0N→主线 NN 这一个方向。

文件命名上 `.md` 文件名带描述（如 `src/01-IPython-Jupyter-Colab入门.md`），`.ipynb` 只用编号（`src/01.ipynb`）。新增章节请沿用此命名并放到 `src/` 下。

每一份 `src/NN-标题.md` 顶部需要包含三个固定元素：
- 一句「Open in Colab」直链（指向同章 `src/NN.ipynb`），格式：`https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/NN.ipynb`
- 一份**硬件门槛**标注，明确本章 ipynb 在哪种 Colab 运行时上能跑通（如「T4（15 GB）✅」/「需要 L4 或 A100，T4 显存不足」/「概念章，CPU 即可」）。读者不切到对应运行时就贸然 Run All 会浪费时间。
- 一份目录（TOC），列出本章二级标题及主要三级标题，方便长文档导航

此外，如果本章用到了某项**预备知识**（P0N），就在**硬件门槛标注的下方**补一行 `> 〔预备知识〕…… ——若不熟悉，建议先读 P0N。` 内联标注（紧贴硬件门槛、TOC 之前），说明本章首次密集用到的背景知识及对应的预备章节，方便读者按需回查。这条**只对主线 NN 正式章节有效**——预备知识 P0N 章节本身就是按 P01→P05 顺序编排的入门读物（正文里允许互引其它 P0N，见「章节结构约定」），顶部不放这类〔预备知识〕标注（P0N 顶部仍需 Open in Colab、硬件门槛、TOC 三件）。注意这与 `README.md` 学习路径里那行「首次用到」标注是同一件事的两处落点，两边要同步维护。

`README.md` 学习路径表格里的对应链接则一律写成 `[OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/NN.ipynb)`，文档与 ipynb 链接写成 `./src/NN-...md` / `./src/NN.ipynb`。

`README.md` 中维护一份「学习路径」表格清单，每完成新章节需把对应行的「链接」列（文档 / ipynb / OpenInColab 文字链接）填上，并把「状态」列标 ✅。

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
- **尽量用口语化表达**，把笔记写得像跟同事白板边讲边推演，而不是教科书式的"陈述句堆砌"。具体几条参考：
  - 能用「咱们」「我们」「你」拉近距离就别一直「读者」「使用者」。
  - 用反问 / 设问把读者带入思考：「那为什么不直接 X？因为……」「这一步要是省了会怎样？答案是……」，比一上来就抛结论更容易跟得上推导。
  - 适度使用「其实」「说白了」「换句话说」「反过来想」「顺手提一句」这类口语连接词来标注节奏切换；但**不要每段都来一句**，密度过高反而像废话。
  - 口语化≠不严谨：定义、公式、形状标注、踩坑结论这几处仍然要写得字斟句酌；口语只用在引入概念、过渡段落、直觉解释上，**不要稀释技术内容本身**。
  - **俏皮表达有个「慎用清单」**——它们偶尔点一句有味道，但草稿里有反复堆砌、过密的倾向，审阅时几乎都会被换成平实说法，所以下笔时就克制：
    - **「算账」类比喻别反复用**：「把账算细一点」「把这笔账算清楚」「把话说公道」这类一章里出现两三次就腻，换成「说到底 / 把缘由讲清楚 / 客观地说」等直白说法。
    - **网络梗、夸张绝对词少用**：`真·xxx`（网梗前缀）、`不要钱 / 免费 / 白赚`（形容「几乎零成本」）、`命门 / 必踩`（夸大某处的重要性）——分别改成「真正 xxx」「没有额外的成本 / 总量持平」「最容易写错的地方 / 容易踩到的坑」。
    - **太随意的口语动词收一收**：「它的玩法是」→「它的核心思想是」，「（上一章）也留了话」→「也提到」，「捏进一个栈」→「揉进一个栈」。
    - 判据：同一个俏皮说法在一章里出现 ≥2 次，或它把一个技术结论说得比实际更绝对，就该换平实表述。
- **术语统一**（这些词在多章里反复被同向替换，下笔时直接用右栏，省得审阅时再改）：
  - **表达力 → 表达能力**（统一不用「表达力」）。
  - **实证 → 验证 / 实验依据**（「深层稳定性实证」「亲手实证」「实证理由」一律写成「验证」「亲手验证」「实验依据」）。
  - 后续发现新的同向高频替换，继续往这张表里补。
- **正文里少把英文术语当动词用**——名词保留英文、动作尽量说中文：写「头 1 专管指代」不写「头 1 专 attend 指代」（attend / embed 当动词时换成「关注 / 查向量」）。
- **正文（非 ipynb 代码）里引用代码级标识符时，配上概念名或数学记号**，别让读者对着裸变量名猜：写「输出投影 $W_O$」不写光秃秃的「o_proj」；确需点出代码里的名字时用「输出投影 $W_O$（代码里叫 `o_proj`）」这种「概念在前、标识符在后」的形式。这条只管正文叙述；ipynb 的 `#` 注释里直接用变量名不受限。
- 章节按「**是什么 → 架构图/关系 → 表格对比 → 实战示例 → 关键概念回顾 → 本章小结 → 预告下一章**」的节奏组织。
- **章节用语统一**：一个 md 文件 = **一章**；同一 md 文件内的不同二级标题 = **一节**。所以：
  - **「本章」指代整个 md 文件**，「本节」**只**指代当前所在的二级标题小节，不要混用（不要用「本节」当成「本章」的同义词）。每章末尾的总结小节统一叫**本章小结**（不是「本节小结」），它总结的是整章。
  - **引用同一 md 内的其他小节**统一写成「第 N 节」（如「第 6 节」）或「第 N.M 节」（如「第 4.2 节」）；范围写「第 N-M 节」（如「第 4-6 节」）。**不要**使用 `§N` 这种 LaTeX 风格的简写。**例外**：纯英文上下文（如配图里的英文标题、英文 caption、英文代码注释）允许使用 `§N` 简写，因为此时中文「第 N 节」会破坏整体英文排版。
  - **跨章引用**加上章节编号前缀，如「P05 第 2.3 节」、「P03 第 6 节」——尤其是写在代码块/docstring 里时，prefix 能让单独阅读 ipynb 的人也知道指向哪一章。
  - **不要再用中文数字写小节引用**（如「第七节」「第六节」）——统一用阿拉伯数字 + 空格分隔的「第 N 节」形式，便于以后章节顺序调整时机械替换。章节标题里的 `## 一、` / `## 二、` 等本身用中文数字不受此约束。
  - **跨章引用整章时统一写成阿拉伯数字 + 空格分隔的「第 N 章」**（如「第 1 章」「第 2 章」），**不要**用中文数字「第一章」「第二章」、也不要补零写「第 01 章」——理由同小节引用：便于以后章节顺序调整时机械替换。**例外（唯一）是各章自己的 H1 标题保持原样**——`# 第一章：…` / `# 第二章：…` 等标题里的中文数字不动（标题是章节的"门牌"，按中文阅读习惯保留中文数字），只把**正文 / 注释 / docstring 里指向别章的引用**统一成「第 N 章」。
- Notebook 行内 `#` 注释**故意写得啰嗦**——解释参数选择、踩坑提示，不要为求「整洁」精简。
- **概念讲解不跳步**。读者基线是「具有基础线性代数知识」；其余术语（softmax、交叉熵、layer norm、KV cache、RoPE 等）**首次出现必须先一两句话定义再使用**。
- **配示例与公式辅助理解**：
  - 公式用 LaTeX 语法（`$...$` 行内、`$$...$$` 块级），关键变量在公式后用一句话注明形状或含义。
  - 抽象原理尽量配最小可手算的数值示例（2×2 矩阵、长度为 3 的序列），让读者能在草稿纸上跟着算。
  - 形状变换（reshape、transpose、broadcasting）每一步标注张量形状，如 `(B, L, D) → (B, L, H, D/H) → (B, H, L, D/H)`。

### 架构图 / 原理图

仓库引入 [`fireworks-tech-graph`](https://github.com/yizhiyanhua-ai/fireworks-tech-graph) skill 用来生成 SVG 架构图、流程图、原理图，已安装在 `.claude/skills/fireworks-tech-graph/`，依赖系统包 `librsvg2-bin`（提供 `rsvg-convert`）。绘图时遵循以下约定：

- **统一使用默认 style 1（Flat Icon）**——白底、淡色填充、彩色边框、Helvetica/PingFang 字体；除非有特别理由，否则不要切换到 Dark Terminal、Blueprint 等其他 style，保持全仓库视觉一致。
- **生成方式**：调 `python3 .claude/skills/fireworks-tech-graph/scripts/generate-from-template.py <template-type> <output.svg> '<json>'`，再 `rsvg-convert -w <实际像素> <svg> -o <tmp.png> && pngquant --quality=100 --strip --force --output <png> <tmp.png>` 导出 PNG（系统依赖 `librsvg2-bin` + `pngquant`）。**优先保证清晰度**，不再卡 PNG 体积上限；`--quality=100` 让 pngquant 只在能保真的前提下做压缩，避免 8-bit 调色板把抗锯齿边缘搞脏。
- **PNG 导出固定用 Noto Sans CJK SC 字体**：渲染 PNG 实际用哪个字体不取决于 SVG 里写的字体栈靠前那些（`PingFang SC`、`Helvetica` 等多半只在 macOS 上有），而取决于**渲染机 fontconfig 能解析到谁**——不指定时会 fallback 到文泉驿正黑之类，字形偏点阵、且各机不一致。为让中文规整、全仓库一致，统一把 **`Noto Sans CJK SC`**（等宽文字用 **`Noto Sans Mono CJK SC`**）放到 SVG `<style>` / 手写 `font-family` / 脚本 `FONT`、`MONO` 常量字体栈的**最前面**。渲染前先确认字体已装，没有就装依赖：`apt-get install -y fonts-noto-cjk`。换字体后 Noto 的字宽和原先 fallback 字体不同，**务必复核有没有撑框 / 错位**（见下方人眼复核条）。
- **导出宽度按信息密度选**：**默认 `-w 2400`**（绝大多数图都用它——多容器 / 并排矩阵 / 长 sublabel / 节点不算少 / 带斜排列头或图例的，统统走这档）；只有**节点非常少、文本很稀疏**（≤ 6 个节点，且没有斜排列头、图例这类细节）才降到 `-w 1800`；**节点非常多 / 极宽的横向流程**（密集矩阵、>15 节点、超长并排）升到 `-w 3000`。这套数值是按 retina 屏和 PDF 打印需要的实际像素定的，直接选最终值传给 `rsvg-convert -w`，不用再乘倍率。
- **画面留白要"刚好"**：SVG 设计时让内容在 viewBox 里**四周留一圈适度留白**——边距太小（接近 0）会让框线压在边缘看着压抑；留白太多则正文显得空旷。一个简单参考：标题与最近的内容上沿留 ~20 px，最外层节点离左右边界 ~40 px、距底部 ~30 px；JSON 里的 `width`/`height` 跟着内容实际占用范围调整，不要无脑沿用 960×600 默认值。
- **生成后必须人眼复核 PNG**：`rsvg-convert` 不会检测越界——viewBox 太小、legend 装不下、节点重叠都不会报错，但 PNG 里会出现"被裁掉一半的字"。每张图渲染完都要用 Read 工具看一眼，确认（a）底部 legend 整条都在画面内（b）节点之间无重叠（c）容器内的标签没有被里面的子节点压住（d）主图与 legend、注解之间没有大段空白（e）**所有文字都"看得清"——不是"勉强能辨认"**（f）**字号都达标、缩放到正文宽度后仍清晰**——没有 ≤ 12px 的小字、没有"得放大才认得出"的标签（字号基线见下方「文字字号不要太小」条）（g）换用 Noto 字体后**没有因字宽变化导致的新撑框 / 错位 / 截断**。
- **文字颜色不要太淡**：fireworks-tech-graph 默认模板里 `type_label_fill` 是 `#9ca3af`（gray-400）、`text_muted` / `section_sub_fill` 是 `#94a3b8`（slate-400）、`text_secondary` / `arrow_label_fill` / `legend_fill` 是 `#6b7280`（gray-500），这套配色在 Figma / mockup 里看着"高级灰"，但 GitHub PNG 缩放到正文宽度后字会糊。**最低对比度基线**：所有正文级别（节点 sublabel、axis label、legend、arrow label、caption / footnote）一律用 `#374151`（gray-700）或更深；带强调意味的 type_label / section_sub 用 `#334155`（slate-700）或更深；唯一允许 gray-600 / slate-600 量级的是**带边框的装饰性标签**（节点头上方的小 caps tag、有底色衬托的角标），因为有色块衬底会拉回对比。手写 SVG 时直接用上面这套色；调模板时如果发现还在用 `#9ca3af` / `#94a3b8` / `#6b7280` 这一类的 gray-400/500，**渲完一定要看一眼是否需要再调深**。**加粗文字另有一条"上限"**：加粗的中性黑灰字（font-weight ≥ 600 / bold，如标题、主节点 label）**灰度不要超过 gray-700**——即不比 `#374151` 更黑，最深就到 gray-700 为止；gray-900 `#111827`、gray-800 `#1f2937`、slate-800 `#1e293b` 这些更黑的中性色，**加粗时一律提到 `#374151`**，否则缩到正文宽度后大块加粗字显得过重、发死。这条只管"上限"、且**只针对中性黑灰**：上面那条"最低对比度基线"管的是**非加粗文字**，它仍照旧（gray-700 或更深，允许更黑来拉对比）；**饱和强调色**（深蓝 `#1e3a8a`、深红 `#991b1b`、深紫 `#5b21b6` 等语义色）不受这条约束，照常可用；slate-700 `#334155` 与 gray-700 亮度基本相等，视作达标、不必再动。
- **文字字号不要太小**：PNG 会被 GitHub / PDF 缩放到正文宽度显示，SVG 里太小的字缩完就糊成"勉强能辨认"。**最低字号基线（SVG 坐标系下）**：正文级文字（节点 sublabel、axis label、legend、arrow label、caption、公式行）**≥ 14px**；最小的辅助文字（角标 tag、坐标刻度这类）**≥ 13px**、**绝不低于 13px**；标题、主节点 label 要明显更大（标题 ~24px、主 label ≥ 15px）。配合默认 `-w 2400` 的导出宽度，缩放后才看得清。手写 SVG / 调脚本时若发现有 ≤ 12px 的字，**渲完务必看一眼是否要调大**；调大字号后注意检查有没有撑框，必要时把对应的框 / 间距一起放大（**只动尺寸、不改文字内容**）。
- **几个生成器常见的"看不见的坑"**（都来自 fireworks-tech-graph 的 `generate-from-template.py`）：
    - **legend 是纵向布局**，每条占 22 px。viewBox 高度必须 ≥ `legend_y + 22 × N + 20`，否则末尾几条会被静默裁掉（最容易踩）。
    - **container 的 `subtitle` 字段会与内部节点重叠**——容器副标题画在节点位置上方一点，节点稍矮就盖住它。要么不写 `subtitle`、要么把 `container.height` 拉大 ~30 px 给副标题让位。
    - **container 内节点的 `type_label` 也是上压**，节点本身要至少 60 px 高才装得下「TYPE_LABEL + label」两行；放 50 px 高的节点会被 type_label 遮住。
    - **手写 SVG（不走模板）时**底部 legend / caption 别紧贴 H 边——`pngquant` 偶尔会让边缘 1–2 px 颜色塌陷，看起来像被裁。统一留 ≥ 18 px。
- **文件按章节分目录存放在仓库根目录的 `assets/<NN>/`**（如 `assets/01/`、`assets/P05/`）：每张图同时提交 `.svg`（源文件，便于以后微调）和 `.png`（实际引用的位图）。文件名不再带章节前缀（目录已表达），改为 `用途.svg` / `用途.png`，例如 `assets/01/stack.png`、`assets/02/generate-pipeline.png`、`assets/P05/v-q-tree.png`。如果某章配图需要 build 脚本，统一放在同一子目录下，命名 `build_diagrams.py`。
- **在 `.md` 中通过相对路径引用 PNG**：章节 md 现在位于 `src/`、`assets/` 在仓库根，写 `![alt 文本](../assets/<NN>/用途.png)`。GitHub 上 PNG 渲染最稳定；`.svg` 在 GitHub README 里的 inline 渲染对外部字体不友好，所以默认引用 PNG。
- **不替换原有 ASCII 框图**：现有 `.md` 里的 ASCII 流程图保留（适合监控终端、纯文本 diff、复制粘贴），新生成的 SVG/PNG 作为视觉辅助插在 ASCII 之后，让两种形态各自发挥所长。
- **新增/修改章节时主动配图**：每章在「架构总览」「流水线/工作流」「关键算法步骤」这类抽象概念处至少配 1 张图；超长章节（>500 行）建议 3–5 张分散在不同小节。

### 公式的 GitHub 渲染避坑

GitHub 用 MathJax 渲染数学公式，但 markdown 解析先于 MathJax，下列写法已踩过坑：

- **`$$` 块级公式独占一行，前后各留空行**——否则部分 GitHub 渲染路径会吞掉它。
- **多行公式（matrix / cases / aligned）的行分隔符用 `\cr`，不要用 `\\`**——GitHub 的「markdown → MathJax」是两段管线，CommonMark 反转义会把 `$$` 里的 `\\` 还原成**单个** `\`（变成一个控制空格），于是 `\begin{pmatrix}…\\…\end{pmatrix}` / `\begin{cases}…\\…\end{cases}` 的换行失效、整块**塌成一行**（实测 MathML 只剩一个 `<mtr>`、原本两行被并进相邻单元格）。坑在于本仓库的 KaTeX（PDF）管线**不**反转义 `\\`、照常换行，所以 PDF 看着没问题、GitHub 网页却是错的——两边不一致最难发现。三种写法实测：`\\` → KaTeX 2 行 / GitHub **1 行**；`\\\\` → KaTeX **3 行**（多一空行）/ GitHub 2 行；**`\cr` → 两边都 2 行**。`\cr` 是「反斜杠 + 字母」命令，CommonMark 不碰它，KaTeX 与 MathJax 3 都渲成正确的行分隔，是唯一跨两条管线都对的写法。`check-github-render` skill 会扫 `$$`/`$` 内的 `\\` 并报 `backslash-rowbreak-eaten`。
- **多行矩阵 / `cases` / `aligned` 一律放进 `$$` 块级，不要塞进行内 `$...$`**——`$...$` 在 GitHub 上按 inline/text style 渲染，里头的 `\begin{pmatrix}…\end{pmatrix}`（或 `cases` / `aligned` / `array`，乃至裸 `\cr` / `\\` 行分隔符）经常**塌行、错位、或干脆不堆叠成矩阵**；同一段 TeX 换进独占一行的 `$$` 块级（display style）就正常。最隐蔽的是：本仓库 `check-github-render --visual` 出的 contact-sheet 是**服务端强制 display style** 渲染的，所以预览 PDF 上行内矩阵看着是对的、GitHub 网页上却塌掉——靠那份 PDF 复核**发现不了**这个坑（要么肉眼上 GitHub 看，要么靠下面的静态规则）。所以但凡是多行的东西，写法上就**只放 `$$` 块**：需要在行文中间插一个小矩阵时，把它提成独立的 `$$`（前后空行），别图省事写进行内 `$...$`。`check-github-render` 会扫行内 `$...$` 内的 `\begin{pmatrix|cases|aligned|array}` / `\cr` / `\\` 并报 `multirow-in-inline-math`（`$$` 块级豁免）。
- **`$...$` 行内公式与中文之间留半角空格**（如「其中 $\ell_i$ 是 logit」），紧贴中文易识别失败。**这条对中文标点同样适用**：`，。；：）（——「」` 等紧贴 `$` 也会让 MathJax 识别失败。例如 `差 1，$\exp$ 之后` 的逗号紧贴 `$`，会让 `$\exp$` 渲染成字面字符；正确写法是 `差 1， $\exp$ 之后` 或干脆把逗号挪开。同理 `（temperature）$T$ 后是` → `（temperature） $T$ 后是`，`$\exp(\ell_i)$——把` → `$\exp(\ell_i)$ ——把`。规则要在 `$` **两侧**都成立。
- **`\left\{ ... \right\}` 在 MathJax 3 上能正常渲染**——GitHub 现在用的是 MathJax 3，实测 `\left\{` 接受良好（早期 MathJax 2 才会报 `Missing or unrecognized delimiter for \left`，这条历史警告已经过时）。仍然推荐用普通 `\{ ... \}` 配 `\left/\right`：等价、更简洁、对个别旧渲染器也兼容（如 GitLab 的 KaTeX 配置、某些离线阅读器）。
- **`\text{}` 内不要放下划线（无论 `_` 还是 `\_` 都不行）**——GitHub markdown 在 `$$` 块内仍会把 `\_` 还原为 `_`，而 MathJax 的 `\text{}` 不接受裸 `_`，会报 `'_' allowed only in math mode`。下划线分隔的标识符改用连字符（`\text{next-token}`）或拆开成 `\text{next}\_\text{token}` 让 `_` 变成下标。
- **数组/列表里带显式正负号的数都要包成 `{+1.23}` / `{-1.23}`**——`,\ -1.23` / `,\ +1.23` 这种写法 MathJax 会把 `-` / `+` 当二元运算符，渲染成 `, − 1.23,` / `, + 1.23,`（前后自动加空格、看起来像断成两段）。用 `{}` 包起来让符号变一元号即可：`,\ {-1.23},` / `,\ {+1.23},`。**这条对正号和负号对称生效**——只要写出显式 `+` 或 `-`、就一律用 `{}` 包；数本身是无符号正数（`0.42`、`1.0`）则不用管。容易漏的地方：交替梯度 / 振荡序列里写出的 `(0.1,\ +1.0), (0.1,\ -1.0)`，正号那侧最容易忘。
- **不要用 `\,` / `\!` / `\;` / `\>` 这类"反斜杠 + ASCII 标点"的间距宏**——CommonMark 的 backslash-escape 会把它们还原成裸标点（`\,`→`,`、`\!`→`!`、`\;`→`;`、`\>`→`>`），实测在 `$$` 块内也会发生。结果 `[\,0.42` 渲染成 `[, 0.42`、`\mathbb{E}_\tau\!\left[...\right]` 在 `\tau` 和左方括号中间会冒出一个字面 `!`。要细空格用 `\thinspace` / `\negthinspace` / `\quad` / `\qquad` 这种"反斜杠 + 字母"的命名宏；**不要直接依赖默认间距**——默认间距对 CJK 混排来说普遍偏紧，相邻符号容易被挤成一团，作者想要的"轻轻挪一格"那种留白必须显式给。
- **双竖线 ‖ 一律写 `\Vert`，不要用 `\|`**——同样是 CommonMark backslash-escape 的坑：`\|` 在 markdown 解析阶段会被还原成裸 `|`，MathJax 拿到的是单线，所以 `$D_{\text{KL}}(p \| q)$` **在 GitHub 上渲染成单竖线**（`D_KL(p|q)`），与 KL 散度的双竖线约定不符；而本仓库 KaTeX 管线的 `unescape_math_in_md()` 只反转义 `\_` / `\*` / `\$`，不碰 `\|`，所以同一份 md 在 **PDF 里渲染成双竖线**——两套渲染结果不一致，肉眼一看就出戏。`\Vert` 是多字母命令、CommonMark 不会去转义，两条管线都稳定渲染成 `‖`。同理 L2 范数 `\|\theta\|^2` 也写成 `\Vert\theta\Vert^2`。KL 散度的双竖线两侧建议各加一个 `\thinspace`（`p \thinspace\Vert\thinspace q`）让分布符号与竖线之间有点呼吸空间；范数包单变量则不加（`\Vert\theta\Vert` 紧贴更像"模长"的视觉单元）。
- **不等号 ≠ 在源码里一律写 `\ne`，不要写 `\mathrel{\char"2260}` / `\neq` / `\not=` / 字面 `≠` / `\unicode{x2260}`**——`\ne` 是唯一同时被两条管线接受的写法（细节见下）。本仓库这块踩过坑：GitHub 用 MathJax，本地 PDF 用 KaTeX → MathML → WeasyPrint，两个引擎对 ≠ 的"原生"支持完全错位——`\ne` 在 MathJax 上渲得很干净，但 KaTeX 把它实现成「斜杠 `\rlap` 叠在 `=` 上」，WeasyPrint 渲染这个叠加结构时斜杠和等号会错位（斜杠漏到左边、`=` 被甩开甚至整个丢掉，求和/积分的缩小下标 scriptstyle 里 `=0` 会掉到下一行）；反过来 `\char"2260` 在 KaTeX 上拿到干净的整字 U+2260，但 MathJax 3 根本不识别 `\char` 命令，会把它当字面文本渲成红字 `\char"2260`；字面 `≠` / `\unicode{x2260}` 则反过来——MathJax 行，KaTeX 报 "No character metrics" 直接看不见。**解决方案**：在 `markdown-to-pdf` 管线里加了一道源码→KaTeX 的自动改写，把 `\ne` / `\neq` 替换成 `\mathrel{\char"2260}` 再喂给 KaTeX，从此 PDF 拿到的是字形 ≠、GitHub 拿到的是原始 `\ne`，两边都对。所以作者**只管在源文件里写 `\ne`**，不需要去关心 `\char` 这层细节（也千万别手写 `\char`——那会立刻坏掉 GitHub 渲染）。`check-github-render` 的 `ne-non-canonical` 规则会扫描源码里上述各种非规范 ≠ 写法，命中后提示统一回 `\ne`。
- **数学公式里的方括号 `[ ]` 内侧统一加 `\thinspace`**——`[2.0, -1.0, 0.5]`、`\mathbb{E}\left[ R(\tau) \right]`、`\big[ y \log \sigma(z) + ... \big]` 这类形态默认情况下首/末元素会紧贴方括号渲成 `[2.0,...,0.5]`、`E[R(τ)]`，视觉上挤成一坨；统一在 `[` 右侧和 `]` 左侧各加一个 `\thinspace`，让首/末元素与方括号之间留一点呼吸空间，渲成 `[ 2.0, ..., 0.5 ]`、`E[ R(τ) ]`。这条对所有 `[`/`]` 变体都适用：裸 `[`、`\big[` / `\Big[` / `\bigg[` / `\Bigg[`、`\left[ ... \right]`。空括号 `[]`、`[]` 内只有一项极短符号（如计数标识）的极少数情况可以例外。**注意**这条规则只针对 `[ ]`——圆括号 `( )` 在数学里通常表示函数参数 / 优先级分组，紧贴是惯例，不要加 `\thinspace`（写成 `f( x )` / `(a + b)` 反而违反直觉）。
- **表格单元格内的公式不能含裸 `|`**——会被当列分隔符吞掉，改用 `\mid` 或挪出表格放块级。
- **下标 / 上标里出现 `<` 或 `>` 要写成 `\lt` / `\gt`**——`x_{<t}` 这种写法（NLP 里"前 t 个 token"的标准记号）GitHub 渲染会报 `Extra open brace or missing close brace`。原因是 markdown 解析先于 MathJax，`<t}` 被当作疑似 HTML 标签处理，干扰了花括号配对。改写成 `x_{\lt t}` / `x_{\gt t}` 即可；另一种写法是直接换成 `x_{1:t-1}` 完全避开 `<`。
- **inline `$...$` 里出现 `}_<char>` 这种"右花括号紧跟下标"的形态时（下标可以是字母、数字或 `{`，如 `}_V`、`}_t`、`}_0`、`}_{pos}`），下面两种场景要把这个 `_` 转义成 `\_`**——原因是 CommonMark 把 `}` 视作标点：`}_<char>` 刚好同时满足"左 flanking"和"前接标点可开启 emphasis"两个条件，于是这个 `_` 变成一个 emphasis 起始分隔符；只要再找到一个配对的 `_` 让 italics 真的成型，整段（或整行）就 fallback 成纯文本、所有下标全没。配对的 `_` 来自两处，对应两种触发场景：
  - **场景 (a)：同一段 inline math 内含方括号 `[ ]`**——`[` / `]` 内部往往还有别的 `_`（或本身被 markdown 当强调上下文），给这个 `}_` 提供了配对，单段内部就成型。**典型踩坑**：`$\mathcal{L}_V(\phi) = \mathbb{E}\left[(V_\phi(s_t) - G_t)^2\right]$` 里第一个 `_V`（前面是 `}`、后面是大写字母）就是触发点——渲染成 `\mathcal{L}V(\phi) = \mathbb{E}\left[(V\phi(s_t) - G_t)^2\right]`，下标全被吃掉；把首个 `_V` 写成 `\_V` 即可恢复（剩下的 `_\phi`、`_t` 不强制转义）。
  - **场景 (b)：同一行里有 ≥2 段独立的 inline `$...$`、且其中 ≥2 段各带一个 `}_<char>` 下标**——这时一段里的 `}_` 当起始分隔符、另一段里的 `}_` 当闭合分隔符，**跨段配对**，中间夹的中文连同两端 `$` 一起被吞，整行所有下标全没（**不需要任何方括号**）。典型踩坑：`当中心词时用 $\mathbf{v}_c$ 、当上下文词时用 $\mathbf{u}_o$` —— `}_c` 和 `}_o` 跨段配对，渲染成 `\mathbf{v}c`、`\mathbf{u}o`。**修法**：把这一行里每一个 `}_<char>` 的 `_` 都转义成 `\_`（`$\mathbf{v}\_c$`、`$\mathbf{u}\_o$`）。
- **其余形态都不必转义**：单独一段、且这一行只此一段带 `}_` 下标的简单 inline math（如整行只有一个 `$V_\phi$`、`$\pi_\theta$`，或单段 `$\mathbf{x}_i^\top \mathbf{x}_j$` 不含 `[` 也不和别段配对）实测都能正确渲染；下标前缀是反斜杠或字母的（`\nabla_\theta`、`\sum_a`、`G_t`、`s_t`、`\pi_\theta`——`_` 前面不是 `}` 而是字母，属 intraword，CommonMark 不当分隔符）也都安全。**实在不放心 / 公式特别长**：把公式抽到 `$$...$$` 块级（独占一行、前后空行）也能绕开，因为 markdown emphasis 不跨越块级数学边界。`$$` 块级公式则一直没这个问题——里头的 `_` 不需要转义。**这两种场景 `check-github-render` skill 都会报 `emphasis-eats-subscript`**（场景 b 报"N separate inline-math spans on this line…"），改前先跑一遍省得靠肉眼。

### 非公式（markdown 层）的 GitHub 渲染避坑

下面这些坑与数学无关，纯粹是 GitHub（cmark-gfm）/ 本仓库 markdown-it-py 解析 markdown 行内语法时的规则，CJK 混排里尤其容易踩：

- **`**粗体**` / `*斜体*` 紧贴全角标点会触发 CommonMark flanking 失效**——最常踩的中文场景：粗体以全角标点（`）`「」』】 等）收尾、紧接着又是中日韩文字，如 `**输出层（lm_head）**都变大`、`**subword（子词）**在词表…`。此时闭合的 `**` 前接标点、后接 CJK 字母，规则「闭合分隔符前接标点时须后接空白或标点」不成立 → `**` 不构成合法闭合、被原样当字面输出（GitHub cmark-gfm 与本仓库 markdown-it-py 管线都如此）；对称地，起始 `**` 若"后接标点且被前面字母紧贴"也会失效。**修法**：把括注 / 标点移到强调之外，让强调收在字母或汉字上——`**输出层**（lm_head）都变大`、`**subword**（子词）在词表…`。**排查**（逐行用 CommonMark 渲染，看 `**` 是否残留到正文，按三反引号跳过代码块）：

      python3 - <<'PY'
      import re, pathlib
      from markdown_it import MarkdownIt
      md = MarkdownIt('commonmark')
      for p in sorted(pathlib.Path('src').glob('*.md')):
          inc = False
          for i, l in enumerate(p.read_text().split('\n'), 1):
              if l.strip().startswith('```'): inc = not inc; continue
              if inc or '**' not in re.sub(r'`[^`]*`', '', l): continue
              if '**' in re.sub(r'<code>.*?</code>', '', md.renderInline(l)):
                  print(f"{p}:{i}: {l.strip()}")
      PY
- **同一行、或表格同一单元格内，不要出现 2 个以上裸 `~`**——GitHub 用的 cmark-gfm 的 strikethrough 扩展接受**单个** `~` 作为分隔符（不只是规范里的 `~~`），同一作用域里偶数个裸 `~` 会两两配对、把中间内容渲成中划线，奇数个则前两两配对、最后一个被吞。作用域默认是整行，但**表格里收窄到单元格**——单元格被 `|` 隔开、各自独立解析，所以 `| ~2 | ~50 小时 |` 这种每格只有一个 `~` 的写法是安全的，不会跨格配对。典型踩坑是在同一作用域里用 `~` 当范围号（`2~4 分钟`、`7B ~ 70B`、`0.1 ~ 0.3`）——`2~4` 会被吃成 `24`、整段加粗内容被划掉。**约定**：范围号一律用 en dash `–`（`2–4 分钟`、`7B–70B`、`0.1–0.3`），近似值用 `≈N`（`≈50 小时`）；**孤立单 `~`（整行 / 整格只此一个，如 `~16 GB`）可以保留**。`~~text~~`（真的想划掉）不在此列。**排查**：下面这段（跳过 inline code 与 `~~`，表格行按 `|` 拆单元格）应无输出：

      python3 <<'PY'
      import re, pathlib
      for p in sorted(pathlib.Path('src').glob('*.md')):
          for i, l in enumerate(p.read_text().splitlines(), 1):
              s = re.sub(r'`[^`]*`', '', l).replace('~~', '')
              cells = s.split('|') if '|' in s else [s]
              if any(c.count('~') >= 2 for c in cells):
                  print(f"{p}:{i}: {l}")
      PY
- **list bullet 里不要直接套三反引号围栏**——markdown-it 在「围栏缩进恰好等于 list 内容列」的边界下会把三反引号当 inline code span 分隔符，整段压成 `<p><code>...</code></p>`，PDF 里换行全没。改用 **6 空格缩进的 indented code block**（= list 内容列 2 + 标准缩进 4），CommonMark 严格保证识别为 `<pre><code>`。**排查命令**：

      grep -nP '<p><code>\s*$' dist/*.html   # 看渲染产物：被误识别成行内 code span 的位置
      grep -nP '^ +```' src/*.md             # 看源文件：list 内非零缩进的三反引号

### matplotlib 图标题/坐标轴避坑：一律用英文

Colab 默认运行时的 matplotlib 字体是 `DejaVu Sans`，**不包含中文字形**。在 ipynb 里写 `plt.title('训练 loss 曲线')` 这种含中文的字符串，PNG 输出会把每个汉字渲染成乱码（□），同时控制台刷一堆 `UserWarning: Glyph XXXXX missing from font(s) DejaVu Sans` 警告。

约束：

- **所有 matplotlib 字符串参数（`title`/`xlabel`/`ylabel`/`suptitle`/`label`/`text`/`annotate` 等）一律用英文**。技术名词本来就英文（loss、accuracy、epoch、warmup、lr...），改起来通常一句直译就够。
- 这条**只针对 matplotlib 渲染出的字符**——`print(...)` 输出、cell 注释、markdown 文档都不受限，照常用中文。
- 不要靠 "在每个 ipynb 顶部装 `fonts-noto-cjk` + 设 `rcParams`" 来绕过：会让每章 ipynb 多一个 `apt-get install` cell（违背 "Run All 即可" 的体验），且字体安装失败时 fallback 还是乱码。**统一用英文是最稳的做法**。
- 新增/修改章节后，扫一眼有没有违规：`grep -nP "(plt\.title|set_title|plt\.xlabel|plt\.ylabel|set_xlabel|set_ylabel|suptitle|axvline.*label|axhline.*label).*[\x{4e00}-\x{9fff}]" *.ipynb` 应当无输出。

### 目录锚点（TOC anchor）的 GitHub slug 规则

每章顶部 TOC 里写 `[3.4 Top-p（nucleus）采样](#34-top-pnucleus采样)` 这种锚点链接，`#...` 后面的串必须与 GitHub 给该标题自动生成的 slug **逐字一致**——否则点击 TOC 跳不过去。GitHub 的 slug 规则有两条容易踩坑：

- **ASCII 强制小写**：标题里写 `LLM`、slug 里就是 `llm`。`## 九、warmup + cosine：LLM 训练的默认调度配方` 的锚点是 `#九warmup--cosinellm-训练的默认调度配方`（不是 `...LLM-训练...`）。
- **CJK 标点不当作分隔符、前后不插连字符**：`（` `）` `：` `，` `。` 这类全角标点会被**直接删除**，前后的字符**贴在一起**，**不**会变成连字符。所以 `## 3.4 Top-p（nucleus）采样` 的 slug 是 `#34-top-pnucleus采样`（`p`、`nucleus`、`采样` 三段紧贴），不是 `#34-top-pnucleus-采样`。
- 反过来 **ASCII 空格会变成连字符，`-` 保留为连字符；`+` 被剥离、不贡献连字符**。连续多个分隔符不合并，会塌成多个连字符保留：`warmup + cosine` 在 slug 里是 `warmup--cosine`（两个连字符——加号被剥离，前后两个空格各出一个 hyphen）。

写新章节 TOC 时不靠手算，最稳的做法是**先把章节推到一个本地分支或草稿 PR，去 GitHub 网页打开看一眼自动生成的 heading 锚点是什么**（GitHub 渲染后悬停标题左侧会显示 `#...` 链接图标，复制即可），再回填到 TOC。markdown-to-pdf 的 PDF 走的是 `markdown-it-py` 的 `anchor` 插件、按 GitHub slug 规则生成，理论上一致——但 GitHub 那边是 ground truth，最终以网页表现为准。

## Git 工作流

**仅当通过 Claude Code on the web 开发时**：沿用其默认流程，每个 session 在 harness 分配的 `claude/...` 工作分支上开发与推送，由用户自行决定何时合并到 `main`。

通过本地 Claude Code CLI 或其他方式开发时不受此约束，按你自己的分支策略来即可。

Claude 代为提交时的 commit 规范：
- **Author** 设为 `weiqiang <weiqiangnd@gmail.com>`（用 `git commit --author=...`，不要修改全局 git config）
- commit message 末尾加一行 `Co-Authored-By: Claude <模型名> <noreply@anthropic.com>`，**模型名填当前 session 实际运行的 Claude 版本**（例：`Opus 4.7 (1M context)`、`Sonnet 4.6`），不要照抄旧 commit 的字符串
- 不要在 message 里贴 `https://claude.ai/code/session_...` 链接
