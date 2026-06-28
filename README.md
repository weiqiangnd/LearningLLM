# LearningLLM

## 仓库介绍

这是一份面向 0 基础大模型（LLM）学习者的教程与代码实例。每一章对应一个 Markdown 文档（理论与逐行代码讲解）和一个 Jupyter Notebook（可直接运行的示例），两者**互为对照**——`.md` 回答「为什么这样做」，`.ipynb` 回答「代码每一行做什么」，公式与代码片段保持一致。所有示例默认在 **Google Colab** 中运行（免费 T4 GPU 即可起步）。

仓库特色：

- **`.md` + `.ipynb` 双轨并行**：每章一份概念讲解、一份可直接 Run All 的代码，互为对照、术语一致。
- **部分章节附「实战代码详解」**：代码量较大的章节额外提供一份 `NN-附录-实战代码详细讲解.md`，按 cell 功能块逐行拆解对应 ipynb——讲清每段代码做什么、入参 / 出参 / 中间变量及张量形状怎么变，适合不熟悉 Python / PyTorch 的读者对照阅读；入口在学习路径表对应章节的「链接」列。
- **示例模型为 Qwen3-8B**：默认在 Colab 免费 T4（15 GB）上用 4-bit 量化跑通 Qwen3-8B；显存吃紧的进阶章节才升 L4 / A100。
- **概念不跳步**：读者基线只要求「会 Python + 矩阵乘法」，softmax / KL / MDP 等首次出现都先定义再使用，并配最小可手算的数值例子（2×2 矩阵、长度为 3 的序列）。
- **数学公式先建立直觉再严格证明**：涉及复杂数学的地方，先用例子 / 图把直觉讲清，再给出严格的定义与推导证明——既不让公式吓退初学者，也不牺牲该有的严谨。
- **张量形状变换全程标注**：reshape / transpose / broadcasting 每一步都给张量形状，如 `(B, L, D) → (B, L, H, D/H) → (B, H, L, D/H)`。
- **关键部分配图辅助理解**：架构总览、流水线 / 工作流、关键算法步骤等抽象处都配示意图，与文字、公式互为补充。
- **预备知识按需回查**：阶段 0 的 P0N 章节与主线 NN 解耦，无需一次读完——主线在首次用到时会用〔预备知识〕标注提醒。

## 学习路径

本仓库面向**完全 0 基础**的中文学习者，目标是从「不会 PyTorch」走到「能读懂主流 LLM 论文、能跑通 SFT/RLHF/多模态训练、能搭 RAG/Agent 应用」。整条路径分为 **0 ~ 9 共 10 个阶段**，越靠后越偏工程与应用。

硬件方面：默认 Colab 免费版 **T4** 起步；标注 ⚡ 的章节需要 Colab Pro 的 **L4 / A100**（如分布式、长上下文训练、SFT/RLHF 全流程、多模态训练）。每章 `.md` 顶部会显式标注本章硬件门槛，读者无需 Pro 也能完整阅读 `.md` 理解原理。

每章状态列：✅ 表示已完成（点击「链接」列即可阅读文档 / 打开 Notebook / 在 Colab 中运行）；空白表示尚未撰写。

### 阶段 0：预备知识（Prerequisites）

读者基线：会写 Python、知道矩阵乘法。其余概念（autograd、softmax、KL 散度、SGD、MDP）从这里补齐。**预备知识章节用 `P0N` 编号**，与主线 `NN` 解耦。

**预备知识章节并不要求一次读完**：建议读者在主线**首次用到**对应背景知识时再回头补齐——下方各阶段会在这些"首次出现点"用 `〔预备知识〕` 标注，已经熟悉的可跳过。

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| P01 PyTorch 与张量：shape / dtype / device / broadcasting / autograd | 张量四个核心属性（shape / dtype / device / requires_grad）、张量创建与形状操作（reshape / transpose / permute / squeeze）、broadcasting 规则、`@` 与 `*` 的区别、autograd 计算图与 `.backward()`、`torch.no_grad` / `detach`、梯度累加与 `zero_grad()`，配合最小可手算的数值例子（含用 autograd 验证 sigmoid 导数恒等式）。 | [文档](./src/P01-PyTorch与张量.md) · [ipynb](./src/P01.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P01.ipynb) | ✅ |
| P02 神经网络最小闭环：手写一个 MLP + 训练循环（前向、反向、优化器三步） | MLP 数学定义与"为什么必须有非线性激活"、用 `nn.Module` 组织模型、`nn.Linear` 内部 $y = xW^\top + b$ 的形状约定、损失函数（MSE / BCE / CrossEntropy）与优化器（SGD / AdamW）选型、训练循环三步「forward → backward → step」+ `zero_grad`，并在 `make_moons` 数据上完整训练 MLP（含 loss 曲线、决策边界可视化），用一个去掉 ReLU 的反例验证"非线性激活不可省略"。 | [文档](./src/P02-神经网络最小闭环.md) · [ipynb](./src/P02.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P02.ipynb) | ✅ |
| P03 概率与信息论够用版：softmax、log-likelihood、熵、交叉熵、KL、MLE | 从 logits → softmax → 概率分布的完整链路（含数值稳定的 log-softmax 与 logsumexp）、温度对分布尖锐 / 扁平的影响、似然 vs 概率、log-likelihood 与 MLE、熵 $H(p)$ 与"分布有多确定"、交叉熵 $H(p,q)$ 与负对数似然 NLL 在 one-hot 标签下的等价性、`F.cross_entropy` 接收 raw logits 的工程铁律（含"传 softmax 进去会出什么错"反例）、KL 散度的定义与不对称性可视化（forward KL vs reverse KL），最后用一个 5 类分类器验证"训练 = 最小化交叉熵 = 等价 MLE"。 | [文档](./src/P03-概率与信息论够用版.md) · [ipynb](./src/P03.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P03.ipynb) | ✅ |
| P04 优化器与学习率调度：SGD / Momentum / Adam / AdamW；warmup + cosine | 从 SGD → Momentum → Adam → AdamW 一路推演每一步的修正动机（"长椭圆山谷"上的优化轨迹对比图）、Adam 的一阶矩 / 二阶矩 / bias correction（手写 Adam 与 PyTorch 内置一致性验证）、AdamW 的 weight decay 解耦、lr 是最敏感超参（lr 太大 / 适中 / 太小三条 loss 曲线对比）、warmup 缓解 Adam 早期方差估计不准与"大初始化 + 大 lr"导致的 NaN、cosine 退火的形状与下界设置、`warmup + cosine` 这套 LLM 训练事实默认调度的完整代码模板（含 `LambdaLR` 与 `optimizer.step()` / `scheduler.step()` 的调用顺序）。 | [文档](./src/P04-优化器与学习率调度.md) · [ipynb](./src/P04.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P04.ipynb) | ✅ |
| P05 强化学习够用版：MDP、policy / value、REINFORCE、policy gradient（为后续 PPO/GRPO 打底） | MDP 五元组与马尔可夫性、trajectory 与累计折扣回报 $G_t$ 、策略 $\pi(a \mid s)$ 与状态价值 $V^\pi$ / 动作价值 $Q^\pi$ / 优势 $A^\pi$ 的关系、目标函数 $J(\theta) = \mathbb{E}\_\tau[R(\tau)]$ 与 policy gradient 定理（"对 log 概率求导乘上回报"的直觉解读）、REINFORCE 算法、加 baseline 降方差的来历，以及把 RL 语言映射到 LLM 对齐（状态 = prompt + 已生成 token、动作 = 下一个 token、奖励来自 RM / 可验证规则、KL 约束防止跑偏）。实战在 `gymnasium` 的 CartPole-v1 上从随机策略 baseline 跑到 REINFORCE 接近满分 500，并对比加 / 不加 baseline 的曲线方差；最后用一个迷你"LLM 风格" REINFORCE 闭环把 PPO / GRPO 的核心思想用 50 行代码验证一遍。 | [文档](./src/P05-强化学习够用版.md) · [ipynb](./src/P05.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P05.ipynb) | ✅ |

### 阶段 1：把大模型用起来

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 01 环境与工具：IPython / Jupyter / Colab + 在 T4 上跑通 Qwen3-8B | 介绍 IPython / Jupyter / Colab 的关系与使用，讨论 Colab 上的 GPU 选型与大模型选型，并在 T4 上用 4-bit 量化跑通 Qwen3-8B 的对话生成。 | [文档](./src/01-IPython-Jupyter-Colab入门.md) · [ipynb](./src/01.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/01.ipynb) | ✅ |
| 02 生成参数与采样策略：logits / softmax / temperature / top-p/top-k / 思考模式 | 拆解 `generate()` 的完整工作流（logits → temperature → top-k / top-p → softmax → 采样），逐个讲解 `do_sample` / `temperature` / `top_p` / `top_k` / `repetition_penalty` 的作用与数学定义，并通过 6 组对比实验（贪心 vs 采样、不同 temperature、不同 top_p、是否抑制重复、思考模式 vs 非思考模式、复现性）让读者直观理解每个旋钮的效果。 | [文档](./src/02-生成参数与采样策略.md) · [ipynb](./src/02.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/02.ipynb) | ✅ |

〔预备知识〕02 章首次用到 softmax 与概率分布——若不熟悉，建议先读 P03。

### 阶段 2：Transformer 架构精讲

目标：从 token 进入模型到 logits 输出，每一层张量怎么变换、每个组件为什么这样设计，最后能从零实现一个 mini-GPT。

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 03 Tokenizer：BPE / BBPE / 词表 / 特殊 token；用 tokenizers 库训一个小 tokenizer | 从「模型只认识整数 id」讲清 tokenizer 在流水线两端的位置，对比 char / word / subword 三种切分粒度的取舍（词表大小 vs 序列长度 vs OOV），用最小可手算的 hug/pug 例子推导 BPE 的训练（统计高频相邻对 → 合并 → 记规则）与编码（按学习顺序套规则、生僻词也能切），讲解 BBPE 如何用「先转 UTF-8 字节 + 256 字节打底」根除 OOV，以及词表大小权衡与特殊 token 不被 BPE 拆开的两条铁律；实战加载 Qwen3 tokenizer 观察中英文 / 代码 / emoji 的切法、纯 Python 手写 BPE 跑通算法、用 tokenizers 库从零训一个 byte-level BPE 验证「永不 OOV + 完美还原」，并量化中英文的 token 压缩率差异。 | [文档](./src/03-Tokenizer.md) · [ipynb](./src/03.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/03.ipynb) | ✅ |
| 04 Embedding 与位置编码：token embedding / word2vec / sinusoidal / learned / RoPE / ALiBi | token embedding 本质是「one-hot 乘矩阵 = 按行查表」，embedding 矩阵与 lm_head 互为转置可共享权重（weight tying）；借 word2vec（分布假说、CBOW / skip-gram + 负采样）讲清 embedding 为何带语义结构、king − man + woman ≈ queen 为何成立，并厘清与 LLM embedding 的异同；论证自注意力对顺序「视而不见」（置换等变）故位置编码是刚需，梳理「改输入（绝对）vs 改注意力（相对）」两条路线，逐个对比 sinusoidal / learned / RoPE / ALiBi 的原理与取舍；实战实现查表与 weight tying、训 word2vec 验证类比、验证置换等变性、画 sinusoidal 热力图、验证 RoPE 相对位置、构造 ALiBi 偏置，并读 Qwen3 config 看真实 rope_theta。 | [文档](./src/04-Embedding与位置编码.md) · [ipynb](./src/04.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/04.ipynb) | ✅ |
| 05 从 RNN 到 attention：seq2seq、Bahdanau / Luong attention（"attention 为什么会出现"） | 从 RNN 的循环隐藏状态与长程依赖软肋讲起，搭出 seq2seq 的 encoder–decoder 骨架并点出它的信息瓶颈（整句话压进一个定长 context、源句越长越记不住），由此引出 attention 的「打分→softmax→加权求和」三步套路；对比 Bahdanau（加性）与 Luong（乘性点积）两大流派，并铺垫到 self-attention / 第 6 章。实战训一个带 attention 的 GRU seq2seq 做数字串反转，画出反对角线对齐矩阵，并用「准确率随长度变化」的对比验证瓶颈。 | [文档](./src/05-从RNN到attention.md) · [ipynb](./src/05.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/05.ipynb) · [实战代码详解](./src/05-附录-实战代码详细讲解.md) | ✅ |
| 06 Scaled Dot-Product Attention：Q/K/V、softmax、causal mask | 把第 5 章的 attention 三步套路平移到 self-attention（序列对自己做注意力，一步融合全局信息且天然可并行）：用「软字典检索」类比讲清 Q/K/V 三个角色及其线性投影来历，从点积方差推出 $\sqrt{d_k}$ 缩放为何能防 softmax 饱和，拼出完整公式 $\text{softmax}(QK^\top/\sqrt{d_k})V$ ，再讲 causal mask 如何把上三角置 $-\infty$ 实现自回归「不偷看未来」，并顺带交代 padding mask 与 $O(L^2)$ 复杂度。这套「全是矩阵乘 + 一个 softmax」的算子，正是 Transformer 里被调用最频繁、也最能在 GPU 上并行的核心。实战从可手算玩具例子起步，封装 `scaled_dot_product_attention` 并与官方实现对齐，可视化缩放与 mask 的效果，收尾拼成一个最小 `SelfAttention` 模块。 | [文档](./src/06-Scaled-Dot-Product-Attention.md) · [ipynb](./src/06.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/06.ipynb) | ✅ |
| 07 Multi-Head Attention 与 MQA / GQA：分头、拼接、形状变换全过程 | 把第 6 章的「一个头」并排成 $H$ 个：单头只有一个注意力分布，而加权求和本质上是平均、会抹平细节，多头则把 $d_{\text{model}}$ 切成 $H$ 个子空间各看一种关系；以「切头 → 各头独立 attention → 拼接 → 输出投影」四步拆解机制，全程标注 `[B,L,d_model] → [B,L,H,d_k] → [B,H,L,d_k] → … → [B,L,d_model]` 的 reshape / transpose 形状变换，并算清「四个投影参数与头数无关、多头计算量和一个大头持平」这笔账；再从推理 KV cache 瓶颈引出 MQA / GQA（query 头数恒为 $H$ 、只削减 K/V 头数， $G=H$ 即 MHA、 $G=1$ 即 MQA）。实战从零搭 `MultiHeadAttention`、与官方 `nn.MultiheadAttention` 对齐验证、可视化不同头的关注模式、改造成 `GroupedQueryAttention`，最后读 Qwen3-8B config 看到真实的 32 query 头 / 8 KV 头 GQA。 | [文档](./src/07-Multi-Head-Attention与MQA-GQA.md) · [ipynb](./src/07.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/07.ipynb) | ✅ |
| 08 FFN、残差连接、LayerNorm / RMSNorm / SwiGLU | 补全 Transformer block 的另一半：FFN（升维→激活→降维的逐位置前馈网络，position-wise、占一个 block 约 2/3 参数，补上注意力欠缺的逐 token 非线性），残差连接 $x+\text{Sublayer}(x)$ （反传时 $+1$ 直通项修出一条梯度高速公路、让几十层训得动，并使每层默认从恒等映射起步），归一化（为何用 LayerNorm 而非 BatchNorm、再到只除均方根不减均值的 RMSNorm），以及 SwiGLU（用 SiLU 门控升级 FFN，把 $d_{\text{ff}}$ 收到约 $\frac{8}{3}d_{\text{model}}$ 对齐参数量）；激活从 ReLU→GELU→SiLU。实战从零搭 FFN、用 40 层堆叠验证残差对底层梯度的救命作用、手写 LayerNorm / RMSNorm 并和官方对齐、实现 SwiGLU 算清参数账，最后拼出一个 Pre-LN + RMSNorm + SwiGLU 的完整 Transformer block 并读 Qwen3-8B config 看真实配置。 | [文档](./src/08-FFN-残差连接与归一化.md) · [ipynb](./src/08.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/08.ipynb) | ✅ |
| 09 Transformer 整体架构：Encoder-Decoder vs Decoder-only；Pre-LN vs Post-LN | 把第 3-8 章的零件总装成完整模型。先用一张演变时间线梳理语言模型从 n-gram → 神经语言模型 → word2vec → RNN/attention → Transformer → 大模型一路怎么走来，讲清为什么是 Transformer 赢了；再用一张全景图（以 Qwen3-8B 为标尺）讲清 Decoder-only 骨架 input_ids → embedding → N 个形状守恒的 block → final norm → lm_head 里每个组件的形状变换与参数量（整模型约 8.2B）；接着深入 Encoder-Decoder 两栈结构与 cross-attention，讲清为什么主流大模型都收敛到 Decoder-only，再算清 Pre-LN vs Post-LN 的取舍，并用残差流视角把整章串成一张图，最后交代这套骨架怎么放大、有哪些现代变体（QK-norm、长上下文 RoPE、MoE）。实战拼出 MiniDecoderLM、验证逐层形状守恒与参数账、用 24 层深模型验证 Pre-LN / Post-LN 的训练稳定性差异，并读 Qwen3-8B config 把全景图数字对上。 | [文档](./src/09-Transformer整体架构.md) · [ipynb](./src/09.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/09.ipynb) · [实战代码详解](./src/09-附录-实战代码详细讲解.md) | ✅ |
| 10 自回归语言建模目标、cross-entropy loss、teacher forcing 与推理采样的关系 | 用概率链式法则把「一句话的概率」精确分解成「逐词预测」，论证「让真实语料最可能」（MLE）= 最大化对数似然 = 最小化负对数似然 = 最小化 cross-entropy（同一目标的四个剖面）；逐位置拆 cross-entropy loss 并引出 perplexity；再讲 teacher forcing（训练喂真值 + 因果掩码让一句话所有位置一次并行算 loss）及其代价 exposure bias；最后把训练与推理统一到同一个分布 $P_\theta(\cdot\mid\text{前文})$ ——训练塑造它、推理从它串行采样，第 2 章的 greedy / temperature / top-p 都只是「怎么读这个分布」。实战在 Qwen3-8B 上手算交叉熵对齐 `F.cross_entropy`、算真实文本的自回归 loss 与 perplexity、验证「一次并行前向 = 逐前缀串行前向」、手写贪心自回归循环演示 exposure bias。 | [文档](./src/10-自回归语言建模目标.md) · [ipynb](./src/10.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/10.ipynb) | ✅ |
| 11 论文精读：Attention Is All You Need 逐节解读 |  |  |  |
| 12a 论文串读（上）：GPT 系列演进——GPT-1 → GPT-2 → GPT-3，decoder-only 路线如何确立 |  |  |  |
| 12b 论文串读（下）：LLaMA / Qwen 的现代改动逐项拆解（RoPE / RMSNorm / SwiGLU / GQA / MoE） |  |  |  |
| 13 从零实现 mini-GPT：tiny-shakespeare 上完成训练 → 生成闭环 |  |  |  |

〔预备知识〕04 章首次**密集**用到张量 shape / broadcasting / autograd——若不熟悉，建议先读 P01。

〔预备知识〕05 章实战首次走完「前向 + 反向 + 优化器」完整训练循环（训一个 seq2seq），**大量**用到优化器与训练三步——若不熟悉，建议先读 P02 与 P04。

〔预备知识〕10 章首次**密集**用到极大似然估计、熵 / 交叉熵（cross-entropy 是自回归语言建模的核心损失）——若不熟悉，建议先读 P03。

### 阶段 3：推理工程

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 14 KV cache：原理、显存占用估算、从零加上 KV cache |  |  |  |
| 15a 量化（上）：数值精度与量化基础——fp16 / bf16 / int8 / int4、对称 vs 非对称、per-tensor / per-channel |  |  |  |
| 15b 量化（下）：训练后量化算法——GPTQ / AWQ / bitsandbytes（NF4）原理与对比 |  |  |  |
| 16 批处理与 continuous batching；FlashAttention 直觉 |  |  |  |
| 17 推测解码（speculative / Medusa / EAGLE）概览 |  |  |  |
| 18 推理框架选型：vLLM / SGLang / llama.cpp / Ollama |  |  |  |
| 19 权重格式与生态：safetensors / GGUF / HuggingFace Hub 工作流 |  |  |  |

### 阶段 4：预训练全景

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 20 预训练数据：清洗、去重、配比、合成数据 |  |  |  |
| 21 Scaling law 与 Chinchilla：参数量 / 数据量 / 算力的最优配比 |  |  |  |
| 22 MoE 架构：Mixtral、DeepSeek-V3 的路由与负载均衡 |  |  |  |
| 23 长上下文：RoPE 外推、YaRN、sliding window、attention sink |  |  |  |
| 24a 分布式训练（上）：数据并行——DDP / FSDP / ZeRO 三级显存优化（用单卡 L4 跑最小 demo） ⚡ |  |  |  |
| 24b 分布式训练（下）：模型并行——张量并行（TP）/ 流水线并行（PP）与 3D 并行组合 ⚡ |  |  |  |
| 25 训练加速：混合精度、梯度累积、gradient checkpointing |  |  |  |
| 26 Continue pretraining 与领域自适应 ⚡ |  |  |  |

### 阶段 5：后训练与对齐

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 27 对话模板与 chat template：special token、role、tool 调用格式 |  |  |  |
| 28 SFT 原理：数据格式（Alpaca / ShareGPT）、loss masking、lr / epochs 选择 |  |  |  |
| 29 SFT 实战：在 L4/A100 上 LoRA-SFT Qwen3-8B（≈ 跑通一遍真实后训练流程） ⚡ |  |  |  |
| 30 奖励模型（RM）：偏好数据、Bradley-Terry、RM 训练与评估 ⚡ |  |  |  |
| 31a RLHF（上）：PPO 原理——从 policy gradient 到 PPO（clip、advantage、actor-critic）、KL 约束 |  |  |  |
| 31b RLHF（下）：工程实战——在 RM 上跑 PPO 全流程、reward hacking 与训练稳定性 ⚡ |  |  |  |
| 32a DPO 系列（上）：DPO 原理——从 RLHF 目标推导出 DPO 损失（无需 RM 的偏好对齐） ⚡ |  |  |  |
| 32b DPO 系列（下）：DPO 变体横向对比——IPO / KTO / SimPO 的动机与差异 ⚡ |  |  |  |
| 33 GRPO 与 RLVR（可验证奖励）：DeepSeek-R1 训练范式 ⚡ |  |  |  |
| 34 Reasoning 模型与 test-time compute：o1 / R1 思路、长 CoT、self-consistency |  |  |  |
| 35 Constitutional AI / RLAIF：用 AI 反馈替代人类标注 |  |  |  |
| 36 模型蒸馏：logit 蒸馏、SFT 蒸馏、reasoning 蒸馏 |  |  |  |

〔预备知识〕30 章首次**密集**用到「成对偏好上的最大似然」（Bradley-Terry 模型，本质是 sigmoid + log-likelihood）——若不熟悉，建议先读 P03。

〔预备知识〕31a 章**大量**用到 MDP / policy gradient / PPO（整章都建立在 policy gradient 之上），并首次把 **KL 散度**当作训练约束——若不熟悉，建议先读 P05（KL 散度见 P03）。

### 阶段 6：参数高效微调实战

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 37 LoRA 原理与从零手写：低秩分解、合并权重、scaling |  |  |  |
| 38 QLoRA：把 29 章的 LoRA-SFT 加上 4-bit 量化，在 T4 上跑通 Qwen3-8B 微调 |  |  |  |
| 39 Adapter / Prefix-Tuning / IA³ 速览 |  |  |  |
| 40 模型合并（Model Merging）：DARE / TIES / SLERP |  |  |  |

### 阶段 7：多模态大模型

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 41 多模态总览：模态对齐目标、CLIP 双塔、对比学习 |  |  |  |
| 42 视觉 encoder：ViT 与 SigLIP |  |  |  |
| 43 VLM 架构演进：LLaVA / Qwen-VL / InternVL 的连接器设计 |  |  |  |
| 44 多模态训练全流程：vision pretrain → 模态对齐 → 多模态 SFT ⚡ |  |  |  |
| 45 多模态偏好对齐：在 VLM 上做 DPO / RLHF ⚡ |  |  |  |
| 46 语音多模态：Whisper、端到端语音模型（Qwen-Audio 等） |  |  |  |
| 47 多模态评测与典型应用（OCR、文档问答、agent for GUI） |  |  |  |

〔预备知识〕41 章的对比学习（CLIP 的 InfoNCE）本质是相似度矩阵上的 softmax / 交叉熵——若不熟悉，建议先读 P03。

### 阶段 8：应用层

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 48 Prompt 工程与 CoT / few-shot / self-consistency |  |  |  |
| 49 嵌入模型与 reranker：训练原理、对比学习、选型 |  |  |  |
| 50a RAG（上）：检索基础——文档分块、embedding、向量库、相似度检索 |  |  |  |
| 50b RAG（下）：进阶与评测——reranker、查询改写 / HyDE、RAG 评测（faithfulness / 命中率） |  |  |  |
| 51 Function calling / Tool use / MCP |  |  |  |
| 52 Agent：ReAct、规划、多轮工具调用、记忆 |  |  |  |

### 阶段 9：评测、安全与生态

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 53 评测体系：perplexity、benchmark（MMLU / GSM8K / HumanEval）、LMSYS Arena、LLM-as-judge |  |  |  |
| 54 安全：jailbreak、prompt injection、对齐失效、红队测试 |  |  |  |
| 55 HuggingFace 生态全景：Transformers / Datasets / Accelerate / TRL / PEFT |  |  |  |

> 路径会随大模型生态演进调整。每完成一章，对应行的「链接」与「状态」列会同步更新。

## 推荐运行环境

| 项目 | 推荐 |
|------|------|
| 运行平台 | Google Colab（免费 T4 GPU 起步） |
| 编辑器 | VS Code + Google Colab 扩展 |
| Python | 3.10+ |
| 核心依赖 | `transformers` / `accelerate` / `bitsandbytes`（示例代码会自动安装） |

## 阅读建议

1. 按章节顺序阅读 `.md` 文档理解概念
2. 点击章节「链接」列里的 **OpenInColab** 直接打开对应 Notebook（或在 VS Code 中连接 Colab 运行时）
3. 文档与代码相互对照——`.md` 里有原理解析，`.ipynb` 注释里有逐行说明

> **关于数学公式渲染**：仓库里**每一章实际上有四份**，分工不同：
>
> - `src/<name>.md` —— **源文档**，优先保障 **GitHub 网页上的渲染效果**。GitHub 走「markdown 解析 → MathJax 渲染」两段管线，会先把 `\_` 等转义字符还原再交给 MathJax，所以 `$...$` / `$$...$$` 大多能正常显示。但本地 markdown 阅读器（Typora、VS Code 预览、Obsidian 等）多数走 KaTeX，且**先抽数学块再渲染**，少了 GitHub 的那一步反转义——结果 `\mathbb{E}\_\pi` 里的 `\_` 进 KaTeX 就变字面下划线，**下标视觉丢失**。所以 src 里的 md 适合在 GitHub 上读，**不建议在 KaTeX 类本地阅读器里打开**。
> - `dist/<name>.md` —— **KaTeX 兼容版**。除了 `$...$` 内的 `\_` `\*` `\$` 被反转义为 `_` `*` `$`，与 src 字节一致。VS Code 预览、Obsidian 等任何 KaTeX 前端都能直接渲染。**注意它在 GitHub 上反而不渲染数学**（少了反转义的下划线会被 emphasis 吃掉）——这是有意的分工。
> - `dist/<name>.html` —— **自包含 HTML**。KaTeX 服务端渲染、所有图片 base64 内嵌、字体 / 样式全打包，离线浏览器双击即看，跨平台一致。
> - `dist/<name>.pdf` —— 上面那份 HTML 经 WeasyPrint 出的 PDF，便于打印 / 离线归档。
>
> 这套 `dist/` 三件套由 `.claude/skills/markdown-to-pdf/` 一条命令从 src 重新生成；生成过程会做跨阶段的数学一致性校验（原始 md、KaTeX md、HTML 渲染三方对齐），任何异常都会阻止 PDF 生成，确保 dist 里看到的公式与作者写下的一致。本地阅读时按上面四份的"渲染场景"挑一份就好。

## 反馈与交流

欢迎通过 [Issue](https://github.com/weiqiangnd/LearningLLM/issues) 提问、讨论或纠错。

## License

本仓库采用**双协议**授权，方便代码与文字内容按各自最适合的方式被复用：

- **代码**（`.ipynb` 中的代码 cell、未来可能加入的脚本）采用 [MIT License](https://opensource.org/licenses/MIT)：任何人可自由使用、修改、再分发，仅需保留版权声明。全文见 [`LICENSE-MIT`](./LICENSE-MIT)。
- **文字与图片内容**（`.md` 文档、`assets/` 下的 SVG / PNG）采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)：可自由分享与改编，需署名（注明来源仓库）。全文见 [`LICENSE-CC-BY-4.0`](./LICENSE-CC-BY-4.0)。

仓库根目录的 [`licensee`](./licensee) 文件给出了双协议的简要说明与指引。
