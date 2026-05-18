# LearningLLM

## 仓库介绍

这是一份大模型（LLM）学习过程中的笔记与代码示例。每一章对应一个 Markdown 文档（理论与逐行代码讲解）和一个 Jupyter Notebook（可直接运行的示例），两者**互为对照**——`.md` 回答「为什么这样做」，`.ipynb` 回答「代码每一行做什么」，公式与代码片段刻意逐字一致。所有示例默认在 **Google Colab** 中运行（免费 T4 GPU 即可起步）。

仓库特色：

- **`.md` + `.ipynb` 双轨并行**：每章一份概念讲解、一份可直接 Run All 的代码，互为对照、术语一致。
- **真实模型而非玩具示例**：默认在 Colab 免费 T4（15 GB）上用 4-bit 量化跑通 Qwen3-8B；显存吃紧的进阶章节才升 L4 / A100。
- **概念不跳步**：读者基线只要求「会 Python + 矩阵乘法」，softmax / KL / MDP 等首次出现都先定义再使用，并配最小可手算的数值例子（2×2 矩阵、长度为 3 的序列）。
- **形状变换全程标注**：reshape / transpose / broadcasting 每一步都给张量形状，如 `(B, L, D) → (B, L, H, D/H) → (B, H, L, D/H)`。
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
| P02 神经网络最小闭环：手写一个 MLP + 训练循环（前向、反向、优化器三步） | MLP 数学定义与"为什么必须有非线性激活"、用 `nn.Module` 组织模型、`nn.Linear` 内部 $y = xW^\top + b$ 的形状约定、损失函数（MSE / BCE / CrossEntropy）与优化器（SGD / AdamW）选型、训练循环三步「forward → backward → step」+ `zero_grad`，并在 `make_moons` 数据上完整训练 MLP（含 loss 曲线、决策边界可视化），用一个去掉 ReLU 的反例实证"非线性激活不可省略"。 | [文档](./src/P02-神经网络最小闭环.md) · [ipynb](./src/P02.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P02.ipynb) | ✅ |
| P03 概率与信息论够用版：softmax、log-likelihood、熵、交叉熵、KL、MLE | 从 logits → softmax → 概率分布的完整链路（含数值稳定的 log-softmax 与 logsumexp）、温度对分布尖锐 / 扁平的影响、似然 vs 概率、log-likelihood 与 MLE、熵 $H(p)$ 与"分布有多确定"、交叉熵 $H(p,q)$ 与负对数似然 NLL 在 one-hot 标签下的等价性、`F.cross_entropy` 接收 raw logits 的工程铁律（含"传 softmax 进去会出什么错"反例）、KL 散度的定义与不对称性可视化（forward KL vs reverse KL），最后用一个 5 类分类器实证"训练 = 最小化交叉熵 = 等价 MLE"。 | [文档](./src/P03-概率与信息论够用版.md) · [ipynb](./src/P03.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P03.ipynb) | ✅ |
| P04 优化器与学习率调度：SGD / Momentum / Adam / AdamW；warmup + cosine | 从 SGD → Momentum → Adam → AdamW 一路推演每一步的修正动机（"长椭圆山谷"上的优化轨迹对比图）、Adam 的一阶矩 / 二阶矩 / bias correction（手写 Adam 与 PyTorch 内置一致性验证）、AdamW 的 weight decay 解耦、lr 是最敏感超参（lr 太大 / 适中 / 太小三条 loss 曲线对比）、warmup 缓解 Adam 早期方差估计不准与"大初始化 + 大 lr"导致的 NaN、cosine 退火的形状与下界设置、`warmup + cosine` 这套 LLM 训练事实默认调度的完整代码模板（含 `LambdaLR` 与 `optimizer.step()` / `scheduler.step()` 的调用顺序）。 | [文档](./src/P04-优化器与学习率调度.md) · [ipynb](./src/P04.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P04.ipynb) | ✅ |
| P05 强化学习够用版：MDP、policy / value、REINFORCE、policy gradient（为后续 PPO/GRPO 打底） | MDP 五元组与马尔可夫性、trajectory 与累计折扣回报 $G_t$ 、策略 $\pi(a \mid s)$ 与状态价值 $V^\pi$ / 动作价值 $Q^\pi$ / 优势 $A^\pi$ 的关系、目标函数 $J(\theta) = \mathbb{E}\_\tau[R(\tau)]$ 与 policy gradient 定理（"对 log 概率求导乘上回报"的直觉解读）、REINFORCE 算法、加 baseline 降方差的来历，以及把 RL 语言映射到 LLM 对齐（状态 = prompt + 已生成 token、动作 = 下一个 token、奖励来自 RM / 可验证规则、KL 约束防止跑偏）。实战在 `gymnasium` 的 CartPole-v1 上从随机策略 baseline 跑到 REINFORCE 接近满分 500，并对比加 / 不加 baseline 的曲线方差；最后用一个迷你"LLM 风格" REINFORCE 闭环把 PPO / GRPO 的核心思想用 50 行代码实证一遍。 | [文档](./src/P05-强化学习够用版.md) · [ipynb](./src/P05.ipynb) · [OpenInColab](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/src/P05.ipynb) | ✅ |

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
| 03 Tokenizer：BPE / BBPE / 词表 / 特殊 token；用 tokenizers 库训一个小 tokenizer |  |  |  |
| 04 Embedding 与位置编码：token embedding / sinusoidal / learned / RoPE / ALiBi |  |  |  |
| 05 从 RNN 到 attention：seq2seq、Bahdanau / Luong attention（"attention 为什么会出现"） |  |  |  |
| 06 Scaled Dot-Product Attention：Q/K/V、softmax、causal mask |  |  |  |
| 07 Multi-Head Attention 与 MQA / GQA：分头、拼接、形状变换全过程 |  |  |  |
| 08 FFN、残差连接、LayerNorm / RMSNorm / SwiGLU |  |  |  |
| 09 Transformer 整体架构：Encoder-Decoder vs Decoder-only；Pre-LN vs Post-LN |  |  |  |
| 10 自回归语言建模目标、cross-entropy loss、teacher forcing 与推理采样的关系 |  |  |  |
| 11 论文精读：Attention Is All You Need 逐节解读 |  |  |  |
| 12 论文串读：GPT / LLaMA / Qwen 系列关键改动（RoPE / RMSNorm / SwiGLU / GQA / MoE） |  |  |  |
| 13 从零实现 mini-GPT：tiny-shakespeare 上完成训练 → 生成闭环 |  |  |  |

〔预备知识〕04 章首次密集用到张量 shape / broadcasting / autograd——若不熟悉，建议先读 P01。
〔预备知识〕10 章首次用到极大似然估计、熵 / 交叉熵 / KL 散度——若不熟悉，建议先读 P03。
〔预备知识〕13 章首次走完「前向 + 反向 + 优化器」完整训练循环——若不熟悉，建议先读 P02 与 P04。

### 阶段 3：推理工程

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 14 KV cache：原理、显存占用估算、从零加上 KV cache |  |  |  |
| 15 量化：fp16 / bf16 / int8 / int4；GPTQ / AWQ / bitsandbytes 对比 |  |  |  |
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
| 24 分布式训练总览：DDP / FSDP / ZeRO / TP / PP（用单卡 L4 跑最小 demo） ⚡ |  |  |  |
| 25 训练加速：混合精度、梯度累积、gradient checkpointing |  |  |  |
| 26 Continue pretraining 与领域自适应 ⚡ |  |  |  |

### 阶段 5：后训练与对齐

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 27 对话模板与 chat template：special token、role、tool 调用格式 |  |  |  |
| 28 SFT 原理：数据格式（Alpaca / ShareGPT）、loss masking、lr / epochs 选择 |  |  |  |
| 29 SFT 实战：在 L4/A100 上 LoRA-SFT Qwen3-8B（≈ 跑通一遍真实后训练流程） ⚡ |  |  |  |
| 30 奖励模型（RM）：偏好数据、Bradley-Terry、RM 训练与评估 ⚡ |  |  |  |
| 31 RLHF 全景：PPO 在 RM 上的应用、KL 约束、reward hacking ⚡ |  |  |  |
| 32 DPO 系列：DPO / IPO / KTO / SimPO（无需 RM 的偏好对齐） ⚡ |  |  |  |
| 33 GRPO 与 RLVR（可验证奖励）：DeepSeek-R1 训练范式 ⚡ |  |  |  |
| 34 Reasoning 模型与 test-time compute：o1 / R1 思路、长 CoT、self-consistency |  |  |  |
| 35 Constitutional AI / RLAIF：用 AI 反馈替代人类标注 |  |  |  |
| 36 模型蒸馏：logit 蒸馏、SFT 蒸馏、reasoning 蒸馏 |  |  |  |

〔预备知识〕31 章首次用到 MDP / policy gradient / PPO——若不熟悉，建议先读 P05。

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

### 阶段 8：应用层

| 标题 | 涵盖内容 | 链接 | 状态 |
|------|---------|------|------|
| 48 Prompt 工程与 CoT / few-shot / self-consistency |  |  |  |
| 49 嵌入模型与 reranker：训练原理、对比学习、选型 |  |  |  |
| 50 RAG：分块、向量库、检索增强、reranker、评测 |  |  |  |
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

> **关于数学公式渲染**：本仓库 `.md` 文件**优先保障在 GitHub 上的渲染效果**。GitHub 走的是「markdown 解析 → MathJax 渲染」两段管线，会先把 `\_` 等转义字符还原再交给 MathJax，行内 / 块级公式（`$...$` / `$$...$$`）大多能正常显示。但本地 markdown 阅读器（Typora、VS Code 预览、Obsidian、各种网页版 markdown viewer 等）的渲染机制各异——有的用 KaTeX、有的用不同配置的 MathJax、有的根本不渲染数学，因此**部分公式（尤其是带 `\_` 转义、`\left\{`、`\text{}` 中带下划线等写法）在这些阅读器里可能显示异常**。如果你在本地阅读时遇到公式渲染问题，可以直接看 `dist/` 目录下的 PDF 文档——PDF 走 KaTeX 服务端渲染、字体内嵌、样式与 GitHub 网页一致，跨平台稳定。

## 反馈与交流

欢迎通过 [Issue](https://github.com/weiqiangnd/LearningLLM/issues) 提问、讨论或纠错。

## License

本仓库采用**双协议**授权，方便代码与文字内容按各自最适合的方式被复用：

- **代码**（`.ipynb` 中的代码 cell、未来可能加入的脚本）采用 [MIT License](https://opensource.org/licenses/MIT)：任何人可自由使用、修改、再分发，仅需保留版权声明。全文见 [`LICENSE-MIT`](./LICENSE-MIT)。
- **文字与图片内容**（`.md` 文档、`assets/` 下的 SVG / PNG）采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)：可自由分享与改编，需署名（注明来源仓库）。全文见 [`LICENSE-CC-BY-4.0`](./LICENSE-CC-BY-4.0)。

仓库根目录的 [`LICENSE`](./LICENSE) 文件给出了双协议的简要说明与指引。
