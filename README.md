# LearningLLM

这是一份大模型（LLM）学习过程中的笔记与代码示例。每一章对应一个 Markdown 文档（理论与逐行代码讲解）和一个 Jupyter Notebook（可直接运行的示例）。所有示例默认在 **Google Colab** 中运行（免费 T4 GPU 即可起步）。

## 学习路径

本仓库面向**完全 0 基础**的中文学习者，目标是从「不会 PyTorch」走到「能读懂主流 LLM 论文、能跑通 SFT/RLHF/多模态训练、能搭 RAG/Agent 应用」。整条路径分为 **0 ~ 9 共 10 个阶段**，越靠后越偏工程与应用。

硬件方面：默认 Colab 免费版 **T4** 起步；标注 ⚡ 的章节需要 Colab Pro 的 **L4 / A100**（如分布式、长上下文训练、SFT/RLHF 全流程、多模态训练）。每章 `.md` 顶部会显式标注本章硬件门槛，读者无需 Pro 也能完整阅读 `.md` 理解原理。

### 阶段 0：先修知识（Prerequisites）

读者基线：会写 Python、知道矩阵乘法。其余概念（autograd、softmax、KL 散度、SGD、MDP）从这里补齐。**先修章节用 `P0N` 编号**，与主线 `NN` 解耦。

```
P01  PyTorch 与张量：shape / dtype / device / broadcasting / autograd
P02  神经网络最小闭环：手写一个 MLP + 训练循环（前向、反向、优化器三步）
P03  概率与信息论够用版：softmax、log-likelihood、熵、交叉熵、KL、MLE
P04  优化器与学习率调度：SGD / Momentum / Adam / AdamW；warmup + cosine
P05  强化学习够用版：MDP、policy / value、REINFORCE、policy gradient（为后续 PPO/GRPO 打底）
```

### 阶段 1：把大模型用起来

```
01  环境与工具：IPython / Jupyter / Colab + 在 T4 上跑通 Qwen3-8B          ← 已完成
02  生成参数与采样策略：logits / softmax / temperature / top-p/top-k / 思考模式 ← 已完成
```

### 阶段 2：Transformer 架构精讲

目标：从 token 进入模型到 logits 输出，每一层张量怎么变换、每个组件为什么这样设计，最后能从零实现一个 mini-GPT。

```
03  Tokenizer：BPE / BBPE / 词表 / 特殊 token；用 tokenizers 库训一个小 tokenizer
04  Embedding 与位置编码：token embedding / sinusoidal / learned / RoPE / ALiBi
05  从 RNN 到 attention：seq2seq、Bahdanau / Luong attention（"attention 为什么会出现"）
06  Scaled Dot-Product Attention：Q/K/V、softmax、causal mask
07  Multi-Head Attention 与 MQA / GQA：分头、拼接、形状变换全过程
08  FFN、残差连接、LayerNorm / RMSNorm / SwiGLU
09  Transformer 整体架构：Encoder-Decoder vs Decoder-only；Pre-LN vs Post-LN
10  自回归语言建模目标、cross-entropy loss、teacher forcing 与推理采样的关系
11  论文精读：Attention Is All You Need 逐节解读
12  论文串读：GPT / LLaMA / Qwen 系列关键改动（RoPE / RMSNorm / SwiGLU / GQA / MoE）
13  从零实现 mini-GPT：tiny-shakespeare 上完成训练 → 生成闭环
```

### 阶段 3：推理工程

```
14  KV cache：原理、显存占用估算、从零加上 KV cache
15  量化：fp16 / bf16 / int8 / int4；GPTQ / AWQ / bitsandbytes 对比
16  批处理与 continuous batching；FlashAttention 直觉
17  推测解码（speculative / Medusa / EAGLE）概览
18  推理框架选型：vLLM / SGLang / llama.cpp / Ollama
19  权重格式与生态：safetensors / GGUF / HuggingFace Hub 工作流
```

### 阶段 4：预训练全景

```
20  预训练数据：清洗、去重、配比、合成数据
21  Scaling law 与 Chinchilla：参数量 / 数据量 / 算力的最优配比
22  MoE 架构：Mixtral、DeepSeek-V3 的路由与负载均衡
23  长上下文：RoPE 外推、YaRN、sliding window、attention sink
24  分布式训练总览：DDP / FSDP / ZeRO / TP / PP（重在读懂，不要求多卡复现）⚡
25  训练加速：混合精度、梯度累积、gradient checkpointing
26  Continue pretraining 与领域自适应 ⚡
```

### 阶段 5：后训练与对齐

```
27  对话模板与 chat template：special token、role、tool 调用格式
28  SFT 原理：数据格式（Alpaca / ShareGPT）、loss masking、lr / epochs 选择
29  SFT 实战：在 L4/A100 上 LoRA-SFT Qwen3-8B（≈ 跑通一遍真实后训练流程） ⚡
30  奖励模型（RM）：偏好数据、Bradley-Terry、RM 训练与评估 ⚡
31  RLHF 全景：PPO 在 RM 上的应用、KL 约束、reward hacking ⚡
32  DPO 系列：DPO / IPO / KTO / SimPO（无需 RM 的偏好对齐） ⚡
33  GRPO 与 RLVR（可验证奖励）：DeepSeek-R1 训练范式 ⚡
34  Reasoning 模型与 test-time compute：o1 / R1 思路、长 CoT、self-consistency
35  Constitutional AI / RLAIF：用 AI 反馈替代人类标注
36  模型蒸馏：logit 蒸馏、SFT 蒸馏、reasoning 蒸馏
```

### 阶段 6：参数高效微调实战

```
37  LoRA 原理与从零手写：低秩分解、合并权重、scaling
38  QLoRA：在 T4 上跑通 Qwen3-8B 微调
39  Adapter / Prefix-Tuning / IA³ 速览
40  模型合并（Model Merging）：DARE / TIES / SLERP
```

### 阶段 7：多模态大模型

```
41  多模态总览：模态对齐目标、CLIP 双塔、对比学习
42  视觉 encoder：ViT 与 SigLIP
43  VLM 架构演进：LLaVA / Qwen-VL / InternVL 的连接器设计
44  多模态训练全流程：vision pretrain → 模态对齐 → 多模态 SFT ⚡
45  多模态偏好对齐：在 VLM 上做 DPO / RLHF ⚡
46  语音多模态：Whisper、端到端语音模型（Qwen-Audio 等）
47  多模态评测与典型应用（OCR、文档问答、agent for GUI）
```

### 阶段 8：应用层

```
48  Prompt 工程与 CoT / few-shot / self-consistency
49  嵌入模型与 reranker：训练原理、对比学习、选型
50  RAG：分块、向量库、检索增强、reranker、评测
51  Function calling / Tool use / MCP
52  Agent：ReAct、规划、多轮工具调用、记忆
```

### 阶段 9：评测、安全与生态

```
53  评测体系：perplexity、benchmark（MMLU / GSM8K / HumanEval）、LMSYS Arena、LLM-as-judge
54  安全：jailbreak、prompt injection、对齐失效、红队测试
55  HuggingFace 生态全景：Transformers / Datasets / Accelerate / TRL / PEFT
```

> 路径会随大模型生态演进调整。每完成一章，下面「已完成章节」清单会追加链接。

## 已完成章节

### 第 01 章：IPython、Jupyter、Google Colab 入门

- 文档：[`01-IPython-Jupyter-Colab入门.md`](./01-IPython-Jupyter-Colab入门.md)
- 代码：[`01.ipynb`](./01.ipynb) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/01.ipynb)
- 涵盖内容：介绍 IPython / Jupyter / Colab 的关系与使用，讨论 Colab 上的 GPU 选型与大模型选型，并在 T4 上用 4-bit 量化跑通 Qwen3-8B 的对话生成。

### 第 02 章：生成参数与采样策略

- 文档：[`02-生成参数与采样策略.md`](./02-生成参数与采样策略.md)
- 代码：[`02.ipynb`](./02.ipynb) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiqiangnd/LearningLLM/blob/main/02.ipynb)
- 涵盖内容：拆解 `generate()` 的完整工作流（logits → temperature → top-k / top-p → softmax → 采样），逐个讲解 `do_sample` / `temperature` / `top_p` / `top_k` / `repetition_penalty` 的作用与数学定义，并通过 6 组对比实验（贪心 vs 采样、不同 temperature、不同 top_p、是否抑制重复、思考模式 vs 非思考模式、复现性）让读者直观理解每个旋钮的效果。

## 仓库结构

```
LearningLLM/
├── README.md
├── 01-IPython-Jupyter-Colab入门.md       # 第 01 章：环境与工具
├── 01.ipynb                              # 第 01 章：可运行示例
├── 02-生成参数与采样策略.md              # 第 02 章：生成参数与采样策略
├── 02.ipynb                              # 第 02 章：可运行示例
└── ...                                   # 后续章节持续更新（先修 P01~P05、主线 03~55，详见上方学习路径）
```

## 推荐运行环境

| 项目 | 推荐 |
|------|------|
| 运行平台 | Google Colab（免费 T4 GPU 起步） |
| 编辑器 | VS Code + Google Colab 扩展 |
| Python | 3.10+ |
| 核心依赖 | `transformers` / `accelerate` / `bitsandbytes`（示例代码会自动安装） |

## 阅读建议

1. 按章节顺序阅读 `.md` 文档理解概念
2. 点击章节旁的 **Open in Colab** 徽章直接打开对应 Notebook（或在 VS Code 中连接 Colab 运行时）
3. 文档与代码相互对照——`.md` 里有原理解析，`.ipynb` 注释里有逐行说明

## 反馈与交流

欢迎通过 [Issue](https://github.com/weiqiangnd/LearningLLM/issues) 提问、讨论或纠错。
