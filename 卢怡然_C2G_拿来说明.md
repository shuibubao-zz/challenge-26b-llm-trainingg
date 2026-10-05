# 卢怡然 C2G 拿来说明

> 原则要求：**严禁从零写 Transformer**。说明你「拿了什么、改了什么、为什么改」。
> 本文逐条列清来源、改动、以及**我判定的风险**。凡我核过出处的一律给出文件路径；核不到的明确标注。

---

## 零、一句话总结

| | 内容 |
|---|---|
| **拿** | OpenAI 官方 `parameter-golf` 的 baseline `train_gpt.py`（架构 + 优化器 + 量化 + 打包），以及 Matthew Li 的滑窗评测机制（record `2026-03-19_SlidingWindowEval`，1.2244 → 1.1925） |
| **改** | 把滑窗评测里的 `stride` 从一个**写死的经验常数 64**，换成**训练结束后现场标定的量**：先测出逐位置损失曲线 ρ(p)，再按给定的评测算力预算挑出「拿到 95% 收益所需的最省步长」 |
| **为什么改** | stride=64 要付 `1024/64 = 16 倍`前向算力。ρ(p) 对上下文的边际收益**强递减**，绝大部分收益在远短于 960 token 的上下文处就已拿到 —— 付 16 倍很可能是在买一个几乎平坦的尾巴 |

---

## 一、逐条拿来清单

### 1. 架构：官方 baseline `train_gpt.py`（全部照搬，未改）

来源：`openai/parameter-golf` → `train_gpt.py`（随包附在 `materials/parameter-golf-main.zip`），
**也是** `records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py` 的同一份代码。

> 精确表述（我核过 sha256，两者**不是**逐字节相同，不要把话说满）：
> 两份均为 1126 行、47686 / 47642 字节，sha256 分别为 `c6cae5e7…` 与 `e77cf46e…`，
> 唯一差异是第 4 行 docstring 一处措辞（"To keep readable for newcomers, let's make sure" vs
> "must never be longer than"）。**代码本体完全一致**。
> 完整哈希与 diff 见 `evidence/base_sha256.txt`。

拿的具体部件：

| 部件 | 位置 | 我是否改动 |
|---|---|---|
| `RMSNorm` / `Rotary` / `apply_rotary_emb` | 上游 501–552 行 | 未改 |
| `CausalSelfAttention`（GQA + query RMSNorm + `q_gain`） | 上游 556–605 行 | 未改 |
| `MLP`（relu²，来自 modded-nanogpt） | 上游 608–617 行 | 未改 |
| `Block`（`resid_mix` / `attn_scale` / `mlp_scale`） | 上游 621–646 行 | 未改 |
| `GPT`（U-Net 式 skip_weights + tied embedding + logit softcap） | 上游 649–730 行 | 仅**新增** `forward_logits`，forward 本体未动 |
| `Muon` 优化器 + Newton-Schulz 系数 | 上游 96–178 行 | 未改 |
| int8 量化 + zlib 打包流程 | 上游 282–428 行 | 未改 |

> `forward_logits` 是唯一的结构性新增，而且它只是把 `GPT.forward` 的
> 「logits → mean cross-entropy」在 `mean` 之前 return 出来，**数学上完全等价**，
> 滑窗评测需要逐位置 NLL，没有它拿不到 ρ(p)。

### 2. 评测机制：Matthew Li 的滑窗评测（照搬核心逻辑）

来源：`records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`，
函数 `eval_val_sliding()`（该 record 文件 837–929 行），作者 GitHub ID `mattqlf`。

拿的部分：
- 窗口起点 `0, S, 2S, ...`，窗口长度 = `train_seq_len`
- **窗口 0 全部位置计分，其余窗口只计分尾部 S 个位置**（源码注释 `s = 0 if ws == 0 else max(wlen - stride, 0)`）
- 这条规则保证每个字节**恰好被计分一次**，不会重复计数

改的部分：
1. `stride` 由调用方传入（来自拐点标定），不再是常量 64
2. 新增 forward token 计数日志，让「评测算力开销」可被外部审计
3. `batch_seqs` 可按显存调节

### 3. 方法论来源：把 33 个历史 record 当成「公开的消融实验库」

随包 ZIP 里有 **33 个历史提交**，每个都带 `submission.json`（真实 BPB / 字节数 / 时间戳）。
我把它当成一个别人已经替我跑过的消融数据集来读，而不是当作「可以抄的答案」。
其中对本题直接相关的两条：

| record | 报告 BPB | 与我的关系 |
|---|---|---|
| `2026-03-17_NaiveBaseline` | 1.2244 | 零点。训练和架构都不动时的朴素评测结果 |
| `2026-03-19_SlidingWindowEval` | 1.1925 | 我要优化的那个 −0.0319。**训练完全没变**，Pure eval-side 收益 |

> ⚠️ **我读出来的一处不可靠处（必须在提案里讲清楚）**
> 该 record 的 README 声称「pre-quant BPB 基本相同（1.2172 vs 1.2196），收益全部来自评测」。
> 但它记录的两个 run 的 `step_stop` **不同**（13,780 vs 13,450），
> 且 pre-quant 数字取自训练循环里**朴素** `eval_val` 的最后一次快照，post-quant 才用滑窗。
> 也就是说这两个 pre-quant 数字**既不同步也不同协议**，拿它们论证「训练没变」是不严的。
> 真正能站住的只有一句话：**两条跑道训练配置相同、都塞进了 16MB**。
> 我把这一点写进提案的「已知局限」，不跟着照抄它的论证方式。

### 4. 工程范式：我自己 C4 的 `challenge-submit-guard`

这是第三次跨挑战复用：零第三方依赖、内置 `--self-test`、规则外置、证据进 `evidence/`。
在 C2G 里的体现是 `pg_lab/selftest.py`（7 个断言）和全文可复算的产物生成链。

---

## 二、没拿什么（以及为什么）

| 方向 | 为什么不选 |
|---|---|
| **换优化器（MuonEq-R 等）** | 收益和 Factor 交互复杂，需要真算力才能调；eval-side 改动**零训练算力**，在我拿不到 GPU 的阶段是唯一能推进的方向 |
| **SP8192 tokenizer** | 要重新跑数据预处理 + 重训练，成本高一个量级；且它和滑窗评测**正交**，不冲突，应先做便宜的那个 |
| **量化（GPTQ / int6）** | 与 artifact 字节预算耦合，改错会直接超 16MB；同样需要多轮真跑 |
| **Test-Time Training（TTT）** | 所有 top-5 record 都在用，是最明显的下一根杠杆；但它**消耗评测算力**，与「省评测算力」的目标正面冲突，必须先把评测侧的收支账算清再谈 |

> 一句话：**在所有 −0.03 量级的改动里，滑窗评测是唯一一个训练预算为零的。**
> 在算力受限时，先做便宜的那个。这一步的目的是把评测侧本来被浪费的算力挤出来，
> 交给后面真正吃算力的手段（TTT、长窗口、depth recurrence）。

---

## 三、我做的唯一原创部分（以及可检验的方式）

**「评测步长的拐点标定」**：把 `stride` 从一个 magic number 变成一个**在提交脚本内部由数据决定的量**。

实现位置：`卢怡然_C2G_train_gpt.py` → `measure_position_profile()` / `pick_stride_by_knee()` /
`resolve_eval_stride()`。决策依据是现场测的 ρ(p)，以及一个**显式声明的算力预算** `EVAL_COST_BUDGET`。

这样做有一个额外的诚实收益：**「为什么是 64」这个问题从此有了答案**。
历史方案把 64 写死，评审无从判断它是否过拟合到某个特定模型+特定预算；
改成现场标定后，预算是输入、选择是输出，取舍被显式化。

### 我为此补的一个验证（这是最容易翻车的地方）

标定用的是**训练分布**的 token（不在验证集上调参 —— CHALLENGE.md 明确禁止），
但评测发生在**验证分布**上。为此我做了 `pg_lab/crossvalidate.py`，
检验链条是「stride 只依赖 ρ 的形状而不依赖水平」：
- 实测 train/val 两条 ρ 的**水平**差异有多大
- 归一化后的**形状**差异有多大
- 分别喂进交付脚本的同一个函数，选出的 stride 是否一致、实际 BPB 是否一致

结论见 `卢怡然_C2G_ablation.md` §5。**如果形状也不一致，那这套标定就不可用** —— 这是我给自己设的判据。

---

## 四、坐标系：我的 Worker 与 33 条历史 record 的关系

```
 1.2244  官方 baseline（朴素评测）
   |
   |  −0.0319   Matthew Li 的滑窗（stride=64，16x 评测算力）
   v
 1.1925
   |
   |  我方目标：拿到同样的 BPB，付出更小的评测算力代价
   v
 1.19xx  ← 本方案目标区间（待 8×H100 验证）
```

本方案**不声称能超过 1.1925**。它声称的是：在同等 BPB 下，把评测算力从 16× 降到
——如果 ρ 的形状如本地小尺度实验所示—— 约 2–4×，
从而把省下的评测时间预算（上限 10 分钟，与训练预算分开计）留给 TTT 等真正吃算力的手段。

**这是一个"把钱从一处挪到另一处"的方案，不是一个"凭空多出收益"的方案。**
