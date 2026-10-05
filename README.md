# 卢怡然 C2G —— 参数高尔夫（Parameter Golf）

> 赛道：OpenAI Parameter Golf `track_10min_16mb`
> 交付日期：2026-10-04 ｜ 截止：2026-12-31

---

## ⚠️ 先看这一节：本包**没有**官方尺度的成绩

| 项 | 状态 |
|---|---|
| 8×H100 上的 BPB | **无。从未跑过。** |
| 官方榜单排名 | **未上榜** |
| 提交到官方 repo | 未做 |

**原因**：交付者本机**无 GPU**（`nvidia-smi` 不存在）、**无外网**（FineWeb / Gutenberg 均不可达）。
这不是托词，是实测结果。所有 BPB 相关的 `submission.json` 字段一律填 `null` 并标 `status: not_run`。

**本包提供的是**：一个可运行的提交脚本（有单元测试）+ 一套在 CPU 上真跑过的、
n=3 seed 的对照实验 + 完整的方法学与诚实声明。

---

## 一、方案一句话

**不改训练、不改架构、不改量化。只改评测：把 `eval stride` 从写死的常数 64，
改成训练结束后用一小段校准 token 现场标定的量。**

```
官方 baseline  = 窗口 W = 步长 S = train_seq_len   →  位置 0 的 token 是零上下文预测
滑窗评测      = W = train_seq_len，S < W           →  每个 token 至少带 W−S 个上下文
代价倍数      = W / S                              →  官方 stride=64 / W=1024 即 16×
```

朴素评测是 `S = W` 的一个特例，所以整个协议族只有两个自由度，可以画出一根「代价—收益」曲线。

**要解决的问题**：历史方案付了 16× 的评测算力；如果 ρ(p)（逐位置期望损失）对上下文的
边际收益是强递减的，那么 16× 里有一大部分可能买的是一条几乎平坦的尾巴。

---

## 二、目录

### 交付文档（CHALLENGE.md 必交项）

| 文件 | 说明 |
|---|---|
| **`卢怡然_C2G_方案草案.md`** | Level 1 算力申请用，回答 4 个门槛问题（1979 字） |
| `卢怡然_C2G_方案设计.md` | 目标 Level、技术选型、实验矩阵、风险判据 |
| **`卢怡然_C2G_train_gpt.py`** | 最终提交脚本（1415 行，由官方 baseline 程序化生成） |
| `卢怡然_C2G_submission.json` | 提交元数据（8×H100 字段为 null，标注 `not_run`） |
| `卢怡然_C2G_logs/` | 3 seed 训练日志（`trainlog_seed{1337,42,2026}.txt/.json` + 语料 manifest） |
| `卢怡然_C2G_ablation.md` | 消融实验报告（协议族 + 拐点 + 负结果 + 交叉验证） |
| `卢怡然_C2G_leaderboard.md` | 与 baseline / SOTA 的对比（含官方 30 条真实记录） |
| **`卢怡然_C2G_AI日志.md`** | AI 使用全过程 + 反向举证 |
| `卢怡然_C2G_拿来说明.md` | 拿了什么 / 改了什么 / 没拿什么 |
| **`卢怡然_C2G_AAR.md`** | 复盘：3 个失败 + 1 个明确标注的未做实验 |

### 代码与证据

| 路径 | 说明 |
|---|---|
| `pg_lab/` | **CPU 可跑的实验台**（详细见 `pg_lab/README.md`） |
| `pg_lab/selftest.py` | 对提交脚本决策函数的单元测试，**7/7 通过** |
| `pg_lab/build_submission.py` | 由官方 baseline 程序化生成提交脚本 + 统一 diff |
| `evidence/` | 原始输出留档：分析输出、交叉验证、官方榜单表、哈希、diff |
| `_upstream/` | 官方 repo 解压副本（**不进自己的 GitHub 仓库**，已在 `.gitignore`） |

---

## 三、核心实测结果（CPU 小尺度，n=3 seed）

> ⚠️ **坐标系不同**：494K 参数、字节级 vocab=256、4MB 本地语料。
> **绝对 BPB 与官方不可比**，只有 Δ 有参考价值。

### ρ(p) 的形状 —— 本方案的立论

| 位置区间 | 0–7 | 8–31 | 32–63 | 64–127 | 128–255 | 256–383 | 384–495 | 496–511 |
|---|---|---|---|---|---|---|---|---|
| BPB | 2.8512 | 2.1302 | 2.0295 | 1.9899 | 1.9794 | 1.9773 | **1.9770** | 1.9799 |

**前 8 个位置就掉掉 82% 的可降幅度，到 64–127 已拿下 98.5%。**
最末尾（496–511）反而回升 0.0029 —— 这解释了为什么极小 stride 会变差。

### 协议族消融

| 代价倍数 | 1.00× | 1.14× | 1.33× | 1.60× | **2.00×** | 2.67× | 4.00× | 8.00× | **16.00×** |
|---|---|---|---|---|---|---|---|---|---|
| ΔBPB | 0 | −0.02375 | −0.02544 | −0.02596 | **−0.02614** | −0.02618 | −0.02610 | −0.02545 | **−0.02460** |

**2× 代价拿到 99.8% 的最大收益。16×（官方 stride=64 的等价代价）比 2× 差 0.00154 BPB**
（配对标准差 0.00050，3 个 seed 符号一致）。

### 拐点

| 想要 50% 收益 | 想要 90% | 想要 95% | 想要 99% | 想要 100% |
|---|---|---|---|---|
| 1.14× | 1.14× | 1.33× | 1.60× | 2.67× |

### 负结果：窗口拉到训练长度之外（W=1024，训练窗口 512）

整条曲线都是灾难，ΔBPB 从 **+0.166 恶化到 +0.467**，越努力越糟。
**这直接否掉了"省下的算力拿去买更长评测窗口"的想法**（除非同时改训练或加 RoPE 缩放）。

### 交叉验证：用训练分布标定、在验证分布生效？

| 检验 | 结果 |
|---|---|
| ρ 的**水平**差（val − train） | **+0.884 nats/byte**（巨大） |
| 五档预算下两种标定选出的 stride | **全部一致**（256） |
| 两种标定算出的验证集 BPB 差 | **0.00e+00** |

水平差了 0.88 nats，决策却完全一致 ——
**"stride 只依赖形状不依赖水平"这个论证成立**，所以敢用训练分布标定（规避验证集调参）。

### 方法学自检

ρ 重建 vs 逐字节金标准直接评测：**最大误差 2.09×10⁻³ BPB**（约为效应量的 8%）。

---

## 四、复现

```bash
# 环境：Python 3.13 + torch(cpu) + numpy，离线可跑
cd pg_lab

python build_corpus.py --out data                                    # ~10s，构建确定性语料
python train.py --seed 1337 --steps 1200 --lr 5e-3 --threads 14 --out runs
python train.py --seed 42    --steps 1200 --lr 5e-3 --threads 14 --out runs
python train.py --seed 2026  --steps 1200 --lr 5e-3 --threads 14 --out runs

python analyze.py       --ckpt runs/ckpt_seed1337.pt --windows 512 1024
python crossvalidate.py --ckpt runs/ckpt_seed1337.pt --window 512
python report.py        --dir analysis
python selftest.py                                                   # 7/7

# 重新生成提交脚本与 diff
python build_submission.py
```

单个 seed 训练约 8 分钟（16 线程 CPU）；三个 seed 全流程约 45 分钟。

---

## 五、拿到 H100 后怎么跑

```bash
# H1 复现 baseline（必须通过才继续）
EVAL_STRIDE_MODE=naive torchrun --standalone --nproc_per_node=8 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
    MAX_WALLCLOCK_SECONDS=600 卢怡然_C2G_train_gpt.py

# H2 同一权重扫协议族（不重训，最便宜的一步）
EVAL_STRIDE_MODE=fixed EVAL_STRIDE=256 ...

# H3 拐点标定
EVAL_STRIDE_MODE=knee EVAL_COST_BUDGET=2.0 KNEE_GAIN_FRAC=0.95 ...
```

判据见 `卢怡然_C2G_方案草案.md` §问题 4。**baseline 复现失败就停，不往下走。**

---

## 六、诚实清单（未做的事）

- [ ] **未在 8×H100 上跑过任何一次**（无 GPU）
- [ ] **未复现官方 baseline BPB**（无算力）
- [ ] **未提交到官方榜单**
- [ ] **未测「训练窗口 1024 + 评测窗口 1024」的真实配置** —— §三的负结果是训练窗口 512 下的，
      这个缺口在 `卢怡然_C2G_AAR.md` §三明确标注，可补（约 2 倍 CPU 时间）
- [ ] **未做 16MB artifact 打包验证**（改动只增代码约 62KB，baseline 余量约 136KB，理论安全但未实测）
- [ ] 未做 Level 2 以上的任何组合优化

**这份清单里没有任何一项是靠文档篇幅掩盖过去的。**
