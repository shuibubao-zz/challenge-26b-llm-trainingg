# pg_lab —— C2G 的本地实验台（CPU 可跑）

> **它解决什么问题**：我在做 C2G 时手上没有 H100、也没有网络能下载 FineWeb。
> 于是不可能复现官方 BPB 绝对值。
> 但「评测协议」这个变量的**相对效应**不需要 H100 也能测 —— 这就是 pg_lab 存在的理由。

## 一句话：它测什么

同一份训练好的权重、同一份 held-out 字节，**只改评测方式**，看 BPB 怎么变。

```
官方 baseline 评测    = 把验证集切成互不重叠的长度 L 窗口，窗口内每个位置都计分
                       → 位置 0 的 token 是「零上下文」预测，平均上下文仅 ~L/2

滑窗评测 (stride S)   = 窗口起点 0,S,2S,...，窗口 0 全计分，其余只计分尾部 S 个
                       → 每个被计分的 token 至少有 L−S 个上下文
                       → 代价：前向 token 数变成原来的 L/S 倍
```

**朴素评测 = S = L 的特例**，所以整个协议族只有 S 一个自由度（外加可选的超长窗口 W）。

## 核心手法：一次测量得到整个协议族

关键发现：**协议的 BPB 可以从「逐位置期望损失曲线」ρ(p) 解析算出**。

ρ(p) = 窗口第 p 个位置的期望交叉熵（nats/byte），用**重叠窗口**测一次即可。
测完之后：

- 朴素评测 BPB = `mean(ρ) / ln2`
- 滑窗 stride=S 的 BPB = `mean(ρ[W−S : W]) / ln2`（稳态值；窗口 0 的边缘效应按有限长公式修正）
- 代价倍数 = `W / S`

于是**扫一遍 S 不需要跑 N 遍评测**，而且因为用了所有重叠窗口采样，方差还更低。

> 这个捷径会不会错？会——如果「位置分布可交换」不成立。
> 所以 `analyze.py` 里同时实现了两种**逐字节金标准直接评测**
> （`direct_eval_naive` / `direct_eval_sliding`），与 ρ 重建值逐项对照。
> 对齐误差有多大，在 `analysis/analysis_seed*.json` 的 `direct_check` 字段里明写着。

## 文件

| 文件 | 作用 |
|---|---|
| `build_corpus.py` | 从本机 `.md/.txt/.py` 构建**确定性**语料，产出 `data/{train,val}.bin` + `manifest.json`（含 sha256） |
| `pgpt.py` | 官方 baseline GPT 的忠实缩小型（结构 1:1，尺寸缩小， vocab=256 裸字节） |
| `train.py` | 确定性 CPU 训练，写 `runs/ckpt_seedN.pt` + `runs/trainlog_seedN.json` |
| `evalproto.py` | ρ(p) 的定义、协议族的解析公式、Pareto 与拐点分析 |
| `analyze.py` | 加载 ckpt → 测 ρ → 扫协议 → **与直接评测交叉验证** → 写 JSON |
| `crossvalidate.py` | 检验「用训练分布标定 stride、在验证分布生效」是否站得住，**直接 import 交付脚本的决策函数** |
| `report.py` | 汇总多 seed，产出 Markdown 消融表 |
| `selftest.py` | 对交付脚本 `卢怡然_C2G_train_gpt.py` 的两个决策函数做单元测试（7 个断言） |
| `build_submission.py` | 由官方 baseline **程序化生成**提交脚本 + 统一 diff |

## 复现

```bash
pip install torch numpy

python build_corpus.py --out data                     # ~10s
python train.py --seed 1337 --steps 1200 --lr 5e-3 --threads 14 --out runs
python analyze.py --ckpt runs/ckpt_seed1337.pt --val data/val.bin --windows 512 1024
python crossvalidate.py --ckpt runs/ckpt_seed1337.pt
python report.py --dir analysis
python selftest.py
```

## 诚实的边界（必须知道）

1. **语料不是 FineWeb**。它是本机 `node_modules` 的 Markdown + Python stdlib 源码。
   所以 BPB 绝对值**不可与官方榜单比较**，只有相对差（ΔBPB）有参考意义。
2. **模型小得多**（494K 参数 vs 官方 ~10M+；字节级 vocab 256 vs SP1024）。
   小模型对小尺度上下文的依赖形态可能与大模型不同，**这可能低估也可能高估长上下文的收益**。
   extrapolation 到 1024 窗口的那组数据尤其要打折扣。
3. **训练预算完全不同**（1200 步 vs 13450 步）。模型的 ρ 曲线形状依赖于训练充分程度。
4. 因此本实验的结论定位为**方向性 + 量级证据**，不是对 8×H100 结果的预测。

## 环境

Python 3.13 · torch 2.14.1+cpu · numpy 2.5.3 · 16 线程 CPU · **无 GPU、无网络**（离线可跑）
