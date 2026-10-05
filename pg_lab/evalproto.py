"""
profile.py — 评测协议实验台的核心：**一次测量，得到整个协议族**。

## 统一的协议族（这是本方法论的关键简化）

官方 baseline 的评测方式：
    把验证集切成互不重叠、长度为 L 的窗口，窗口内**每个位置**都计分。
    于是位置 0 的 token 是"零上下文"预测，平均上下文只有 ~L/2。

Matthew Li 的滑窗评测（滑窗 = stride S < L）：
    窗口起点为 0, S, 2S, ...，每个窗口只计分**尾部 S 个位置**。

把两者写成同一个公式：**协议由窗口长 W 与步长 S 两个参数决定**：
    - 基线 = W = S = L（互不重叠）
    - 滑窗 = W = L, S < L（每个被计分的 token 至少有 W − S 个上下文）
被测接连 DA(S) 的加权平均位置分布：
    - 窗口 0：位置 0..W-1 全计分（否则前 W−S 个字节永远不被计分）
    - 窗口 i>=1：只计分位置 [W−S, W-1]

核心测量对象：**逐位置期望损失曲线** ρ(p) = E[窗口内第 p 个位置的交叉熵]（nats/byte）
只要把 ρ(p) 测准（p = 0..W-1），整个协议族的 BPB 都能解析计算出来，不需要每个 F 单独跑一遍。

    BPB_bytes(nats) = nats / ln(2)          # 字节级建模下精确成立

## 代价模型
    窗口数 ≈ N / S，每窗口前向 W 个 token ⇒ 前向 token 总数 ≈ N·W/S
    基线 前向 token 数 = N                  ⇒ 代价倍数 C(W,S) = W / S
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch

LN2 = math.log(2.0)


# ---------------------------------------------------------------- ρ(p) 测量

@torch.no_grad()
def measure_position_profile(model, val_bytes: torch.Tensor, window: int, stride_min: int,
                             batch_windows: int = 32, device: str = "cpu",
                             val_limit: Optional[int] = None, verbose: bool = True):
    """
    在验证集上滑窗前向，累积「每个位置的期望 nats/byte」。
    返回 dict: {window, stride_min, profile[np array length window], n_scored, n_windows,
                forward_tokens, val_bytes_used}
    """
    was_training = model.training
    model.eval()
    data = val_bytes if val_limit is None else val_bytes[:val_limit]
    n = data.numel()
    if n < window + 1:
        raise ValueError(f"验证集太短: {n} < window+1")

    max_start = n - 1 - window
    starts = list(range(0, max_start + 1, stride_min))
    prof = torch.zeros(window, dtype=torch.float64)
    cnt = torch.zeros(window, dtype=torch.float64)

    total = len(starts)
    for i in range(0, total, batch_windows):
        chunk = starts[i: i + batch_windows]
        xs = torch.stack([data[s: s + window] for s in chunk]).to(torch.long)
        ys = torch.stack([data[s + 1: s + window + 1] for s in chunk]).to(torch.long)
        _, per_pos = model(xs, ys)          # (B, W) nats，逐位置
        prof += per_pos.to(torch.float64).sum(dim=0)
        cnt += per_pos.shape[0]
        if verbose and (i // batch_windows) % 20 == 0:
            print(f"    ρ 测量 {i}/{total} 窗口", flush=True)

    model.train(was_training)
    return {
        "window": window,
        "stride_min": stride_min,
        "profile": (prof / cnt.clamp(min=1)).numpy(),
        "n_windows": total,
        "forward_tokens": int(total * window),
        "val_bytes_used": int(n),
    }


# ---------------------------------------------------------------- 协议族解析

def protocol_bpb(profile, window: int, stride: int, n_val_bytes: int) -> dict:
    """
    给定 ρ(p)（长度 = window），计算协议 (window, stride) 的 BPB。
    严格处理窗口 0 的边缘效应（与 Matthew Li 实现 `s = 0 if ws == 0 else ...` 一致）。
    """
    import numpy as np
    prof = profile
    W = window
    S = stride
    if S > W:
        raise ValueError("stride 不能大于 window")

    n_win = max(1, int((n_val_bytes - 1) // S))
    tail_mean = float(prof[W - S:].mean())          # 稳态窗口的期望
    first_mean = float(prof.mean())                 # 窗口 0 全部位置都计分
    scored_first = W
    scored_tail = n_win - 1
    # 加权平均
    total_weighted = first_mean * scored_first + tail_mean * scored_tail * S
    total_tokens = scored_first + scored_tail * S
    bpb = (total_weighted / total_tokens) / LN2

    return {
        "window": W,
        "stride": S,
        "bpb": bpb,
        "bpb_steady_state": tail_mean / LN2,
        "min_context": W - S,
        "cost_multiplier": W / S,
        "forward_tokens": int(n_win * W),
        "n_windows": n_win,
        "edge_window_share": scored_first / total_tokens,
    }


def sweep_protocols(profile, window: int, strides, n_val_bytes: int) -> list:
    return [protocol_bpb(profile, window, s, n_val_bytes) for s in strides]


def naive_reference(profile) -> dict:
    """基线协议 = stride == window 时的取值，用作 ΔBPB 的零点。"""
    return {"bpb": float(profile.mean()) / LN2, "min_context": 0, "cost_multiplier": 1.0}


# ---------------------------------------------------------------- 拐点 / Pareto

def knee_analysis(strides, gains, costs) -> dict:
    """
    gains[i] = 相对基线的 BPB 下降量（正数=更好）；costs[i] = 代价倍数。
    返回：达到最大收益 X% 所需的最小代价（拐点 adepts）。
    """
    gmax = max(gains)
    out = {"max_gain_bpb": gmax, "frontier": [], "knee": {}}
    for frac in (0.50, 0.80, 0.90, 0.95, 0.99):
        target = frac * gmax
        hit = [(c, g, s) for s, g, c in zip(strides, gains, costs) if g >= target]
        if hit:
            c, g, s = min(hit)   # 代价最小者
            out["knee"][f"p{int(frac*100)}"] = {"stride": s, "cost_multiplier": c, "gain_bpb": g}
    out["frontier"] = [{"stride": s, "cost_multiplier": c, "gain_bpb": g}
                       for s, g, c in zip(strides, gains, costs)]
    return out


def pareto_frontier(rows: list) -> list:
    """rows: [{cost_multiplier, bpb, window, stride}] → 保留 Pareto 最优（代价小且 BPB 小）。"""
    pts = sorted(rows, key=lambda r: (r["cost_multiplier"], r["bpb"]))
    best = []
    best_bpb = float("inf")
    for r in pts:
        if r["bpb"] < best_bpb - 1e-12:
            best.append(r)
            best_bpb = r["bpb"]
    return best


# ---------------------------------------------------------------- IO helper

def save_profile(obj, path: str):
    import numpy as np
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path + ".npy", obj["profile"])
    meta = {k: v for k, v in obj.items() if k != "profile"}
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def load_profile(path: str):
    import numpy as np
    with open(path + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["profile"] = np.load(path + ".npy")
    return meta
