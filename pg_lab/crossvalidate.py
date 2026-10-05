"""
crossvalidate.py — 用本地实测数据检验**交付脚本的核心决策逻辑**能否站得住。

要回答一个问题（这也是方案最大的风险点）：
    交付脚本是在**训练分布**的一段 token 上标定 eval stride 的（刻意不在验证集上调参）。
    凭什么相信这样标定出来的 stride 到了验证分布上还是对的？

论证链条：stride 的选择只依赖 rho 的**形状**（随上下文下降的相对速度），不依赖它的**水平**。
这里把链条拆开逐步验证：
    V1  同一份权重下，rho_val 与 rho_train 的**水平**是否不同？（预期显著不同）
    V2  归一化后的**形状**是否接近？（预期很接近）
    V3  分别用 rho_val / rho_train 送入提交脚本的 pick_stride_by_knee()，选出的 stride 是否一致？
    V4  两者各自算出的验证集 BPB 是否一致？
若 V1 成立而 V2/V3/V4 也成立，则"用训练分布标定、在验证分布生效"得到支持。

本文件**直接 import 交付脚本**里的 pick_stride_by_knee / stride_candidates，
所以本地实验台就是交付件的测试装置 —— 不是两套各写一遍的逻辑。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import torch

import evalproto as EP
from analyze import measure_profile
from train import load_bin, load_model

LN2 = math.log(2.0)
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.abspath(os.path.join(HERE, "..", "卢怡然_C2G_train_gpt.py"))


def load_submission_module():
    for name in ("sentencepiece",):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.SentencePieceProcessor = object
                sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("c2g_submit_cv", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["c2g_submit_cv"] = m
    spec.loader.exec_module(m)
    return m


def shape_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """比较形状而非水平：先各自减去末段均值（去掉水平差），再比较残差曲线。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a2 = a - a[-64:].mean()
    b2 = b - b[-64:].mean()
    return {
        "level_diff_nats": float(a.mean() - b.mean()),
        "shape_l2": float(np.sqrt(np.mean((a2 - b2) ** 2))),
        "shape_max_abs": float(np.max(np.abs(a2 - b2))),
        "shape_scale": float(np.std(a2)),
    }


def run(ckpt: str, val_path: str, train_path: str, window: int, stride_min: int,
        batch_windows: int, budgets, out_dir: str, threads: int):
    torch.set_num_threads(threads)
    sm = load_submission_module()
    model = load_model(ckpt)
    val = load_bin(val_path)
    train_raw = load_bin(train_path)

    # 训练集.n. calibrate 片段：取 train.bin 的中段（避开可能的头尾偏置）
    mid = train_raw.numel() // 2
    calib_train = train_raw[mid: mid + 1_500_000]

    print("测量 rho_val ...", flush=True)
    prof_val, meta_val = measure_profile(model, val, window, stride_min, batch_windows)
    print("测量 rho_train ...", flush=True)
    prof_train, meta_train = measure_profile(model, calib_train, window, stride_min, batch_windows)

    V1V2 = shape_distance(prof_val, prof_train)
    print(f"  水平差 nats/byte = {V1V2['level_diff_nats']:+.5f}  "
          f"形状残差 L2 = {V1V2['shape_l2']:.6f} (曲线自身标准差 {V1V2['shape_scale']:.6f})", flush=True)

    cands = sm.stride_candidates(window, stride_min)
    rows = []
    for budget in budgets:
        s_val, info_val = sm.pick_stride_by_knee(torch.tensor(prof_val), window, budget, 0.95, cands)
        s_train, info_train = sm.pick_stride_by_knee(torch.tensor(prof_train), window, budget, 0.95, cands)
        # 用 rho_val 算两种 stride 实际落在验证集上的 BPB
        bpb_val = EP.protocol_bpb(prof_val, window, s_val, val.numel())["bpb"]
        bpb_train_calib = EP.protocol_bpb(prof_val, window, s_train, val.numel())["bpb"]
        rows.append({
            "budget_cost_multiplier": budget,
            "stride_from_val_rho": s_val,
            "stride_from_train_rho": s_train,
            "stride_agree": s_val == s_train,
            "val_bpb_if_stride_from_val_rho": bpb_val,
            "val_bpb_if_stride_from_train_rho": bpb_train_calib,
            "bpb_gap": abs(bpb_val - bpb_train_calib),
            "chosen_cost": window / s_train,
        })
        print(f"  预算 {budget:>4.1f}x: stride(val 标定)={s_val:<5d} stride(train 标定)={s_train:<5d} "
              f"一致={s_val == s_train}  ΔBPB={abs(bpb_val - bpb_train_calib):.2e}", flush=True)

    out = {
        "ckpt": os.path.basename(ckpt),
        "window": window,
        "stride_min": stride_min,
        "level_and_shape": V1V2,
        "budgets": rows,
        "meta": {"val": meta_val, "train": meta_train},
        "rho_val_head": prof_val[::max(1, window // 32)].tolist(),
        "rho_train_head": prof_train[::max(1, window // 32)].tolist(),
    }
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(ckpt).replace("ckpt_", "").replace(".pt", "")
    p = os.path.join(out_dir, f"crossval_{name}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("->", p, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--train", default="data/train.bin")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride-min", type=int, default=32)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--budgets", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    run(a.ckpt, a.val, a.train, a.window, a.stride_min, a.batch_windows, a.budgets, a.out, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
