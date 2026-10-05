"""
analyze.py — 加载 checkpoint，测量 ρ(p)，扫协议族，做 Pareto / 拐点分析。

## 方法学自检（很重要）
本文件的所有协议对比都是**从 ρ(p) 解析算出**的，省下大量算力。
为了让这个捷径可信，代码同时实现了两种**逐字节直接评测**：
    direct_eval_naive()   —— 真正把验证集切成互不重叠窗口，逐窗口计分
    direct_eval_sliding() —— 真正按 stride 走滑窗，只计分窗口尾部
然后与 ρ 解析值做**逐项对照**。两者若吻合，说明 ρ 重建可信；
若不吻合，说明假设（位置分布可交换）不成立，必须退回直接评测。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import evalproto as EP
from train import load_bin, load_model

LN2 = math.log(2.0)


# ---------------------------------------------------------------- 直接评测（金标准）

@torch.no_grad()
def direct_eval_naive(model, data: torch.Tensor, window: int, batch_windows: int = 32):
    """金标准 A：互不重叠切分，窗口内每个位置都计分 —— 完全复刻官方 eval_val()。"""
    n = (data.numel() - 1) // window * window
    xs, ys = [], []
    for s in range(0, n, window):
        xs.append(data[s: s + window])
        ys.append(data[s + 1: s + window + 1])
    xs = torch.stack(xs).to(torch.long)
    ys = torch.stack(ys).to(torch.long)
    tot, cnt = 0.0, 0
    was = model.training
    model.eval()
    for i in range(0, xs.shape[0], batch_windows):
        _, pp = model(xs[i:i + batch_windows], ys[i:i + batch_windows])
        tot += float(pp.to(torch.float64).sum())
        cnt += int(pp.numel())
    model.train(was)
    return (tot / cnt) / LN2


@torch.no_grad()
def direct_eval_sliding(model, data: torch.Tensor, window: int, stride: int,
                        batch_windows: int = 32):
    """金标准 B：Matthew Li 的评测评测实现 —— 窗口 0 全计分，其余只计分尾部 stride 个。"""
    if stride >= window:
        return direct_eval_naive(model, data, window, batch_windows)
    W, S = window, stride
    n = data.numel()
    starts = list(range(0, n - 1 - W + 1, S))
    xs, ys, masks = [], [], []
    for j, s in enumerate(starts):
        xs.append(data[s: s + W])
        ys.append(data[s + 1: s + W + 1])
        m = torch.zeros(W, dtype=torch.bool)
        # 与 Matthew Li 完全一致：`s = 0 if ws == 0 else max(wlen - stride, 0)`
        if j == 0:
            m[:] = True
        else:
            m[W - S:] = True
        masks.append(m)
    xs = torch.stack(xs).to(torch.long)
    ys = torch.stack(ys).to(torch.long)
    ms = torch.stack(masks)
    tot, cnt = 0.0, 0
    was = model.training
    model.eval()
    for i in range(0, xs.shape[0], batch_windows):
        sl = slice(i, i + batch_windows)
        _, pp = model(xs[sl], ys[sl])
        p64 = pp.to(torch.float64)
        sel = ms[sl]
        tot += float((p64 * sel).sum())
        cnt += int(sel.sum())
    model.train(was)
    return (tot / cnt) / LN2


@torch.no_grad()
def measure_profile(model, data: torch.Tensor, window: int, stride_min: int, batch_windows: int):
    """返回 (profile np.ndarray[window], meta)。profile[p] = 位置 p 的期望 nats/byte。"""
    n = data.numel()
    starts = list(range(0, n - 1 - window + 1, stride_min))
    prof = torch.zeros(window, dtype=torch.float64)
    cnt = 0
    model.eval()
    t0 = time.time()
    for i in range(0, len(starts), batch_windows):
        chunk = starts[i:i + batch_windows]
        xs = torch.stack([data[s: s + window] for s in chunk]).to(torch.long)
        ys = torch.stack([data[s + 1: s + window + 1] for s in chunk]).to(torch.long)
        _, pp = model(xs, ys)
        prof += pp.to(torch.float64).sum(dim=0)
        cnt += pp.shape[0]
    dt = time.time() - t0
    return (prof / cnt).numpy(), {"n_windows": len(starts), "window": window,
                                  "seconds": round(dt, 2),
                                  "tokens_per_sec": round(len(starts) * window / max(dt, 1e-6))}


# ---------------------------------------------------------------- 主分析

STRIDES_512 = [512, 448, 384, 320, 256, 192, 128, 96, 64, 32]
STRIDES_1024 = [1024, 768, 512, 384, 256, 128, 64]


def analyze_seed(ckpt: str, val_path: str, windows=(512, 1024), val_limit=None,
                 batch_windows=32, stride_min=32, do_direct_check=True,
                 out_dir="analysis"):
    model = load_model(ckpt)
    data = load_bin(val_path)
    if val_limit:
        data = data[:val_limit]
    n = data.numel()
    res = {"ckpt": os.path.basename(ckpt), "val_bytes": int(n), "windows": {}, "direct_check": {}}

    for W in windows:
        smin = stride_min if W <= 512 else 64
        prof, meta = measure_profile(model, data, W, smin, batch_windows)
        strides = STRIDES_512 if W == 512 else STRIDES_1024
        rows = EP.sweep_protocols(prof, W, strides, n)
        res["windows"][str(W)] = {
            "profile": prof.tolist(),
            "meta": meta,
            "protocols": rows,
            "naive_bpb": EP.naive_reference(prof)["bpb"],
        }
        print(f"  window={W} 完成：{meta['n_windows']} 窗口，{meta['seconds']}s，"
              f"{meta['tokens_per_sec']:,} tok/s", flush=True)

    if do_direct_check:
        # 用 512 窗口做 ρ 重建 vs 真实直接评测的交叉验证
        W = 512
        prof = np.asarray(res["windows"]["512"]["profile"])
        checks = []
        for S in [512, 256, 128, 64]:
            pred = EP.protocol_bpb(prof, W, S, n)
            actual = direct_eval_sliding(model, data, W, S, batch_windows)
            checks.append({"window": W, "stride": S,
                           "reconstructed_bpb": pred["bpb"],
                           "direct_bpb": actual,
                           "abs_diff": abs(pred["bpb"] - actual)})
            print(f"  交叉验证 stride={S}: ρ重建={pred['bpb']:.6f}  直接评测={actual:.6f}  "
                  f"Δ={abs(pred['bpb']-actual):.2e}", flush=True)
        naive_actual = direct_eval_naive(model, data, W, batch_windows)
        checks.append({"window": W, "stride": 512, "kind": "pure_naive_partition",
                       "reconstructed_bpb": float(prof.mean()) / LN2,
                       "direct_bpb": naive_actual,
                       "abs_diff": abs(float(prof.mean()) / LN2 - naive_actual)})
        res["direct_check"]["512"] = checks

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(ckpt).replace("ckpt_", "").replace(".pt", "")
    p = os.path.join(out_dir, f"analysis_{name}.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {p}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--windows", type=int, nargs="*", default=[512, 1024])
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--stride-min", type=int, default=32)
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--no-direct-check", action="store_true")
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    analyze_seed(a.ckpt, a.val, tuple(a.windows), a.val_limit, a.batch_windows,
                 a.stride_min, not a.no_direct_check, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
