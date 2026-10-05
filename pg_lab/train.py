"""
train.py — CPU 上的确定性小模型训练，产出 checkpoint 供 profile.py 做协议对比。

关键点：**训练过程对所有 arm 完全一致**。本实验唯一的自变量是"评测协议"，
训练必须逐 bit 可复现（固定 seed → 固定 init → 固定数据顺序）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from pgpt import Config, GPT, build_optimizer, lr_schedule

LN2 = math.log(2.0)


def load_bin(path: str) -> torch.Tensor:
    raw = open(path, "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)


def make_batches(data: torch.Tensor, seq_len: int, batch_seqs: int, step: int, seed: int):
    """按 step 确定性取样：用可复现的伪随机起点，避免整轮 shuffle 的开销。"""
    g = torch.Generator().manual_seed(seed * 1000003 + step)
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_seqs,), generator=g)
    xs = torch.stack([data[s: s + seq_len] for s in starts.tolist()]).to(torch.long)
    ys = torch.stack([data[s + 1: s + seq_len + 1] for s in starts.tolist()]).to(torch.long)
    return xs, ys


def train_one(seed: int, steps: int, seq_len: int, batch_seqs: int, lr: float,
              weight_decay: float, warmup: int, data_path: str, out_dir: str,
              cfg_overrides: dict, log_every: int = 100, val_path: str = "",
              val_every: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

    cfg = Config(**cfg_overrides) if cfg_overrides else Config()
    model = GPT(cfg)
    opt = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    train = load_bin(data_path)

    hist = []
    t0 = time.time()
    print(f"[seed {seed}] params={model.n_params():,} seq={seq_len} batch_seqs={batch_seqs} steps={steps}", flush=True)
    for step in range(steps):
        cur_lr = lr_schedule(step, steps, warmup, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        xs, ys = make_batches(train, seq_len, batch_seqs, step, seed)
        loss, _ = model(xs, ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            msg = f"[seed {seed}] step {step+1}/{steps} loss {loss.item():.4f} bpb {loss.item()/LN2:.4f} lr {cur_lr:.5f} elapsed {time.time()-t0:.1f}s"
            print(msg, flush=True)
            hist.append({"step": step + 1, "train_loss": float(loss.item()),
                         "train_bpb": float(loss.item()) / LN2, "lr": cur_lr,
                         "elapsed_s": round(time.time() - t0, 2)})

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"ckpt_seed{seed}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.param_groups(), "seed": seed,
                "steps": steps, "train_seconds": round(time.time() - t0, 2)}, ckpt)
    json.dump({"seed": seed, "history": hist, "params": model.n_params(),
               "train_seconds": round(time.time() - t0, 2),
               "steps": steps, "seq_len": seq_len, "batch_seqs": batch_seqs,
               "tokens_seen": steps * seq_len * batch_seqs},
              open(os.path.join(out_dir, f"trainlog_seed{seed}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[seed {seed}] 完成 -> {ckpt}", flush=True)
    return {"seed": seed, "ckpt": ckpt, "params": model.n_params(),
            "train_seconds": round(time.time() - t0, 2)}


def load_model(ckpt: str) -> GPT:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**obj["cfg"])
    m = GPT(cfg)
    m.load_state_dict(obj["model"])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--mlp-mult", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    ov = {"num_layers": a.layers, "model_dim": a.dim, "num_heads": a.heads,
          "num_kv_heads": a.kv_heads, "mlp_mult": a.mlp_mult}
    train_one(a.seed, a.steps, a.seq_len, a.batch_seqs, a.lr, a.weight_decay,
              a.warmup, a.data, a.out, ov, a.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
