"""
build_corpus.py — 构建 C2G 本地实验台所用的确定性语料。

为什么需要本地语料：
    官方榜单的评估集是 FineWeb validation，本实验环境无法访问（离线 + 无 H100），
    因此本实验台**不追求复现官方 BPB 绝对值**，只测量"评测协议"带来的**相对变化**。
    相对量（同一份权重、同一份 held-out 数据，只换评测协议）与语料本身弱相关，
    这是本实验的有效性前提，也在 README 里写明。

为什么用字节级（vocab=256）：
    官方指标 BPB = bits per byte，本来就是字节口径。
    用字节级建模时 bits/byte = loss_nats / ln(2) **精确成立**，无任何 tokenizer 换算误差，
    消除了一个混淆变量。

确定性保证：
    语料由「按路径排序的文件列表 + 逐文件字节」拼成，
    manifest.json 记录 sha256 与字节数，换机器只要 manifest 一致即为同一份数据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOTS = [
    r"D:\develop\NodeJS\node_modules",
    r"C:\Users\卢怡然\.workbuddy\binaries\python\versions\3.13.12\Lib",
]
DEFAULT_EXTS = {".md", ".txt", ".py", ".rst"}


def iter_files(roots, exts):
    seen = set()
    files = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # 排除噪声目录（测试用例、二进制资产），保持语料干净
            dirnames[:] = [d for d in dirnames if d not in
                           {"node_modules", "__pycache__", ".git", "test", "tests", "vendor", "dist", "build"}]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    p = os.path.join(dirpath, fn)
                    key = os.path.normcase(os.path.abspath(p))
                    if key in seen:
                        continue
                    seen.add(key)
                    files.append(p)
    files.sort()
    return files


def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def build(roots, exts, val_every: int, train_cap: int, val_cap: int, out_dir: str, progress=True):
    files = iter_files(roots, exts)
    if not files:
        raise SystemExit("未在指定根目录下找到任何语料文件")

    train_parts, val_parts = [], []
    train_n = val_n = 0
    used = []
    for idx, p in enumerate(files):
        # 用「文件下标取模」做留出划分，而非随机抽样 —— 保证任何机器上划分一致
        bucket = "val" if idx % val_every == 0 else "train"
        limitb = val_cap if bucket == "val" else train_cap
        cur = val_n if bucket == "val" else train_n
        if cur >= limitb:
            continue
        raw = _read_bytes(p)
        if not raw:
            continue
        take = raw[: limitb - cur]
        (val_parts if bucket == "val" else train_parts).append(take)
        if bucket == "val":
            val_n += len(take)
        else:
            train_n += len(take)
        used.append({"path": p, "split": bucket, "bytes": len(take)})
        if progress and len(used) % 200 == 0:
            print(f"  ...已收集 {len(used)} 个文件 train={train_n/1e6:.2f}MB val={val_n/1e6:.2f}MB", flush=True)
        if train_n >= train_cap and val_n >= val_cap:
            break

    train = b"".join(train_parts)
    val = b"".join(val_parts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.bin").write_bytes(train)
    (out / "val.bin").write_bytes(val)

    def sha(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    manifest = {
        "roots": [os.path.abspath(r) for r in roots],
        "extensions": sorted(exts),
        "val_every": val_every,
        "train_cap_bytes": train_cap,
        "val_cap_bytes": val_cap,
        "train_bytes": len(train),
        "val_bytes": len(val),
        "train_sha256": sha(train),
        "val_sha256": sha(val),
        "n_files_used": len(used),
        "filelist_sha256": sha(json.dumps([u["path"] for u in used], ensure_ascii=False).encode("utf-8")),
        "files": used[:400],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="构建确定性本地语料")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--exts", nargs="*", default=sorted(DEFAULT_EXTS))
    ap.add_argument("--val-every", type=int, default=8, help="每 N 个文件留出 1 个做验证集")
    ap.add_argument("--train-cap", type=int, default=4_000_000)
    ap.add_argument("--val-cap", type=int, default=400_000)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    print("构建语料：roots=%d exts=%s" % (len(args.roots), args.exts), flush=True)
    m = build(args.roots, set(args.exts), args.val_every, args.train_cap, args.val_cap, args.out)
    print("\n完成：")
    for k in ("train_bytes", "val_bytes", "train_sha256", "val_sha256", "n_files_used", "filelist_sha256"):
        print(f"  {k}: {m[k]}")
    print(f"\n输出目录: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
