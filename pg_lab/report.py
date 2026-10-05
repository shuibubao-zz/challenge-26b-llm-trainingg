"""
report.py — 汇总 3 个 seed 的 analysis_*.json，产出消融表 / 拐点表 / Pareto 表（Markdown + JSON）。

所有数字都来自真实测量的 rho(p)，没有任何一项是手填的。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

LN2 = math.log(2.0)


def load_all(d: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "analysis_seed*.json"))):
        rows.append(json.load(open(p, encoding="utf-8")))
    return rows


def md_table(headers, rows, aligns=None):
    out = ["| " + " | ".join(headers) + " |"]
    if aligns is None:
        aligns = ["---"] * len(headers)
    out.append("| " + " | ".join(aligns) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build(reports, budget: float):
    lines = []
    # ---- 1. 每个 window 的协议族表 ----
    windows = sorted({w for r in reports for w in r["windows"].keys()}, key=int)
    for W in windows:
        # 收集所有 stride
        strides = sorted({row["stride"] for r in reports for row in r["windows"][W]["protocols"]},
                         reverse=True)
        naive_all = [r["windows"][W]["naive_bpb"] for r in reports]
        head = ["stride", "最小上下文", "代价倍数", "窗口平均数",
                "BPB 均值", "BPB 标准差", "ΔBPB vs 朴素"]
        rows = []
        for S in strides:
            vals = []
            for r in reports:
                m = {x["stride"]: x for x in r["windows"][W]["protocols"]}
                vals.append(m[S]["bpb"])
            mu = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cost = int(W) / S
            rows.append([S, int(W) - S, f"{cost:.2f}x", len(vals),
                         f"{mu:.5f}", f"±{sd:.5f}", f"{mu - st.mean(naive_all):+.5f}"])
        lines.append(f"### 窗口长度 W = {W}\n")
        lines.append(md_table(head, rows, ["---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        lines.append("")

    # ---- 2. 跨 window 的 Pareto 前沿 ----
    all_rows = []
    for W in windows:
        for r in reports:
            for x in r["windows"][W]["protocols"]:
                all_rows.append((int(W), x["stride"], x["cost_multiplier"], x["bpb"]))
    agg = {}
    for W, S, C, B in all_rows:
        agg.setdefault((W, S), {"cost": C, "bpb": []})["bpb"].append(B)
    pts = [{"window": W, "stride": S, "cost": v["cost"], "bpb": st.mean(v["bpb"]),
            "bpb_std": st.stdev(v["bpb"]) if len(v["bpb"]) > 1 else 0.0}
           for (W, S), v in agg.items()]
    pts.sort(key=lambda p: (p["cost"], p["bpb"]))
    frontier, best = [], float("inf")
    for p in pts:
        if p["bpb"] < best - 1e-12:
            frontier.append(p)
            best = p["bpb"]
    rows = [[p["window"], p["stride"], int(p["window"]) - p["stride"], f"{p['cost']:.2f}x",
             f"{p['bpb']:.5f}", f"±{p['bpb_std']:.5f}"] for p in frontier]
    lines.append("### Pareto 前沿（代价更低且 BPB 更低的点）\n")
    lines.append(md_table(["窗口 W", "stride", "最小上下文", "代价倍数", "BPB 均值", "BPB 标准差"],
                          rows, ["---:", "---:", "---:", "---:", "---:", "---:"]))
    lines.append("")

    # ---- 3. 拐点：达到最优收益 X% 所需的最小代价 ----
    # 注意：增益必须**配对**计算 —— 同一个 seed 的 (朴素 − 协议)，再跨 seed 取平均。
    # 用「全局朴素均值 − 单 seed 协议值」会把 seed 间的水平差混进增益里，是错的。
    base = st.mean([r["windows"]["512"]["naive_bpb"] for r in reports])
    paired = {}          # (W, stride) -> [per-seed gain]
    for W in windows:
        for r in reports:
            nb = r["windows"][W]["naive_bpb"]
            for x in r["windows"][W]["protocols"]:
                paired.setdefault((int(W), x["stride"]), {"cost": x["cost_multiplier"], "g": []})
                paired[(int(W), x["stride"])]["g"].append(nb - x["bpb"])
    gains = [(v["cost"], st.mean(v["g"]), W, S) for (W, S), v in paired.items()]
    gmax = max(g for _, g, _, _ in gains)
    rows = []
    for frac in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
        target = frac * gmax
        hit = [(c, g, W, S) for c, g, W, S in gains if g >= target - 1e-15]
        if not hit:
            continue
        c, g, W, S = min(hit)
        rows.append([f"{int(frac*100)}%", f"{frac*gmax:.5f}", W, S, int(W) - S,
                     f"{c:.2f}x", f"{g:.5f}"])
    lines.append("### 拐点分析：拿到最优收益的 X% 最少要付多少代价\n")
    lines.append(md_table(["收益比例", "目标 ΔBPB", "窗口 W", "最省 stride", "最小上下文",
                           "代价倍数", "实得 ΔBPB"], rows,
                          ["---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    lines.append("")

    # ---- 4. rho 形状摘要 ----
    lines.append("### rho(p) 形状（窗口上下 dropout 最剧烈的地方在哪）\n")
    head = ["位置区间"] + [f"seed{i+1}" for i in range(len(reports))] + ["均值"]
    buckets = [(0, 8), (8, 32), (32, 64), (64, 128), (128, 256), (256, 384), (384, 496), (496, 512)]
    rows = []
    per_seed = []
    for r in reports:
        prof = r["windows"]["512"]["profile"]
        per_seed.append([sum(prof[a:b]) / (b - a) / LN2 for a, b in buckets])
    for bi, (a, b) in enumerate(buckets):
        vals = [ps[bi] for ps in per_seed]
        rows.append([f"{a}–{b-1}", *[f"{v:.4f}" for v in vals], f"{st.mean(vals):.4f}"])
    lines.append(md_table(head, rows))
    lines.append("")

    # ---- 5. 配对显著性检验：便宜方案 vs 昂贵方案 ----
    W512 = 512
    def per_seed_bpb(stride):
        out = []
        for r in reports:
            m = {x["stride"]: x["bpb"] for x in r["windows"][str(W512)]["protocols"]}
            out.append(m[stride])
        return out

    def paired_test(a_stride, b_stride):
        """a 相对 b 的配对差：正 = a 更好（BPB 更低）。"""
        a, b = per_seed_bpb(a_stride), per_seed_bpb(b_stride)
        d = [bi - ai for ai, bi in zip(a, b)]     # b - a，正表示 a 更低
        n = len(d)
        mu = st.mean(d)
        sd = st.stdev(d) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")
        t = mu / se if se and se == se else float("nan")
        return {"a": a_stride, "b": b_stride, "cost_a": W512 / a_stride, "cost_b": W512 / b_stride,
                "mean_diff": mu, "std": sd, "stderr": se, "t": t, "n": n}

    tests = [paired_test(256, 32),    # 2x  vs 16x（16x 正是官方 stride=64/1024 的等价代价）
             paired_test(384, 32),    # 1.33x vs 16x
             paired_test(448, 32),    # 1.14x vs 16x
             paired_test(192, 256)]   # 最优 2.67x vs 2x（检验"最优是否显著优于 2x"）
    rows = [[f"{t['cost_a']:.2f}x (stride={t['a']})", f"{t['cost_b']:.2f}x (stride={t['b']})",
             f"{t['mean_diff']:+.5f}", f"{t['std']:.5f}", f"{t['stderr']:.5f}",
             f"{t['t']:.2f}", t["n"]] for t in tests]
    lines.append("### 配对显著性检验（同一 seed 的两个协议相减，n=3）\n")
    lines.append(md_table(["便宜方案", "昂贵方案", "BPB 差（正=便宜方案更好）",
                           "配对标准差", "标准误", "t", "n"], rows,
                          ["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    lines.append("")
    lines.append("> n=3 时 t 的自由度只有 2，不能当作严格的 p 值用。"
                 "这里列出 t 只为说明**差的符号在 3 个 seed 上是否一致**，"
                 "真正的显著性要靠 8×H100 上更多 seed。\n")
    return "\n".join(lines), {"pareto": frontier, "max_gain": gmax, "naive_bpb": base,
                              "paired_tests": tests}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis")
    ap.add_argument("--out", default="analysis/消融附录.md")
    ap.add_argument("--json-out", default="analysis/summary.json")
    ap.add_argument("--budget", type=float, default=4.0)
    a = ap.parse_args()
    reports = load_all(a.dir)
    if not reports:
        raise SystemExit(f"未在 {a.dir} 找到 analysis_seed*.json")
    text, meta = build(reports, a.budget)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    json.dump(meta, open(a.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"seed 数: {len(reports)}   最大可获得 ΔBPB: {meta['max_gain']:.5f}   朴素 BPB: {meta['naive_bpb']:.5f}")
    print("写出:", a.out)
    print(text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
