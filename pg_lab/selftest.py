"""
selftest.py — 对提交脚本里两个**纯函数**做单元测试（可在无 GPU 的机器上跑）。

被测对象：C:/Users/卢怡然/Desktop/C2G/卢怡然_C2G_train_gpt.py 里的
    stride_candidates()  —— 候选 stride 生成
    pick_stride_by_knee() —— 拐点选择器的决策逻辑
这两个函数是「rho -> stride」的全部逻辑所在，必须可独立验证。
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.abspath(os.path.join(HERE, "..", "卢怡然_C2G_train_gpt.py"))
LN2 = math.log(2.0)

_results = []


def case(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")


def load_module():
    # 提交脚本面向 8xH100 环境，import 了 sentencepiece 等本实验台用不到的依赖。
    # 被测的两个函数不依赖它们，因此这里注入桩模块，保证自检在任何机器上都能跑。
    import types

    for name in ("sentencepiece",):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.SentencePieceProcessor = object
                sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("c2g_submit", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["c2g_submit"] = m
    spec.loader.exec_module(m)
    return m


def make_profile(window: int, kind: str) -> torch.Tensor:
    """造一条合成 rho(p)：给你想要的形状，用来检验选择器是否做出正确决策。"""
    p = torch.arange(window, dtype=torch.float64)
    if kind == "flat":
        prof = torch.ones(window, dtype=torch.float64) * 2.0
    elif kind == "fast_saturate":
        # 前 128 个位置快速下降，之后完全饱和
        prof = 2.0 - 0.5 * torch.clamp(p / 128.0, max=1.0)
    elif kind == "slow":
        prof = 2.0 - 0.5 * (p / (window - 1.0))
    else:
        raise ValueError(kind)
    return prof


def main() -> int:
    m = load_module()
    print("unit test: ", os.path.basename(SCRIPT))

    def t_syntax_smoke():
        assert hasattr(m, "pick_stride_by_knee"), "缺少 pick_stride_by_knee"
        assert hasattr(m, "stride_candidates"), "缺少 stride_candidates"
        assert hasattr(m, "eval_val_sliding"), "缺少 eval_val_sliding"
        assert hasattr(m, "measure_position_profile"), "缺少 measure_position_profile"

    def t_candidates():
        c = m.stride_candidates(1024, 64)
        assert 1024 in c, "候选必须包含 window 自身（= 朴素评测的退化情形）"
        assert 64 in c, "候选必须包含 stride_min"
        assert all(1024 % s == 0 for s in c), "候选必须整除 window（保证窗口对齐）"
        assert c == sorted(c, reverse=True), "候选应降序（由贵到便宜）"

    def t_flat_no_gain():
        """rho 完全平坦 => 滑窗没有任何收益 => 应退化为朴素评测（stride == window）。"""
        W = 1024
        prof = make_profile(W, "flat")
        s, info = m.pick_stride_by_knee(prof, W, 4.0, 0.95, m.stride_candidates(W, 64))
        assert abs(info["best_gain"]) < 1e-9, f"平坦 rho 的最优收益应为 0，实得 {info['best_gain']}"
        assert s == W, f"无收益时应退化为朴素 stride={W}，实得 {s}"

    def t_fast_saturate_cheap():
        """rho 在 128 token 内饱和 => 在 4x 预算内应该选到便宜的那个，而不是 16x 的 stride=64。"""
        W = 1024
        prof = make_profile(W, "fast_saturate")
        s, info = m.pick_stride_by_knee(prof, W, 4.0, 0.95, m.stride_candidates(W, 64))
        best = info["best_gain"]
        got = info["chosen"]["gain"]
        assert got >= 0.95 * best - 1e-9, f"未拿到 95% 最优收益: got={got} best={best}"
        assert W / s <= 4.0 + 1e-9, f"超出算力预算: cost={W/s}"
        # 关键断言：绝不该为了最后一丝收益付 16 倍代价
        assert s > 64, f"rho 已在前 128 token 饱和，不应选最贵的 stride=64，实得 {s}"

    def t_budget_zero():
        """预算 = 1x（不允许任何额外开销）=> 只能选 window 自身。"""
        W = 1024
        prof = make_profile(W, "fast_saturate")
        s, info = m.pick_stride_by_knee(prof, W, 1.0, 0.95, m.stride_candidates(W, 64))
        assert s == W, f"预算 1x 时只能朴素评测，实得 {s}"

    def t_gain_monotone():
        """收益应随 stride 变小而单调上升（更多上下文不会更差 —— 对合成 rho 成立）。"""
        W = 512
        prof = make_profile(W, "slow")
        rows = m.pick_stride_by_knee(prof, W, 1e9, 0.0, m.stride_candidates(W, 32))[1]["all"]
        rows_sorted_by_stride = sorted(rows, key=lambda r: -r["stride"])
        gains = [r["gain"] for r in rows_sorted_by_stride]
        assert all(g2 >= g1 - 1e-12 for g1, g2 in zip(gains, gains[1:])), f"收益非单调: {gains}"

    def t_naive_reference_matches_mean():
        """朴素评测参考值必须等于整条 rho 的均值 / ln2 —— 这是所有 ΔBPB 的零点。"""
        W = 512
        prof = make_profile(W, "slow")
        info = m.pick_stride_by_knee(prof, W, 1e9, 0.0, m.stride_candidates(W, 32))[1]
        expect = float(prof.mean()) / LN2
        assert abs(info["naive_bpb"] - expect) < 1e-12, f"naive 参考值错误: {info['naive_bpb']} vs {expect}"

    case("脚本存在且导出全部所需函数", t_syntax_smoke)
    case("候选 stride 生成：含 window、整除、降序", t_candidates)
    case("rho 平坦 -> 无收益 -> 退化为朴素评测", t_flat_no_gain)
    case("rho 快速饱和 -> 预算内选便宜解，不付 16x", t_fast_saturate_cheap)
    case("预算 1x -> 强制朴素评测", t_budget_zero)
    case("收益随 stride 单调", t_gain_monotone)
    case("朴素参考值 = mean(rho)/ln2", t_naive_reference_matches_mean)

    bad = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
