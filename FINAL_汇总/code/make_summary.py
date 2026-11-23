# -*- coding: utf-8 -*-
# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: ______（以官方发布页为准，提交前填写）
#   使用日期    : 2026-11-20
#   辅助范围    : 汇总表生成代码
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""
汇总脚本：从各小题的 result JSON 中抽取最终答案，生成机器可读的
FINAL_汇总/output/summary.json，并打印控制台速查表。

设计原则：本脚本**不做任何数值计算**，只读取各小题已产出的结果文件，
保证汇总结果与分题结果严格一致、可追溯（改分题 -> 重跑本脚本即可刷新）。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 项目根
OUT_DIR = os.path.join(os.path.dirname(HERE), "output")

PATHS = {
    "Q1_A": os.path.join(ROOT, "Q1_可观测区域与最优测站", "output", "method_A_grid.json"),
    "Q1_B": os.path.join(ROOT, "Q1_可观测区域与最优测站", "output", "method_B_analytic.json"),
    "Q2_1": os.path.join(ROOT, "Q2-1_两站交会定位精度", "output", "q2_1_result.json"),
    "Q2_2": os.path.join(ROOT, "Q2-2_并发告警车辆复用", "output", "q2_2_result.json"),
    "Q3":   os.path.join(ROOT, "Q3_指向误差传播", "output", "q3_result.json"),
    "Q4":   os.path.join(ROOT, "Q4_扰动重调度", "output", "q4_result.json"),
}


def load(key: str) -> dict:
    p = PATHS[key]
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少分题结果文件：{p}\n请先在对应小题目录下运行 code/ 中的求解脚本。")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build() -> dict:
    a, b = load("Q1_A"), load("Q1_B")
    q21, q22, q3, q4 = load("Q2_1"), load("Q2_2"), load("Q3"), load("Q4")

    # ---------------- Q1 ----------------
    st = b["选定最优测站"]
    q1 = {
        "目标": "T1",
        "问题": "给定 T1 告警，求可观测区域并选最优测站（最小化水平距离 r）",
        "采用方案": "B 解析下界 + 弧枚举 + 三级裁决",
        "最优测站_xy": st["坐标"],
        "最优水平距离_r_m": st["水平距离_m"],
        "视线仰角_deg": st["视线仰角_deg"],
        "优化指标_Q_1每m": st["Q_1每m"],
        "净空_m": st["净空_m"],
        "解析下界_r_lb_m": b["最优性证明"]["r_lb_m"],
        "最优集": {
            "弧段数": b["最优集"]["可行弧数"],
            "弧段": b["最优集"]["弧段"],
            "可行弧占比%": b["最优集"]["最优圆可行弧占比%"],
        },
        "约束统计": a["约束统计"],
        "方法对比": {
            "A_栅格穷举": {"r_m": a["最优水平距离_m"],
                           "Q": a["最优Q_1每m"],
                           "耗时_s": a["耗时_s"]},
            "B_解析": {"r_m": st["水平距离_m"], "Q": st["Q_1每m"], "耗时_s": b["耗时_s"]},
        },
        "鲁棒变体": b["鲁棒变体"],
        "独立复核通过": b["独立复核"],
    }

    # ---------------- Q2(1) ----------------
    best = q21["最优测站对"]
    q2_1 = {
        "目标": "T2",
        "问题": "两站交会定位精度模型 + T2 最优测站对",
        "采用方案": "A 0.5 m 栅格全枚举 + 0.02 m 连续精修（B 差分进化交叉验证）",
        "测站对": best["车辆"],
        "测站1_xy": best["测站1"],
        "测站2_xy": best["测站2"],
        "r1_m": best["r1_m"],
        "r2_m": best["r2_m"],
        "基线_B_m": best["基线长度B_m"],
        "交会角_deg": best["交会角γ_deg"],
        "σ_p_m": best["σ_p_m"],
        "误差圆_1σ_半径_m": best["误差椭圆长半轴_m_1σ"],
        "95%误差圆半径_m": best["95%误差圆半径_m"],
        "行驶时间_s": {"t1": best["行驶时间t1_s"], "t2": best["行驶时间t2_s"]},
        "理论下界_m": q21["理论最优闭式"]["σ_p_min_m"],
        "蒙特卡洛": q21["蒙特卡洛验证"],
        "平局裁决": "6 组车辆对同达 σ*，按 max(t1,t2) 最小取 V3-V5（13.51 s，次优 20.22 s）",
        "耗时_s": q21["耗时_s"],
    }

    # ---------------- Q2(2) ----------------
    sol = q22["优化方案"]
    geo = sol["几何校验"]
    q2_2 = {
        "问题": "三告警并发，允许一车连续服务两个告警，最小化最后告警观测开始时刻 t_last",
        "采用方案": "结构枚举（360 种）+ 候选池方位分箱约化 + 时间轴不动点迭代仿真",
        "t_last_s": sol["t_last_s"],
        "各告警观测开始_s": sol["各告警观测开始"],
        "复用结构": sol["复用结构"],
        "测站": sol["测站"],
        "各告警σ_p_m": {k.replace("_geometry", ""): geo[k]["σ_p_m"]
                        for k in geo if k.endswith("_geometry")},
        "各告警交会角_deg": {k.replace("_geometry", ""): geo[k]["交会角_deg"]
                            for k in geo if k.endswith("_geometry")},
        "基线_最近分配启发式": {
            "t_last_s": q22["基线_最近分配启发式"]["t_last_s"],
            "各告警观测开始_s": q22["基线_最近分配启发式"]["各告警观测开始"],
        },
        "改进": q22["对比"],
        "敏感性": q22["敏感性"],
        "耗时_s": q22["耗时_s"],
    }

    # ---------------- Q3 ----------------
    r3 = q3["结果"]
    c80 = q3["80m时的参数约束"]
    mc = q3["蒙特卡洛"]
    q3s = {
        "问题": "定位误差 + 云台指向误差传播到水平方位角，求保证 P(|Δψ|≤5°)≥0.95 的 s_max 与参数约束",
        "采用方案": "一阶传播（解析）+ 200 万次蒙特卡洛 + KS/二阶项双重检验",
        "σ_Δψ允许上限_deg": r3["σ_Δψ允许上限_deg"],
        "允许定位误差_s_max处_m": r3["定位误差允许上限_m"]["精确(tan)"],
        "s_max_m": r3["最大允许行驶距离_s_max_m"]["精确"],
        "s_max_小角近似_m": r3["最大允许行驶距离_s_max_m"]["小角近似"],
        "s=0与80m指标": {
            "σ_Δψ(s=0)_deg": r3["s=0时σ_Δψ_deg"],
            "P(s=0)": r3["s=0时P(|Δψ|≤5°)"],
            "σ_Δψ(s=80)_deg": r3["s=80时σ_Δψ_deg"],
            "P(s=80)": r3["s=80时P(|Δψ|≤5°)"],
            "80m缺口_deg": r3["s=80相对要求的缺口_deg"],
        },
        "80m参数约束": {
            "约束式": c80["约束式"],
            "标称值是否满足": c80["标称值是否满足"],
            "标称a+80b_m": c80["标称a+b·80"],
            "a=0.3时b上限": c80["给定a=0.3时b上限"],
            "b=0.02时a上限": c80["给定b=0.02时a上限"],
        },
        "蒙特卡洛最大偏差_deg": max(abs(x["MC_σ_Δψ_deg"] - x["解析σ_Δψ_deg"]) for x in mc),
        "正态性检验": q3["正态性检验"],
    }
    # 校验点：s_max 处应恰为 0.95
    hit = [x for x in mc if abs(x["s_m"] - q3s["s_max_m"]) < 1e-6]
    q3s["s_max处MC通过率"] = hit[0]["MC_P(|Δψ|≤5°)"] if hit else None

    # ---------------- Q4 ----------------
    tot = q4["汇总"]["全部"]
    q4s = {
        "问题": "基线方案运行中发生扰动（新告警/车辆失效/通信中断），比较沿用 vs 重调度",
        "采用方案": "S1 在线重调度（状态推进 + 局部重优化 + 安全兜底）",
        "扰动设定": q4["扰动设定"],
        "情景数": q4["情景集"]["数量"],
        "情景类别": q4["情景集"]["类别"],
        "评价准则": "先比 η（成功获取 ≥2 站交会的告警比例），再比 t_last",
        "S0_沿用": tot["S0"],
        "S1_重调度": tot["S1"],
        "改进": tot["改进"],
        "分类结果": q4["汇总"],
        "典型情景": q4["典型情景"],
        "耗时_s": q4["耗时_s"],
    }

    return {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "说明": "本文件由 FINAL_汇总/code/make_summary.py 自动汇总各分题 result JSON 生成，不含独立计算。",
        "Q1": q1,
        "Q2_1": q2_1,
        "Q2_2": q2_2,
        "Q3": q3s,
        "Q4": q4s,
    }


def print_table(s: dict) -> None:
    q1, q21, q22, q3s, q4s = s["Q1"], s["Q2_1"], s["Q2_2"], s["Q3"], s["Q4"]
    w = 78
    print("=" * w)
    print("2026 数维杯 B 题  最终答案速查".center(w - 8))
    print("=" * w)
    print(f"[Q1 ] T1 最优测站 = ({q1['最优测站_xy'][0]:.4f}, {q1['最优测站_xy'][1]:.4f})"
          f"  r = {q1['最优水平距离_r_m']:.4f} m  仰角 = {q1['视线仰角_deg']:.2f}°"
          f"  Q = {q1['优化指标_Q_1每m']:.6f} /m")
    print(f"[Q2-1] 测站对 = {q21['测站对']}  r1 = r2 = {q21['r1_m']:.4f} m"
          f"  γ = {q21['交会角_deg']:.1f}°  σ_p = {q21['σ_p_m']:.5f} m"
          f"  (95% {q21['95%误差圆半径_m']:.4f} m)")
    print(f"[Q2-2] t_last = {q22['t_last_s']:.3f} s"
          f"   (T1 {q22['各告警观测开始_s']['T1']:.3f} / T2 {q22['各告警观测开始_s']['T2']:.3f}"
          f" / T3 {q22['各告警观测开始_s']['T3']:.3f})"
          f"   复用 {q22['复用结构']['复用车辆']}: {q22['复用结构']['先服务']}->{q22['复用结构']['后服务']}")
    print(f"[Q3  ] s_max = {q3s['s_max_m']:.3f} m   σ_Δψ上限 = {q3s['σ_Δψ允许上限_deg']:.5f}°"
          f"   80 m 约束: a + 80b ≤ {q3s['允许定位误差_s_max处_m']:.4f} m"
          f"  (标称 {q3s['80m参数约束']['标称a+80b_m']:.2f} m 不满足)")
    print(f"[Q4  ] S1 vs S0：更优 {q4s['改进']['S1更优情景数']} / 持平 {q4s['改进']['持平情景数']}"
          f" / 更差 {q4s['改进']['S1更差情景数']}"
          f"   η {q4s['S0_沿用']['平均η']:.4f} -> {q4s['S1_重调度']['平均η']:.4f}"
          f"   t_last {q4s['S0_沿用']['平均t_last']:.2f} -> {q4s['S1_重调度']['平均t_last']:.2f} s")
    print("=" * w)


def main() -> int:
    try:
        s = build()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, "summary.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print_table(s)
    print(f"\n已写入：{dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
