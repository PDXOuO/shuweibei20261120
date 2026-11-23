# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-20
#   辅助范围    : 误差传播与蒙特卡洛代码
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""问题3 定位与指向误差对水平方位精度的影响

模型
----
真值：车辆在 P，目标 T 完全已知，水平距离 r = 40 m 固定。
估计：P̂ = P + (δx, δy)，δx, δy ~ N(0, σ(s)²) 独立，σ(s) = a + b·s（题设 a=0.3, b=0.02, s 为累计行驶距离）。
云台零偏 ε_b ~ N(0, 0.8°²)，控制误差 ε_c ~ N(0, 0.6°²)，与定位误差相互独立。

Δψ = 指令方位角 − 真实方位角 = [α(P̂→T) − α(P→T)] + ε_b + ε_c，再卷绕到 [−180°, 180°)。

一阶（delta 法）传播：
    α = atan2(y_T−y_P, x_T−x_P)
    ∂α/∂x_P = (y_T−y_P)/r² = sinα / r ,   ∂α/∂y_P = −(x_T−x_P)/r² = −cosα / r
    δψ_geo ≈ (sinα·δx − cosα·δy)/r  ⇒  σ_geo = σ(s)/r     （与 α 无关，各向同性）
于是
    σ_Δψ(s) = sqrt( (σ(s)/r)²·(180/π)² + 0.8² + 0.6² )   [deg]
要求 P(|Δψ| ≤ 5°) ≥ 0.95，正态近似下等价于 5/σ_Δψ ≥ z_0.975 = 1.95996。

运行：python "Q3_指向误差传播/code/solve.py"
输出：output/q3_result.json、figures/*.png
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(HERE), "output")
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

R = 40.0          # 车辆到目标的水平距离 m
A, B = 0.3, 0.02  # 定位误差标准差模型 σ(s) = a + b s
S_B = 0.8         # 云台零偏 deg
S_C = 0.6         # 云台控制误差 deg
TOL = 5.0         # 允许偏差 deg
CONF = 0.95
Z = float(stats.norm.ppf(0.5 + CONF / 2))     # 1.95996

# ---------------------------------------------------------------- 解析模型
def sigma_pos(s, a=A, b=B):
    return a + b * s


def sigma_geo_deg(s, r=R, a=A, b=B):
    """一维横向偏差 σ(s) 张角的标准差。用 arctan 保持与 r·tanθ 的严格互逆（差异 0.07%）。"""
    return np.degrees(np.arctan(sigma_pos(s, a, b) / r))


def sigma_psi_deg(s, r=R, a=A, b=B):
    return np.sqrt(sigma_geo_deg(s, r, a, b) ** 2 + S_B ** 2 + S_C ** 2)


def prob_within(s, tol=TOL, r=R, a=A, b=B):
    """正态近似下的 P(|Δψ| ≤ tol)。"""
    return 2 * stats.norm.cdf(tol / sigma_psi_deg(s, r, a, b)) - 1


# 目标精度对应的 σ_Δψ 上限
sig_max = TOL / Z
sig_geo_max = math.sqrt(sig_max ** 2 - S_B ** 2 - S_C ** 2)
sig_pos_max = R * math.tan(math.radians(sig_geo_max))          # 精确换算（用 tan，不用小角近似）
sig_pos_max_small = R * math.radians(sig_geo_max)              # 小角近似
s_max = (sig_pos_max - A) / B
s_max_small = (sig_pos_max_small - A) / B

# 80 m 时必须满足的 (a,b) 约束
sig_pos_allow = sig_pos_max          # 定位误差标准差允许上限
b_max_at_a = (sig_pos_allow - A) / 80.0
a_max_at_b = sig_pos_allow - 80.0 * B
b_max_at_a0 = sig_pos_allow / 80.0

# ---------------------------------------------------------------- 蒙特卡洛
rng = np.random.default_rng(20261120)
N = 2_000_000
Tp = np.array([0.0, 0.0])            # 目标置于原点（模型旋转不变，不失一般性）
Pp = np.array([-R, 0.0])             # 车辆真值在目标正西 40 m，真实方位角 = 0°

mc_rows = []
for s in (0.0, 10.0, 20.0, 40.0, 60.0, s_max, 80.0, 100.0):
    sg = sigma_pos(s)
    dx = rng.normal(0, sg, N)
    dy = rng.normal(0, sg, N)
    # 非线性：用估计位置直接算方位，再叠加云台两项误差，最后卷绕
    psi = np.degrees(np.arctan2(Tp[1] - (Pp[1] + dy), Tp[0] - (Pp[0] + dx))) - 0.0
    psi = psi + rng.normal(0, S_B, N) + rng.normal(0, S_C, N)
    psi = np.mod(psi + 180.0, 360.0) - 180.0
    p_emp = float(np.mean(np.abs(psi) <= TOL))
    mc_rows.append({
        "s_m": round(s, 3),
        "σ(s)_m": round(sg, 4),
        "解析σ_Δψ_deg": round(float(sigma_psi_deg(s)), 4),
        "MC_σ_Δψ_deg": round(float(np.std(psi)), 4),
        "解析P(|Δψ|≤5°)": round(float(prob_within(s)), 5),
        "MC_P(|Δψ|≤5°)": round(p_emp, 5),
        "绝对误差": round(abs(p_emp - float(prob_within(s))), 5),
        "MC_95%分位_deg": round(float(np.percentile(np.abs(psi), 95)), 4),
        "偏度": round(float(stats.skew(psi)), 5),
        "峰度": round(float(stats.kurtosis(psi)), 5),
        "卷绕样本占比%": round(float(np.mean(np.abs(psi) > 179.0) * 100), 5),
    })

# KS 检验：Δψ 与同均值方差正态的差异（s=66.9 与 80）
ks = {}
for s in (s_max, 80.0):
    sg = sigma_pos(s)
    dx = rng.normal(0, sg, 300_000)
    dy = rng.normal(0, sg, 300_000)
    psi = np.degrees(np.arctan2(-dy, 40.0 - dx)) + rng.normal(0, S_B, 300_000) + rng.normal(0, S_C, 300_000)
    psi = np.mod(psi + 180, 360) - 180
    z = (psi - psi.mean()) / psi.std(ddof=1)
    k = stats.kstest(z, "norm")
    ks[f"s={s:.2f}"] = {"KS统计量": round(float(k.statistic), 5),
                        "p值": round(float(k.pvalue), 5),
                        "结论": "不能拒绝正态" if k.pvalue > 0.05 else "拒绝正态"}

# 二阶项检查：线性近似 vs 精确的角度误差标准差
second_order = []
for s in (20.0, 66.92, 80.0):
    sg = sigma_pos(s)
    dx = rng.normal(0, sg, 400_000)
    dy = rng.normal(0, sg, 400_000)
    psi_geo = np.degrees(np.arctan2(-dy, 40.0 - dx))
    lin = np.degrees(-dy / 40.0)
    second_order.append({"s_m": s, "精确σ_geo_deg": round(float(np.std(psi_geo)), 5),
                         "一阶σ_geo_deg": round(float(sigma_geo_deg(s)), 5),
                         "相对偏差%": round(100 * (float(np.std(psi_geo)) / float(sigma_geo_deg(s)) - 1), 4),
                         "偏度": round(float(stats.skew(psi_geo)), 5)})

# ---------------------------------------------------------------- 误差源贡献
contrib = []
for s in (0.0, 20.0, 40.0, 60.0, s_max, 80.0):
    v_pos = sigma_geo_deg(s) ** 2
    v_b = S_B ** 2
    v_c = S_C ** 2
    tot = v_pos + v_b + v_c
    contrib.append({"s_m": round(s, 2),
                    "定位项%": round(100 * v_pos / tot, 2),
                    "零偏项%": round(100 * v_b / tot, 2),
                    "控制项%": round(100 * v_c / tot, 2),
                    "σ_Δψ_deg": round(float(sigma_psi_deg(s)), 4)})

# 等贡献点（定位项方差 = 云台两项方差之和）
v_gimbal = S_B ** 2 + S_C ** 2
s_equal = (R * math.tan(math.radians(math.sqrt(v_gimbal))) - A) / B

res = {
    "问题": "3 定位与指向误差对水平方位精度的影响",
    "参数": {"r_m": R, "a_m": A, "b_m每m": B, "云台零偏σ_deg": S_B,
           "云台控制σ_deg": S_C, "容差_deg": TOL, "置信度": CONF, "z": round(Z, 6)},
    "模型": {
        "角度误差一阶传播": "δψ_geo ≈ (sinα·δx − cosα·δy)/r ⇒ σ_geo = σ(s)/r（与方位角无关）",
        "合成": "σ_Δψ(s) = sqrt(σ_geo_deg(s)² + 0.8² + 0.6²)",
        "判据": "P(|Δψ|≤5°) ≥ 0.95 ⟺ 5/σ_Δψ ≥ 1.95996 ⟺ σ_Δψ ≤ 2.5510°",
    },
    "结果": {
        "σ_Δψ允许上限_deg": round(sig_max, 5),
        "定位误差允许上限_m": {"精确(tan)": round(sig_pos_max, 5),
                        "小角近似": round(sig_pos_max_small, 5)},
        "最大允许行驶距离_s_max_m": {"精确": round(s_max, 3), "小角近似": round(s_max_small, 3),
                             "相对差%": round(100 * (s_max_small - s_max) / s_max, 4)},
        "s=0时σ_Δψ_deg": round(float(sigma_psi_deg(0)), 4),
        "s=0时P(|Δψ|≤5°)": float(f"{prob_within(0):.6f}"),
        "s=80时σ_Δψ_deg": round(float(sigma_psi_deg(80)), 4),
        "s=80时P(|Δψ|≤5°)": round(float(prob_within(80)), 5),
        "s=80相对要求的缺口_deg": round(float(sigma_psi_deg(80) - sig_max), 4),
    },
    "80m时的参数约束": {
        "约束式": "a + 80·b ≤ %.4f m（等价于 80b ≤ %.4f − a，a,b ≥ 0）" % (sig_pos_allow, sig_pos_allow),
        "含义": "行驶 80 m 后定位误差标准差不得超过该值，才能保证 P(|Δψ|≤5°) ≥ 0.95",
        "标称值是否满足": bool(A + 80 * B <= sig_pos_allow + 1e-12),
        "标称a+b·80": round(A + 80 * B, 4),
        "给定a=0.3时b上限": round(b_max_at_a, 6),
        "给定b=0.02时a上限": round(a_max_at_b, 4),
        "a=0时b上限": round(b_max_at_a0, 6),
        "可行域描述": "a ∈ [0, %.4f]，且 b ≤ (%.4f − a)/80；三角形可行域，a 越小允许的 b 越大" % (sig_pos_allow, sig_pos_allow),
    },
    "蒙特卡洛": mc_rows,
    "正态性检验": ks,
    "二阶项检验": second_order,
    "误差源贡献": contrib,
    "定位项与云台项等贡献点_m": round(s_equal, 3),
}

with open(os.path.join(OUT, "q3_result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 图1
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=155)
ss = np.linspace(0, 120, 600)
ax = axes[0]
ax.plot(ss, sigma_psi_deg(ss), "b-", lw=2, label="σ_Δψ(s)")
ax.plot(ss, sigma_geo_deg(ss), "b--", lw=1.1, label="定位项 σ_geo(s)")
ax.axhline(S_B, color="#ff7f0e", ls=":", lw=1.2, label="云台零偏 0.8°")
ax.axhline(S_C, color="#2ca02c", ls=":", lw=1.2, label="云台控制 0.6°")
ax.axhline(sig_max, color="r", ls="--", lw=1.4, label=f"允许上限 {sig_max:.3f}°")
ax.axvline(s_max, color="k", ls="-.", lw=1.0)
ax.plot([s_max], [sig_max], "r*", ms=15, mec="k")
ax.annotate(f"s_max={s_max:.2f} m", (s_max, sig_max), textcoords="offset points",
            xytext=(8, -22), fontsize=8.5)
ax.set_xlabel("累计行驶距离 s / m"); ax.set_ylabel("标准差 / °")
ax.set_title("各误差源标准差随行驶距离的增长", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left"); ax.grid(ls=":", alpha=0.5)
ax.set_ylim(0, 4.2)

ax = axes[1]
ax.plot(ss, prob_within(ss), "b-", lw=2, label="P(|Δψ|≤5°)（正态近似）")
ax.axhline(CONF, color="r", ls="--", lw=1.2, label="要求 0.95")
ax.axvline(s_max, color="k", ls="-.", lw=1.0)
ax.plot([80], [prob_within(80)], "rv", ms=9, label=f"s=80 m 时 {prob_within(80):.3f}")
ax.plot([0], [prob_within(0)], "g^", ms=9, label=f"s=0 时 {prob_within(0):.3f}")
ax.set_xlabel("累计行驶距离 s / m"); ax.set_ylabel("概率")
ax.set_title("满足置信要求的最大行驶距离", fontsize=10)
ax.legend(fontsize=8, loc="lower left"); ax.grid(ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q3_sigma_vs_s.png"))
plt.close(fig)

# ---------------------------------------------------------------- 图2 分布与贡献
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=155)
ax = axes[0]
for s, c in ((s_max, "#1f77b4"), (80.0, "#d62728")):
    sg = sigma_pos(s)
    dx = rng.normal(0, sg, 400_000); dy = rng.normal(0, sg, 400_000)
    psi = np.degrees(np.arctan2(-dy, 40.0 - dx)) + rng.normal(0, S_B, 400_000) + rng.normal(0, S_C, 400_000)
    psi = np.mod(psi + 180, 360) - 180
    ax.hist(psi, bins=180, density=True, histtype="step", lw=1.4, color=c,
            label=f"MC 直方图 s={s:.1f} m")
    xx = np.linspace(-10, 10, 400)
    ax.plot(xx, stats.norm.pdf(xx, 0, sigma_psi_deg(s)), "--", color=c, lw=1.0,
            label=f"正态近似 σ={sigma_psi_deg(s):.3f}°")
ax.axvline(5, color="k", ls=":"); ax.axvline(-5, color="k", ls=":")
ax.set_xlim(-9, 9)
ax.set_xlabel("Δψ / °"); ax.set_ylabel("概率密度")
ax.set_title("Δψ 的经验分布 vs 正态近似（虚线为 ±5° 门限）", fontsize=10)
ax.legend(fontsize=7); ax.grid(ls=":", alpha=0.4)

ax = axes[1]
s_grid = np.linspace(0, 80, 200)
vp = sigma_geo_deg(s_grid) ** 2
vb = np.full_like(s_grid, S_B ** 2)
vc = np.full_like(s_grid, S_C ** 2)
tot = vp + vb + vc
ax.stackplot(s_grid, 100 * vp / tot, 100 * vb / tot, 100 * vc / tot,
             labels=["车辆定位", "云台零偏", "云台控制"],
             colors=["#4c72b0", "#dd8452", "#55a868"], alpha=0.85)
ax.axvline(s_equal, color="k", ls="--", lw=1.0)
ax.annotate(f"等贡献点 s≈{s_equal:.1f} m", (s_equal, 50), textcoords="offset points",
            xytext=(6, 40), fontsize=8)
ax.set_xlabel("累计行驶距离 s / m"); ax.set_ylabel("方差贡献 / %")
ax.set_ylim(0, 100); ax.set_xlim(0, 80)
ax.set_title("三个误差源的方差贡献随行驶距离的变化", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q3_distribution.png"))
plt.close(fig)

print(json.dumps(res, ensure_ascii=False, indent=2))
