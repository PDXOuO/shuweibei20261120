# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-20
#   辅助范围    : 误差传播公式实现、枚举与差分进化调用
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""问题2(1) 两站交会定位精度模型与 T2 的最优测站对

模型（一阶误差传播）
------------------
测站 i 对目标的方位角 α_i，测量误差 δα_i ~ N(0, σ²)，σ = 1°，两站独立。
观测方程在真值处线性化：
        ∂α_i/∂p = (-(y_T-y_i), (x_T-x_i)) / r_i²  =  n_i^T / r_i,
        n_i = (-sin α_i, cos α_i)（视线法向单位向量）
=>      n_i^T δp = r_i δα_i
=>      A δp = b,  A=[n_1^T;n_2^T],  b=[r_1 δα_1, r_2 δα_2]^T
=>      Cov(δp) = σ² A^{-1} diag(r_1²,r_2²) A^{-T}
=>      det(A) = ±sin γ  =>  tr Cov = σ² (r_1²+r_2²)/sin²γ
精度指标：σ_p = sqrt(tr Cov) = σ·sqrt(r_1²+r_2²)/|sin γ|（位置误差 RMS，旋转不变）

解法
----
A) 0.5 m 栅格枚举（10 种车辆组合全枚举）+ 0.02 m 坐标下降精修
B) 差分进化（4 维连续全局搜索，以 A 的解作为初始种群成员之一）+ Nelder-Mead 精修

为控制计算量，行驶时间用 0.25 m 栅格上的最短路场 + 最近邻查表（误差 <= 0.083 s）。

运行：python "Q2-1_两站交会定位精度/code/solve.py"
输出：output/q2_1_result.json、output/q2_1_pairs.csv、figures/*.png
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from common import model as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(HERE), "output")
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SIGMA_DEG = 1.0
SIGMA = math.radians(SIGMA_DEG)
T = M.TARGETS["T2"]
TLIM = M.TRAVEL_LIMIT
t_start = time.perf_counter()

# ---------------------------------------------------------------- 精度模型
def sigma_p(s1, s2, sigma=SIGMA):
    s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    r1 = np.hypot(s1[..., 0] - T[0], s1[..., 1] - T[1])
    r2 = np.hypot(s2[..., 0] - T[0], s2[..., 1] - T[1])
    a1 = np.arctan2(T[1] - s1[..., 1], T[0] - s1[..., 0])
    a2 = np.arctan2(T[1] - s2[..., 1], T[0] - s2[..., 0])
    sg = np.abs(np.sin(a1 - a2))
    return sigma * np.sqrt(r1 ** 2 + r2 ** 2) / np.maximum(sg, 1e-12)


def geometry(s1, s2):
    r1 = float(np.hypot(s1[0] - T[0], s1[1] - T[1]))
    r2 = float(np.hypot(s2[0] - T[0], s2[1] - T[1]))
    a1 = math.degrees(math.atan2(T[1] - s1[1], T[0] - s1[0])) % 360
    a2 = math.degrees(math.atan2(T[1] - s2[1], T[0] - s2[0])) % 360
    d = abs(M.wrap180(a1 - a2))
    gamma = d if d <= 90 else 180.0 - d
    B = float(np.hypot(s1[0] - s2[0], s1[1] - s2[1]))
    return r1, r2, B, gamma, a1, a2


def cov_matrix(s1, s2, sigma=SIGMA):
    r1, r2, B, gamma, a1, a2 = geometry(s1, s2)
    n1 = np.array([-math.sin(math.radians(a1)), math.cos(math.radians(a1))])
    n2 = np.array([-math.sin(math.radians(a2)), math.cos(math.radians(a2))])
    A = np.vstack([n1, n2])
    D = np.diag([r1 ** 2, r2 ** 2])
    Ai = np.linalg.inv(A)
    return sigma ** 2 * Ai @ D @ Ai.T


# ---------------------------------------------------------------- 行驶时间查表
STEP_T = 0.25
gx = np.arange(STEP_T / 2, 100.0, STEP_T)
GY, GX = np.meshgrid(gx, gx, indexing="ij")
PTS_T = np.column_stack([GX.ravel(), GY.ravel()])
TT_FIELD = {}
for v, p0 in M.VEH0.items():
    TT_FIELD[v] = (M.PATH.dist_field(p0, PTS_T) / M.SPEED).reshape(GX.shape)


def travel_time(v, x, y):
    """最近邻查表（0.25 m 栅格，时间误差 <= 0.083 s）。"""
    ix = int(round((x - STEP_T / 2) / STEP_T))
    iy = int(round((y - STEP_T / 2) / STEP_T))
    ix = min(max(ix, 0), GX.shape[1] - 1)
    iy = min(max(iy, 0), GX.shape[0] - 1)
    return float(TT_FIELD[v][iy, ix])


def feasible(p, v, tlim=TLIM):
    m, _, _, _ = M.observable(np.array([p[0]]), np.array([p[1]]), T)
    if not m[0]:
        return False
    return travel_time(v, p[0], p[1]) <= tlim


# ---------------------------------------------------------------- 方法A 栅格枚举
STEP = 0.5
xs = np.arange(STEP / 2, 100.0, STEP)
X, Y = np.meshgrid(xs, xs)
PTS = np.column_stack([X.ravel(), Y.ravel()])
mask, r_all, el_all, az_all = M.observable(X, Y, T)
mask = mask.ravel()
near = mask & (r_all.ravel() <= 35.0)
pts = PTS[near]
print(f"[候选] 可观测点 {mask.sum()}/{mask.size}，其中 r<=35m 的 {len(pts)} 个", flush=True)

feas = {}
for v in M.VEH0:
    feas[v] = np.array([travel_time(v, p[0], p[1]) <= TLIM for p in pts])
    print(f"  {v}: 30s 内可达 {feas[v].sum()} 个", flush=True)

rows = []
best_A = None
for v1, v2 in itertools.combinations(M.VEH0, 2):
    A = pts[feas[v1]]
    Bp = pts[feas[v2]]
    if len(A) == 0 or len(Bp) == 0:
        rows.append({"车辆": f"{v1}-{v2}", "可行": False})
        continue
    rA = np.hypot(A[:, 0] - T[0], A[:, 1] - T[1])
    rB = np.hypot(Bp[:, 0] - T[0], Bp[:, 1] - T[1])
    aA = np.arctan2(T[1] - A[:, 1], T[0] - A[:, 0])
    aB = np.arctan2(T[1] - Bp[:, 1], T[0] - Bp[:, 0])
    best = (np.inf, None, None)
    CH = 512
    for i0 in range(0, len(A), CH):
        sl = slice(i0, min(i0 + CH, len(A)))
        sg = np.abs(np.sin(aA[sl, None] - aB[None, :]))
        val = SIGMA * np.sqrt(rA[sl, None] ** 2 + rB[None, :] ** 2) / np.maximum(sg, 1e-12)
        k = int(np.argmin(val))
        ii, jj = divmod(k, val.shape[1])
        if val[ii, jj] < best[0]:
            best = (float(val[ii, jj]), A[i0 + ii], Bp[jj])
    s1, s2 = best[1], best[2]
    r1, r2, B, gamma, a1, a2 = geometry(s1, s2)
    rows.append({"车辆": f"{v1}-{v2}", "可行": True,
                 "s1": [round(float(s1[0]), 3), round(float(s1[1]), 3)],
                 "s2": [round(float(s2[0]), 3), round(float(s2[1]), 3)],
                 "r1": round(r1, 3), "r2": round(r2, 3),
                 "基线B": round(B, 3), "交会角γ": round(gamma, 2),
                 "t1": round(travel_time(v1, *s1), 3), "t2": round(travel_time(v2, *s2), 3),
                 "σ_p_m_栅格": round(best[0], 4)})
    if best_A is None or best[0] < best_A[0]:
        best_A = (best[0], np.array(s1, float), np.array(s2, float), v1, v2)

print(f"[方法A] 最优 {best_A[3]}-{best_A[4]}: σ_p={best_A[0]:.4f} m  ({time.perf_counter()-t_start:.1f}s)", flush=True)


def local_refine(s1, s2, v1, v2, half=0.6, h=0.02, rounds=3):
    s1 = np.array(s1, float); s2 = np.array(s2, float)
    g = np.arange(-half, half + 1e-12, h)
    off = np.column_stack([np.repeat(g, len(g)), np.tile(g, len(g))])
    for _ in range(rounds):
        cand = s1[None, :] + off
        ok = np.array([feasible(c, v1) for c in cand])
        if ok.any():
            vals = np.where(ok, sigma_p(cand, s2[None, :]), np.inf)
            s1 = cand[int(np.argmin(vals))]
        cand = s2[None, :] + off
        ok = np.array([feasible(c, v2) for c in cand])
        if ok.any():
            vals = np.where(ok, sigma_p(s1[None, :], cand), np.inf)
            s2 = cand[int(np.argmin(vals))]
    return s1, s2


s1r, s2r = local_refine(best_A[1], best_A[2], best_A[3], best_A[4])
sig_A = float(sigma_p(s1r, s2r))
print(f"[方法A+] 精修后 σ_p={sig_A:.5f} m  ({time.perf_counter()-t_start:.1f}s)", flush=True)

# ---------------------------------------------------------------- 平局处理
# 几何最优解只取决于 (r, γ)，与"由哪两辆车执行"无关；只要两车各自能在 30 s 内到达
# 相应测站，多组车辆即可取得同一 σ*。故需按"部署最快"裁决：
#   R1: 取 max(行驶时间) 最小者（响应最快，也给云台转动留更多余量）
#   R2: 再并列时取车辆编号字典序最小者（保证唯一、可复现）
tie_cands = []
for row in rows:
    if not row.get("可行"):
        continue
    if row["σ_p_m_栅格"] <= best_A[0] * 1.005:
        v1_, v2_ = row["车辆"].split("-")
        a, b = local_refine(np.array(row["s1"], float), np.array(row["s2"], float), v1_, v2_)
        s_ = float(sigma_p(a, b))
        tie_cands.append({"车辆": row["车辆"], "σ_p": round(s_, 5),
                          "s1": [round(float(a[0]), 4), round(float(a[1]), 4)],
                          "s2": [round(float(b[0]), 4), round(float(b[1]), 4)],
                          "t1": round(travel_time(v1_, *a), 3),
                          "t2": round(travel_time(v2_, *b), 3),
                          "max_t": round(max(travel_time(v1_, *a), travel_time(v2_, *b)), 3)})
tie_cands.sort(key=lambda d: (round(min(d["σ_p"], 1e9), 4), d["max_t"], d["车辆"]))
print(f"[平局] σ* 附近共 {len(tie_cands)} 组车辆组合：", flush=True)
for d in tie_cands:
    print("   ", d["车辆"], d["σ_p"], "max_t=", d["max_t"], flush=True)
if tie_cands:
    b0 = tie_cands[0]
    v1t, v2t = b0["车辆"].split("-")
    s1r = np.array(b0["s1"], float); s2r = np.array(b0["s2"], float)
    best_A = (float(b0["σ_p"]), s1r, s2r, v1t, v2t)
    sig_A = float(b0["σ_p"])


# ---------------------------------------------------------------- 方法B 差分进化
def make_obj(v1, v2):
    def obj(x):
        s1 = x[:2]; s2 = x[2:]
        pen = 0.0
        m1, _, _, _ = M.observable(np.array([s1[0]]), np.array([s1[1]]), T)
        m2, _, _, _ = M.observable(np.array([s2[0]]), np.array([s2[1]]), T)
        if not m1[0]:
            pen += 100.0
        if not m2[0]:
            pen += 100.0
        if pen:
            return pen
        t1 = travel_time(v1, s1[0], s1[1]); t2 = travel_time(v2, s2[0], s2[1])
        if t1 > TLIM or t2 > TLIM:
            pen += 100.0 + 10.0 * (max(t1 - TLIM, 0) + max(t2 - TLIM, 0))
        return float(sigma_p(s1, s2)) + pen
    return obj


top_pairs = sorted([r for r in rows if r.get("可行")], key=lambda r: r["σ_p_m_栅格"])[:4]
best_B = None
rngde = np.random.default_rng(11)
for row in top_pairs:
    v1, v2 = row["车辆"].split("-")
    obj = make_obj(v1, v2)
    x0 = np.array([row["s1"][0], row["s1"][1], row["s2"][0], row["s2"][1]], float)
    init = np.vstack([x0, rngde.uniform(3, 97, (48, 4))])
    de = differential_evolution(obj, [(2, 98)] * 4, seed=11, maxiter=150,
                                popsize=12, tol=1e-9, polish=False, init=init)
    res = minimize(obj, de.x, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 6000, "maxfev": 20000})
    xx = res.x if res.fun < de.fun else de.x
    val = float(sigma_p(xx[:2], xx[2:]))
    if res.fun < 100 and (best_B is None or val < best_B[0]):
        best_B = (val, np.array(xx[:2]), np.array(xx[2:]), v1, v2)
    print(f"[DE] {v1}-{v2}: σ_p={val:.5f} m  ({time.perf_counter()-t_start:.1f}s)", flush=True)

if best_B is None:
    best_B = (np.inf, s1r, s2r, best_A[3], best_A[4])

# ---------------------------------------------------------------- 对裁决出的车辆对做连续精修
if tie_cands:
    v1p, v2p = best_A[3], best_A[4]
    obj_p = make_obj(v1p, v2p)
    res_p = minimize(obj_p, np.concatenate([s1r, s2r]), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 20000, "maxfev": 40000})
    if res_p.fun < 100:
        s1p, s2p = np.array(res_p.x[:2]), np.array(res_p.x[2:])
        val_p = float(sigma_p(s1p, s2p))
        if val_p <= sig_A + 1e-12:
            s1r, s2r, sig_A = s1p, s2p, val_p
    print(f"[精修] {v1p}-{v2p}: σ_p={sig_A:.6f} m, γ={geometry(s1r, s2r)[3]:.3f}°", flush=True)

    # σ_p 在最优附近沿"更优方向"几乎无梯度（sinγ 在 90° 附近极平坦），为得到可复现的
    # 规范解，把两站投影到严格 γ=90° 的位置：保持 r=r_lb，绕目标旋转半角。
    ph1 = math.atan2(s1r[1] - T[1], s1r[0] - T[0])
    ph2 = math.atan2(s2r[1] - T[1], s2r[0] - T[0])
    dphi = M.wrap180(math.degrees(ph1 - ph2))
    mid = ph2 + math.radians(dphi) / 2
    r_now = float(np.hypot(s1r[0] - T[0], s1r[1] - T[1]))
    for r_use in (r_now, (T[2] - M.CAM_H) / math.tan(math.radians(M.EL_MAX))):
        a1n = np.array([T[0] + r_use * math.cos(mid + math.pi / 4),
                        T[1] + r_use * math.sin(mid + math.pi / 4)])
        a2n = np.array([T[0] + r_use * math.cos(mid - math.pi / 4),
                        T[1] + r_use * math.sin(mid - math.pi / 4)])
        if feasible(a1n, v1p) and feasible(a2n, v2p):
            s1r, s2r = a1n, a2n
            sig_A = float(sigma_p(s1r, s2r))
            print(f"[规范解] 投影到 γ=90°: r={r_use:.4f}, σ_p={sig_A:.6f} m", flush=True)
            break

# ---------------------------------------------------------------- 最终取优
# 只有当连续解法明显更优（> 0.1 mm）时才改用它，否则保留"部署最快"的平局裁决结果
use_B = best_B[0] < sig_A - 1e-4
sig_star, s1, s2, v1, v2 = (best_B if use_B else (sig_A, s1r, s2r, best_A[3], best_A[4]))
t1 = M.PATH.travel_time(M.VEH0[v1], tuple(s1))
t2 = M.PATH.travel_time(M.VEH0[v2], tuple(s2))
r1, r2, B, gamma, a1, a2 = geometry(s1, s2)
Cov = cov_matrix(s1, s2)
evals, evecs = np.linalg.eigh(Cov)
sig_major, sig_minor = float(np.sqrt(evals[1])), float(np.sqrt(evals[0]))
angle_major = float(np.degrees(math.atan2(evecs[1, 1], evecs[0, 1])))

# ---------------------------------------------------------------- 蒙特卡洛验证
rng = np.random.default_rng(2026)
NMC = 200000
d1 = rng.normal(0, SIGMA, NMC)
d2 = rng.normal(0, SIGMA, NMC)
ang1 = math.atan2(T[1] - s1[1], T[0] - s1[0]) + d1
ang2 = math.atan2(T[1] - s2[1], T[0] - s2[0]) + d2
ux1, uy1 = np.cos(ang1), np.sin(ang1)
ux2, uy2 = np.cos(ang2), np.sin(ang2)
den = ux1 * uy2 - uy1 * ux2
dx, dy = s2[0] - s1[0], s2[1] - s1[1]
tpar = (dx * uy2 - dy * ux2) / np.where(np.abs(den) < 1e-15, np.nan, den)
px = s1[0] + tpar * ux1
py = s1[1] + tpar * uy1
err = np.hypot(px - T[0], py - T[1])
err = err[np.isfinite(err)]
mc = {
    "样本数": int(err.size),
    "经验RMS_m": round(float(np.sqrt(np.mean(err ** 2))), 5),
    "一阶模型σ_p_m": round(float(sig_star), 5),
    "相对偏差%": round(100 * (float(np.sqrt(np.mean(err ** 2))) / float(sig_star) - 1), 3),
    "经验95%分位_m": round(float(np.percentile(err, 95)), 5),
    "经验中位数_m": round(float(np.median(err)), 5),
}

# ---------------------------------------------------------------- 退化分析
deg = []
for g in (1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 150, 175, 179):
    gg = min(g, 180 - g)
    deg.append({"交会角γ_deg": g,
                "σ_p_m_r20": round(float(SIGMA * math.sqrt(2) * 20.0 / math.sin(math.radians(gg))), 3)})

res = {
    "问题": "2(1) 两站交会定位精度与 T2 最优测站对",
    "目标T2": {"θ": 150, "R": 45, "H": 35, "坐标": [round(float(c), 4) for c in T]},
    "误差模型": {
        "观测误差": "δα_i ~ N(0,(1°)²) 独立",
        "线性化": "n_i^T δp = r_i δα_i, n_i=(-sinα_i, cosα_i)",
        "协方差": "Cov(δp)=σ²A^{-1}diag(r1²,r2²)A^{-T}, det A=±sinγ",
        "迹公式": "tr Cov = σ²(r1²+r2²)/sin²γ",
        "精度指标": "σ_p=sqrt(tr Cov)=σ·sqrt(r1²+r2²)/|sinγ|（位置误差RMS，旋转不变，单位m）",
    },
    "候选集": {"栅格步长_m": STEP, "可观测点数_r<=35m": int(len(pts)),
             "各车30s可达点数": {v: int(feas[v].sum()) for v in M.VEH0}},
    "各车辆对_栅格枚举": rows,
    "方法对比": {"A_栅格枚举+精修_σ_p": round(sig_A, 5),
              "B_差分进化+精修_σ_p": round(best_B[0], 5) if np.isfinite(best_B[0]) else None,
              "相对差%": round(100 * (best_B[0] - sig_A) / sig_A, 4) if np.isfinite(best_B[0]) else None,
              "采用": "B" if use_B else "A"},
    "最优测站对": {
        "车辆": [v1, v2],
        "测站1": [round(float(s1[0]), 4), round(float(s1[1]), 4)],
        "测站2": [round(float(s2[0]), 4), round(float(s2[1]), 4)],
        "r1_m": round(r1, 4), "r2_m": round(r2, 4),
        "基线长度B_m": round(B, 4),
        "交会角γ_deg": round(gamma, 4),
        "方位角α1_deg": round(a1, 4), "方位角α2_deg": round(a2, 4),
        "行驶时间t1_s": round(t1, 3), "行驶时间t2_s": round(t2, 3),
        "σ_p_m": round(float(sig_star), 5),
        "误差椭圆长半轴_m_1σ": round(sig_major, 5),
        "误差椭圆短半轴_m_1σ": round(sig_minor, 5),
        "长轴方向_deg": round(angle_major, 3),
        "95%误差圆半径_m": round(2.4477 * sig_major, 4),
        "GDOP_eff_m每deg": round(float(sig_star) / SIGMA_DEG, 4),
        "仰角1_deg": round(float(M.elev_deg(r1, T)), 3),
        "仰角2_deg": round(float(M.elev_deg(r2, T)), 3),
    },
    "蒙特卡洛验证": mc,
    "退化分析_σ_p随γ": deg,
    "平局组合_同达σstar": tie_cands,
    "鲁棒变体_仰角留余量": [
        {"仰角上限_deg": cap,
         "r_lb_m": round((T[2] - M.CAM_H) / math.tan(math.radians(cap)), 4),
         "σ_p_m": round(float(SIGMA * math.sqrt(2) * (T[2] - M.CAM_H) / math.tan(math.radians(cap))), 5),
         "σ_p相对增量%": round(100 * ((T[2] - M.CAM_H) / math.tan(math.radians(cap)) /
                                ((T[2] - M.CAM_H) / math.tan(math.radians(80.0))) - 1), 3),
         "可达弧占比%": round(100 * float(M.observable(
             T[0] + (T[2] - M.CAM_H) / math.tan(math.radians(cap)) * np.cos(np.linspace(0, 2 * np.pi, 7200)),
             T[1] + (T[2] - M.CAM_H) / math.tan(math.radians(cap)) * np.sin(np.linspace(0, 2 * np.pi, 7200)),
             T, el=(M.EL_MIN, cap + 1e-6))[0].mean()), 2)}
        for cap in (80.0, 78.0, 75.0, 70.0)
    ],
    "理论最优闭式": {
        "结论": "σ_p 在 r1=r2=r_lb 且 γ=90° 时取最小 = σ·√2·r_lb",
        "r_lb_m": round((T[2] - M.CAM_H) / math.tan(math.radians(90 - (90 - M.EL_MAX))), 5),
        "σ_p_min_m": round(float(SIGMA * math.sqrt(2) * (T[2] - M.CAM_H) / math.tan(math.radians(M.EL_MAX))), 5),
        "误差椭圆": "γ=90° 且 r1=r2 时协方差各向同性，误差圆半径 = r·σ = %.5f m" % (5.8188 * SIGMA),
    },
    "耗时_s": round(time.perf_counter() - t_start, 2),
}

with open(os.path.join(OUT, "q2_1_result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, "q2_1_pairs.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------------------------------------------------------------- 图1
fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), dpi=150)
ax = axes[0]
ax.set_facecolor("#f2f2f2")
for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#d5d5d5",
                           edgecolor="#666", lw=0.8, zorder=3))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, k, ha="center", va="center", fontsize=7.5, zorder=4)
ax.imshow(np.where(mask.reshape(X.shape), 1.0, np.nan), origin="lower",
          extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap="Greens", alpha=0.32, vmin=0, vmax=2, zorder=1)
for v, p in M.VEH0.items():
    ax.plot(*p, "s", ms=6, mfc="#1f77b4", mec="k", zorder=6)
    ax.annotate(v, p, textcoords="offset points", xytext=(5, 3), fontsize=7.5)
ax.plot(*T[:2], "^", ms=12, mfc="#d62728", mec="k", zorder=7, label="T2")
for s, v, c in ((s1, v1, "#ff7f0e"), (s2, v2, "#9467bd")):
    ax.plot(*s, "*", ms=14, mfc=c, mec="k", zorder=8)
ax.add_patch(Rectangle((0, 55), 36, 42, fill=False, edgecolor="#333", ls=":", lw=1.0, zorder=9))
ax.annotate("放大区", (36, 55), fontsize=7, ha="right", va="bottom", color="#333")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
ax.set_title("全区域：T2 可观测区（绿）与测站位置", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower left")

ax = axes[1]
ax.set_facecolor("#f2f2f2")
gx2 = np.arange(0, 36.01, 0.1)
gy2 = np.arange(55, 97.01, 0.1)
G2X, G2Y = np.meshgrid(gx2, gy2)
m2, r2, e2, _ = M.observable(G2X, G2Y, T)
ax.imshow(np.where(m2, 1.0, np.nan), origin="lower",
          extent=(gx2[0], gx2[-1], gy2[0], gy2[-1]), cmap="Greens", alpha=0.35, vmin=0, vmax=2, zorder=1)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(T[0] + 5.8188 * np.cos(th), T[1] + 5.8188 * np.sin(th), "r--", lw=1.0,
        label="仰角=80° 临界圆 r=5.8188 m", zorder=4)
for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#d5d5d5",
                           edgecolor="#666", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((x1 - 1.5, y1 - 1.5), x2 - x1 + 3, y2 - y1 + 3, fill=False,
                           edgecolor="#999", ls="--", lw=0.6, zorder=3))
ax.plot([s1[0], s2[0]], [s1[1], s2[1]], "k--", lw=1.0, zorder=5, label=f"基线 B={B:.2f} m")
for (s, v, c, off) in ((s1, v1, "#ff7f0e", (-16, -26)), (s2, v2, "#9467bd", (6, 10))):
    ax.plot(*s, "*", ms=16, mfc=c, mec="k", zorder=8)
    ax.plot([s[0], T[0]], [s[1], T[1]], "-", color=c, lw=1.2, zorder=6)
    ax.annotate(f"{v}\n({s[0]:.2f},{s[1]:.2f})", s, textcoords="offset points",
                xytext=off, fontsize=7)
ax.plot(*T[:2], "^", ms=12, mfc="#d62728", mec="k", zorder=9, label="T2")
ax.annotate(f"γ={gamma:.1f}°", (T[0] + 1.2, T[1] - 4.2), fontsize=8.5, color="#b30000")
ax.set_xlim(0, 36); ax.set_ylim(55, 97); ax.set_aspect("equal")
ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
ax.set_title(f"放大：r1=r2={r1:.3f} m, γ={gamma:.1f}°, σ_p={sig_star:.4f} m", fontsize=9.5)
ax.legend(fontsize=6.8, loc="upper right")

ax = axes[2]
gv = np.linspace(2, 88, 240)
rv = np.linspace(5.9, 40, 240)
GV, RV = np.meshgrid(gv, rv)
SP = SIGMA * np.sqrt(2) * RV / np.sin(np.radians(GV))
cs = ax.contourf(GV, RV, np.log10(SP), levels=26, cmap="viridis_r")
cb = fig.colorbar(cs, ax=ax); cb.set_label("lg(σ_p / m)", fontsize=9)
ax.plot(90, 5.8188, "r*", ms=16, mec="k", label="本解 (γ=90°, r=r_lb)")
ax.set_xlabel("交会角 γ / °"); ax.set_ylabel("目标距离 r1=r2 / m")
ax.set_title("σ_p=σ·√(r1²+r2²)/|sinγ|：γ→0°/180° 发散", fontsize=9.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q2-1_station_pair.png"))
plt.close(fig)

# ---------------------------------------------------------------- 图2
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=150)
ax = axes[0]
ok_rows = [r for r in rows if r.get("可行")]
names = [r["车辆"] for r in ok_rows]
vals = [r["σ_p_m_栅格"] for r in ok_rows]
order = np.argsort(vals)
ax.barh([names[i] for i in order], [vals[i] for i in order], color="#4c72b0")
ax.set_xscale("log")
ax.set_xlabel("σ_p / m（对数）")
ax.set_title("10 种车辆组合各自可达的最优 σ_p", fontsize=10)
for i, o in enumerate(order):
    ax.text(vals[o] * 1.06, i, f"{vals[o]:.3f}", va="center", fontsize=7.5)

ax = axes[1]
th = np.linspace(0, 2 * np.pi, 400)
ca, sa = math.cos(math.radians(angle_major)), math.sin(math.radians(angle_major))
K = 800
ex = K * (sig_major * np.cos(th) * ca - sig_minor * np.sin(th) * sa)
ey = K * (sig_major * np.cos(th) * sa + sig_minor * np.sin(th) * ca)
ax.plot(ex, ey, "r-", lw=1.5, label="1σ 误差椭圆（放大 800 倍）")
ax.plot(0, 0, "k+", ms=12)
ax.axhline(0, color="#ddd", lw=0.6); ax.axvline(0, color="#ddd", lw=0.6)
ax.set_aspect("equal")
ax.set_xlabel("东向偏差 / m"); ax.set_ylabel("北向偏差 / m")
ax.set_title(f"最优构型误差椭圆\n长半轴 {sig_major:.4f} m, 短半轴 {sig_minor:.4f} m", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q2-1_pairs_ellipse.png"))
plt.close(fig)

print(json.dumps({k: res[k] for k in ("方法对比", "最优测站对", "蒙特卡洛验证")},
                 ensure_ascii=False, indent=2))
