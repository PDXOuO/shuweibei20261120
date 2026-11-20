"""问题1 方法B：解析下界 + 边界弧枚举 + 连续精修

核心思路（给出最优性证明，而不只是数值搜索）：
  三项约束中，仰角约束给出水平距离的硬下界
        r >= r_lb = (H - 2) / tan(80°) = 7.58206 m,
  因为 Q = 1/r 关于 r 严格单调递减，故 Q <= 1/r_lb 对一切可行测站成立。
  若在圆 r = r_lb 上存在同时满足"可停留 + 视线不遮挡"的点，则这些点即为全局最优解，
  且最优集是一条（或数条）圆弧 —— 平局是结构性的，必须给出裁决规则。

  同时给出"鲁棒变体"：把仰角上界收紧到 78°（留 2° 余量），看 Q 的损失，用于讨论
  "数学最优"与"工程可用"的差别。

运行：python "Q1_可观测区域与最优测站/code/method_B_analytic.py"
输出：output/method_B_analytic.json、figures/Q1_optimal_arc.png
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from common import model as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Wedge  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(HERE), "output")
FIG = os.path.join(os.path.dirname(HERE), "figures")
T = M.TARGETS["T1"]
t0 = time.perf_counter()

# ---------------------------------------------------------------- 1 仰角下界
r_lb = (T[2] - M.CAM_H) / math.tan(math.radians(M.EL_MAX))
Q_star = 1.0 / r_lb

# ---------------------------------------------------------------- 2 最优圆上的可行弧
NA = 36000
phi = np.linspace(0.0, 360.0, NA, endpoint=False)      # 目标 -> 测站的方位（自正东逆时针）
px = T[0] + r_lb * np.cos(np.radians(phi))
py = T[1] + r_lb * np.sin(np.radians(phi))
# 临界圆上仰角恰为 80°，浮点抖动会让 el 有 1e-13 量级的越界，故给 1e-6 度容差
EL_TOL = 1e-6
ok, r_chk, el_chk, az_chk = M.observable(px, py, T, el=(M.EL_MIN, M.EL_MAX + EL_TOL))

# 连续弧段提取
idx = np.nonzero(ok)[0]
arcs = []
dphi = 360.0 / NA
if idx.size:
    # 环形扫描：0->1 为弧起点，1->0 为弧终点（首尾环形处理）
    ob = ok.astype(np.int8)
    prev = np.concatenate([[ob[-1]], ob[:-1]])
    starts = np.nonzero((ob == 1) & (prev == 0))[0]
    ends = np.nonzero((ob == 0) & (prev == 1))[0]
    if ob.all():
        starts, ends = np.array([0]), np.array([0])
    for s0 in starts:
        # 找 s0 之后第一个终点
        cands = ends[ends != s0]
        if cands.size == 0:
            continue
        e0 = cands.min() if cands.min() > s0 else cands.max()
        if e0 <= s0:
            e0 += NA
        length = (e0 - s0) * dphi
        if length < 0.5:
            continue
        arcs.append({"起_deg": round(float(s0 * dphi), 2),
                     "止_deg": round(float((e0 % NA) * dphi), 2),
                     "弧长_deg": round(float(length), 2)})
    arcs.sort(key=lambda a: -a["弧长_deg"])

# ---------------------------------------------------------------- 3 平局裁决
# 规则：在最优集内最大化"障碍与边界净空"，再以"离调度中心 O 最近"作第二级；
#       再以"离 80° 临界圆的弧中点最近"作第三级（数值稳定）。净空反映抗定位误差能力。
def clearance(x, y):
    d_bld = float(M.building_clearance(np.array([x]), np.array([y]))[0]) - M.R_VEH
    d_bd = float(min(x - M.R_VEH, M.AREA - M.R_VEH - x, y - M.R_VEH, M.AREA - M.R_VEH - y))
    return min(d_bld, d_bd)


cand = np.column_stack([px[idx], py[idx]])
phi_ok = phi[idx]
clr = np.array([clearance(x, y) for x, y in cand])
d_O = np.hypot(cand[:, 0] - 50.0, cand[:, 1] - 50.0)

order = np.lexsort((np.arange(len(cand)), d_O, -np.round(clr, 6)))
# 净空优先（四舍五入到 1e-6 避免浮点抖动），再离 O 近，再取序号小
best_i = int(order[0])
best = cand[best_i]
best_phi = float(phi_ok[best_i])
best_clr = float(clr[best_i])
best_r = float(np.hypot(best[0] - T[0], best[1] - T[1]))
best_el = float(M.elev_deg(best_r, T))
best_az = float(M.azimuth_to(best[0], best[1], T))
best_Q = 1.0 / best_r

# 对照：若只按"离 O 最近"裁决（可说明平局规则影响）
i_O = int(np.argmin(d_O))
alt_O = cand[i_O]

# 对照：净空最大的点
i_clr = int(np.argmax(clr))
alt_clr = cand[i_clr]

# ---------------------------------------------------------------- 4 独立复核
# 用 1 mm 步长的视线采样复核最优测站（不依赖 los_blocked 的解析判据）
def brute_los(p, target, step=0.001):
    p = np.asarray(p, float); target = np.asarray(target, float)
    L = np.linalg.norm(target - p)
    ts = np.linspace(0, 1, max(int(L / step), 3))
    pts = p[None, :] + ts[:, None] * (target - p)[None, :]
    for (x1, x2, y1, y2, h) in M.BUILDINGS.values():
        inside = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) &
                  (pts[:, 1] <= y2) & (pts[:, 2] >= 0) & (pts[:, 2] <= h))
        if inside.any():
            return True
    return False


verify = {
    "可停留": bool(M.free_space(np.array([best[0]]), np.array([best[1]]))[0]),
    "解析判据_视线遮挡": bool(M.los_blocked(np.array([[best[0], best[1], M.CAM_H]]), T)[0]),
    "独立采样_视线遮挡": bool(brute_los(np.array([best[0], best[1], M.CAM_H]), T)),
    "仰角在范围内": bool(M.EL_MIN <= best_el <= M.EL_MAX),
}

# ---------------------------------------------------------------- 5 鲁棒变体
rob = {}
for el_cap in (78.0, 75.0, 70.0):
    rr = (T[2] - M.CAM_H) / math.tan(math.radians(el_cap))
    th = np.linspace(0, 2 * np.pi, 20000)
    qx = T[0] + rr * np.cos(th); qy = T[1] + rr * np.sin(th)
    okk, _, _, _ = M.observable(qx, qy, T, el=(M.EL_MIN, el_cap + EL_TOL))
    rob[f"仰角上限{el_cap:g}deg"] = {
        "r_lb_m": round(rr, 4), "Q_1每m": round(1 / rr, 6),
        "可行弧占比%": round(100 * okk.mean(), 2),
        "相对Q损失%": round(100 * (1 - (1 / rr) / Q_star), 3),
    }

res = {
    "方法": "B 解析下界 + 弧枚举",
    "T1": {"坐标": [round(c, 4) for c in T]},
    "最优性证明": {
        "逻辑": "仰角 <= 80° => r >= (H-2)/tan80° ；Q=1/r 单调减 => Q <= 1/r_lb；最优圆上存在可行点 => 全局最优",
        "r_lb_m": round(r_lb, 6),
        "Q_ub_1每m": round(Q_star, 6),
    },
    "最优集": {"可行弧数": len(arcs), "弧段": arcs,
               "最优圆可行弧占比%": round(100 * float(ok.mean()), 2)},
    "裁决规则": ("① 在最优集内最大化障碍/边界净空（抗定位误差、远离遮挡阴影边界）；"
                 "② 净空并列时取离调度中心 O 最近（部署距离短）；③ 仍并列取目标方位角最小者（可复现）"),
    "选定最优测站": {
        "坐标": [round(float(best[0]), 4), round(float(best[1]), 4)],
        "水平距离_m": round(best_r, 4),
        "视线仰角_deg": round(best_el, 4),
        "测站方位角_deg": round(best_az, 4),
        "目标->测站方位_deg": round(best_phi, 4),
        "Q_1每m": round(best_Q, 6),
        "净空_m": round(best_clr, 4),
        "到O距离_m": round(float(np.hypot(best[0] - 50, best[1] - 50)), 4),
        "到O行驶时间_s": round(M.PATH.travel_time(M.VEH0["V4"], tuple(best)), 3),
    },
    "其他裁决下的解": {
        "仅按离O最近": {"坐标": [round(float(alt_O[0]), 4), round(float(alt_O[1]), 4)],
                    "Q_1每m": round(1 / float(np.hypot(alt_O[0] - T[0], alt_O[1] - T[1])), 6)},
        "仅按净空最大": {"坐标": [round(float(alt_clr[0]), 4), round(float(alt_clr[1]), 4)],
                    "净空_m": round(float(clr[i_clr]), 4)},
    },
    "独立复核": verify,
    "鲁棒变体": rob,
    "耗时_s": round(time.perf_counter() - t0, 3),
}

os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
with open(os.path.join(OUT, "method_B_analytic.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 绘图：最优弧
VIEW = (66.0, 100.0, 54.0, 92.0)
fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=170)
ax.set_facecolor("#f4f4f4")
for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
    if x2 < VIEW[0] or x1 > VIEW[1] or y2 < VIEW[2] or y1 > VIEW[3]:
        continue          # 视野外的建筑不画，避免把文本框撑出画布
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#cfcfcf",
                           edgecolor="#555", lw=0.9, zorder=3))
    ax.add_patch(Rectangle((x1 - 1.5, y1 - 1.5), x2 - x1 + 3, y2 - y1 + 3, fill=False,
                           edgecolor="#999", ls="--", lw=0.7, zorder=3))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{k} (h={h:g})", ha="center", va="center",
            fontsize=8, zorder=4, clip_on=True)

th = np.linspace(0, 2 * np.pi, 720)
ax.plot(T[0] + r_lb * np.cos(th), T[1] + r_lb * np.sin(th), "r-", lw=1.5,
        label=f"最优圆 r=r_lb={r_lb:.3f} m（仰角=80°）", zorder=5)
# 高亮可行弧（绿色粗线）
ob = ok.astype(np.int8)
prev = np.concatenate([[ob[-1]], ob[:-1]])
for s0 in np.nonzero((ob == 1) & (prev == 0))[0]:
    e0 = s0
    while ob[e0 % NA] == 1 and (e0 - s0) < NA:
        e0 += 1
    seg = np.arange(s0, e0) % NA
    ax.plot(px[seg], py[seg], "-", color="#2ca02c", lw=4.5, alpha=0.7, zorder=6,
            solid_capstyle="round")
ax.plot(*T[:2], "^", ms=13, mfc="#d62728", mec="k", zorder=8, label="目标 T1")
ax.plot(*best, "*", ms=17, mfc="gold", mec="k", zorder=9,
        label=f"选定测站 ({best[0]:.2f}, {best[1]:.2f})")
ax.plot([best[0], T[0]], [best[1], T[1]], "k-", lw=1.0, zorder=7)
ax.annotate(f"r={best_r:.3f} m, 仰角={best_el:.2f}°\nQ=1/r={best_Q:.5f} /m",
            best, textcoords="offset points", xytext=(10, -34), fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.95), zorder=10)
ax.plot(50, 50, "ko", ms=5, zorder=8, clip_on=False)
ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
ax.set_title("问题1 最优集是 r=r_lb 上的一段圆弧（绿）\n圆上各点 Q 相同 → 必须规定平局裁决规则", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q1_optimal_arc.png"))
plt.close(fig)

print(json.dumps(res, ensure_ascii=False, indent=2))
