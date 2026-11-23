# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: ______（以官方发布页为准，提交前填写）
#   使用日期    : 2026-11-20
#   辅助范围    : 网格搜索与绘图代码
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""问题1 方法A：均匀网格扫描法

思路：在 0.2 m 网格上对全区域逐点判定三项约束（几何可停留 / 视线不遮挡 / 仰角可行），
得到 T1 的可观测区域离散近似；在该离散集上直接取 Q = 1/r 的最大值。

优点：实现简单、可同时给出区域图像、无需处理边界拓扑。
缺点：最优值受网格分辨率限制（O(h) 误差），且无法给出"最优性证明"。

运行：python "Q1_可观测区域与最优测站/code/method_A_grid.py"
输出：output/method_A_grid.json、figures/Q1_obs_region_T1.png、figures/Q1_zoom.png
"""
from __future__ import annotations

import json
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
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(HERE), "output")
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

H = 0.2                     # 网格步长 m
T = M.TARGETS["T1"]
xs = np.arange(0.0, M.AREA + 1e-9, H)
ys = np.arange(0.0, M.AREA + 1e-9, H)
X, Y = np.meshgrid(xs, ys)

t0 = time.perf_counter()
mask, r, elv, az = M.observable(X, Y, T)
t_grid = time.perf_counter() - t0

Q = np.where(mask, 1.0 / np.maximum(r, 1e-9), np.nan)
i, j = np.unravel_index(np.nanargmax(Q), Q.shape)
x_best, y_best = X[i, j], Y[i, j]
r_best = float(r[i, j])
az_best = float(az[i, j])
el_best = float(elv[i, j])

# ---- 约束逐项统计（用于说明"哪一条约束在起作用"）
free = M.free_space(X, Y)
r_all = M.horiz_dist(X, Y, T)
el_all = M.elev_deg(np.maximum(r_all, 1e-9), T)
el_ok = (el_all >= M.EL_MIN) & (el_all <= M.EL_MAX) & (r_all > 0)

# 视线判定（全网格一次性算，供统计"仅被遮挡"的面积）
P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, M.CAM_H)])
blk = M.los_blocked(P, T).reshape(X.shape)

stat = {
    "区域内网格点": int(X.size),
    "几何可停留": int(free.sum()),
    "仰角可行": int((free & el_ok).sum()),
    "可观测(三项全满足)": int(mask.sum()),
    "仅因遮挡被排除": int((free & el_ok & ~mask).sum()),
    "可观测面积_m2": round(float(mask.sum()) * H * H, 2),
    "可停留面积_m2": round(float(free.sum()) * H * H, 2),
}

# ---- 平局分析：与最优值相差 0.1% 以内的网格点
tol = 1e-3
tied = mask & (r <= r_best * (1 + tol))
ti, tj = np.nonzero(tied)
tie_pts = np.column_stack([X[ti, tj], Y[ti, tj]])
if len(tie_pts) > 1:
    d = np.hypot(tie_pts[:, 0] - T[0], tie_pts[:, 1] - T[1])
    tie_info = {
        "近优网格点数": int(len(tie_pts)),
        "近优集半径范围_m": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "近优集方位角范围_deg": [round(float(np.degrees(np.arctan2(tie_pts[:, 1] - T[1], tie_pts[:, 0] - T[0])).min()), 2),
                            round(float(np.degrees(np.arctan2(tie_pts[:, 1] - T[1], tie_pts[:, 0] - T[0])).max()), 2)],
    }
else:
    tie_info = {"近优网格点数": int(len(tie_pts))}

res = {
    "方法": "A 均匀网格扫描",
    "网格步长_m": H,
    "目标": {"T1坐标": [round(c, 4) for c in T]},
    "约束统计": stat,
    "最优测站_网格": [round(float(x_best), 3), round(float(y_best), 3)],
    "最优水平距离_m": round(r_best, 4),
    "最优视线仰角_deg": round(el_best, 4),
    "最优Q_1每m": round(1.0 / r_best, 6),
    "最优方位角_deg": round(az_best, 3),
    "仰角上限对应的理论最小水平距离_m": round((T[2] - M.CAM_H) / np.tan(np.radians(M.EL_MAX)), 6),
    "平局分析": tie_info,
    "耗时_s": round(t_grid, 2),
}

with open(os.path.join(OUT, "method_A_grid.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ 绘图
fig, ax = plt.subplots(figsize=(8.2, 7.6), dpi=160)
ax.set_facecolor("#efefef")   # 浅灰底：白色=不可观测，避免与建筑混淆
for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#d9d9d9",
                           edgecolor="#666666", lw=0.9, zorder=2))
    ax.add_patch(Rectangle((x1 - 1.5, y1 - 1.5), x2 - x1 + 3, y2 - y1 + 3, fill=False,
                           edgecolor="#999999", ls="--", lw=0.7, zorder=2))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{k}\nh={h:g}m", ha="center", va="center",
            fontsize=8, color="#333333", zorder=3)

Zm = np.where(mask, 1.0 / np.maximum(r, 1e-9), np.nan)
im = ax.imshow(Zm, origin="lower", extent=(0, M.AREA, 0, M.AREA), cmap="viridis",
               vmin=0.01, vmax=0.08, alpha=0.85, zorder=1)
cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cb.set_label("观测质量 Q = 1/r  (1/m)", fontsize=9)

ax.plot(*T[:2], marker="^", ms=13, mfc="#d62728", mec="k", zorder=6, label="目标 T1")
ax.plot(*M.VEH0["V1"], "s", ms=7, mfc="#1f77b4", mec="k", zorder=6)
for k, p in M.VEH0.items():
    ax.annotate(k, p, textcoords="offset points", xytext=(5, 4), fontsize=7, zorder=7)
ax.plot(x_best, y_best, "*", ms=17, mfc="gold", mec="k", zorder=8,
        label=f"最优测站 ({x_best:.2f}, {y_best:.2f})")
ax.plot([x_best, T[0]], [y_best, T[1]], "w-", lw=1.2, zorder=5)
ax.plot(50, 50, "ko", ms=5, zorder=6)
ax.annotate("O 调度中心", (50, 50), textcoords="offset points", xytext=(6, -10), fontsize=8)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_title(f"问题1 目标 T1 的可观测区域（{H} m 网格，Q=1/r 着色）\n"
             f"最优测站 r={r_best:.2f} m, 仰角={el_best:.2f}°, Q={1/r_best:.4f} 1/m", fontsize=10)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_xticks(range(0, 101, 20))
ax.set_yticks(range(0, 101, 20))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q1_obs_region_T1.png"))
plt.close(fig)

# 局部放大：展示仰角约束圆与遮挡阴影边界
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=180)
x1, x2 = max(58, x_best - 30), 100
y1, y2 = max(50, y_best - 26), 100
gx = np.arange(x1, x2, 0.1)
gy = np.arange(y1, y2, 0.1)
GX, GY = np.meshgrid(gx, gy)
m2, r2, e2, _ = M.observable(GX, GY, T)
ax.imshow(np.where(m2, 1, 0), origin="lower", extent=(gx[0], gx[-1], gy[0], gy[-1]),
          cmap="Greens", alpha=0.55, vmin=0, vmax=1.6)
ax.contour(GX, GY, r2, levels=[7.582, 10, 15, 20], colors="#444444", linewidths=0.8,
           linestyles=["-", "--", "--", "--"])
for k, (bx1, bx2, by1, by2, h) in M.BUILDINGS.items():
    ax.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, facecolor="#d9d9d9",
                           edgecolor="#666", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((bx1 - 1.5, by1 - 1.5), bx2 - bx1 + 3, by2 - by1 + 3, fill=False,
                           edgecolor="#888", ls="--", lw=0.7, zorder=3))
th = np.linspace(0, 2 * np.pi, 720)
ax.plot(T[0] + 7.582 * np.cos(th), T[1] + 7.582 * np.sin(th), "r--", lw=1.0,
        label="仰角=80° 临界圆 (r=7.582 m)")
ax.plot(*T[:2], "^", ms=12, mfc="#d62728", mec="k", zorder=6, label="T1")
ax.plot(x_best, y_best, "*", ms=16, mfc="gold", mec="k", zorder=8, label="最优测站")
ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1]); ax.set_aspect("equal")
ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
ax.set_title("T1 最优测站邻域放大（绿色=可观测，红色虚线=仰角约束边界）", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q1_zoom.png"))
plt.close(fig)

print(json.dumps(res, ensure_ascii=False, indent=2))
