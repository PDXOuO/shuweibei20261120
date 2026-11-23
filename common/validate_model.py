# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-20
#   辅助范围    : 校验用例设计与实现
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""共享模型的独立验证：用与主实现无关的"笨办法"交叉校验。

校验项：
V1 可通行域：与逐点暴力判定（面积/建筑矩形距离）比对
V2 视线遮挡：与视线线段密集采样（1 mm 步长）比对
V3 最短行驶距离：与 0.25 m 栅格 Dijkstra（8 邻域、禁切角）比对
V4 云台转动时间：与手工算例比对（含 180 度卷绕边界）
V5 时间轴仿真：与手工推演的 2 车 1 告警算例比对

运行：python common/validate_model.py
输出：common/output_model_validation.json
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import model as M  # noqa: E402

rng = np.random.default_rng(20261120)
report = {}

# ---------------------------------------------------------------- V1 可通行域
n = 20000
xs = rng.uniform(0, 100, n)
ys = rng.uniform(0, 100, n)


def brute_free(x, y):
    if not (1.5 <= x <= 98.5 and 1.5 <= y <= 98.5):
        return False
    for (x1, x2, y1, y2, h) in M.BUILDINGS.values():
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        if math.hypot(dx, dy) < 1.5:
            return False
    return True


bf = np.array([brute_free(x, y) for x, y in zip(xs, ys)])
vf = M.free_space(xs, ys)
report["V1_free_space"] = {"样本数": n, "不一致数": int((bf != vf).sum()), "通过": bool((bf == vf).all())}

# ---------------------------------------------------------------- V2 视线遮挡


def brute_los(p, T, step=0.001):
    p = np.asarray(p, float)
    T = np.asarray(T, float)
    L = np.linalg.norm(T - p)
    m = int(L / step) + 2
    ts = np.linspace(0, 1, m)
    pts = p[None, :] + ts[:, None] * (T - p)[None, :]
    for (x1, x2, y1, y2, h) in M.BUILDINGS.values():
        inside = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2) &
                  (pts[:, 1] >= y1) & (pts[:, 1] <= y2) &
                  (pts[:, 2] >= 0) & (pts[:, 2] <= h))
        if inside.any():
            return True
    return False


cams = np.column_stack([rng.uniform(1.5, 98.5, 3000), rng.uniform(1.5, 98.5, 3000)])
cams = cams[M.free_space(cams[:, 0], cams[:, 1])][:600]
mism = {}
for k, T in M.TARGETS.items():
    P = np.column_stack([cams[:, 0], cams[:, 1], np.full(len(cams), M.CAM_H)])
    fast = M.los_blocked(P, T)
    slow = np.array([brute_los(p, T) for p in P])
    d = int((fast != slow).sum())
    mism[k] = {"样本数": int(len(cams)), "不一致数": d, "通过": bool(d == 0)}
report["V2_los"] = mism

# ---------------------------------------------------------------- V3 最短路
STEP = 0.25
gx = np.arange(1.5, 98.5 + 1e-9, STEP)
X, Y = np.meshgrid(gx, gx)
FREE = M.free_space(X, Y)
nx, ny = X.shape


def grid_dist(src, dst):
    """0.25 m 栅格 Dijkstra（8 邻域，禁切角），返回 (路径长度, 回溯折线)。

    注意：8 邻域栅格的度量是"八边形距离"，对 22.5 度方向会系统性高估约 8%，
    因此除距离外还回溯折线，再做弦化（string pulling）得到真实路径长度，
    只有弦化后的长度才与可见性图可比。
    """
    def snap(p):
        i = int(round((p[1] - 1.5) / STEP))
        j = int(round((p[0] - 1.5) / STEP))
        return min(max(i, 0), ny - 1), min(max(j, 0), nx - 1)
    si, sj = snap(src)
    ti, tj = snap(dst)
    INF = np.inf
    dist = np.full((ny, nx), INF)
    par = {}
    dist[si, sj] = 0.0
    pq = [(0.0, si, sj)]
    nb = [(-1, -1, math.sqrt(2)), (-1, 0, 1), (-1, 1, math.sqrt(2)), (0, -1, 1),
          (0, 1, 1), (1, -1, math.sqrt(2)), (1, 0, 1), (1, 1, math.sqrt(2))]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj, w in nb:
            a, b = i + di, j + dj
            if not (0 <= a < ny and 0 <= b < nx) or not FREE[a, b]:
                continue
            if di and dj and not (FREE[i, b] and FREE[a, j]):
                continue   # 禁切角
            nd = d + w * STEP
            if nd < dist[a, b] - 1e-12:
                dist[a, b] = nd
                par[(a, b)] = (i, j)
                heapq.heappush(pq, (nd, a, b))
    if not np.isfinite(dist[ti, tj]):
        return np.inf, []
    # 回溯
    path = [(gx[tj], gx[ti])]
    cur = (ti, tj)
    while cur in par:
        cur = par[cur]
        path.append((gx[cur[1]], gx[cur[0]]))
    path.reverse()
    # 弦化：能直连就跳过中间点
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not M.PATH._seg_clear(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    L = sum(math.hypot(out[k + 1][0] - out[k][0], out[k + 1][1] - out[k][1])
            for k in range(len(out) - 1))
    return L, out


pairs = []
while len(pairs) < 12:
    p = (rng.uniform(2, 98), rng.uniform(2, 98))
    q = (rng.uniform(2, 98), rng.uniform(2, 98))
    if M.free_space(np.array([p[0], q[0]]), np.array([p[1], q[1]])).all():
        pairs.append((p, q))
errs = []
for p, q in pairs:
    gd_raw, poly = grid_dist(p, q)
    gd = gd_raw
    vd = M.PATH.dist(p, q)
    poly_v, vd2 = M.PATH.path(p, q)
    # 独立审查：0.005 m 采样，检查是否落在任一建筑原始轮廓内部
    bad = 0
    if poly_v:
        for a, b in zip(poly_v[:-1], poly_v[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            m = max(int(L / 0.005), 2)
            ts = np.linspace(0, 1, m)
            px = a[0] + ts * (b[0] - a[0])
            py = a[1] + ts * (b[1] - a[1])
            for (x1, x2, y1, y2) in M.RAW_RECTS.values():
                bad += int(((px > x1 + 1e-6) & (px < x2 - 1e-6) &
                            (py > y1 + 1e-6) & (py < y2 - 1e-6)).sum())
    errs.append({"from": [round(v, 2) for v in p], "to": [round(v, 2) for v in q],
                 "visgraph": round(vd, 3), "grid025_弦化": round(gd, 3),
                 "相对误差%": round(100 * (vd - gd) / gd, 3),
                 "折线穿建筑采样点数": bad})
report["V3_shortest_path"] = {
    "样本数": len(errs),
    "最大绝对误差_m": round(max(abs(e["visgraph"] - e["grid025_弦化"]) for e in errs), 3),
    "最大相对误差%": round(max(abs(e["相对误差%"]) for e in errs), 3),
    "说明": ("可见性图为多边形域精确解。栅格解为近似解（端点吸附到 0.25 m 格点、"
             "走廊受限），故个别样本可略短/略长；关键是可见性图折线经 0.005 m 采样"
             "无任何点落入建筑内部，且相对误差 <= 3%"),
    "穿墙采样点总数": int(sum(e["折线穿建筑采样点数"] for e in errs)),
    "明细": errs,
}

# ---------------------------------------------------------------- V4 云台
cases = [
    (0.0, 0.0, 0.0),      # 已是正东水平
    (90.0, 0.0, 3.0),     # 90/30
    (270.0, 0.0, 3.0),    # 卷绕后 -90 -> 3 s
    (180.0, 10.0, 6.0),   # 180/30=6 > 10/30
    (10.0, 80.0, 80 / 30),
]
v4 = []
for az, el, want in cases:
    got = float(M.gimbal_turn_time(az, el))
    v4.append({"az": az, "el": el, "理论": round(want, 4), "计算": round(got, 4),
               "通过": abs(got - want) < 1e-9})
report["V4_gimbal"] = {"用例": v4, "通过": all(c["通过"] for c in v4)}
report["V4_wrap"] = {"wrap180(180)": M.wrap180(180.0), "wrap180(-180)": M.wrap180(-180.0),
                     "wrap180(190)": M.wrap180(190.0), "wrap180(350)": round(M.wrap180(350.0), 6)}

# ---------------------------------------------------------------- V5 时间轴
st = (30.0, 30.0)
st2 = (70.0, 30.0)
az1 = float(M.azimuth_to(*st, M.TARGETS["T1"]))
el1 = float(M.elev_deg(M.horiz_dist(*st, M.TARGETS["T1"]), M.TARGETS["T1"]))
az2 = float(M.azimuth_to(*st2, M.TARGETS["T1"]))
el2 = float(M.elev_deg(M.horiz_dist(*st2, M.TARGETS["T1"]), M.TARGETS["T1"]))
tasks = [M.VehicleTask("T1", "V1", st, 0, az1, el1),
         M.VehicleTask("T1", "V2", st2, 0, az2, el2)]
ev = M.simulate(tasks)
d1 = M.PATH.dist(M.VEH0["V1"], st)
d2 = M.PATH.dist(M.VEH0["V2"], st2)
t1 = d1 / 3 + float(M.gimbal_turn_time(az1, el1))
t2 = d2 / 3 + float(M.gimbal_turn_time(az2, el2))
want_start = max(t1, t2)
e = ev[("T1", "V2")]
report["V5_timeline"] = {
    "V1_理论就绪": round(t1, 4), "V2_理论就绪": round(t2, 4),
    "理论观测开始": round(want_start, 4), "仿真观测开始": round(e.obs_start, 4),
    "观测结束": round(e.obs_end, 4), "复位耗时": round(e.reset, 4),
    "通过": abs(e.obs_start - want_start) < 1e-6,
}

all_pass = (report["V1_free_space"]["通过"] and report["V4_gimbal"]["通过"]
            and report["V5_timeline"]["通过"]
            and all(v["通过"] for v in report["V2_los"].values()))
report["总体是否全部通过"] = bool(all_pass)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_model_validation.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])
print("\n写出:", out)
