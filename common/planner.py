"""调度规划器：问题2(2) 与问题4 共用的"测站候选池 + 结构枚举 + 启发式 + 仿真"。

设计要点
--------
1. 可观测区域离散为 1.0 m 栅格；对每个 (告警, 车辆) 组合，把候选站按"从该车初始位置
   出发的就绪时刻 t_ready"排序，再按 5° 方位分箱，只保留每箱最快的一个。
   由于同一方位箱内的站对交会角的影响不超过 5°，而目标函数只关心时间，这一约化对
   "最晚观测开始时间"是近似精确的（最终解再做连续精修与精确校验）。
2. 点对交会的最早就绪时刻：max(t1,t2)，约束 γ >= γ_min（交会角下限，抑制病态几何）。
3. 站间距离按可见性图最短路精确计算，并按候选池对缓存。
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np

from . import model as M

BIN_DEG = 5.0
GRID = 1.0
TMAX = 70.0        # 候选站的就绪时刻上限（超过则对最小化 t_last 无意义）


class Pool:
    """某告警 × 某车辆的候选测站集合（已按方位分箱约化）。"""

    __slots__ = ("alert", "veh", "x", "y", "t", "rot", "ready", "az", "r", "el")

    def __init__(self, alert, veh, x, y, t, rot, ready, az, r, el):
        self.alert = alert
        self.veh = veh
        self.x, self.y = x, y
        self.t, self.rot, self.ready = t, rot, ready
        self.az, self.r, self.el = az, r, el

    def __len__(self):
        return len(self.x)

    def station(self, i):
        return (float(self.x[i]), float(self.y[i]))


_POOLS: dict[tuple[str, str], Pool] = {}
_DIST: dict[tuple[int, int], np.ndarray] = {}
_GRID_CACHE: dict[str, dict] = {}


def alert_grid(alert, step=GRID, tmax=TMAX):
    """告警的可观测区域栅格（带缓存）。"""
    key = f"{alert}|{step}|{tmax}"
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    T = M.TARGETS[alert]
    g = np.arange(step / 2, M.AREA, step)
    X, Y = np.meshgrid(g, g)
    mask, r, el, az = M.observable(X, Y, T)
    m = mask.ravel()
    d = {"X": X.ravel()[m], "Y": Y.ravel()[m], "r": r.ravel()[m],
         "el": el.ravel()[m], "az": az.ravel()[m], "T": T}
    _GRID_CACHE[key] = d
    return d


def build_pool(alert, veh, start_pos=None, tmax=TMAX, bin_deg=BIN_DEG, step=GRID,
               rmax=np.inf, avail=0.0) -> Pool:
    key = (alert, veh, round(step, 3), round(tmax, 3), round(bin_deg, 3),
           round(float(rmax), 3), round(float(avail), 3),
           None if start_pos is None else (round(start_pos[0], 3), round(start_pos[1], 3)))
    if key in _POOLS:
        return _POOLS[key]
    d = alert_grid(alert, step, tmax)
    pos = start_pos if start_pos is not None else M.VEH0[veh]
    tt = M.PATH.dist_field(pos, np.column_stack([d["X"], d["Y"]])) / M.SPEED
    rot = M.gimbal_turn_time(d["az"], d["el"])
    ready = avail + tt + rot
    keep = (ready <= tmax) & (d["r"] <= rmax)
    x, y, az, r, el = d["X"][keep], d["Y"][keep], d["az"][keep], d["r"][keep], d["el"][keep]
    t, ro, rd = tt[keep], rot[keep], ready[keep]
    # 按就绪时刻排序，再按方位分箱保留每箱最快者
    order = np.argsort(rd)
    binid = np.floor(np.mod(az, 360.0) / bin_deg).astype(int)
    seen = set()
    sel = []
    for i in order:
        b = int(binid[i])
        if b in seen:
            continue
        seen.add(b)
        sel.append(i)
    sel = np.array(sel, dtype=int)
    pool = Pool(alert, veh, x[sel], y[sel], t[sel], ro[sel], rd[sel],
                az[sel], r[sel], el[sel])
    _POOLS[key] = pool
    return pool


def dist_matrix(pa: Pool, pb: Pool):
    """两候选池之间的最短行驶距离矩阵（精确最短路，带缓存）。"""
    key = (id(pa), id(pb))
    if key in _DIST:
        return _DIST[key][0]
    pts = np.column_stack([pb.x, pb.y])
    D = np.empty((len(pa), len(pb)))
    for i in range(len(pa)):
        D[i] = M.PATH.dist_field((pa.x[i], pa.y[i]), pts)
    _DIST[key] = (D, pa, pb)      # 同时持有对象引用，防止 id 被回收复用
    return D


def reset_caches():
    """当目标/车辆状态发生变化（如问题4 的新告警）时必须清空缓存。"""
    _POOLS.clear()
    _DIST.clear()
    _GRID_CACHE.clear()


def gamma_mat(pa: Pool, pb: Pool):
    """交会角矩阵 γ = 两方位线夹角的锐角部分。"""
    d = np.abs(pa.az[:, None] - pb.az[None, :])
    return np.minimum(d, 180.0 - d)


def pair_best(pa: Pool, pb: Pool, gmin):
    """在 pa × pb 中选 max(ready) 最小、且 γ >= gmin 的一对。"""
    G = gamma_mat(pa, pb)
    R = np.maximum(pa.ready[:, None], pb.ready[None, :])
    ok = G >= gmin
    if not ok.any():
        return np.inf, None, None
    R = np.where(ok, R, np.inf)
    k = int(np.argmin(R))
    i, j = divmod(k, R.shape[1])
    return float(R[i, j]), int(i), int(j)


def pair_best_given(pb: Pool, az_query, gmin):
    """给定"另一测站的方位角"（可为数组），求 pb 中满足 γ>=gmin 且 ready 最小者。

    返回 (ready 数组, 索引数组)。用于把"共享车辆的第一站位置"作为参数扫过。
    """
    az_query = np.atleast_1d(np.asarray(az_query, float))
    d = np.abs(az_query[:, None] - pb.az[None, :])
    G = np.minimum(d, 180.0 - d)
    ok = G >= gmin
    R = np.where(ok, pb.ready[None, :], np.inf)
    idx = np.argmin(R, axis=1)
    val = R[np.arange(len(az_query)), idx]
    return val, idx


# ---------------------------------------------------------------- 结构枚举
def enumerate_structures():
    """6 席位 / 5 车的全部结构：1 辆车复用 2 席 + 4 辆车各 1 席。

    返回 [(复用车辆, 告警序对(a_first,a_second), 先手伙伴, 后手伙伴)]，共 360 种。
    """
    alerts = ["T1", "T2", "T3"]
    out = []
    for v in M.VEHICLE_IDS:
        others = [x for x in M.VEHICLE_IDS if x != v]
        for a1, a2 in itertools.permutations(alerts, 2):
            a3 = [a for a in alerts if a not in (a1, a2)][0]
            for p1 in others:
                for p2 in [x for x in others if x != p1]:
                    out.append((v, (a1, a2), a3, p1, p2))
    return out


def eval_structure(struct, gmin, obs_dur=M.OBS_DUR):
    """给定结构，求最小化 max(观测开始) 的测站配置。

    决策：复用车辆的第一站 s（在其候选池中扫）、两站的伙伴站、第三告警的两站。
    返回 dict（含 t_last 与各站索引）或 None（不可行）。
    """
    v, (a1, a2), a3, p1, p2 = struct
    Pv1 = build_pool(a1, v)
    Pp1 = build_pool(a1, p1)
    Pv2 = build_pool(a2, v)
    Pp2 = build_pool(a2, p2)
    # a3 的两辆车 = 剩余两辆
    rest = [x for x in M.VEHICLE_IDS if x not in (v, p1, p2)]
    if len(rest) != 2:
        return None
    Pa3a = build_pool(a3, rest[0])
    Pa3b = build_pool(a3, rest[1])
    t3, i3, j3 = pair_best(Pa3a, Pa3b, gmin)
    if not np.isfinite(t3):
        return None

    # 伙伴站在"给定主站方位"下的最优就绪时刻
    b1, j1 = pair_best_given(Pp1, Pv1.az, gmin)      # 长度 = len(Pv1)
    b2, j2 = pair_best_given(Pp2, Pv2.az, gmin)      # 长度 = len(Pv2)
    D12 = dist_matrix(Pv1, Pv2)                      # 复用车辆 站1 -> 站2

    best = (np.inf, None)
    for i in range(len(Pv1)):
        if not np.isfinite(b1[i]):
            continue
        start1 = max(Pv1.ready[i], b1[i])
        c1 = start1 + obs_dur + Pv1.rot[i]
        ready2 = c1 + D12[i] / M.SPEED + Pv2.rot     # 向量
        start2 = np.maximum(ready2, b2)
        k = int(np.argmin(start2))
        if not np.isfinite(start2[k]):
            continue
        obj = max(start1, start2[k], t3)
        if obj < best[0]:
            best = (float(obj), (i, int(j1[i]), k, int(j2[k]), i3, j3))
    if best[1] is None:
        return None
    i, j, k, m, i3b, j3b = best[1]
    return {
        "t_last": best[0],
        "structure": struct,
        "a3": a3, "a3_veh": rest,
        "stations": {
            (a1, v): Pv1.station(i), (a1, p1): Pp1.station(j),
            (a2, v): Pv2.station(k), (a2, p2): Pp2.station(m),
            (a3, rest[0]): Pa3a.station(i3b), (a3, rest[1]): Pa3b.station(j3b),
        },
        "t3_part": t3,
    }


def structure_times(detail, obs_dur=M.OBS_DUR):
    """把 eval_structure 的结果交给精确仿真，返回 (t_last, events)。"""
    tasks = []
    order = {}
    for (alert, veh), st in detail["stations"].items():
        T = M.TARGETS[alert]
        az = float(M.azimuth_to(st[0], st[1], T))
        el = float(M.elev_deg(M.horiz_dist(st[0], st[1], T), T))
        k = order.get(veh, 0)
        order[veh] = k + 1
        tasks.append(M.VehicleTask(alert, veh, st, k, az, el))
    # 复用车辆的第二站必须 order 更大：按"与初始位置的距离"排序不稳健，
    # 这里用 eval_structure 给出的先手/后手关系显式设定
    v, (a1, a2), a3, p1, p2 = detail["structure"]
    for t in tasks:
        if t.vehicle == v:
            t.order = 0 if t.alert == a1 else 1
    events = M.simulate(tasks)
    return max(e.obs_start for e in events.values()), events


# ---------------------------------------------------------------- 基线启发式
def nearest_heuristic(alerts=("T1", "T2", "T3"), gmin=15.0, verbose=False):
    """最近分配启发式：按告警发布顺序，逐个把"当前最早可用 + 最近"的两辆车派过去。

    规则：
      1. 处理告警 a 时，对每辆车计算 earliest = max(可用时刻, 0) + 到候选池的最小就绪时刻；
      2. 取 earliest 最小的两辆（若同一辆已被本告警占用则跳过）；
      3. 第一辆车放到使 earliest 最小的站；第二辆车在与之交会角 >= gmin 的站中取最早者；
      4. 观测开始 = max(两车就绪)，结束后按复位规则更新该车可用时刻；
      5. 若两辆车不可得（例如只剩一辆），等待最早可用车辆。
    """
    state = {v: (M.VEH0[v], 0.0) for v in M.VEHICLE_IDS}
    plan = []      # (alert, veh, station, order)
    order_ct = {v: 0 for v in M.VEHICLE_IDS}
    for a in alerts:
        chosen = []
        for _ in range(2):
            cands = []
            for v in M.VEHICLE_IDS:
                if v in [c[0] for c in chosen]:
                    continue
                pos, avail = state[v]
                pool = build_pool(a, v, start_pos=pos)
                if len(pool) == 0:
                    continue
                cands.append((avail + float(pool.ready.min()), v, pool, avail))
            if not cands:
                return None
            cands.sort(key=lambda c: c[0])
            _, v, pool, avail = cands[0]
            if not chosen:
                i = int(np.argmin(pool.ready))
            else:
                paz = chosen[0][2]
                val, idx = pair_best_given(pool, np.array([paz]), gmin)
                if not np.isfinite(val[0]):
                    # 无满足交会角的站，退化为最近站
                    i = int(np.argmin(pool.ready))
                else:
                    i = int(idx[0])
            st = pool.station(i)
            az = float(M.azimuth_to(st[0], st[1], M.TARGETS[a]))
            el = float(M.elev_deg(M.horiz_dist(st[0], st[1], M.TARGETS[a]), M.TARGETS[a]))
            chosen.append((v, st, az, el))
            plan.append((a, v, st, order_ct[v], az, el))
            order_ct[v] += 1
        # 用精确仿真确定该告警的开始时刻并回填可用时刻
        sub = [M.VehicleTask(p[0], p[1], p[2], p[3], p[4], p[5]) for p in plan]
        ev = M.simulate(sub, vehicle_state={v: (state[v][0], state[v][1]) for v in M.VEHICLE_IDS})
        for (al, ve), e in ev.items():
            state[ve] = (e.station, e.avail)
        if verbose:
            print("   ", a, [(c[0], [round(q, 2) for q in c[1]]) for c in chosen])
    ev = M.simulate([M.VehicleTask(p[0], p[1], p[2], p[3], p[4], p[5]) for p in plan],
                    vehicle_state={v: (M.VEH0[v], 0.0) for v in M.VEHICLE_IDS})
    return ev, plan
