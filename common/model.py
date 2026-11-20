"""2026 数维杯 B 题 —— 共享几何 / 运动学模型

坐标系：原点在西南角，x 东、y 北，单位 m。
本模块只做"与小题无关"的公共建模，所有小题脚本统一 import 本模块，
保证四个小题的口径（建筑遮挡、云台约束、行驶距离、时间轴）完全一致。

主要口径：
1. 观测车为半径 1.5 m 的圆盘，圆心坐标即测站坐标；
   圆心须落在 [1.5, 98.5]^2 内，且到任一建筑矩形的最小距离 >= 1.5 m。
2. 相机位于圆心、离地 2 m；云台仰角可行域 [-10, 80] 度。
3. 视线与建筑棱柱（含边、面、顶点）相切或接触均计为遮挡（闭区间判定）。
4. 云台方位/俯仰最大角速度均为 30 deg/s，两轴同时转动，耗时取较长者；
   行驶中云台保持正东水平（方位 0、俯仰 0）。
5. 车辆速度 3 m/s，路径按"绕建筑膨胀矩形的可见性图最短路"计算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------- 基本常数
AREA = 100.0            # 正方形区域边长
R_VEH = 1.5             # 观测车等效圆盘半径
SPEED = 3.0             # 行驶速度 m/s
CAM_H = 2.0             # 相机离地高度 m
EL_MIN, EL_MAX = -10.0, 80.0   # 云台仰角可行范围 (deg)
OMEGA_GIMBAL = 30.0     # 云台角速度 deg/s
OBS_DUR = 5.0           # 联合观测时长 s
TRAVEL_LIMIT = 30.0     # 问题2(1) 的行驶时间上限 s

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# 建筑 (x1, x2, y1, y2, h)
BUILDINGS = {
    "B1": (22.0, 34.0, 16.0, 26.0, 20.0),
    "B2": (63.0, 79.0, 59.0, 71.0, 28.0),
    "B3": (10.0, 24.0, 65.0, 76.0, 12.0),
    "B4": (72.0, 86.0, 22.0, 34.0, 24.0),
    "B5": (41.0, 51.0, 38.0, 46.0, 32.0),
}
BUILDING_IDS = list(BUILDINGS)

# 车辆初始位置
VEH0 = {
    "V1": (15.0, 14.0),
    "V2": (88.0, 10.0),
    "V3": (14.0, 86.0),
    "V4": (88.0, 68.0),
    "V5": (50.0, 93.0),
}
VEHICLE_IDS = list(VEH0)

# 基准告警 (theta deg, R m, H m)
ALERTS_SPEC = {
    "T1": (30.0, 40.0, 45.0),
    "T2": (150.0, 45.0, 35.0),
    "T3": (285.0, 38.0, 55.0),
}


def alert_position(theta_deg: float, R_m: float, H_m: float) -> tuple[float, float, float]:
    """告警 -> 三维目标点。方位角自正东逆时针量取。"""
    t = theta_deg * D2R
    return (50.0 + R_m * math.cos(t), 50.0 + R_m * math.sin(t), H_m)


# 目标三维坐标
TARGETS = {k: alert_position(*v) for k, v in ALERTS_SPEC.items()}
TARGET_XY = {k: (v[0], v[1]) for k, v in TARGETS.items()}


# ---------------------------------------------------------------- 可通行域
def inflated_rects(margin: float = R_VEH):
    """建筑矩形按车辆半径外扩（圆角以欧氏距离体现，此处给出外包矩形，用于距离判定）。"""
    out = {}
    for k, (x1, x2, y1, y2, h) in BUILDINGS.items():
        out[k] = (x1 - margin, x2 + margin, y1 - margin, y2 + margin)
    return out


RAW_RECTS = {k: (v[0], v[1], v[2], v[3]) for k, v in BUILDINGS.items()}   # 建筑原始轮廓
RECTS = inflated_rects()                                                   # 圆心禁入区（外扩 1.5 m）


def point_rect_dist(x, y, rect):
    """点到矩形的欧氏距离（矩形外为正，内部为 0）。"""
    x1, x2, y1, y2 = rect
    dx = np.maximum(np.maximum(x1 - x, 0.0), x - x2)
    dy = np.maximum(np.maximum(y1 - y, 0.0), y - y2)
    return np.hypot(dx, dy)


def free_space(x, y):
    """车辆圆心是否可停留（向量化）：在区域内，且到建筑原始轮廓距离 >= 1.5 m。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = (x >= R_VEH) & (x <= AREA - R_VEH) & (y >= R_VEH) & (y <= AREA - R_VEH)
    for rect in RAW_RECTS.values():
        ok &= point_rect_dist(x, y, rect) >= R_VEH - 1e-12
    return ok


def building_clearance(x, y):
    """圆心到建筑（未膨胀）的最小水平净距，用于诊断。"""
    best = np.full(np.shape(x), np.inf)
    for k, (x1, x2, y1, y2, h) in BUILDINGS.items():
        best = np.minimum(best, point_rect_dist(x, y, (x1, x2, y1, y2)))
    return best


# ---------------------------------------------------------------- 视线遮挡
def los_blocked(P, T):
    """判断相机点 P=(...,3) 到目标 T=(3,) 的视线是否被任一建筑棱柱遮挡。

    遮挡判据：线段与闭长方体 [x1,x2]x[y1,y2]x[0,h] 的交集非空（含相切/接触）。
    向量化：P 可为 (N,3)。
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    T = np.asarray(T, dtype=float)
    d = T[None, :] - P
    n = P.shape[0]
    blocked = np.zeros(n, dtype=bool)

    for (x1, x2, y1, y2, h) in BUILDINGS.values():
        lo = np.array([x1, y1, 0.0])
        hi = np.array([x2, y2, h])
        tmin = np.full(n, -np.inf)
        tmax = np.full(n, np.inf)
        paral_out = np.zeros(n, dtype=bool)

        for k in range(3):
            pk = P[:, k]
            dk = d[:, k]
            par = np.abs(dk) < 1e-12
            safe = np.where(par, 1.0, dk)
            t1 = (lo[k] - pk) / safe
            t2 = (hi[k] - pk) / safe
            a = np.minimum(t1, t2)
            b = np.maximum(t1, t2)
            inside = (pk >= lo[k] - 1e-12) & (pk <= hi[k] + 1e-12)
            paral_out |= par & (~inside)
            tmin = np.where(par, tmin, np.maximum(tmin, a))
            tmax = np.where(par, tmax, np.minimum(tmax, b))

        hit = (~paral_out) & (tmax >= tmin - 1e-12) & (tmax >= -1e-12) & (tmin <= 1.0 + 1e-12)
        blocked |= hit

    return blocked


def horiz_dist(x, y, target):
    tx, ty = target[0], target[1]
    return np.hypot(np.asarray(x) - tx, np.asarray(y) - ty)


def elev_deg(r, target, cam_h=CAM_H):
    h = target[2] - cam_h
    return np.degrees(np.arctan2(h, np.asarray(r)))


def azimuth_to(x, y, target):
    """测站 -> 目标的方位角（自正东逆时针，[0,360)）。"""
    return np.mod(np.degrees(np.arctan2(target[1] - np.asarray(y), target[0] - np.asarray(x))), 360.0)


def observable(x, y, target, el=(EL_MIN, EL_MAX)):
    """三项约束同时满足则可视（向量化）。返回 (mask, r, elevation, dirs)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = horiz_dist(x, y, target)
    elv = elev_deg(np.maximum(r, 1e-9), target)
    m = free_space(x, y) & (r > 0) & (elv >= el[0]) & (elv <= el[1])
    # 视线遮挡只对候选点计算，节省时间
    if np.any(m):
        idx = np.nonzero(m.ravel())[0]
        P = np.column_stack([x.ravel()[idx], y.ravel()[idx], np.full(idx.size, CAM_H)])
        blk = los_blocked(P, target)
        mm = m.ravel().copy()
        mm[idx] = ~blk
        m = mm.reshape(m.shape)
    return m, r, elv, azimuth_to(x, y, target)


# ---------------------------------------------------------------- 云台
def wrap180(a):
    """把角度卷绕到 (-180, 180]。"""
    a = np.mod(np.asarray(a, dtype=float) + 180.0, 360.0) - 180.0
    a = np.where(np.isclose(a, -180.0), 180.0, a)
    return a if a.ndim else float(a)


def gimbal_turn_time(az_target, el_target, az_now=0.0, el_now=0.0):
    """两轴同时转动，耗时取较长者。"""
    daz = np.abs(wrap180(az_target - az_now))
    dele = np.abs(el_target - el_now)
    return np.maximum(daz, dele) / OMEGA_GIMBAL


def gimbal_reset_time(az_now, el_now):
    return np.maximum(np.abs(wrap180(-az_now)), np.abs(el_now)) / OMEGA_GIMBAL


# ---------------------------------------------------------------- 最短行驶距离
class PathMetric:
    """在"区域内部 + 绕开建筑膨胀矩形"的多边形域中求最短路。

    节点 = 各膨胀矩形外扩 1e-6 后的 4 个角点。最短路只可能在角点处折弯，
    故 dist(p,q) = min_v [ d(p,v) + |v-q| ]（v 与 q 通视），或直接通视时取直线。
    对同一出发点做一次 Dijkstra，即可批量评估到任意端点的距离。
    """

    def __init__(self):
        nodes = []
        for (x1, x2, y1, y2) in RECTS.values():
            e = 1e-6
            for xx in (x1 - e, x2 + e):
                for yy in (y1 - e, y2 + e):
                    if R_VEH < xx < AREA - R_VEH and R_VEH < yy < AREA - R_VEH:
                        nodes.append((xx, yy))
        self.nodes = np.array(sorted(set(nodes)), dtype=float)
        self.n = len(self.nodes)
        # 邻接矩阵（通视即可连边）
        A = np.full((self.n, self.n), np.inf)   # 不通视必须为 inf，否则会被当作零成本边
        for i in range(self.n):
            A[i, i] = 0.0
            for j in range(i + 1, self.n):
                if self._seg_clear(self.nodes[i], self.nodes[j]):
                    A[i, j] = A[j, i] = np.hypot(*(self.nodes[i] - self.nodes[j]))
        self.A = A

    # --- 基础判定 -------------------------------------------------
    @staticmethod
    def _seg_rect_blocked(p, q, rect):
        """线段 pq 是否穿过膨胀矩形内部（贴边/相切不算穿过）。"""
        x1, x2, y1, y2 = rect
        p = np.asarray(p, float)
        q = np.asarray(q, float)
        d = q - p
        t0, t1 = 0.0, 1.0
        for k, (lo, hi) in enumerate(((x1, x2), (y1, y2))):
            if abs(d[k]) < 1e-15:
                if p[k] <= lo or p[k] >= hi:
                    return False
            else:
                ta = (lo - p[k]) / d[k]
                tb = (hi - p[k]) / d[k]
                if ta > tb:
                    ta, tb = tb, ta
                t0 = max(t0, ta)
                t1 = min(t1, tb)
                if t0 > t1:
                    return False
        return (t1 - t0) > 1e-9

    def _seg_clear(self, p, q):
        return not any(self._seg_rect_blocked(p, q, r) for r in RECTS.values())

    # --- 距离 -----------------------------------------------------
    def dist_many(self, src, targets):
        """src: (2,)，targets: (M,2) -> (M,) 最短路长度。"""
        src = np.asarray(src, float)
        tgt = np.asarray(targets, float).reshape(-1, 2)
        # 单源 Dijkstra（节点数 <= 20，直接朴素实现）
        INF = np.inf
        d = np.array([np.hypot(*(src - nd)) if self._seg_clear(src, nd) else INF for nd in self.nodes])
        fixed = np.zeros(self.n, dtype=bool)
        for _ in range(self.n):
            cand = np.where(fixed, INF, d)
            i = int(np.argmin(cand))
            if cand[i] == INF:
                break
            fixed[i] = True
            nd = d[i] + self.A[i]
            d = np.where(fixed, d, np.minimum(d, nd))
        best = np.full(len(tgt), INF)
        for i in range(self.n):
            if not np.isfinite(d[i]):
                continue
            hop = np.hypot(tgt[:, 0] - self.nodes[i, 0], tgt[:, 1] - self.nodes[i, 1])
            ok = np.array([self._seg_clear(self.nodes[i], t) for t in tgt])
            best = np.where(ok, np.minimum(best, d[i] + hop), best)
        direct_ok = np.array([self._seg_clear(src, t) for t in tgt])
        straight = np.hypot(tgt[:, 0] - src[0], tgt[:, 1] - src[1])
        best = np.where(direct_ok, np.minimum(best, straight), best)
        return best

    def dist(self, src, dst):
        return float(self.dist_many(src, [dst])[0])

    # --- 向量化版本：一次求"某点到一大批端点"的最短路（用于栅格场） --------
    @staticmethod
    def _seg_rect_blocked_vec(p, Q, rect):
        """点 p 到一批端点 Q (N,2) 的线段是否穿过矩形内部（贴边不算）。"""
        Q = np.asarray(Q, float)
        x1, x2, y1, y2 = rect
        n = Q.shape[0]
        d = Q - np.asarray(p, float)[None, :]
        t0 = np.zeros(n)
        t1 = np.ones(n)
        valid = np.ones(n, dtype=bool)
        for k, (lo, hi) in enumerate(((x1, x2), (y1, y2))):
            dk = d[:, k]
            pk = float(p[k])
            par = np.abs(dk) < 1e-15
            outside = par & ((pk <= lo) | (pk >= hi))
            safe = np.where(par, 1.0, dk)
            ta = (lo - pk) / safe
            tb = (hi - pk) / safe
            t0 = np.where(par, t0, np.maximum(t0, np.minimum(ta, tb)))
            t1 = np.where(par, t1, np.minimum(t1, np.maximum(ta, tb)))
            valid &= ~outside
        return valid & (t1 - t0 > 1e-9)

    def _blocked_any(self, p, Q):
        b = np.zeros(Q.shape[0], dtype=bool)
        for r in RECTS.values():
            b |= self._seg_rect_blocked_vec(p, Q, r)
        return b

    def dist_field(self, src, pts):
        """src: (2,)，pts: (N,2) -> (N,) 最短路长度（向量化）。"""
        pts = np.asarray(pts, float).reshape(-1, 2)
        INF = np.inf
        d = np.array([np.hypot(*(np.asarray(src, float) - nd)) if self._seg_clear(src, nd) else INF
                      for nd in self.nodes])
        fixed = np.zeros(self.n, dtype=bool)
        for _ in range(self.n):
            cand = np.where(fixed, INF, d)
            i = int(np.argmin(cand))
            if cand[i] == INF:
                break
            fixed[i] = True
            nd = d[i] + self.A[i]
            d = np.where(fixed, d, np.minimum(d, nd))
        best = np.full(pts.shape[0], INF)
        for i in range(self.n):
            if not np.isfinite(d[i]):
                continue
            hop = np.hypot(pts[:, 0] - self.nodes[i, 0], pts[:, 1] - self.nodes[i, 1])
            clear = ~self._blocked_any(self.nodes[i], pts)
            best = np.where(clear, np.minimum(best, d[i] + hop), best)
        straight = np.hypot(pts[:, 0] - src[0], pts[:, 1] - src[1])
        best = np.where(~self._blocked_any(src, pts), np.minimum(best, straight), best)
        return best

    def path(self, src, dst):
        """返回最短路的折线顶点，便于独立审查是否穿墙。"""
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        INF = np.inf
        d = np.array([np.hypot(*(src - nd)) if self._seg_clear(src, nd) else INF for nd in self.nodes])
        pre = [-1] * self.n
        fixed = np.zeros(self.n, dtype=bool)
        for _ in range(self.n):
            cand = np.where(fixed, INF, d)
            i = int(np.argmin(cand))
            if cand[i] == INF:
                break
            fixed[i] = True
            for j in range(self.n):
                if not fixed[j] and np.isfinite(self.A[i, j]):
                    nd = d[i] + self.A[i, j]
                    if nd < d[j] - 1e-12:
                        d[j] = nd
                        pre[j] = i
        if self._seg_clear(src, dst):
            best, best_i = np.hypot(*(dst - src)), -1
        else:
            best, best_i = INF, None
        for i in range(self.n):
            if np.isfinite(d[i]) and self._seg_clear(self.nodes[i], dst):
                val = d[i] + np.hypot(*(dst - self.nodes[i]))
                if val < best:
                    best, best_i = val, i
        if best_i is None:
            return None, INF
        if best_i == -1:
            return [tuple(src), tuple(dst)], best
        chain = []
        k = best_i
        while k != -1:
            chain.append(tuple(self.nodes[k]))
            k = pre[k]
        chain.reverse()
        return [tuple(src)] + chain + [tuple(dst)], best

    def travel_time(self, src, dst):
        return self.dist(src, dst) / SPEED


PATH = PathMetric()


# ---------------------------------------------------------------- 时间轴仿真
@dataclass
class VehicleTask:
    alert: str
    vehicle: str
    station: tuple
    order: int          # 该车的第几个任务（0,1,...）
    az: float = 0.0
    el: float = 0.0


@dataclass
class TaskEvent:
    alert: str
    vehicle: str
    station: tuple
    depart: float
    arrive: float
    rot: float
    ready: float
    obs_start: float
    obs_end: float
    reset: float
    avail: float
    az: float
    el: float


def simulate(tasks, vehicle_state=None, obs_dur=OBS_DUR, max_iter=200, tol=1e-9):
    """给定任务序列，按"依次观测"规则推导完整时间轴。

    vehicle_state: {vid: (pos, avail_time)}，缺省为初始位置、0 时刻可用。
    tasks: list[VehicleTask]，同一车辆按 order 递增。
    返回 (events: dict[(alert, vehicle)] -> TaskEvent, info)
    """
    if vehicle_state is None:
        vehicle_state = {v: (VEH0[v], 0.0) for v in VEHICLE_IDS}

    # 每辆车的任务链
    chain = {}
    for t in tasks:
        chain.setdefault(t.vehicle, []).append(t)
    for v in chain:
        chain[v].sort(key=lambda t: t.order)

    # 每辆车的每段行驶：上一任务结束位置/时刻
    leg_info = {}   # id(task) -> (depart, dist)
    for v, lst in chain.items():
        pos, avail = vehicle_state[v]
        for t in lst:
            d = PATH.dist(pos, t.station)
            leg_info[id(t)] = (avail, d)
            # 该段结束时刻依赖观测开始，先占位，下面迭代
            pos = t.station
            avail = None

    # 迭代求不动点：obs_start 只增不减，收敛即得最早可行时刻
    start = {id(t): 0.0 for t in tasks}
    for _ in range(max_iter):
        changed = False
        # 先按车辆链推出各任务的 ready 时刻
        ready = {}
        for v, lst in chain.items():
            pos, avail = vehicle_state[v]
            for t in lst:
                depart = avail
                d = PATH.dist(pos, t.station)
                arrive = depart + d / SPEED
                rot = float(gimbal_turn_time(t.az, t.el))
                ready[id(t)] = arrive + rot
                s = start[id(t)]
                avail = s + obs_dur + float(gimbal_reset_time(t.az, t.el))
                pos = t.station
        # 再按告警耦合：同一告警两车须同时就绪
        by_alert = {}
        for t in tasks:
            by_alert.setdefault(t.alert, []).append(id(t))
        for a, ids in by_alert.items():
            m = max(max(ready[i], start[i]) for i in ids)
            for i in ids:
                if m > start[i] + tol:
                    start[i] = m
                    changed = True
        if not changed:
            break
    else:
        raise RuntimeError("时间轴迭代未收敛：任务依赖可能存在环（互等死锁）")

    events = {}
    for v, lst in chain.items():
        pos, avail = vehicle_state[v]
        for t in lst:
            d = PATH.dist(pos, t.station)
            depart = avail
            arrive = depart + d / SPEED
            rot = float(gimbal_turn_time(t.az, t.el))
            s = start[id(t)]
            reset = float(gimbal_reset_time(t.az, t.el))
            e = TaskEvent(t.alert, v, t.station, depart, arrive, rot, arrive + rot,
                          s, s + obs_dur, reset, s + obs_dur + reset, t.az, t.el)
            events[(t.alert, v)] = e
            avail = e.avail
            pos = t.station
    return events


def alert_start_times(events, alerts=None):
    out = {}
    for (a, v), e in events.items():
        out[a] = max(out.get(a, -np.inf), e.obs_start)
    if alerts is not None:
        out = {a: out.get(a, np.inf) for a in alerts}
    return out


if __name__ == "__main__":
    # 自检
    print("TARGETS:", {k: tuple(round(c, 3) for c in v) for k, v in TARGETS.items()})
    rng = np.random.default_rng(0)
    g = np.linspace(0.5, 99.5, 199)
    X, Y = np.meshgrid(g, g)
    for k, T in TARGETS.items():
        m, r, elv, _ = observable(X, Y, T)
        frac = m.mean()
        print(f"{k}: 可视网格占比={frac:.3f}  r范围=[{r[m].min():.2f},{r[m].max():.2f}]")
    print("V1->(50,50) dist=%.3f" % PATH.dist(VEH0["V1"], (50, 50)))
    print("V1->(88,68) dist=%.3f" % PATH.dist(VEH0["V1"], (88, 68)))
