"""问题4 扰动下的重调度与方案比较

基线：问题2(2) 得到的最优调度（t_last = 10.655 s，V3 复用 T1→T3）。

扰动（t = 5 s 发生）
-------------------
D1 新告警 T4：(θ, R, 50)，θ ∈ [0,360)，R ∈ [30,50]，需 2 辆车；
D2 单车失效：5 辆车等概率，失效车退出服务（不阻碍通行）；
D3 通信中断 [5, t_rec]，t_rec ∈ (5,30]：中断期间车辆继续执行当前任务，
   新指令只能在 t_rec 之后送达 → 重调度生效时刻 t_eff = max(5, t_rec)。

规则
----
· 扰动前已开始观测的告警视为已完成（T1 在 2.678 s 开始 → 完成，不可重调度）；
· 已派出但尚未开始观测的车辆可重新分配（V1 等待中、V2 在途、V4 已指向但未观测）；
· 两方案在 t=0 都不预知扰动，重调度仅在扰动实际发生后才触发。

评价（窗口 [0,60] s）
--------------------
N = 窗口内需观测的告警数（含新增）；k = 成功启动观测的告警数（两车均到达并完成指向才算成功）；
η = k/N；t_last = 最晚观测开始时刻（无法在窗口内开始的记 60 s）。先比 η，再比 t_last。

策略
----
S0 保留原方案：T1/T2/T3 的原分配不动（失效车的任务直接作废），新告警用"最早空出的两辆车"补做；
S1 重调度：在 t_eff 用实时状态对所有"未开始观测"的告警重新分配（穷举 + 精修）。

运行：python "Q4_扰动重调度/code/solve.py"
输出：output/q4_scenarios.csv、output/q4_result.json、figures/*.png
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from common import model as M          # noqa: E402
from common import planner as P        # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(HERE), "output")
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

GMIN = 15.0
WIN = 60.0
T_DIS = 5.0
T_START = time.perf_counter()

# ---------------------------------------------------------------- 读入问题2(2) 基线
q2_path = os.path.join(ROOT, "Q2-2_并发告警车辆复用", "output", "q2_2_result.json")
with open(q2_path, encoding="utf-8") as f:
    q2 = json.load(f)
BASE_ST = {}
for k, v in q2["优化方案"]["测站"].items():
    a, veh = k.split("-")
    BASE_ST[(a, veh)] = tuple(v)
V_REUSE = q2["优化方案"]["复用结构"]["复用车辆"]
A_FIRST = q2["优化方案"]["复用结构"]["先服务"]
A_SECOND = q2["优化方案"]["复用结构"]["后服务"]


def make_tasks(stations, order_hint=None):
    tasks, cnt = [], {}
    for (a, v), st in stations.items():
        T = M.TARGETS[a]
        az = float(M.azimuth_to(st[0], st[1], T))
        el = float(M.elev_deg(M.horiz_dist(st[0], st[1], T), T))
        k = cnt.get(v, 0)
        cnt[v] = k + 1
        tasks.append(M.VehicleTask(a, v, tuple(st), k, az, el))
    if order_hint:
        for t in tasks:
            if t.vehicle in order_hint:
                t.order = order_hint[t.vehicle].index(t.alert)
    return tasks


BASE_EV = M.simulate(make_tasks(BASE_ST, {V_REUSE: [A_FIRST, A_SECOND]}))
BASE_START = {}
for (a, v), e in BASE_EV.items():
    BASE_START[a] = max(BASE_START.get(a, -np.inf), e.obs_start)

# ---------------------------------------------------------------- 状态推进
def path_len_polyline(poly):
    return [0.0] + list(itertools.accumulate(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)))


def point_on_polyline(poly, dist):
    cum = path_len_polyline(poly)
    dist = min(max(dist, 0.0), cum[-1])
    for i in range(len(poly) - 1):
        if cum[i + 1] >= dist - 1e-12:
            seg = cum[i + 1] - cum[i]
            t = 0.0 if seg < 1e-12 else (dist - cum[i]) / seg
            return (poly[i][0] + t * (poly[i + 1][0] - poly[i][0]),
                    poly[i][1] + t * (poly[i + 1][1] - poly[i][1]))
    return poly[-1]


def vehicle_state_at(events, t_eff, failed=None):
    """推进到 t_eff，返回 {veh: (pos, avail, busy_until, committed_alert)}。"""
    st = {}
    for v in M.VEHICLE_IDS:
        if v == failed:
            continue
        evs = [(a, e) for (a, vv), e in events.items() if vv == v]
        if not evs:
            st[v] = (M.VEH0[v], 0.0, 0.0, None)
            continue
        evs.sort(key=lambda x: x[1].depart)
        # 找到 t_eff 时正在执行/即将执行的那一段
        cur = None
        for a, e in evs:
            if t_eff <= e.avail + 1e-9:
                cur = (a, e)
                break
        if cur is None:
            a, e = evs[-1]
            st[v] = (e.station, e.avail, 0.0, None)
            continue
        a, e = cur
        if t_eff <= e.depart:
            st[v] = (events_origin(events, v), t_eff, 0.0, a)
        elif t_eff <= e.arrive:
            poly, _ = M.PATH.path(events_origin(events, v), e.station)
            pos = point_on_polyline(poly, (t_eff - e.depart) * M.SPEED)
            st[v] = (pos, t_eff, 0.0, a)          # 在途：可立即改派
        elif t_eff < e.obs_start:
            st[v] = (e.station, t_eff, 0.0, a)    # 已到站/已指向但未观测：可改派
        else:
            st[v] = (e.station, e.avail, e.avail - t_eff, a)   # 观测中/复位中：须等完成
    return st


def events_origin(events, v):
    evs = [e for (a, vv), e in events.items() if vv == v]
    evs.sort(key=lambda e: e.depart)
    return evs[0].station if evs else M.VEH0[v]


_READY_CACHE: dict = {}


# ---------------------------------------------------------------- 通用求解：给定告警集与车辆状态
def solve_pending(alerts, vstate, gmin=GMIN, win=WIN):
    """对 pending 告警重新分配（穷举车辆对 + 站内最优配对），返回 {alert: [(veh, station), ...]}。

    vstate: {veh: (pos, avail)}  ——  可立即改派的车辆（自由或等待中）
    每辆车最多承担 2 个 pending 告警（按可用时刻顺序）。
    """
    if not alerts:
        return {}, {}
    vlist = list(vstate.keys())
    if len(vlist) < 2:
        return {}, {}
    # 每个 (告警, 车辆) 的候选池
    pools = {}
    for a in alerts:
        for v in vlist:
            pos, av = vstate[v]
            pools[(a, v)] = P.build_pool(a, v, start_pos=pos, avail=av,
                                         step=1.0, bin_deg=20.0)
    # 每个 (告警, 车辆对) 的最优配对
    opt = {}
    for a in alerts:
        for u, w in itertools.combinations(vlist, 2):
            t0, i, j = P.pair_best(pools[(a, u)], pools[(a, w)], gmin)
            if np.isfinite(t0):
                opt[(a, (u, w))] = (t0, i, j)
    if not opt:
        return {}, {}
    best = None
    pair_choices = {}
    for a in alerts:
        ks = sorted([k for k in opt if k[0] == a], key=lambda k: opt[k][0])[:4]
        pair_choices[a] = ks
        if not ks:
            return {}, {}
    for combo in itertools.product(*[pair_choices[a] for a in alerts]):
        # 每辆车的任务数 <= 2
        cnt = {}
        for (a, (u, w)) in combo:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[w] = cnt.get(w, 0) + 1
        if any(c > 2 for c in cnt.values()):
            continue
        # 需要为承担 2 个告警的车确定先后 → 枚举告警整体顺序（<=3! 种）
        for perm in itertools.permutations(range(len(alerts))):
            avail = {v: vstate[v][1] for v in vlist}
            pos = {v: vstate[v][0] for v in vlist}
            stations = {}
            starts = {}
            ok = True
            for idx in perm:
                a, (u, w) = combo[idx]
                pu, pw = pools[(a, u)], pools[(a, w)]
                if avail[u] <= vstate[u][1] + 1e-9 and avail[w] <= vstate[w][1] + 1e-9:
                    # 首次执行：直接用池中预先算好的配对
                    t0, i, j = opt[(a, (u, w))]
                else:
                    # 至少一辆车已做过别的任务 → 从新位置重算 ready（带缓存）
                    P2 = []
                    for vv, pp in ((u, pu), (w, pw)):
                        kk = (a, vv, round(pos[vv][0], 2), round(pos[vv][1], 2), round(avail[vv], 3))
                        r = _READY_CACHE.get(kk)
                        if r is None:
                            r = avail[vv] + M.PATH.dist_field(
                                pos[vv], np.column_stack([pp.x, pp.y])) / M.SPEED + pp.rot
                            _READY_CACHE[kk] = r
                        P2.append(r)
                    R = np.maximum(P2[0][:, None], P2[1][None, :])
                    G = P.gamma_mat(pu, pw)
                    R = np.where(G >= gmin, R, np.inf)
                    k = int(np.argmin(R))
                    i, j = divmod(k, R.shape[1])
                    t0 = float(R[i, j])
                    if not np.isfinite(t0):
                        ok = False
                        break
                start = max(t0, avail[u], avail[w])
                if start > win:
                    ok = False
                    break
                su, sw = pu.station(i), pw.station(j)
                stations[(a, u)] = su
                stations[(a, w)] = sw
                starts[a] = start
                for vv, pp, ss, ii in ((u, pu, su, i), (w, pw, sw, j)):
                    avail[vv] = start + M.OBS_DUR + float(pp.rot[ii])
                    pos[vv] = ss
            if not ok:
                continue
            tl = max(starts.values()) if starts else np.inf
            if best is None or (len(starts), -tl) > (best[0], -best[1]):
                best = (len(starts), tl, dict(stations), dict(starts))
    if best is None:
        return {}, {}
    return best[2], best[3]


# ---------------------------------------------------------------- 方案评估
def evaluate_plan(stations, vstate, order_hint=None):
    """用精确仿真评估：只有"两车都到位并完成指向"的告警才算成功。

    返回 (k, t_last, events, starts)；k 为成功启动观测的告警数。
    """
    from collections import Counter
    cnt = Counter(a for (a, v) in stations)
    ok_alerts = {a for a, c in cnt.items() if c >= 2}          # 少于 2 车的告警不能交会
    use = {k_: v_ for k_, v_ in stations.items() if k_[0] in ok_alerts}
    if not use:
        return 0, WIN, {}, {}
    tasks = make_tasks(use, order_hint)
    ev = M.simulate(tasks, vehicle_state={v: (vstate[v][0], vstate[v][1]) for v in vstate})
    starts = {}
    for (a, v), e in ev.items():
        starts[a] = max(starts.get(a, -np.inf), e.obs_start)
    good = {a: st_ for a, st_ in starts.items() if st_ <= WIN + 1e-9}
    k = len(good)
    tl = max(good.values()) if good else WIN
    return k, tl, ev, starts


def scenario_run(theta=None, R=None, fail=None, t_rec=5.0, H=50.0):
    """跑一个情景，返回 S0 与 S1 的结果。"""
    t_eff = max(T_DIS, t_rec)
    # 基线在 t_eff 的状态（若车辆在 t_eff 前失效，则从状态中剔除）
    st = vehicle_state_at(BASE_EV, t_eff, failed=fail)
    alerts = [a for a in ("T1", "T2", "T3")]
    done = {a for a, s in BASE_START.items() if s <= T_DIS}      # 扰动前已开始 → 视为完成
    pending = [a for a in alerts if a not in done]
    if theta is not None:
        M.TARGETS["T4"] = M.alert_position(theta, R, H)
        M.TARGET_XY["T4"] = M.TARGETS["T4"][:2]
        P.reset_caches()          # T4 位置随情景变化，必须清空栅格/池/距离缓存
        _READY_CACHE.clear()
        alerts = alerts + ["T4"]

    out = {}
    # ---------- S0 保留原方案
    st0 = {}
    for v, (pos, avail, busy, cur) in st.items():
        st0[v] = (pos, avail)
    # 已完成的告警不再进入仿真（其车辆可自由改派），k 里直接计入
    keep = {}
    for (a, v), st_ in BASE_ST.items():
        if a in done:
            continue
        if v in st0:             # 未失效且已派出的车继续原任务
            keep[(a, v)] = st_
    if theta is not None:
        # 新告警：用最早空出的两辆车，站内最优配对（不触动原方案）
        t4 = {}
        for v in st0:
            t4[v] = P.build_pool("T4", v, start_pos=st0[v][0], avail=st0[v][1])
        best = None
        for u, w in itertools.combinations(list(st0.keys()), 2):
            t0, i, j = P.pair_best(t4[u], t4[w], GMIN)
            if np.isfinite(t0) and (best is None or t0 < best[0]):
                best = (t0, u, w, i, j)
        if best is not None:
            _, u, w, i, j = best
            keep[("T4", u)] = t4[u].station(i)
            keep[("T4", w)] = t4[w].station(j)
    k0p, tl0, ev0, starts0 = evaluate_plan(keep, st0)
    k0 = len(done) + k0p
    if not starts0:                       # 无待观测告警（极端情形）
        tl0 = max(BASE_START.values())
    out["S0"] = {"k": k0, "t_last": tl0, "stations": keep, "starts": starts0, "events": ev0}

    # ---------- S1 重调度
    pend = [a for a in pending]
    if theta is not None:
        pend = pend + ["T4"]
    vstate = {v: (st[v][0], st[v][1]) for v in st}
    new_st, new_starts = solve_pending(pend, vstate)
    keep1 = dict(new_st)                                             # 只重排待观测告警
    k1p, tl1, ev1, starts1 = evaluate_plan(keep1, vstate)
    if not starts1:
        tl1 = max(BASE_START.values())
    # 安全兜底：重调度允许"不改动"，故与"沿用原分配"比较后取优（(k, -t_last) 字典序）
    k0p2, tl0p2, ev0p2, starts0p2 = evaluate_plan({k_: v_ for k_, v_ in keep.items()},
                                                  st0)
    if (k0p2, -tl0p2) > (k1p, -tl1):
        keep1, k1p, tl1, ev1, starts1 = {k_: v_ for k_, v_ in keep.items()}, k0p2, tl0p2, ev0p2, starts0p2
        out["S1_选择"] = "沿用原分配"
    else:
        out["S1_选择"] = "重新优化"
    k1 = len(done) + k1p
    out["S1"] = {"k": k1, "t_last": tl1, "stations": keep1, "starts": starts1, "events": ev1}
    out["meta"] = {"theta": theta, "R": R, "fail": fail, "t_rec": t_rec, "t_eff": t_eff,
                   "done": sorted(done), "pending": pend, "N": len(alerts)}
    return out


# ---------------------------------------------------------------- 情景集
scen = []
# S-单一①：仅新告警（36 个位置）
for th in range(0, 360, 30):
    for R in (30, 40, 50):
        scen.append(("单一-新告警", dict(theta=th, R=R, fail=None, t_rec=5.0)))
# S-单一②：仅单车失效（5 个）
for v in M.VEHICLE_IDS:
    scen.append(("单一-失效", dict(theta=None, R=None, fail=v, t_rec=5.0)))
# S-单一③：仅通信中断（4 个 t_rec，新告警取代表位置 θ=210,R=40）
for tr in (10.0, 15.0, 20.0, 30.0):
    scen.append(("单一-通信", dict(theta=210, R=40, fail=None, t_rec=tr)))
# S-组合：新告警 × 失效 × 通信（分层抽样）
for th in (0, 60, 120, 180, 240, 300):
    for R in (30, 50):
        for v in ("V2", "V4"):
            for tr in (5.0, 20.0):
                scen.append(("组合", dict(theta=th, R=R, fail=v, t_rec=tr)))

rows = []
print(f"[情景] 共 {len(scen)} 个情景 × 2 策略", flush=True)
for i, (tag, kw) in enumerate(scen):
    r = scenario_run(**kw)
    N = r["meta"]["N"]
    for s in ("S0", "S1"):
        st_ = r[s]["starts"]
        n_done = len(r["meta"]["done"])
        eta20 = (n_done + sum(1 for x in st_.values() if x <= 20.0)) / N
        eta30 = (n_done + sum(1 for x in st_.values() if x <= 30.0)) / N
        rows.append({"情景类别": tag, "θ": kw["theta"], "R": kw["R"],
                     "失效车": kw.get("fail") or "-", "t_rec": kw["t_rec"], "N": N,
                     "策略": s, "k": r[s]["k"], "η": round(r[s]["k"] / N, 4),
                     "η@20s窗口": round(eta20, 4), "η@30s窗口": round(eta30, 4),
                     "t_last": round(r[s]["t_last"], 3),
                     "各告警开始": "; ".join(f"{a}:{v:.2f}" for a, v in sorted(st_.items()))})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(scen)}  耗时 {time.perf_counter()-T_START:.1f}s", flush=True)

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "q4_scenarios.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 汇总
def agg(sub):
    out = {}
    for s_ in ("S0", "S1"):
        d = sub[sub["策略"] == s_]
        out[s_] = {"平均η": round(float(d["η"].mean()), 4),
                   "平均η@20s": round(float(d["η@20s窗口"].mean()), 4),
                   "平均η@30s": round(float(d["η@30s窗口"].mean()), 4),
                   "平均t_last": round(float(d["t_last"].mean()), 3),
                   "平均N": round(float(d["N"].mean()), 2)}
    a0 = sub[sub["策略"] == "S0"].reset_index(drop=True)
    a1 = sub[sub["策略"] == "S1"].reset_index(drop=True)
    # 按题意的字典序比较：(η 大者优) → (η 相同则 t_last 小者优)
    e0, e1 = a0["η"].values, a1["η"].values
    t0, t1 = a0["t_last"].values, a1["t_last"].values
    better = (e1 > e0 + 1e-9) | ((np.abs(e1 - e0) <= 1e-9) & (t1 < t0 - 1e-6))
    worse = (e1 < e0 - 1e-9) | ((np.abs(e1 - e0) <= 1e-9) & (t1 > t0 + 1e-6))
    out["改进"] = {"Δ平均η": round(float(out["S1"]["平均η"] - out["S0"]["平均η"]), 4),
                 "Δt_last": round(float(out["S1"]["平均t_last"] - out["S0"]["平均t_last"]), 3),
                 "S1更优情景数": int(better.sum()),
                 "持平情景数": int((~better & ~worse).sum()),
                 "S1更差情景数": int(worse.sum()),
                 "η提升情景数": int((e1 > e0 + 1e-9).sum()),
                 "最大单项改进_s": round(float(np.max(t0 - t1)), 3)}
    return out


summary = {tag: agg(df[df["情景类别"] == tag]) for tag in df["情景类别"].unique()}
summary["全部"] = agg(df)

# 供报告引用的典型情景
def one(tag, **kw):
    r = scenario_run(**kw)
    return {s: {"k": r[s]["k"], "η": round(r[s]["k"] / r["meta"]["N"], 3),
                "t_last": round(r[s]["t_last"], 3),
                "各告警开始": {a: round(v, 2) for a, v in sorted(r[s]["starts"].items())},
                "测站": {f"{a}-{v}": [round(q[0], 2), round(q[1], 2)]
                       for (a, v), q in sorted(r[s]["stations"].items())}}
            for s in ("S0", "S1")} | {"meta": r["meta"]}


examples = {
    "示例A_新告警在东南(θ=300,R=50)": one("单一-新告警", theta=300, R=50, fail=None, t_rec=5.0),
    "示例B_新告警在正西(θ=180,R=30)_V2失效": one("组合", theta=180, R=30, fail="V2", t_rec=5.0),
    "示例C_V1失效": one("单一-失效", theta=None, R=None, fail="V1", t_rec=5.0),
    "示例D_通信中断至30s": one("单一-通信", theta=210, R=40, fail=None, t_rec=30.0),
}

res = {
    "问题": "4 扰动下的重调度与方案比较",
    "基线": {"t_last": round(BASE_START and max(BASE_START.values()), 3),
           "各告警开始": {k: round(v, 3) for k, v in sorted(BASE_START.items())}},
    "扰动设定": {"发生时刻_s": T_DIS, "新告警": "θ∈[0,360), R∈[30,50], H=50",
             "失效": "5 车等概率，退出服务", "通信": "t_rec ∈ (5,30]",
             "重调度生效": "t_eff = max(5, t_rec)"},
    "情景集": {"数量": len(scen), "类别": {t: int((df['情景类别'] == t).sum() / 2)
                                     for t in df['情景类别'].unique()},
             "设计理由": ("新告警位置按 θ 每 30°×R∈{30,40,50} 全网格覆盖（观测区域随方位变化剧烈）；"
                      "失效覆盖全部 5 辆车；通信取 4 个代表时长；组合情景按 θ×R×失效×通信分层抽样。"
                      "评价准则按题意：先 η 后 t_last。")},
    "汇总": summary,
    "典型情景": examples,
    "耗时_s": round(time.perf_counter() - T_START, 1),
}
with open(os.path.join(OUT, "q4_result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------- 图1：η/t_last 随新告警方位变化
sub = df[(df["情景类别"] == "单一-新告警")]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=155)
for s, mk, c in (("S0", "o--", "#d62728"), ("S1", "s-", "#1f77b4")):
    d = sub[sub["策略"] == s]
    for R_, ls in ((30, ":"), (40, "-"), (50, "--")):
        dd = d[d["R"] == R_].sort_values("θ")
        axes[0].plot(dd["θ"], dd["η@20s窗口"], ls, marker="o", ms=3.5, color=c, alpha=0.85,
                     label=f"{s} R={R_}")
        axes[1].plot(dd["θ"], dd["t_last"], ls, marker="o", ms=3.5, color=c, alpha=0.85,
                     label=f"{s} R={R_}")
axes[0].set_xlabel("新告警方位 θ / °"); axes[0].set_ylabel("完成率 η（20 s 窗口）")
axes[0].set_ylim(0.4, 1.05); axes[0].grid(ls=":", alpha=0.5)
axes[0].set_title("单一新告警：完成率随方位变化（60 s 窗口下 η 恒为 1，\n故用 20 s 窗口放大差异）", fontsize=9.5)
axes[0].legend(fontsize=6.2, ncol=2)
axes[1].set_xlabel("新告警方位 θ / °"); axes[1].set_ylabel("t_last / s")
axes[1].grid(ls=":", alpha=0.5)
axes[1].set_title("单一新告警：最晚观测开始时刻", fontsize=10)
axes[1].legend(fontsize=6.2, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q4_new_alert.png"))
plt.close(fig)

# ---------------------------------------------------------------- 图2：分类汇总
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=155)
tags = [t for t in df["情景类别"].unique()]
x = np.arange(len(tags))
w = 0.36
e0 = [summary[t]["S0"]["平均η"] for t in tags]
e1 = [summary[t]["S1"]["平均η"] for t in tags]
axes[0].bar(x - w / 2, e0, w, label="S0 保留原方案", color="#d62728", alpha=0.85)
axes[0].bar(x + w / 2, e1, w, label="S1 重调度", color="#1f77b4", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(tags, fontsize=8)
axes[0].set_ylabel("平均完成率 η"); axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=8); axes[0].grid(axis="y", ls=":", alpha=0.5)
axes[0].set_title("各类扰动下的平均完成率", fontsize=10)
for i, (a, b) in enumerate(zip(e0, e1)):
    axes[0].text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5)
    axes[0].text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7.5)

t0 = [summary[t]["S0"]["平均t_last"] for t in tags]
t1 = [summary[t]["S1"]["平均t_last"] for t in tags]
axes[1].bar(x - w / 2, t0, w, label="S0", color="#d62728", alpha=0.85)
axes[1].bar(x + w / 2, t1, w, label="S1", color="#1f77b4", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(tags, fontsize=8)
axes[1].set_ylabel("平均 t_last / s"); axes[1].legend(fontsize=8)
axes[1].grid(axis="y", ls=":", alpha=0.5)
axes[1].set_title("各类扰动下的平均最晚观测开始时刻", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q4_summary.png"))
plt.close(fig)

# ---------------------------------------------------------------- 图3：典型情景时间轴对比
def gantt(ax, ev, title, tmax=60):
    colors = {"行驶": "#8ecae6", "云台转动": "#ffb703", "等待": "#e0e0e0",
              "观测": "#d62728", "复位": "#90be6d"}
    ypos = {v: i for i, v in enumerate(M.VEHICLE_IDS)}
    for (a, v), e in ev.items():
        y = ypos[v]
        ax.barh(y, e.arrive - e.depart, left=e.depart, height=0.55, color=colors["行驶"], edgecolor="k", lw=0.35)
        ax.barh(y, e.ready - e.arrive, left=e.arrive, height=0.55, color=colors["云台转动"], edgecolor="k", lw=0.35)
        if e.obs_start > e.ready + 1e-6:
            ax.barh(y, e.obs_start - e.ready, left=e.ready, height=0.55, color=colors["等待"],
                    edgecolor="k", lw=0.3, hatch="///")
        ax.barh(y, e.obs_end - e.obs_start, left=e.obs_start, height=0.55, color=colors["观测"], edgecolor="k", lw=0.45)
        ax.barh(y, e.avail - e.obs_end, left=e.obs_end, height=0.55, color=colors["复位"], edgecolor="k", lw=0.35)
        ax.text((e.obs_start + e.obs_end) / 2, y, a, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
    ax.axvline(T_DIS, color="k", ls=":", lw=1.0)
    ax.set_yticks(range(len(M.VEHICLE_IDS))); ax.set_yticklabels(M.VEHICLE_IDS)
    ax.set_xlim(0, tmax); ax.set_xlabel("t / s"); ax.set_title(title, fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.4)


ex = examples["示例B_新告警在正西(θ=180,R=30)_V2失效"]
fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), dpi=155)
r_b = scenario_run(theta=180, R=30, fail="V2", t_rec=5.0)
gantt(axes[0], r_b["S0"]["events"], f"S0 保留原方案：η={ex['S0']['η']}, t_last={ex['S0']['t_last']}s")
gantt(axes[1], r_b["S1"]["events"], f"S1 重调度：η={ex['S1']['η']}, t_last={ex['S1']['t_last']}s")
axes[0].legend(handles=[Patch(facecolor=c, edgecolor="k", label=k) for k, c in
                        (("行驶", "#8ecae6"), ("云台转动", "#ffb703"), ("等待", "#e0e0e0"),
                         ("观测", "#d62728"), ("复位", "#90be6d"))],
               fontsize=7, ncol=5, loc="lower right")
fig.suptitle("问题4 典型情景（θ=180°,R=30 m 新告警 + V2 失效）两策略时间轴对比", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q4_timeline_case.png"))
plt.close(fig)

# 典型情景 B 的逐事件时间轴
rows_b = []
for s_ in ("S0", "S1"):
    for (a, v), e in sorted(r_b[s_]["events"].items(), key=lambda kv: (kv[0][1], kv[1].depart)):
        rows_b.append({"方案": s_, "告警": a, "车辆": v,
                       "测站x": round(e.station[0], 2), "测站y": round(e.station[1], 2),
                       "出发": round(e.depart, 2), "到达": round(e.arrive, 2),
                       "云台耗时": round(e.rot, 2), "就绪": round(e.ready, 2),
                       "观测开始": round(e.obs_start, 2), "观测结束": round(e.obs_end, 2),
                       "复位耗时": round(e.reset, 2), "可再出发": round(e.avail, 2),
                       "等待": round(e.obs_start - e.ready, 2)})
if rows_b:
    with open(os.path.join(OUT, "q4_caseB_timeline.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        w.writeheader(); w.writerows(rows_b)

print("\n完成，用时 %.1f s" % (time.perf_counter() - T_START))
