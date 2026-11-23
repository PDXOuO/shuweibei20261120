# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-20
#   辅助范围    : 结构枚举、时间轴仿真与绘图代码
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""问题2(2) 并发告警下的车辆复用与路径规划

目标：min  t_last = max{三个告警的观测开始时刻}
约束：① 每告警 2 车，测站须在该目标的可观测区域内；
     ② 6 席位 > 5 车 → 至少一辆车复用（本问枚举"1 车复用 2 席"的全部 360 种结构）；
     ③ 严格按"行驶 → 云台转动 → 同步观测 5 s → 复位 → 再出发"的时序规则；
     ④ 交会角 γ ≥ γ_min（抑制病态交会，γ_min 由问题2(1)的退化分析确定）。

方法
----
A) 结构枚举 + 候选池最优配对 + 坐标下降精修（主方法）
B) 最近分配启发式（题目要求的基线）

为说明"纯时间最优"与"可用精度"的张力，另跑两组对照：
   · 质量增强方案：γ ≥ 60° 且 r ≤ 45 m
   · 退化对照：γ ≥ 0.5°（几乎不设交会角约束）

运行：python "Q2-2_并发告警车辆复用/code/solve.py"
输出：output/q2_2_result.json、output/q2_2_timeline.csv、figures/*.png
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

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

T_START = time.perf_counter()
ALERTS = ("T1", "T2", "T3")


# ---------------------------------------------------------------- 基础工具
def tasks_from(stations, v_order=None):
    tasks, cnt = [], {}
    for (alert, veh), st in stations.items():
        T = M.TARGETS[alert]
        az = float(M.azimuth_to(st[0], st[1], T))
        el = float(M.elev_deg(M.horiz_dist(st[0], st[1], T), T))
        k = cnt.get(veh, 0)
        cnt[veh] = k + 1
        tasks.append(M.VehicleTask(alert, veh, tuple(st), k, az, el))
    for t in tasks:
        if v_order and t.vehicle in v_order:
            t.order = v_order[t.vehicle].index(t.alert)
    return tasks


def evaluate(stations, v_order=None):
    ev = M.simulate(tasks_from(stations, v_order))
    starts = {}
    for (a, v), e in ev.items():
        starts[a] = max(starts.get(a, -np.inf), e.obs_start)
    return max(starts.values()), ev, starts


def pair_gamma(stations, alert, T):
    vs = [v for (a, v) in stations if a == alert]
    if len(vs) != 2:
        return None
    az = [float(M.azimuth_to(stations[(alert, v)][0], stations[(alert, v)][1], T)) for v in vs]
    d = abs(M.wrap180(az[0] - az[1]))
    return vs, min(d, 180 - d), az


def sigma_of(stations, alert):
    """按问题2(1)模型（σ=1°）给出该告警的定位精度估计。"""
    T = M.TARGETS[alert]
    g = pair_gamma(stations, alert, T)
    if g is None:
        return None
    vs, gam, _ = g
    r = [float(M.horiz_dist(stations[(alert, v)][0], stations[(alert, v)][1], T)) for v in vs]
    if gam < 1e-6:
        return float("inf")
    return float(np.radians(1.0) * math.hypot(*r) / math.sin(math.radians(gam)))


def refine(stations, v_order, gmin, step=0.5, rounds=8, min_step=0.02):
    st = {k: np.array(v, float) for k, v in stations.items()}
    tl, _, _ = evaluate(st, v_order)
    for _ in range(rounds):
        improved = False
        for key in st:
            T = M.TARGETS[key[0]]
            for dim in (0, 1):
                for sgn in (+1, -1):
                    trial = {k: q.copy() for k, q in st.items()}
                    trial[key][dim] += sgn * step
                    p = trial[key]
                    m, _, _, _ = M.observable(np.array([p[0]]), np.array([p[1]]), T)
                    if not m[0]:
                        continue
                    g = pair_gamma(trial, key[0], T)
                    if g is not None and g[1] < gmin:
                        continue
                    try:
                        nt, _, _ = evaluate(trial, v_order)
                    except RuntimeError:
                        continue
                    if nt < tl - 1e-9:
                        st, tl, improved = trial, nt, True
        if not improved:
            step *= 0.5
            if step < min_step:
                break
    return {k: tuple(q) for k, q in st.items()}, tl


# ---------------------------------------------------------------- 主求解器
def solve_config(gmin, rmax=np.inf, label="", topk=6, verbose=True):
    structs = P.enumerate_structures()
    t0 = time.perf_counter()
    cand = []
    for s in structs:
        # 把 rmax 注入候选池
        d = eval_struct_with_rmax(s, gmin, rmax)
        if d is not None:
            cand.append(d)
    cand.sort(key=lambda d: d["t_last"])
    if verbose:
        print(f"[{label}] 枚举 {len(structs)} 结构, 候选池最优 t_last={cand[0]['t_last']:.3f}s, "
              f"耗时 {time.perf_counter()-t0:.1f}s", flush=True)
    out = []
    for d in cand[:topk]:
        v, (a1, a2), a3, p1, p2 = d["structure"]
        st, tl = refine(d["stations"], {v: [a1, a2]}, gmin)
        out.append((tl, st, d))
    out.sort(key=lambda x: x[0])
    tl, st, d = out[0]
    v, (a1, a2), a3, p1, p2 = d["structure"]
    tl2, ev, starts = evaluate(st, {v: [a1, a2]})
    if verbose:
        print(f"[{label}] 精修后 t_last={tl2:.3f}s, 结构={v}({a1}->{a2}) 伙伴 {p1}/{p2}, "
              f"{a3} 由 {[x for x in M.VEHICLE_IDS if x not in (v,p1,p2)]}", flush=True)
    return {"label": label, "gmin": gmin, "rmax": rmax, "t_last": tl2,
            "stations": st, "structure": d["structure"], "events": ev,
            "starts": starts, "structure_count": len(structs)}


def eval_struct_with_rmax(struct, gmin, rmax):
    """同 planner.eval_structure，但把 rmax 传入候选池构造。"""
    v, (a1, a2), a3, p1, p2 = struct
    Pv1 = P.build_pool(a1, v, rmax=rmax)
    Pp1 = P.build_pool(a1, p1, rmax=rmax)
    Pv2 = P.build_pool(a2, v, rmax=rmax)
    Pp2 = P.build_pool(a2, p2, rmax=rmax)
    rest = [x for x in M.VEHICLE_IDS if x not in (v, p1, p2)]
    if len(rest) != 2:
        return None
    Pa = P.build_pool(a3, rest[0], rmax=rmax)
    Pb = P.build_pool(a3, rest[1], rmax=rmax)
    t3, i3, j3 = P.pair_best(Pa, Pb, gmin)
    if not np.isfinite(t3):
        return None
    b1, j1 = P.pair_best_given(Pp1, Pv1.az, gmin)
    b2, j2 = P.pair_best_given(Pp2, Pv2.az, gmin)
    D12 = P.dist_matrix(Pv1, Pv2)
    best = (np.inf, None)
    for i in range(len(Pv1)):
        if not np.isfinite(b1[i]):
            continue
        start1 = max(Pv1.ready[i], b1[i])
        c1 = start1 + M.OBS_DUR + Pv1.rot[i]
        start2 = np.maximum(c1 + D12[i] / M.SPEED + Pv2.rot, b2)
        k = int(np.argmin(start2))
        if not np.isfinite(start2[k]):
            continue
        obj = max(start1, start2[k], t3)
        if obj < best[0]:
            best = (float(obj), (i, int(j1[i]), k, int(j2[k]), i3, j3))
    if best[1] is None:
        return None
    i, j, k, m, i3b, j3b = best[1]
    return {"t_last": best[0], "structure": struct, "a3": a3, "a3_veh": rest,
            "stations": {(a1, v): Pv1.station(i), (a1, p1): Pp1.station(j),
                         (a2, v): Pv2.station(k), (a2, p2): Pp2.station(m),
                         (a3, rest[0]): Pa.station(i3b), (a3, rest[1]): Pb.station(j3b)}}


# ---------------------------------------------------------------- 三个配置
main = solve_config(15.0, np.inf, "主方案(γ≥15°)")
qual = solve_config(60.0, 45.0, "质量增强(γ≥60°,r≤45m)")
loose = solve_config(0.5, np.inf, "退化对照(γ≥0.5°)")

# ---------------------------------------------------------------- 基线启发式
heur = P.nearest_heuristic(alerts=ALERTS, gmin=15.0)
ev_h, plan_h = heur
starts_h = {}
for (a, v), e in ev_h.items():
    starts_h[a] = max(starts_h.get(a, -np.inf), e.obs_start)
tl_h = max(starts_h.values())

st_opt, ev_opt, starts_opt = main["stations"], main["events"], main["starts"]
tl_opt = main["t_last"]
v_reuse, (a_first, a_second), a3, p1, p2 = main["structure"]
st_h = {(a, v): e.station for (a, v), e in ev_h.items()}


def geo_report(stations):
    out = {}
    for (alert, veh), st in sorted(stations.items()):
        T = M.TARGETS[alert]
        r = float(M.horiz_dist(st[0], st[1], T))
        out[f"{alert}-{veh}"] = {"站": [round(st[0], 3), round(st[1], 3)], "r_m": round(r, 3),
                                 "仰角_deg": round(float(M.elev_deg(r, T)), 2),
                                 "方位_deg": round(float(M.azimuth_to(st[0], st[1], T)), 2)}
    for al in ALERTS:
        g = pair_gamma(stations, al, M.TARGETS[al])
        if g is None:
            continue
        vs, gam, _ = g
        B = float(np.hypot(stations[(al, vs[0])][0] - stations[(al, vs[1])][0],
                           stations[(al, vs[0])][1] - stations[(al, vs[1])][1]))
        out[al + "_geometry"] = {"车辆": vs, "交会角_deg": round(gam, 2), "基线_m": round(B, 3),
                                 "σ_p_m": round(float(sigma_of(stations, al)), 4)}
    return out


geo_opt = geo_report(st_opt)
geo_h = geo_report(st_h)


# ---------------------------------------------------------------- 时间轴导出
def timeline_rows(events, tag):
    rows = []
    for (a, v), e in sorted(events.items(), key=lambda kv: (kv[0][1], kv[1].depart)):
        rows.append({"方案": tag, "告警": a, "车辆": v,
                     "测站x": round(e.station[0], 3), "测站y": round(e.station[1], 3),
                     "出发": round(e.depart, 3), "到达": round(e.arrive, 3),
                     "云台耗时": round(e.rot, 3), "云台就绪": round(e.ready, 3),
                     "观测开始": round(e.obs_start, 3), "观测结束": round(e.obs_end, 3),
                     "复位耗时": round(e.reset, 3), "可再出发": round(e.avail, 3),
                     "等待s": round(e.obs_start - e.ready, 3),
                     "方位角": round(e.az, 2), "仰角": round(e.el, 2)})
    return rows


rows = timeline_rows(ev_opt, "优化") + timeline_rows(ev_h, "最近分配")
with open(os.path.join(OUT, "q2_2_timeline.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

veh_tl = {}
for v in M.VEHICLE_IDS:
    seq = []
    for (a, vv), e in ev_opt.items():
        if vv != v:
            continue
        seq += [(e.depart, "出发"), (e.arrive, "到达测站"), (e.ready, "云台指向完成"),
                (e.obs_start, "观测开始"), (e.obs_end, "观测结束"), (e.avail, "复位完成")]
    seq.sort(key=lambda x: (x[0], x[1]))
    veh_tl[v] = [{"t": round(t, 2), "事件": ev_, "告警": next(e.alert for (a, vv), e in
                                                          ev_opt.items() if vv == v and abs(e.depart - t) < 1e-9)}
                 if ev_ == "出发" else {"t": round(t, 2), "事件": ev_} for t, ev_ in seq]
    # 简化：直接按 (时刻, 事件, 告警) 重排
    seq2 = []
    for (a, vv), e in ev_opt.items():
        if vv != v:
            continue
        seq2 += [(e.depart, "出发", a), (e.arrive, "到达测站", a), (e.ready, "云台指向完成", a),
                 (e.obs_start, "观测开始", a), (e.obs_end, "观测结束", a), (e.avail, "复位完成", a)]
    seq2.sort(key=lambda x: (x[0], ["出发", "到达测站", "云台指向完成", "观测开始",
                                    "观测结束", "复位完成"].index(x[1])))
    veh_tl[v] = [{"t": round(t, 2), "事件": e_, "告警": a} for t, e_, a in seq2]

res = {
    "问题": "2(2) 并发告警下的车辆复用与路径规划",
    "目标函数": "min t_last = max{三个告警的观测开始时刻}",
    "假设": {"交会角下限_deg": 15.0, "理由": "问题2(1)退化分析：γ<15° 时 1/sinγ>3.86，精度不可用"},
    "结构总数": 360,
    "候选池规模": {f"{a}-{v}": len(P.build_pool(a, v)) for a in ALERTS for v in M.VEHICLE_IDS},
    "优化方案": {
        "t_last_s": round(tl_opt, 3),
        "各告警观测开始": {k: round(v, 3) for k, v in sorted(starts_opt.items())},
        "复用结构": {"复用车辆": v_reuse, "先服务": a_first, "后服务": a_second,
                 "伙伴车": [p1, p2], "独立告警": a3,
                 "独立告警车辆": [x for x in M.VEHICLE_IDS if x not in (v_reuse, p1, p2)]},
        "测站": {f"{a}-{v}": [round(s[0], 3), round(s[1], 3)] for (a, v), s in sorted(st_opt.items())},
        "几何校验": geo_opt,
        "车辆时间轴": veh_tl,
    },
    "基线_最近分配启发式": {
        "t_last_s": round(tl_h, 3),
        "各告警观测开始": {k: round(v, 3) for k, v in sorted(starts_h.items())},
        "测站": {f"{a}-{v}": [round(s[0], 3), round(s[1], 3)] for (a, v), s in sorted(st_h.items())},
        "几何校验": geo_h,
    },
    "对比": {"优化_优于基线_s": round(tl_h - tl_opt, 3),
           "相对改进%": round(100 * (tl_h - tl_opt) / tl_h, 2),
           "优化_σ_p均值_m": round(float(np.mean([geo_opt[a + "_geometry"]["σ_p_m"] for a in ALERTS])), 3),
           "基线_σ_p均值_m": round(float(np.mean([geo_h[a + "_geometry"]["σ_p_m"] for a in ALERTS])), 3)},
    "敏感性": {
        "质量增强(γ≥60°,r≤45m)": {"t_last_s": round(qual["t_last"], 3),
                              "相对主方案增加_s": round(qual["t_last"] - tl_opt, 3),
                              "σ_p均值_m": round(float(np.mean([sigma_of(qual["stations"], a) for a in ALERTS])), 3)},
        "退化对照(γ≥0.5°)": {"t_last_s": round(loose["t_last"], 3),
                          "相对主方案减少_s": round(tl_opt - loose["t_last"], 3),
                          "σ_p均值_m": round(float(np.mean([sigma_of(loose["stations"], a) for a in ALERTS])), 3)},
    },
    "耗时_s": round(time.perf_counter() - T_START, 1),
}

with open(os.path.join(OUT, "q2_2_result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 图1 甘特图
fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=160)
colors = {"行驶": "#8ecae6", "云台转动": "#ffb703", "等待": "#e0e0e0",
          "观测": "#d62728", "复位": "#90be6d"}
ypos = {v: i for i, v in enumerate(M.VEHICLE_IDS)}
for (a, v), e in ev_opt.items():
    y = ypos[v]
    ax.barh(y, e.arrive - e.depart, left=e.depart, height=0.55, color=colors["行驶"], edgecolor="k", lw=0.4)
    ax.barh(y, e.ready - e.arrive, left=e.arrive, height=0.55, color=colors["云台转动"], edgecolor="k", lw=0.4)
    if e.obs_start > e.ready + 1e-6:
        ax.barh(y, e.obs_start - e.ready, left=e.ready, height=0.55, color=colors["等待"],
                edgecolor="k", lw=0.3, hatch="///")
    ax.barh(y, e.obs_end - e.obs_start, left=e.obs_start, height=0.55, color=colors["观测"], edgecolor="k", lw=0.5)
    ax.barh(y, e.avail - e.obs_end, left=e.obs_end, height=0.55, color=colors["复位"], edgecolor="k", lw=0.4)
    ax.text((e.obs_start + e.obs_end) / 2, y, a, ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold")
for a, s in starts_opt.items():
    ax.axvline(s, color="#1f77b4", ls=":", lw=0.9)
ax.axvline(tl_opt, color="k", ls="--", lw=1.2)
ax.text(tl_opt + 0.2, len(M.VEHICLE_IDS) - 0.45, f"t_last={tl_opt:.2f}s", fontsize=8)
ax.set_yticks(range(len(M.VEHICLE_IDS))); ax.set_yticklabels(M.VEHICLE_IDS)
ax.set_xlabel("时间 t / s")
ax.set_title("问题2(2) 优化方案车辆时间轴（行驶→云台转动→等待→观测 5 s→复位）", fontsize=10)
ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=k) for k, c in colors.items()],
          fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
ax.set_xlim(0, max(tl_opt, tl_h) + 5)
ax.grid(axis="x", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q2-2_gantt.png"))
plt.close(fig)

# ---------------------------------------------------------------- 图2 布站对比
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9), dpi=150)
cls = {"T1": "#d62728", "T2": "#2ca02c", "T3": "#9467bd"}
for ax, (stations_, title_) in zip(axes, [(st_opt, f"优化方案  t_last={tl_opt:.2f}s"),
                                          (st_h, f"最近分配启发式  t_last={tl_h:.2f}s")]):
    ax.set_facecolor("#f2f2f2")
    for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               facecolor="#d9d9d9", edgecolor="#666", lw=0.8, zorder=2))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, k, ha="center", va="center", fontsize=7, zorder=3)
    for v, p0 in M.VEH0.items():
        ax.plot(*p0, "s", ms=6, mfc="#1f77b4", mec="k", zorder=6)
        ax.annotate(v, p0, textcoords="offset points", xytext=(4, 4), fontsize=7)
    for al in ALERTS:
        ax.plot(*M.TARGETS[al][:2], "^", ms=13, mfc=cls[al], mec="k", zorder=9, label=f"目标 {al}")
    for (a, v), st in stations_.items():
        ax.plot(*st, "o", ms=8, mfc="white", mec=cls[a], mew=1.8, zorder=8)
        ax.plot([st[0], M.TARGETS[a][0]], [st[1], M.TARGETS[a][1]], "-", color=cls[a],
                lw=0.9, alpha=0.75, zorder=5)
        ax.annotate(v, st, textcoords="offset points", xytext=(5, -10), fontsize=6.5, color=cls[a])
        ax.plot(*M.VEH0[v], "s", ms=6, mfc="#1f77b4", mec="k", zorder=6)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
    ax.set_title(title_, fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2, framealpha=0.9)
fig.suptitle("问题2(2) 布站与交会关系（△ 目标，○ 测站，颜色=告警，标号=车辆）", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Q2-2_layout.png"))
plt.close(fig)

print(json.dumps({"优化": res["优化方案"]["各告警观测开始"], "t_last": tl_opt,
                  "基线": res["基线_最近分配启发式"], "对比": res["对比"],
                  "敏感性": res["敏感性"]}, ensure_ascii=False, indent=2))
