# -*- coding: utf-8 -*-
"""为英文论文重绘全部插图（英文标注），数据一律取自各子问题已有输出，不重新优化。

输出：PAPER_论文/en/figures/*.png
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
OUT = os.path.join(PAPER, "en", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

import common.model as M  # noqa: E402

# ---------------------------------------------------------------- 样式
C_BLD = "#d9d9d9"
C_EDGE = "#7f7f7f"
C_OK = "#8ecf8e"
C_OBS = "#2f7f2f"
C_T = ["#C44E52", "#4C72B0", "#DD8452"]
C_V = "#4C72B0"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})


def draw_buildings(ax, labels=True, alpha=1.0):
    """Draw all buildings; label only those whose centre lies inside the current view."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for k, (x1, x2, y1, y2, h) in M.BUILDINGS.items():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=C_BLD,
                               edgecolor=C_EDGE, lw=0.7, alpha=alpha, zorder=2))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if labels and xl[0] <= cx <= xl[1] and yl[0] <= cy <= yl[1]:
            ax.text(cx, cy, k, ha="center", va="center", fontsize=6.5,
                    color="#4d4d4d", zorder=3, clip_on=True)


def setup_xy(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---------------------------------------------------------------- Fig 1
def fig1_observable():
    step = 0.5
    xs = np.arange(step / 2, 100, step)
    X, Y = np.meshgrid(xs, xs)
    T1 = M.TARGETS["T1"]
    mask, r, elv, _ = M.observable(X, Y, T1)
    fig, ax = plt.subplots(figsize=(4.1, 3.8))
    ax.imshow(mask.astype(float), origin="lower", extent=(0, 100, 0, 100),
              cmap=matplotlib.colors.ListedColormap(["#ffffff", C_OK]),
              interpolation="nearest", vmin=0, vmax=1, zorder=1)
    draw_buildings(ax)
    th = np.linspace(0, 2 * np.pi, 400)
    r_lb = (T1[2] - M.CAM_H) / np.tan(np.radians(M.EL_MAX))
    ax.plot(T1[0] + r_lb * np.cos(th), T1[1] + r_lb * np.sin(th),
            ls="--", lw=0.9, color="#333333", zorder=4, label="optimal circle $r_{lb}$")
    ax.plot(*T1[:2], marker="*", ms=9, color=C_T[0], mec="k", mew=0.4, zorder=6,
            label="target T1")
    ax.plot(88.8247, 76.3233, marker="o", ms=5, mfc="none", mec="k", mew=0.9,
            zorder=7, label="selected station")
    setup_xy(ax)
    ax.set_title("Observable stations for T1 (0.5 m grid)", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=6, handlelength=1.4, frameon=True,
              facecolor="white", framealpha=0.9, edgecolor="none")
    ax.text(3, 96, f"observable area {5607.8:.0f} m$^2$", fontsize=6.5, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.6))
    fig.savefig(os.path.join(OUT, "fig1_observable.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def _segments(mask):
    """Yield index arrays of consecutive True runs in a circular mask."""
    n = mask.size
    if mask.all():
        yield np.arange(n)
        return
    idx = np.nonzero(~mask)[0]
    if idx.size == 0:
        yield np.arange(n)
        return
    cuts = list(idx)
    bounds = [(cuts[i] + 1, cuts[(i + 1) % len(cuts)]) for i in range(len(cuts))]
    for a, b in bounds:
        if b <= a:
            b += n
        sel = np.arange(a, b) % n
        if sel.size:
            yield sel


def fig2_arc():
    T1 = M.TARGETS["T1"]
    r_lb = (T1[2] - M.CAM_H) / np.tan(np.radians(M.EL_MAX))
    az = np.arange(0, 360, 0.25)
    px = T1[0] + r_lb * np.cos(np.radians(az))
    py = T1[1] + r_lb * np.sin(np.radians(az))
    ok = M.free_space(px, py)
    P = np.column_stack([px, py, np.full(px.size, M.CAM_H)])
    ok &= ~M.los_blocked(P, T1)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.set_xlim(75.5, 94.5); ax.set_ylim(61.0, 79.6)
    draw_buildings(ax)
    ax.plot(px, py, lw=0.8, ls=":", color="#bbbbbb", zorder=4)
    for seg in _segments(ok):
        ax.plot(px[seg], py[seg], lw=2.4, color="#C44E52", zorder=5, solid_capstyle="round")
    ax.plot(px[~ok], py[~ok], ".", ms=2.2, color="#7a7a7a", zorder=5)
    ax.plot(88.8247, 76.3233, marker="*", ms=13, color="#1a1a1a", zorder=7)
    ax.annotate("selected\n(88.8247, 76.3233)\nclearance 9.674 m",
                xy=(88.8247, 76.3233), xytext=(75.8, 78.6), fontsize=6.2,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#333"), zorder=8)
    ax.plot(*T1[:2], marker="^", ms=6, color="#4C72B0", mec="k", mew=0.4, zorder=7)
    ax.text(T1[0] - 1.9, T1[1] - 2.6, "T1", fontsize=7, color="#2b4f7a")
    ax.plot([], [], lw=2.4, color="#C44E52", label="feasible arc, 78.85% of circle")
    ax.plot([], [], ".", ms=3, color="#7a7a7a", label="blocked by buildings")
    ax.plot([], [], lw=0.8, ls=":", color="#bbbbbb", label="optimal circle $r_{lb}$ = 7.582 m")
    setup_xy(ax); ax.set_xlim(75.5, 94.5); ax.set_ylim(61.0, 79.6)
    ax.legend(loc="upper center", fontsize=6.2, handlelength=1.6, ncol=1,
              bbox_to_anchor=(0.5, -0.16))
    ax.set_title("All points on the optimal circle share $Q$ = 0.131890 m$^{-1}$", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_arc.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3_layout():
    d = json.load(open(os.path.join(ROOT, "Q2-2_并发告警车辆复用", "output",
                                    "q2_2_result.json"), encoding="utf-8"))
    st = d["优化方案"]["测站"]
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    draw_buildings(ax)
    alt_map = {"T1": 0, "T2": 1, "T3": 2}
    for key, (x, y) in st.items():
        alert, veh = key.split("-")
        j = alt_map[alert]
        v0 = M.VEH0[veh]
        ax.annotate("", xy=(x, y), xytext=v0,
                    arrowprops=dict(arrowstyle="->", lw=0.6, color=C_T[j], alpha=0.75),
                    zorder=4)
        ax.plot(x, y, marker="o", ms=5.5, mfc=C_T[j], mec="k", mew=0.4, zorder=6)
        ax.text(x + 1.2, y + 1.2, f"{veh}", fontsize=6, color=C_T[j], zorder=7)
    for t, j in alt_map.items():
        tx, ty = M.TARGET_XY[t]
        ax.plot(tx, ty, marker="*", ms=10, color=C_T[j], mec="k", mew=0.4, zorder=7)
        ax.text(tx + 1.5, ty - 4.0, t, fontsize=7, color=C_T[j], zorder=7)
    for v, (x, y) in M.VEH0.items():
        ax.plot(x, y, marker="s", ms=4, color=C_V, mec="k", mew=0.3, zorder=6)
    ax.plot([], [], marker="s", ls="", color=C_V, label="vehicle start")
    ax.plot([], [], marker="*", ls="", ms=9, color="#888", label="target")
    ax.plot([], [], marker="o", ls="", color="#888", label="station")
    setup_xy(ax)
    ax.set_title("Station layout: V3 serves T1 and T3 from one point", fontsize=8)
    ax.legend(loc="lower left", fontsize=6, handlelength=1.0)
    fig.savefig(os.path.join(OUT, "fig3_layout.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4_gantt():
    rows = []
    with open(os.path.join(ROOT, "Q2-2_并发告警车辆复用", "output",
                           "q2_2_timeline.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["方案"].strip() == "优化":
                rows.append(r)
    if not rows:
        rows = []
        with open(os.path.join(ROOT, "Q2-2_并发告警车辆复用", "output",
                               "q2_2_timeline.csv"), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["观测开始"]))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    h = 0.55
    for i, r in enumerate(rows):
        arr = float(r["到达"]); gr = float(r["云台就绪"])
        s0 = float(r["观测开始"]); s1 = float(r["观测结束"]); back = float(r["可再出发"])
        ax.barh(i, arr, height=h, left=0, color="#c8c8c8", edgecolor="none")
        ax.barh(i, max(gr - arr, 0), height=h, left=arr, color="#e8c26a", edgecolor="none")
        ax.barh(i, max(s0 - gr, 0), height=h, left=gr, color="#dcdcdc",
                edgecolor="#999", lw=0.4, hatch="///")
        ax.barh(i, s1 - s0, height=h, left=s0, color="#4C72B0", edgecolor="none")
        ax.barh(i, max(back - s1, 0), height=h, left=s1, color="#dcdcdc",
                edgecolor="#999", lw=0.4, hatch="///")
        ax.text(s0 + 0.1, i, f"{s0:.2f}", va="center", ha="left", fontsize=6.2,
                color="white", zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r["告警"]} · {r["车辆"]}' for r in rows], fontsize=7)
    ax.set_xlabel("time (s)")
    ax.axvline(10.655, ls="--", lw=0.9, color="#C44E52")
    ax.text(10.75, -0.75, "makespan 10.655 s", fontsize=6.5, color="#C44E52")
    ax.set_ylim(-1.0, len(rows) - 0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c8c8c8", label="drive"),
                       Patch(color="#e8c26a", label="gimbal slew"),
                       Patch(color="#4C72B0", label="observe (5 s)"),
                       Patch(facecolor="#dcdcdc", edgecolor="#999", hatch="///",
                             label="resettle / wait")],
              loc="lower right", fontsize=6, ncol=2, handlelength=1.2)
    ax.set_title("Timeline of the optimal plan (V3 reuses one station)", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig4_gantt.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5_sigma():
    r, a, b = 40.0, 0.3, 0.02
    sb, sc = 0.8, 0.6
    s = np.linspace(0, 100, 400)
    sig = a + b * s
    geo = np.degrees(np.arctan(sig / r))
    tot = np.sqrt(geo ** 2 + sb ** 2 + sc ** 2)
    crit = 5.0 / 1.959964
    smax = (r * np.tan(np.radians(np.sqrt(crit ** 2 - sb ** 2 - sc ** 2))) - a) / b

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    ax.plot(s, tot, lw=1.6, color="#4C72B0", label=r"$\sigma_{\Delta\psi}(s)$")
    ax.axhline(crit, ls="--", lw=1.0, color="#C44E52",
               label=r"criterion $2.551^\circ$")
    ax.axvline(smax, ls=":", lw=0.9, color="#333")
    ax.plot([smax], [crit], marker="o", ms=4, color="#C44E52", zorder=5)
    ax.annotate(f"$s_{{max}}$ = {smax:.2f} m", xy=(smax, crit), xytext=(smax + 6, crit - 0.55),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.plot([80], [np.sqrt(np.degrees(np.arctan(1.9 / r)) ** 2 + sb ** 2 + sc ** 2)],
            marker="s", ms=4, color="#DD8452", zorder=5)
    ax.text(80.5, 3.05, "nominal @ 80 m\n(0.9156)", fontsize=6, color="#DD8452")
    ax.set_xlabel("travelled distance $s$ (m)")
    ax.set_ylabel(r"pointing error $\sigma_{\Delta\psi}$ (deg)")
    ax.set_xlim(0, 100); ax.set_ylim(0.9, 3.7)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4)
    ax.set_title("Error growth versus the 95% criterion", fontsize=8)

    ax2 = axes[1]
    ax2.stackplot(s, geo ** 2, np.full_like(s, sb ** 2), np.full_like(s, sc ** 2),
                  colors=["#4C72B0", "#DD8452", "#8c8c8c"],
                  labels=["localisation", "gimbal bias", "gimbal control"], alpha=0.9)
    ax2.set_xlabel("travelled distance $s$ (m)")
    ax2.set_ylabel(r"variance share (deg$^2$)")
    ax2.set_xlim(0, 100)
    ax2.axvline(19.91, ls=":", lw=0.9, color="#333")
    ax2.text(21, 1.0, "equal-contribution\npoint 19.91 m", fontsize=6)
    ax2.legend(loc="upper left", fontsize=6.5)
    ax2.set_title(r"Variance budget of $\sigma_{\Delta\psi}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_sigma.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig6_scenarios():
    path = os.path.join(ROOT, "Q4_扰动重调度", "output", "q4_scenarios.csv")
    recs = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["情景类别"], r["θ"], r["R"], r["失效车"], r["t_rec"])
            recs.setdefault(key, {})[r["策略"]] = r
    cats = {"单一-新告警": ("new alert", "#4C72B0"),
            "单一-失效": ("vehicle failure", "#C44E52"),
            "单一-通信": ("communication", "#8c8c8c"),
            "组合": ("combined", "#DD8452")}
    fig, ax = plt.subplots(figsize=(4.5, 3.9))
    seen = set()
    for key, d in recs.items():
        if "S0" not in d or "S1" not in d:
            continue
        s0, s1 = d["S0"], d["S1"]
        cat = key[0]
        label, col = cats.get(cat, ("other", "#333"))
        eta_up = float(s1["η"]) > float(s0["η"]) + 1e-9
        ax.scatter(float(s0["t_last"]), float(s1["t_last"]),
                   s=26 if eta_up else 13,
                   facecolor="none" if eta_up else col,
                   edgecolor=col, lw=0.9, alpha=0.95,
                   label=label if label not in seen else None, zorder=4)
        if label not in seen:
            seen.add(label)
    lim = [8, 32]
    ax.plot(lim, lim, ls="--", lw=0.8, color="#666", zorder=1)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("$t_{last}$ of S0 (keep original plan), s")
    ax.set_ylabel("$t_{last}$ of S1 (reschedule), s")
    ax.plot([], [], marker="o", ms=5, mfc="none", mec="#333", ls="", label="completeness improves")
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.0)
    ax.set_title("93 disruption scenarios: rescheduling never delays the makespan",
                 fontsize=7.8)
    fig.savefig(os.path.join(OUT, "fig6_scenarios.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_observable(); print("fig1 ok")
    fig2_arc(); print("fig2 ok")
    fig3_layout(); print("fig3 ok")
    fig4_gantt(); print("fig4 ok")
    fig5_sigma(); print("fig5 ok")
    fig6_scenarios(); print("fig6 ok")
    print("done ->", OUT)


# ---------------------------------------------------------------- Fig 7 (Q2-1)
def fig7_q21():
    d = json.load(open(os.path.join(ROOT, "Q2-1_两站交会定位精度", "output",
                                    "q2_1_result.json"), encoding="utf-8"))
    best = d["最优测站对"]
    s1, s2 = best["测站1"], best["测站2"]
    T = M.TARGETS["T2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    ax = axes[0]
    ax.set_xlim(-1, 30); ax.set_ylim(60, 88)
    draw_buildings(ax)
    ax.plot(*T[:2], marker="*", ms=12, color="#1a1a1a", zorder=7)
    ax.annotate("T2  (11.03, 72.50)\nz = 35 m, above B3 (h = 12 m)",
                xy=T[:2], xytext=(14.0, 85.5), fontsize=6.2,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#333"), zorder=8)
    for s, c, veh in [(s1, "#C44E52", "V3"), (s2, "#4C72B0", "V5")]:
        ax.plot(*s, marker="o", ms=6, mfc=c, mec="k", mew=0.4, zorder=7)
        ax.plot([s[0], T[0]], [s[1], T[1]], lw=0.9, color=c, ls="-", zorder=5)
        ax.text(s[0] - 0.9, s[1] + 1.6, veh, fontsize=7, color=c)
    ax.plot([], [], lw=1.0, color="#666", label="sight line")
    ax.plot([], [], marker="o", ls="", color="#888", label="station")
    ax.legend(loc="lower right", fontsize=6, handlelength=1.2)
    setup_xy(ax); ax.set_xlim(-1, 30); ax.set_ylim(60, 88)
    ax.set_title(f"Two-station geometry: $B$ = {best['基线长度B_m']:.3f} m, "
                 f"$\\gamma$ = {best['交会角γ_deg']:.1f} deg", fontsize=7.6)

    ax2 = axes[1]
    th = np.linspace(0, 2 * np.pi, 400)
    for rad, ls, lab, lw in [(0.2486, "--", "95% circle, 0.2486 m", 1.0),
                             (0.10156, "-", r"$1\sigma$ circle, 0.1016 m", 1.4)]:
        ax2.plot(rad * np.cos(th), rad * np.sin(th), ls=ls, lw=lw, color="#C44E52",
                 label=lab)
    ax2.axhline(0, lw=0.4, color="#bbb"); ax2.axvline(0, lw=0.4, color="#bbb")
    ax2.set_aspect("equal")
    ax2.set_xlabel("east error (m)"); ax2.set_ylabel("north error (m)")
    ax2.set_xlim(-0.32, 0.32); ax2.set_ylim(-0.34, 0.46)
    ax2.legend(loc="upper center", fontsize=6.2, bbox_to_anchor=(0.5, 1.0))
    ax2.set_title(r"Error circle at target: $\sigma_p$ = 0.14362 m", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_q21.png"))
    plt.close(fig)
