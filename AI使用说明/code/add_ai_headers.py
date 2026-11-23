# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-23
#   辅助范围    : 注释块批量插入脚本
#   核验方式    : 队员逐行审阅；插入结果由脚本自行打印核对
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""给全部脚本插入 AI 辅助注释块（竞赛规定第 7 条）。幂等：已含标记则跳过。

注释块置于模块 docstring 之前，不影响 __doc__ 与运行逻辑。
"""
from __future__ import annotations

import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))   # 项目根

MARK = "AI 辅助说明（《数维杯人工智能工具使用规定"

TPL = """# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : {date}
#   辅助范围    : {scope}
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""

TARGETS = [
    ("common/model.py", "2026-11-20", "模块骨架、函数实现与注释"),
    ("common/validate_model.py", "2026-11-20", "校验用例设计与实现"),
    ("common/planner.py", "2026-11-20", "候选池、距离矩阵与结构枚举的实现"),
    ("main.py", "2026-11-20", "统一入口脚本"),
    ("Q1_可观测区域与最优测站/code/method_A_grid.py", "2026-11-20", "网格搜索与绘图代码"),
    ("Q1_可观测区域与最优测站/code/method_B_analytic.py", "2026-11-20", "解析下界推导的代码实现与弧段扫描"),
    ("Q2-1_两站交会定位精度/code/solve.py", "2026-11-20", "误差传播公式实现、枚举与差分进化调用"),
    ("Q2-2_并发告警车辆复用/code/solve.py", "2026-11-20", "结构枚举、时间轴仿真与绘图代码"),
    ("Q3_指向误差传播/code/solve.py", "2026-11-20", "误差传播与蒙特卡洛代码"),
    ("Q4_扰动重调度/code/solve.py", "2026-11-20", "状态推进、重调度评估与情景批量运行"),
    ("FINAL_汇总/code/make_summary.py", "2026-11-20", "汇总表生成代码"),
    ("PAPER_论文/code/fetch_refs.py", "2026-11-20", "Crossref 检索与 bib 生成"),
    ("PAPER_论文/code/make_en_figures.py", "2026-11-20", "英文版插图绘制代码"),
    ("AI使用说明/code/make_docx.py", "2026-11-23", "Markdown→Word 版式转换代码"),
    ("AI使用说明/code/insert_appendix.py", "2026-11-23", "论文附录插入脚本"),
    ("AI使用说明/code/add_ai_headers.py", "2026-11-23", "注释块批量插入脚本"),
]

SH = ("PAPER_论文/code/build.sh", "2026-11-20", "编译流程与两段式编译命令")


def patch(path: str, date: str, scope: str) -> str:
    full = os.path.join(PROJ, path)
    s = io.open(full, encoding="utf-8").read()
    if MARK in s:
        return "skip"
    block = TPL.format(mark=MARK, date=date, scope=scope)
    # 保留编码声明在第一行
    lines = s.split("\n")
    if lines and lines[0].startswith("#!") or (lines and "coding" in lines[0] and lines[0].startswith("#")):
        s = lines[0] + "\n" + block + "\n".join(lines[1:])
    else:
        s = block + s
    io.open(full, "w", encoding="utf-8", newline="\n").write(s)
    return "ok"


if __name__ == "__main__":
    for p, d, sc in TARGETS:
        print(f"{patch(p, d, sc):5s} {p}")
    p, d, sc = SH
    print(f"{patch(p, d, sc):5s} {p}")
