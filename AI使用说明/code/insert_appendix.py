# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: 2026-10-17
#   使用日期    : 2026-11-23
#   辅助范围    : 论文附录插入脚本
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""把附录 C（AI 工具使用情况报告）插入两篇论文，并补参考文献条目。

幂等：重复运行不会重复插入。
  cn/main.tex : 附录插入 \\end{document} 前；文献 [13] 插在 \\bibliography{references} 之后
  en/main.tex : 附录插入 \\end{document} 前；文献 [13] 插入 thebibliography 环境内
"""
from __future__ import annotations

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # AI使用说明/
PROJ = os.path.dirname(ROOT)                 # 项目根

CN_TEX = os.path.join(PROJ, "PAPER_论文", "cn", "main.tex")
EN_TEX = os.path.join(PROJ, "PAPER_论文", "en", "main.tex")
CN_APP = os.path.join(ROOT, "appendix_cn.tex")
EN_APP = os.path.join(ROOT, "appendix_en.tex")

CN_MARK = "附录 C\\quad AI 工具使用情况报告"
EN_MARK = "\\section{AI Tool Usage Report}"

# 用唯一哨兵注释判重（附录 C.5 里也有同样一行条目文字，不能拿条目文字判重）
CN_SENTINEL = "% [AI-REF-13] AI 工具条目，竞赛规定第 5 条，勿删"
EN_SENTINEL = "% [AI-REF-13] AI tool entry, Art. 5 of the competition rules"
CN_REF = (
    CN_SENTINEL + "\n"
    "\\noindent [13] WorkBuddy，DeepSeek-V4.1-Flash，深度求索（DeepSeek），"
    "2026-11-20 至 2026-11-23.\n"
)
EN_REF = (
    EN_SENTINEL + "\n"
    "\\bibitem{workbuddy} WorkBuddy, DeepSeek-V4.1-Flash, DeepSeek, "
    "2026-11-20 to 2026-11-23.\n"
)


def read(p):
    return io.open(p, encoding="utf-8").read()


def write(p, s):
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)


START = {
    "cn": "% ---------------------------------------------------------------- 附录 C",
    "en": "% ---------------------------------------------------------------- AI report",
}


def insert_before_end(s, block, mark, key):
    r"""已存在则整块替换（便于改版），否则插入 \end{document} 之前。"""
    end = s.rindex("\\end{document}")
    start_tok = START[key]
    if start_tok in s:
        i = s.index(start_tok)
        return s[:i] + block.rstrip() + "\n\n" + s[end:], True
    return s[:end] + block.rstrip() + "\n\n" + s[end:], True


def ensure_pkg(s, pkg="longtable"):
    """附录表用 longtable 跨页续排，需在前导区加载。"""
    if "\\usepackage{%s}" % pkg in s or "{%s}" % pkg in s.split("\\begin{document}")[0]:
        return s
    anchor = "\\usepackage{array}"
    if anchor not in s:
        raise SystemExit("找不到 \\usepackage{array}，无法插入 " + pkg)
    return s.replace(anchor, anchor + "\n\\usepackage{%s}" % pkg, 1)


def patch_cn():
    s = read(CN_TEX)
    s = ensure_pkg(s)
    s, added = insert_before_end(s, read(CN_APP), CN_MARK, "cn")
    # 参考文献 [13]：紧跟 \bibliography{references} 之后
    if CN_SENTINEL not in s:
        m = re.search(r"\\bibliography\{references\}\s*\n", s)
        if not m:
            raise SystemExit("找不到 \\bibliography{references}")
        s = s[:m.end()] + CN_REF + s[m.end():]
        added = True
    write(CN_TEX, s)
    return added


def patch_en():
    s = read(EN_TEX)
    s = ensure_pkg(s)
    s, added = insert_before_end(s, read(EN_APP), EN_MARK, "en")
    if EN_SENTINEL not in s:
        i = s.index("\\end{thebibliography}")
        s = s[:i] + EN_REF + s[i:]
        added = True
    write(EN_TEX, s)
    return added


if __name__ == "__main__":
    print("cn 变更:", patch_cn())
    print("en 变更:", patch_en())
