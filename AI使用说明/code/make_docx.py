# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: ______（以官方发布页为准，提交前填写）
#   使用日期    : 2026-11-23
#   辅助范围    : Markdown→Word 版式转换代码
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
"""AI工具使用情况报告.md -> .docx

只做版式转换，不增删任何文字。Markdown 支持的子集：
  标题(#/##/###)、段落、无序列表(- / - [ ])、引用(>)、表格(|)、
  代码块(```)、分隔线(---)、行内粗体(**x**)与行内代码(`x`)。
字体：正文宋体小四 1.5 倍行距，标题黑体，表格五号，代码 Consolas 小五。
"""
from __future__ import annotations

import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "AI工具使用情况报告.md")
DST = os.path.join(ROOT, "AI工具使用情况报告.docx")

SONG, HEI, KAI, MONO = "宋体", "黑体", "楷体", "Consolas"


def set_run_font(run, ascii_font=SONG, ea_font=SONG, size=12, bold=None, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), ea_font)
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)


def shade(cell, hexcolor: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tc_pr.append(shd)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(par, text: str, size=12, base_font=SONG, base_ea=SONG):
    """把 **粗体** 与 `代码` 拆成不同 run。"""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            r = par.add_run(piece[2:-2])
            set_run_font(r, base_font, base_ea, size, bold=True)
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            set_run_font(r, MONO, MONO, size - 1)
        else:
            r = par.add_run(piece)
            set_run_font(r, base_font, base_ea, size)


def body_par(doc, spacing=1.5, first_indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    if first_indent:
        pf.first_line_indent = Pt(first_indent)
    return p


def build(dst=None):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.4)

    normal = doc.styles["Normal"]
    normal.font.name = SONG
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    lines = open(SRC, encoding="utf-8").read().split("\n")
    i, n = 0, len(lines)
    first_h1 = True

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        s = line.strip()

        if not s or s == "---":
            i += 1
            continue

        # ---------------- code block
        if s.startswith("```"):
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.0
            for k, cl in enumerate(block):
                r = p.add_run(cl)
                set_run_font(r, MONO, SONG, 9)
                if k < len(block) - 1:
                    r.add_break()
            continue

        # ---------------- headings
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and first_h1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(10)
                r = p.add_run(text)
                set_run_font(r, HEI, HEI, 20, bold=True)
                first_h1 = False
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12 if level == 2 else 8)
                p.paragraph_format.space_after = Pt(5)
                p.paragraph_format.keep_with_next = True
                r = p.add_run(text)
                set_run_font(r, HEI, HEI, {2: 14, 3: 12}.get(level, 11),
                             bold=True)
            i += 1
            continue

        # ---------------- blockquote
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = body_par(doc, spacing=1.3)
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.right_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, " ".join(buf), size=10.5, base_font=KAI, base_ea=KAI)
            continue

        # ---------------- table
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            tbl = doc.add_table(rows=len(rows), cols=ncol)
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.autofit = True
            for ri, row in enumerate(rows):
                for ci in range(ncol):
                    cell = tbl.cell(ri, ci)
                    cell.text = ""
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.15
                    txt = row[ci] if ci < len(row) else ""
                    add_inline(p, txt, size=9)
                    if ri == 0:
                        for r in p.runs:
                            r.font.bold = True
                        shade(cell, "F2F2F2")
            # 表格后留白
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(4)
            sp.paragraph_format.line_spacing = 1.0
            continue

        # ---------------- list
        m = re.match(r"^-\s+(\[[ xX]\]\s+)?(.*)$", s)
        if m:
            p = body_par(doc, spacing=1.4)
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.4)
            box = m.group(1)
            prefix = ("☐ " if box and "[ ]" in box.replace("x", " ") else "• ")
            r = p.add_run(prefix)
            set_run_font(r, SONG, SONG, 12)
            add_inline(p, m.group(2), size=12)
            i += 1
            continue

        # ---------------- paragraph
        buf = [s]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt == "---" or nxt.startswith(("#", ">", "|", "```"))
                    or re.match(r"^-\s+", nxt)):
                break
            buf.append(nxt)
            i += 1
        p = body_par(doc)
        p.paragraph_format.first_line_indent = Cm(0.74)
        add_inline(p, " ".join(buf), size=12)

    # 页脚页码
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)

    dst = dst or DST
    doc.save(dst)
    return dst


if __name__ == "__main__":
    path = build(sys.argv[1] if len(sys.argv) > 1 else None)
    print("写出:", path, os.path.getsize(path), "bytes")
