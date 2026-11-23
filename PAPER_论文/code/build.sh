#!/usr/bin/env bash
# =============================================================================
# AI 辅助说明（《数维杯人工智能工具使用规定（2026 年修订）》第 7 条）
#   工具名称    : WorkBuddy（桌面端 AI 智能体）
#   版本 / 型号 : DeepSeek-V4.1-Flash
#   开发机构    : 模型 深度求索（DeepSeek）；产品 WorkBuddy
#   版本发布日期: ______（以官方发布页为准，提交前填写）
#   使用日期    : 2026-11-20
#   辅助范围    : 编译流程与两段式编译命令
#   核验方式    : 队员逐行审阅；几何、遮挡、最短路与时间轴口径另由
#                 common/validate_model.py 独立交叉验证（不同实现路径）
#   本文件全部逻辑与结论由参赛队负责
# =============================================================================
# 一键编译两篇论文（中文走 bibtex，英文用手工参考文献）
set -e
cd "$(dirname "$0")/.."

echo "==[1/2] Chinese (CUMCM) =="
cd cn
xelatex -no-pdf -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
xelatex -no-pdf -interaction=nonstopmode main.tex >/dev/null
xelatex -no-pdf -interaction=nonstopmode main.tex >/dev/null
xdvipdfmx main.xdv >/dev/null
echo "   -> cn/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
cd ..

echo "==[2/2] English (MCM/ICM) =="
cd en
xelatex -no-pdf -interaction=nonstopmode main.tex >/dev/null
xelatex -no-pdf -interaction=nonstopmode main.tex >/dev/null
xdvipdfmx main.xdv >/dev/null
echo "   -> en/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
cd ..

echo "done."
