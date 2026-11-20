#!/usr/bin/env bash
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
