# 论文交付：中文（国赛格式）与英文（MCM/ICM 格式）

两篇论文基于同一套已完成的建模与求解结果（见 `FINAL_汇总/README.md`），
**所有数字均取自各子问题的 `output/*.json`**，未在论文写作阶段重新计算，因此与分题结果严格一致。

| 论文 | 文件 | 版式 | 篇幅 | 参考文献 |
|---|---|---|---|---|
| 中文 | `cn/main.pdf` | CUMCM 国赛格式（ctexart，A4，摘要单独成页，GB/T 7714-2015） | 25 页，9 图 8 表 | 12 条，全部 Crossref 核验 |
| 英文 | `en/main.pdf` | MCM/ICM 格式（article，Times New Roman，1 inch 边距，Summary 单独成页） | 24 页，7 图 6 表 | 12 条，全部 Crossref 核验 |

## 一、两篇论文的结构

中文（国赛标准结构）：

```
摘要（单独成页，含四要素与三个创新点）
1 问题重述        1.1 问题背景  1.2 需解决的问题（含数据表）
2 问题分析        2.1 总体思路  2.2 各子问题的关键判断
3 模型假设        6 条，逐条给出理由与"在后文何处使用"
4 符号说明
5 模型建立        5.1 统一几何与运动学基础 → 5.2~5.6 各子问题模型
6 模型求解与结果  6.1 流程与规模 → 6.2~6.6 逐题结果
7 模型检验        7.1 模型层 5 项交叉验证  7.2 灵敏度  7.3 蒙特卡洛与统计检验  7.4 下界自校验
8 模型评价与推广  优点 / 缺点（5 条，均量化）/ 三个推广方向
参考文献（12 条）  附录（支撑材料清单 + 核心代码 3 段）
```

英文（MCM 标准结构）：`Summary` → `Introduction`（background / restatement / approach）→
`Assumptions and Notation` → `Model Development`（shared layer + 5 子问题）→
`Solution and Results` → `Sensitivity and Validation` → `Model Evaluation`（strengths / weaknesses）→
`Conclusion` → `References` → `Appendix`（材料清单 + 代码）。

## 二、文献真实性与核验

12 条参考文献全部通过 **Crossref API** 拉取元数据生成，逐条记录 DOI、题名、刊名、年份、卷期页码，
核验结果存于 `refs_verified.json`。生成脚本：`code/fetch_refs.py`。

其中 3 条在首次自动检索时返回了**错误条目**（DOI 指向了另一篇文章），已人工核验并替换：

| 键 | 首次检索的错误结果 | 修正后 |
|---|---|---|
| `lozano1979` | *Designing Software for Ease of Extension and Contraction*（Parnas, 1979） | Lozano-Pérez & Wesley, *An Algorithm for Planning Collision-Free Paths among Polyhedral Obstacles*, CACM 22(10), DOI 10.1145/359156.359164 |
| `robert2004mc` | *Efficient Monte Carlo Simulation Methods in Statistical Physics* | Robert & Casella, *Monte Carlo Statistical Methods*, 2nd ed., Springer, DOI 10.1007/978-1-4757-4145-2 |
| `gavish1992` | 键名与内容不符 | 改为 `hepner1987`，Hepner & Geering, AIAA GNC, DOI 10.2514/6.1987-2382 |

被删掉的候选：`gdop`（book chapter，Crossref 无作者元数据，`author = Unknown`）、
`evidential`（学位论文，缺 venue/year）——宁缺勿滥。

## 三、图表策略（中英文分别处理）

- **中文论文**：直接使用各分题目录下的原始图（`PAPER_论文/figures/`，中文标注），9 图。
- **英文论文**：原始图的中文标注不可用于英文论文，因此用 `code/make_en_figures.py`
  **重绘了 7 张英文图**（输出到 `en/figures/`）。重绘脚本的数据来源：
  | 英文图 | 数据来源 |
  |---|---|
  | `fig1_observable.png` | `common.model.observable()` 现场重算 0.5 m 网格（0.3 s） |
  | `fig2_arc.png` | 最优圆 0.25° 扫描 + `common.model.los_blocked()` |
  | `fig3_layout.png` | `Q2-2/output/q2_2_result.json` 的测站表 |
  | `fig4_gantt.png` | `Q2-2/output/q2_2_timeline.csv` |
  | `fig5_sigma.png` | 问题三解析式（σ 合成与分量分解） |
  | `fig6_scenarios.png` | `Q4/output/q4_scenarios.csv` |
  | `fig7_q21.png` | `Q2-1/output/q2_1_result.json` 的测站对与误差圆半径 |
  只有 `fig1` 与 `fig2` 需要重算（合计 < 1 s），其余全部读已有结果，不重复优化。

## 四、编译

### 环境

- TeX Live 2026（Windows），`xelatex` + `xdvipdfmx`
- 中文字体：系统 SimSun / SimHei / KaiTi / FangSong（ctex 自动调用）
- 英文正文字体：Times New Roman（`fontspec`），中文论文正文用 ctex 默认宋体
- **注意**：本机 TeX Live 未安装 `fancyvrb`，故两篇论文的代码排版统一改用 `listings`
  （`basicstyle=\ttfamily\scriptsize`，行号 + 边框，`breaklines`）。

### 命令

```bash
# 中文：xelatex → bibtex → xelatex ×2 → xdvipdfmx
cd PAPER_论文/cn
xelatex -no-pdf main.tex && bibtex main
xelatex -no-pdf main.tex && xelatex -no-pdf main.tex
xdvipdfmx main.xdv

# 英文：xelatex ×2 → xdvipdfmx（参考文献手工 thebibliography，无需 bibtex）
cd PAPER_论文/en
xelatex -no-pdf main.tex && xelatex -no-pdf main.tex
xdvipdfmx main.xdv
```

或直接运行 `bash PAPER_论文/code/build.sh`。

### 一个环境坑（记录以免复现）

直接 `xelatex main.tex` 会在最后报
`Error 1 (driver return code) generating output; file main.pdf may not be valid.`
但 PDF 实际已完整写出（页数、字体、图均正常）。这是本机 `xelatex` 调用 `xdvipdfmx` 时
驱动返回码异常，与文档内容无关。改用**两步法**（`xelatex -no-pdf` 生成 `.xdv`，再手工
`xdvipdfmx main.xdv`）即无任何报错，退出码为 0。

另两个踩过的坑：

1. `\IfFileExists{Q1_obs_region_T1.png}` 会因文件名中的 `_` 在文本模式下触发
   `Missing $ inserted`（宏参数不是文件名语境）。插图宏改为直接 `\includegraphics`。
2. `gbt7714-numeric.bst` 的 `\bibitem` 带著者-年标签，若不加载 `natbib`，`\cite` 会输出
   "He et al.(2015)He, Behnad, and Wang" 这样的长串。加载
   `\usepackage[numbers,sort&compress]{natbib}` 后变为上标数字 `[7]`，符合国赛惯例。

## 五、文件清单

```
PAPER_论文/
├── README.md                  本文件
├── refs_verified.json         12 条文献的 DOI/题名/刊名/年份核验记录
├── figures/                   中文论文用图（12 张原始图）
├── code/
│   ├── fetch_refs.py          Crossref 拉取 + 生成 references.bib
│   ├── make_en_figures.py     英文版插图重绘（7 张）
│   └── build.sh               两篇论文一键编译
├── cn/
│   ├── main.tex  main.pdf  references.bib  gbt7714-numeric.bst
│   └── figures/               中文图（副本，供 \graphicspath 解析）
└── en/
    ├── main.tex  main.pdf
    └── figures/               英文重绘图 + 原始图副本
```

## 六、与分题结果的一致性

论文中的每一个关键数字都能回溯到具体文件：

| 论文中的数字 | 来源 |
|---|---|
| Q1 测站 (88.8247, 76.3233)、r=7.5821、Q=0.131890 | `Q1_*/output/method_B_analytic.json` |
| Q1 面积 5607.8 m²、遮挡排除 63 435 点 | `Q1_*/output/method_A_grid.json` |
| Q2(1) V3–V5、σ_p=0.14362、95% 圆 0.2486 | `Q2-1_*/output/q2_1_result.json` |
| Q2(2) t_last=10.655、各站坐标、σ_p 分项 | `Q2-2_*/output/q2_2_result.json` |
| Q3 s_max=66.968、a+80b≤1.6394、MC 偏差 0.0069° | `Q3_*/output/q3_result.json` |
| Q4 88/5/0、η 0.8566→1.0000、t_last 23.99→22.37 | `Q4_*/output/q4_result.json` |
| 5 项交叉验证全通过 | `common/output_model_validation.json` |
