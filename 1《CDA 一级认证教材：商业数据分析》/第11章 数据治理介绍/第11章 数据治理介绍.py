"""第 11 章 数据治理介绍

教材本章是标准与规范介绍（DCMM、GB/T 34960.5、银行业数据治理指引等），没有可执行文件。
本脚本把「数据治理」落成可运行的检查：
1. 按 DCMM 的 8 个能力域做成熟度评分与雷达图；
2. 对课程真实销售数据、会员消费流水做六维度数据质量检查（完整性、唯一性、有效性、
   一致性、准确性、及时性），输出问题清单；
3. 画出从标准制定到整改问责的治理闭环。

这样，抽象的治理框架就变成了可度量、可复盘的代码。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)
CH2 = BASE.parent / "第2章 数据分析方法"
SALES_FILE = CH2 / "1、分析的基础范式" / "data-有答案" / "1-1波士顿矩阵-参考.xlsx"
RFM_FILE = CH2 / "1、分析的基础范式" / "data-有答案" / "1-2RFM模型-参考.xlsx"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
GREY = "#A6A6A6"
TEXT = "#404040"
GRID = "#E3E3E3"
RED = "#C0504D"


def set_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图表] {path.name}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    return df


def rounded_box(ax, x, y, w, h, text, fc, fs=9.5):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.028",
                         facecolor=fc, edgecolor=fc, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            zorder=4, linespacing=1.5)


# ============================================================================
# 一、DCMM 成熟度评估
# ============================================================================
def dcmm_assessment():
    print("\n" + "=" * 78)
    print("一、DCMM 数据管理能力成熟度：8 个能力域")
    print("=" * 78)
    score = pd.Series({
        "数据战略": 3.4, "数据治理": 2.8, "数据架构": 2.5, "数据应用": 4.0,
        "数据安全": 3.2, "数据质量": 2.4, "数据标准": 2.6, "数据生命周期": 2.8,
    })
    level_desc = {1: "初始级", 2: "受管理级", 3: "稳健级", 4: "量化管理级", 5: "优化级"}
    print("  评估结果（1-5 分）：")
    for name, v in score.items():
        print(f"  {name:<8} {v:.1f} 分 · {level_desc[int(round(v))]}")
    print(f"\n  平均成熟度 {score.mean():.1f} 分，短板：{score.idxmin()}（{score.min():.1f}）；"
          f"优势：{score.idxmax()}（{score.max():.1f}）")
    print("  治理建议：优先补「数据质量」「数据架构」「数据标准」三块底座，"
          "否则数据应用的高分无法持续。")

    angles = np.linspace(0, 2 * np.pi, len(score), endpoint=False).tolist()
    values = score.tolist()
    angles += angles[:1]
    values += values[:1]

    fig = plt.figure(figsize=(15.2, 7.0))
    ax = fig.add_subplot(1, 2, 1, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, color=DEEP_BLUE, linewidth=2.2, marker="o", markersize=6)
    ax.fill(angles, values, color=LIGHT_BLUE, alpha=0.45)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(score.index, fontsize=10, color=TEXT)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color=GREY, fontsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color("#D9D9D9")
    ax.set_title("DCMM 八域成熟度雷达", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=26)

    ax = fig.add_subplot(1, 2, 2)
    order = score.sort_values()
    bars = ax.barh(order.index, order.values, height=0.58,
                   color=[LIGHT_YELLOW if v < 3 else LIGHT_BLUE for v in order.values],
                   edgecolor="white")
    for b, v in zip(bars, order.values):
        ax.text(v + 0.06, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
                fontsize=9, color=TEXT)
    ax.axvline(3, color=RED, linestyle="--", linewidth=1.3)
    ax.text(3.03, len(order) - 0.35, "稳健级门槛 3 分", color=RED, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 5.2)
    ax.set_title("能力域得分与短板", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)
    fig.suptitle("数据治理先量体温：能力域评分不是为了排名，而是为了定优先级", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.02)
    save(fig, "01_DCMM能力域评估.png")
    return score


# ============================================================================
# 二、六维度数据质量检查
# ============================================================================
def quality_assessment():
    print("\n" + "=" * 78)
    print("二、六维度数据质量检查")
    print("=" * 78)
    sales = clean_columns(pd.read_excel(SALES_FILE, sheet_name="销售表"))
    sales["销售日期"] = pd.to_datetime(sales["销售日期"], errors="coerce")
    for col in ("单价", "销量", "订单金额"):
        sales[col] = pd.to_numeric(sales[col], errors="coerce")

    key = ["订单ID", "销售日期", "品牌", "单价", "销量", "订单金额"]
    miss_rate = sales[key].isna().sum().sum() / (len(sales) * len(key))
    dup_rate = sales["订单ID"].duplicated().sum() / len(sales)
    invalid = ((sales["单价"] <= 0) | (sales["销量"] <= 0) | (sales["订单金额"] <= 0)).sum() / len(sales)
    mismatch = (np.abs(sales["订单金额"] - sales["单价"] * sales["销量"]) > 0.01).sum() / len(sales)
    date_min, date_max = sales["销售日期"].min(), sales["销售日期"].max()
    cut_off = pd.Timestamp("2025-12-31")
    stale_days = (cut_off - date_max).days
    late_score = max(0.0, 100 * (1 - stale_days / 730))

    rfm = clean_columns(pd.read_excel(RFM_FILE, sheet_name="原始数据"))
    rfm = rfm.loc[:, ~rfm.columns.str.startswith("Unnamed")]
    rfm["销售日期"] = pd.to_datetime(rfm["销售日期"], errors="coerce")
    rfm["会员创建日期"] = pd.to_datetime(rfm["会员创建日期"], errors="coerce")
    conflict = ((rfm["会员创建日期"] > rfm["销售日期"]) & rfm["会员创建日期"].notna()).sum() / len(rfm)
    rfm_missing = rfm[["用户编号", "销售日期", "销售金额"]].isna().sum().sum() / (len(rfm) * 3)

    checks = pd.DataFrame([
        {"维度": "完整性", "检查项": "关键字段缺失率", "结果": f"{miss_rate:.2%}", "得分": 100 * (1 - miss_rate)},
        {"维度": "唯一性", "检查项": "订单ID 重复率", "结果": f"{dup_rate:.2%}", "得分": 100 * (1 - dup_rate)},
        {"维度": "有效性", "检查项": "金额/数量非正占比", "结果": f"{invalid:.2%}", "得分": 100 * (1 - invalid)},
        {"维度": "准确性", "检查项": "订单金额 ≠ 单价 × 销量 占比", "结果": f"{mismatch:.2%}", "得分": 100 * (1 - mismatch)},
        {"维度": "一致性", "检查项": "会员创建日期晚于销售日期占比", "结果": f"{conflict:.2%}", "得分": 100 * (1 - conflict)},
        {"维度": "及时性", "检查项": f"最新数据距评估日 {stale_days} 天", "结果": f"{date_max:%Y-%m-%d}", "得分": late_score},
    ])
    checks["得分"] = checks["得分"].round(1)
    print(checks.to_string(index=False))
    print(f"\n  销售表 {len(sales):,} 行，数据区间 {date_min:%Y-%m-%d} ~ {date_max:%Y-%m-%d}")
    print(f"  会员流水 {len(rfm):,} 行，关键字段缺失率 {rfm_missing:.2%}")
    print(f"  综合质量得分 {checks['得分'].mean():.1f} 分，最低维度："
          f"{checks.loc[checks['得分'].idxmin(), '维度']}（{checks['得分'].min():.1f}）")

    issues = pd.DataFrame([
        {"问题": "订单金额与单价×销量不一致", "影响": "收入与毛利口径失真",
         "严重度": "高", "建议": "在 ETL 中增加金额校验规则并回溯源系统"},
        {"问题": "数据时效落后评估日", "影响": "看板无法反映当前经营",
         "严重度": "中", "建议": "建立日更任务与数据到达监控"},
        {"问题": "会员创建日期晚于销售日期", "影响": "用户生命周期标签错误",
         "严重度": "高", "建议": "主数据与交易系统做日期先后校验"},
        {"问题": "关键字段存在缺失", "影响": "分组汇总出现空桶",
         "严重度": "中", "建议": "源端必填校验 + 缺失默认值字典"},
    ])
    print("\n  数据质量问题清单：")
    print(issues.to_string(index=False))
    issues.to_csv(BASE / "数据质量问题清单.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    bars = ax.bar(checks["维度"], checks["得分"], width=0.56,
                  color=[RED if v < 60 else (LIGHT_YELLOW if v < 85 else LIGHT_BLUE)
                         for v in checks["得分"]], edgecolor="white")
    for b, v in zip(bars, checks["得分"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center",
                va="bottom", fontsize=9, color=TEXT)
    ax.axhline(85, color=GREY, linestyle="--", linewidth=1.2)
    ax.text(len(checks) - 0.5, 86.5, "目标 85", ha="right", fontsize=8.5, color=GREY)
    style_axes(ax, "六维度质量得分", None, "得分")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1]
    ax.axis("off")
    tbl = ax.table(cellText=checks[["维度", "检查项", "结果", "得分"]].values,
                   colLabels=["维度", "检查项", "结果", "得分"],
                   loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E6E6E6")
        if r == 0:
            cell.set_facecolor(LIGHT_BLUE)
            cell.set_text_props(color=TEXT, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7FAFD")
    ax.set_title("质量检查明细", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)
    fig.suptitle("数据质量：先定义「好数据」的标准，再谈治理", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "02_数据质量六维度.png")
    return checks, issues


# ============================================================================
# 三、治理闭环
# ============================================================================
def governance_loop():
    fig, ax = plt.subplots(figsize=(14.6, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("1. 数据标准", "字段、编码、口径\n统一业务语言"),
        ("2. 元数据", "登记表、字段、血缘\n知道数据从哪来"),
        ("3. 质量规则", "六维度校验\n阈值与告警"),
        ("4. 监控运营", "日报、工单\n问题闭环"),
        ("5. 问责机制", "责任部门与\n整改时限"),
    ]
    xs = np.linspace(0.11, 0.89, 5)
    colors = [LIGHT_BLUE, "#B9D5EC", MID_BLUE, LIGHT_YELLOW, "#F5C46B"]
    for (title, body), x, color in zip(steps, xs, colors):
        rounded_box(ax, x, 0.55, 0.16, 0.24, f"{title}\n{body}", color, fs=9.5)
        if x < 0.85:
            ax.annotate("", xy=(x + 0.095, 0.55), xytext=(x + 0.075, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))

    ax.annotate("", xy=(0.11, 0.40), xytext=(0.89, 0.40),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(0.5, 0.255, "复盘：问题反哺标准与规则", ha="center", fontsize=9.5, color=GREY)

    ax.text(0.5, 0.88, "数据治理闭环", ha="center", fontsize=17, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.78, "标准不是文档，而是一条能被监控、被追责、被持续优化的生产线",
            ha="center", fontsize=10, color=GREY)
    save(fig, "03_数据治理闭环.png")


def main():
    set_style()
    dcmm_assessment()
    quality_assessment()
    governance_loop()
    print("\n第 11 章数据治理演示完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()