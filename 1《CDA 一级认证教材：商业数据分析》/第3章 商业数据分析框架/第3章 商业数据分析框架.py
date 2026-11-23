"""第 3 章 商业数据分析框架：三个 Excel 案例的 Python 复现

教材用 Excel 完成三个完整分析案例，本脚本按同样的业务口径重建：
- 案例 1 收入趋势分析：年度 / 月度趋势、同比增速、省份对比，定位增长与下滑来源；
- 案例 2 财务费用分析：总费用、三大期间费用、管理费用明细的环比拆解；
- 案例 3 电商平台精准营销：性别、年龄、品类、支付方式逐层下钻，形成目标客群画像。

图表沿用淡蓝 / 淡黄配色与圆角卡片，保证与教材 Excel 图表的信息结构一致。
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

CASE1 = BASE / "案例1：收入趋势分析" / "案例1：收入趋势分析原始数据.xlsx"
CASE2 = BASE / "案例2：财务费用分析" / "案例2：财务费用分析-原始数据.xlsx"
CASE3 = BASE / "案例3：电商平台的精准营销" / "案例3：电商平台的精准营销-原始数据.xlsx"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
GREY = "#A6A6A6"
LIGHT_GREY = "#D9D9D9"
TEXT = "#404040"
GRID = "#E3E3E3"
RED = "#C0504D"


def set_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#BFBFBF"


def style_axes(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
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


def pct(v):
    return f"{v:+.1%}" if pd.notna(v) else "—"


# ============================================================================
# 案例 1：收入趋势分析
# ============================================================================
def case1_revenue():
    print("\n" + "=" * 78)
    print("案例 1：收入趋势分析")
    print("=" * 78)
    df = clean_columns(pd.read_excel(CASE1, sheet_name="原始数据"))
    df = df.dropna(subset=["订单年份", "订单月份", "销售额"])
    df["订单年份"] = df["订单年份"].astype(int)
    df["订单月份"] = df["订单月份"].astype(int)
    df["订单金额"] = pd.to_numeric(df["销售额"], errors="coerce")

    year = df.groupby("订单年份")["订单金额"].sum()
    year_growth = year.pct_change()
    print(f"  数据 {len(df):,} 行，{df['销售省份'].nunique()} 个省份，"
          f"{df['订单年份'].min()}-{df['订单年份'].max()}")
    print("  年度销售额与增速：")
    print(pd.DataFrame({"销售额": year.map(lambda v: f"{v:,.0f}"),
                        "同比": year_growth.map(pct)}).to_string())

    month = df[df["订单年份"] == 2020].groupby("订单月份")["订单金额"].sum().sort_index()
    month_2019 = df[df["订单年份"] == 2019].groupby("订单月份")["订单金额"].sum()
    month_yoy = (month / month_2019.reindex(month.index) - 1).dropna()
    print(f"\n  2020 年月度销售额（{month.index.min()}-{month.index.max()} 月），"
          f"全年 {month.sum():,.0f}")

    nov = df[(df["订单年份"] == 2020) & (df["订单月份"] == 11)]
    nov_2019 = df[(df["订单年份"] == 2019) & (df["订单月份"] == 11)]
    prov = pd.DataFrame({
        "2020年11月": nov.groupby("销售省份")["订单金额"].sum(),
        "2019年11月": nov_2019.groupby("销售省份")["订单金额"].sum(),
    }).dropna()
    prov["同比"] = prov["2020年11月"] / prov["2019年11月"] - 1
    prov = prov.sort_values("同比")
    print("\n  2020 年 11 月各省份同比（最低 5 个 / 最高 5 个）：")
    print(prov.assign(**{"2020年11月": prov["2020年11月"].map(lambda v: f"{v:,.0f}"),
                          "2019年11月": prov["2019年11月"].map(lambda v: f"{v:,.0f}"),
                          "同比": prov["同比"].map(pct)}).iloc[list(range(5)) + list(range(-5, 0))]
          .to_string())

    sichuan = df[(df["销售省份"] == "四川") & (df["订单年份"].isin([2019, 2020]))]
    sc = sichuan.pivot_table(index="订单月份", columns="订单年份", values="订单金额", aggfunc="sum")
    sc_yoy = sc[2020] / sc[2019] - 1

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    ax = axes[0, 0]
    ax.plot(year.index, year.values / 10000, marker="o", linewidth=2.6, color=DEEP_BLUE,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.fill_between(year.index, year.values / 10000, color=LIGHT_BLUE, alpha=0.4)
    for x, y in zip(year.index, year.values / 10000):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=TEXT)
    style_axes(ax, "年度销售趋势", "订单年份", "销售额（万元）")
    ax.set_xticks(year.index)
    ax.set_ylim(year.min() / 10000 * 0.75, year.max() / 10000 * 1.12)

    ax = axes[0, 1]
    bars = ax.bar(month.index, month.values / 10000, width=0.62, color=LIGHT_BLUE,
                  edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, month.values / 10000):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "2020 年月度销售额", "月份", "销售额（万元）")
    ax.set_xticks(month.index)
    ax.set_ylim(0, month.max() / 10000 * 1.18)

    ax = axes[1, 0]
    colors = [RED if v < 0 else MID_BLUE for v in month_yoy.values]
    bars = ax.bar(month_yoy.index, month_yoy.values, width=0.62, color=colors, edgecolor=colors)
    ax.axhline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, month_yoy.values):
        va = "bottom" if v >= 0 else "top"
        offset = 0.012 if v >= 0 else -0.012
        ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:+.0%}", ha="center", va=va,
                fontsize=8.5, color=TEXT)
    style_axes(ax, "2020 年月度同比增速", "月份", "同比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_xticks(month_yoy.index)
    ax.set_ylim(month_yoy.min() * 1.35, max(month_yoy.max() * 1.25, 0.05))

    ax = axes[1, 1]
    show = prov.iloc[list(range(5)) + list(range(-5, 0))]
    colors = [RED if v < 0 else LIGHT_BLUE for v in show["同比"].values]
    bars = ax.barh(show.index, show["同比"].values, height=0.58, color=colors, edgecolor=colors)
    ax.axvline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, show["同比"].values):
        ax.text(v + (0.01 if v >= 0 else -0.01), b.get_y() + b.get_height() / 2,
                f"{v:+.0%}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "2020 年 11 月各省份同比（两端）", "同比", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_xlim(show["同比"].min() * 1.45, show["同比"].max() * 1.45)

    fig.suptitle("案例 1：收入趋势分析——增长在哪些维度、哪个月份失速", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "01_案例1_收入趋势分析.png")

    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    colors = [RED if v < 0 else LIGHT_YELLOW for v in sc_yoy.values]
    bars = ax.bar(sc_yoy.index, sc_yoy.values, width=0.6, color=colors, edgecolor=colors)
    ax.axhline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, sc_yoy.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.0%}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=TEXT)
    style_axes(ax, "四川省 2020 年各月同比增速", "月份", "同比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_xticks(sc_yoy.index)
    save(fig, "02_案例1_四川月度同比.png")

    print(f"\n  业务解读：2020 年销售额 {year.loc[2020]:,.0f}，同比 {year_growth.loc[2020]:+.1%}；"
          f"月度同比最低点出现在 {month_yoy.idxmin()} 月（{month_yoy.min():+.1%}），"
          f"11 月恢复到 {month_yoy.get(11, float('nan')):+.1%}。")
    print(f"  11 月同比下滑最多的省份：{prov.index[0]}（{prov['同比'].iloc[0]:+.1%}）；"
          f"增长最快的省份：{prov.index[-1]}（{prov['同比'].iloc[-1]:+.1%}）。")
    return df


# ============================================================================
# 案例 2：财务费用分析
# ============================================================================
def case2_cost():
    print("\n" + "=" * 78)
    print("案例 2：财务费用分析")
    print("=" * 78)
    total = clean_columns(pd.read_excel(CASE2, sheet_name="总费用数据"))
    total["月份"] = pd.to_numeric(total["月份"], errors="coerce")
    total["金额"] = pd.to_numeric(total["金额"], errors="coerce")
    total = total.dropna(subset=["月份"])

    period = clean_columns(pd.read_excel(CASE2, sheet_name="财务三大期间费用数据"))
    period = period.iloc[:, :4]
    period["月份"] = pd.to_numeric(period["月份"], errors="coerce")
    period["金额"] = pd.to_numeric(period["金额"], errors="coerce")
    period = period.dropna(subset=["月份", "期间费用项"])

    manage = clean_columns(pd.read_excel(CASE2, sheet_name="管理费用数据"))
    manage.columns = ["年份", "月份", "管理费用项", "金额"]
    manage["月份"] = pd.to_numeric(manage["月份"], errors="coerce")
    manage["金额"] = pd.to_numeric(manage["金额"], errors="coerce")
    manage = manage.dropna(subset=["月份", "管理费用项"])

    total_map = total.set_index("月份")["金额"]
    total_diff = total_map.loc[7] - total_map.loc[6]
    print(f"  总费用：6 月 {total_map.loc[6]:,.0f} -> 7 月 {total_map.loc[7]:,.0f}，"
          f"环比增加 {total_diff:,.0f}（{total_diff / total_map.loc[6]:+.1%}）")

    p = period.pivot_table(index="期间费用项", columns="月份", values="金额", aggfunc="sum")
    p.columns = ["6月", "7月"]
    p["环比差额"] = p["7月"] - p["6月"]
    p["环比增速"] = p["7月"] / p["6月"] - 1
    print("\n  三大期间费用：")
    print(p.assign(**{"6月": p["6月"].map(lambda v: f"{v:,.0f}"),
                      "7月": p["7月"].map(lambda v: f"{v:,.0f}"),
                      "环比差额": p["环比差额"].map(lambda v: f"{v:+,.0f}"),
                      "环比增速": p["环比增速"].map(pct)}).to_string())
    print(f"  校验：三大费用合计环比增加 {p['环比差额'].sum():,.0f}，与总费用差额 {total_diff:,.0f} 一致")

    m = manage.pivot_table(index="管理费用项", columns="月份", values="金额", aggfunc="sum")
    m.columns = ["6月", "7月"]
    m["环比差额"] = m["7月"] - m["6月"]
    m["环比增速"] = m["7月"] / m["6月"] - 1
    m = m.sort_values("环比差额")
    print("\n  管理费用明细（按环比差额升序）：")
    print(m.assign(**{"6月": m["6月"].map(lambda v: f"{v:,.0f}"),
                      "7月": m["7月"].map(lambda v: f"{v:,.0f}"),
                      "环比差额": m["环比差额"].map(lambda v: f"{v:+,.0f}"),
                      "环比增速": m["环比增速"].map(pct)}).to_string())
    top = m["环比差额"].abs().sort_values(ascending=False).index[:3]
    print(f"  主要拉动项：{'、'.join(top)}，合计 {m.loc[top, '环比差额'].sum():+,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    ax = axes[0, 0]
    vals = [total_map.loc[6] / 10000, total_map.loc[7] / 10000]
    bars = ax.bar(["6月", "7月"], vals, width=0.5, color=[LIGHT_BLUE, LIGHT_YELLOW],
                  edgecolor=[LIGHT_BLUE, LIGHT_YELLOW])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}", ha="center", va="bottom",
                fontsize=10, color=TEXT)
    ax.text(0.5, max(vals) * 1.08, f"环比 {total_diff / 10000:+,.1f} 万元（{total_diff / total_map.loc[6]:+.1%}）",
            ha="center", fontsize=10, color=RED,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#FBEAEA", edgecolor=RED))
    style_axes(ax, "总财务费用：6 月 vs 7 月", None, "金额（万元）")
    ax.set_ylim(0, max(vals) * 1.24)

    ax = axes[0, 1]
    x = np.arange(len(p))
    ax.bar(x - 0.19, p["6月"] / 10000, width=0.36, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, label="6月")
    ax.bar(x + 0.19, p["7月"] / 10000, width=0.36, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW, label="7月")
    for xi, row in enumerate(p.itertuples()):
        ax.text(xi - 0.19, row._1 / 10000, f"{row._1 / 10000:.0f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
        ax.text(xi + 0.19, row._2 / 10000, f"{row._2 / 10000:.0f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(p.index)
    style_axes(ax, "三大期间费用对比", None, "金额（万元）")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(0, p[["6月", "7月"]].max().max() / 10000 * 1.18)

    ax = axes[1, 0]
    colors = [RED if v < 0 else MID_BLUE for v in p["环比差额"].values]
    bars = ax.barh(p.index, p["环比差额"].values / 10000, height=0.56, color=colors, edgecolor=colors)
    ax.axvline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, p["环比差额"].values / 10000):
        ax.text(v + (0.15 if v >= 0 else -0.15), b.get_y() + b.get_height() / 2,
                f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
    style_axes(ax, "7 月较 6 月的环比差额", "差额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(p["环比差额"].min() / 10000 * 1.6, p["环比差额"].max() / 10000 * 1.6)

    ax = axes[1, 1]
    mm = m.sort_values("环比差额")
    colors = [RED if v < 0 else LIGHT_YELLOW for v in mm["环比差额"].values]
    bars = ax.barh(mm.index, mm["环比差额"].values / 10000, height=0.6,
                   color=colors, edgecolor="white")
    ax.axvline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, mm["环比差额"].values / 10000):
        ax.text(v + (0.06 if v >= 0 else -0.06), b.get_y() + b.get_height() / 2,
                f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=TEXT)
    style_axes(ax, "管理费用明细环比差额", "差额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(mm["环比差额"].min() / 10000 * 1.85, mm["环比差额"].max() / 10000 * 1.9)

    fig.suptitle("案例 2：财务费用分析——总费用变化由谁拉动", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "03_案例2_财务费用分析.png")
    return p, m


# ============================================================================
# 案例 3：电商平台精准营销
# ============================================================================
def case3_marketing():
    print("\n" + "=" * 78)
    print("案例 3：电商平台精准营销")
    print("=" * 78)
    df = clean_columns(pd.read_excel(CASE3))
    df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(1)
    df = df.dropna(subset=["年龄", "性别"])
    print(f"  订单明细 {len(df):,} 行，客户 {df['客户ID'].nunique():,} 人，"
          f"女性占比 {(df['性别'] == 'Female').mean():.1%}")

    gender = df.drop_duplicates("客户ID")["性别"].value_counts()
    female = df[df["性别"] == "Female"]

    bins = [17, 20, 30, 40, 50, 60, 100]
    labels = ["18-19岁", "20-29岁", "30-39岁", "40-49岁", "50-59岁", "60岁以上"]
    female_age = pd.cut(female["年龄"], bins=bins, labels=labels).value_counts().reindex(labels)
    print("\n  女性客户年龄段分布：")
    print((female_age / female_age.sum()).map(lambda v: f"{v:.1%}").to_string())

    target = female[female["年龄"] >= 50]
    cat = target["品类"].value_counts()
    print(f"\n  50 岁以上女性客户 {len(target):,} 单，品类分布：")
    print(cat.map(lambda v: f"{v:,}（{v / len(target):.1%}）").to_string())

    focus = target[target["品类"].isin(["Clothing", "Cosmetics"])]
    pay = focus["支付方式"].value_counts()
    print(f"\n  50 岁以上女性购买服装/护肤品的支付方式（{len(focus):,} 单）：")
    print(pay.map(lambda v: f"{v:,}（{v / len(focus):.1%}）").to_string())

    print("\n  画像结论：核心人群 = 女性 + 50 岁以上 + 服装/护肤 + 偏好支付方式；"
          "营销动作应围绕该人群的品类偏好与支付习惯设计。")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    ax = axes[0, 0]
    wedges, texts, autotexts = ax.pie(
        gender.values, labels=["女性", "男性"], colors=[LIGHT_BLUE, LIGHT_YELLOW],
        autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT, fontsize=10),
    )
    ax.text(0, 0, f"客户\n{gender.sum():,}", ha="center", va="center", fontsize=11, color=TEXT)
    ax.set_title("购买客户的性别分布", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)

    ax = axes[0, 1]
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE, "#F5C46B", "#C9DAEA", LIGHT_GREY]
    wedges, texts, autotexts = ax.pie(
        female_age.values, labels=female_age.index, colors=colors,
        autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT, fontsize=9.5),
    )
    for t in autotexts:
        t.set_fontsize(8.5)
    ax.text(0, 0, "女性\n年龄段", ha="center", va="center", fontsize=10.5, color=TEXT)
    ax.set_title("女性客户的年龄段分布", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)

    ax = axes[1, 0]
    cc = cat.head(7).sort_values()
    bars = ax.barh(cc.index, cc.values, height=0.6,
                   color=[LIGHT_YELLOW if i % 2 == 0 else LIGHT_BLUE for i in range(len(cc))],
                   edgecolor="white")
    for b, v in zip(bars, cc.values):
        ax.text(b.get_width() + cat.max() * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,}（{v / len(target):.1%}）", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax, "50 岁以上女性客户的品类分布", "订单数", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, cat.max() * 1.38)

    ax = axes[1, 1]
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE]
    bars = ax.bar(pay.index, pay.values, width=0.56, color=colors[:len(pay)],
                  edgecolor="white")
    for b, v in zip(bars, pay.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v / len(focus):.1%}",
                ha="center", va="bottom", fontsize=9, color=TEXT)
    style_axes(ax, "服装/护肤人群的支付方式", None, "订单数")
    ax.set_ylim(0, pay.max() * 1.22)

    fig.suptitle("案例 3：精准营销——从全量客户逐层收窄到可运营人群", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "04_案例3_精准营销客群画像.png")

    steps = [("全量客户", len(df)), ("女性客户", len(female)), ("50 岁以上女性", len(target)),
             ("服装/护肤人群", len(focus))]
    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    max_v = steps[0][1]
    for i, (name, v) in enumerate(steps):
        w = v / max_v
        y = len(steps) - i - 1
        box = FancyBboxPatch((0.5 - w / 2, y + 0.16), w, 0.62,
                             boxstyle="round,pad=0.005,rounding_size=0.08",
                             facecolor=[LIGHT_BLUE, MID_BLUE, LIGHT_YELLOW, "#F5C46B"][i],
                             edgecolor="white", linewidth=1.2)
        ax.add_patch(box)
        ax.text(0.5, y + 0.47, f"{name}  {v:,} 单（{v / max_v:.1%}）", ha="center", va="center",
                fontsize=11, color=TEXT)
        if i > 0:
            ax.annotate("", xy=(0.5, y + 0.86), xytext=(0.5, y + 1.02),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")
    ax.set_title("精准营销客群收窄路径", loc="left", fontsize=15, fontweight="bold",
                 color=TEXT, pad=16)
    save(fig, "05_案例3_客群收窄路径.png")


def main():
    set_style()
    case1_revenue()
    case2_cost()
    case3_marketing()
    print("\n第 3 章全部案例完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()