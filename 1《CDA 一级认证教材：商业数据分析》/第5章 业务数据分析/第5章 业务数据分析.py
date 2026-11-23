"""第 5 章 业务数据分析：银行 Power BI 看板与 WPS 归因分析

教材包含两个银行案例：
1. Power BI 业务看板：440 位理财客户、2500 万月度销售目标，需要回答「目标完成得怎样、
   客户是谁、买什么、偏好什么风险与期限」；
2. WPS 归因分析：贷款申请量 5 月环比下降 14.5%，需要拆到渠道，找出下降由谁贡献。

本脚本用 pandas 计算全部度量值，用 matplotlib 复刻看板与归因图，并把分析结论打印出来。
图表采用淡蓝 / 淡黄配色、圆角 KPI 卡片，元素之间保留统一间距。
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)

BANK = BASE / "第5章  PowerBI业务看板案例-银行案例" / "银行案例数据 - 学员用.xlsx"
ATTR = BASE / "第5章 WPS归因分析案例-银行案例" / "归因分析数据.xlsx"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
GREY = "#A6A6A6"
TEXT = "#404040"
GRID = "#E3E3E3"
RED = "#C0504D"
TARGET = 2500


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


def kpi_card(ax, label, value, sub, color):
    ax.set_axis_off()
    box = FancyBboxPatch((0.03, 0.14), 0.94, 0.70,
                         boxstyle="round,pad=0.01,rounding_size=0.06",
                         facecolor=color, edgecolor=color, linewidth=1.2,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(0.5, 0.66, label, ha="center", va="center", fontsize=10, color=TEXT,
            transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha="center", va="center", fontsize=19, fontweight="bold",
            color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.24, sub, ha="center", va="center", fontsize=8.5, color=TEXT,
            transform=ax.transAxes)


def wordcloud_image(texts, width=900, height=420):
    from wordcloud import WordCloud
    font = None
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
                      "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(candidate):
            font = candidate
            break
    freq = texts.value_counts().to_dict() if isinstance(texts, pd.Series) else texts
    wc = WordCloud(font_path=font, width=width, height=height, background_color="white",
                   colormap="Blues", prefer_horizontal=1.0, max_words=40,
                   relative_scaling=0.4, margin=4)
    wc.generate_from_frequencies(freq)
    return np.array(wc.to_image())


# ============================================================================
# 一、Power BI 银行理财看板
# ============================================================================
def bank_dashboard():
    print("\n" + "=" * 78)
    print("一、Power BI 银行理财业务看板（440 位客户 / 2500 万目标）")
    print("=" * 78)
    xl = pd.read_excel(BANK, sheet_name=None)
    sales = clean_columns(xl["销售数据"])
    risk_map = clean_columns(xl["风险偏好说明表"]).set_index("风险编号")["风险说明"]
    cat_map = clean_columns(xl["品类说明表"]).set_index("品类编号")["品类名称"]
    edu_map = clean_columns(xl["学历说明表"]).set_index("学历编号")["学历名称"]

    sales["风险偏好"] = sales["风险编号"].map(risk_map)
    sales["品类名称"] = sales["品类编号"].map(cat_map)
    sales["学历名称"] = sales["学历编号"].map(edu_map)
    sales["购买金额"] = pd.to_numeric(sales["购买金额"], errors="coerce")
    sales["收益"] = pd.to_numeric(sales["收益"], errors="coerce")

    total = sales["购买金额"].sum()
    profit = sales["收益"].sum()
    customers = sales["客户ID"].nunique() if "客户ID" in sales.columns else len(sales)
    completion = total / 10000 / TARGET
    gap = TARGET - total / 10000
    print(f"  本月累计销售额 {total / 10000:,.1f} 万，目标 {TARGET:,} 万，"
          f"完成率 {completion:.1%}，缺口 {gap:,.1f} 万")
    print(f"  客户 {customers:,} 人，人均购买 {sales['购买金额'].mean() / 10000:.2f} 万，"
          f"累计收益 {profit / 10000:,.1f} 万")
    print("\n  风险偏好结构：")
    risk = sales.groupby("风险偏好")["购买金额"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    print(risk.assign(sum=risk["sum"].map(lambda v: f"{v / 10000:,.1f} 万")).to_string())
    print("\n  品类销售 TOP5：")
    cat = sales.groupby("品类名称")["购买金额"].sum().sort_values(ascending=False)
    print(cat.head(5).map(lambda v: f"{v / 10000:,.1f} 万").to_string())
    print("\n  年龄结构：")
    print(sales["年龄"].value_counts().to_string())
    print("\n  学历结构：")
    print(sales["学历名称"].value_counts().to_string())

    fig = plt.figure(figsize=(17.2, 10.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.15, 1.15], hspace=0.5, wspace=0.42)
    cards = [("本月销售额", f"{total / 10000:,.1f} 万", f"目标 {TARGET} 万"),
             ("目标完成率", f"{completion:.1%}", f"缺口 {gap:,.1f} 万"),
             ("客户数", f"{customers} 人", "仅本月数据"),
             ("人均购买", f"{sales['购买金额'].mean() / 10000:.2f} 万", f"累计收益 {profit / 10000:,.0f} 万")]
    for i, (label, value, sub) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        kpi_card(ax, label, value, sub, LIGHT_BLUE if i % 2 == 0 else LIGHT_YELLOW)

    ax = fig.add_subplot(gs[1, 0:2])
    risk_order = risk.index.tolist()
    wedges, texts, autotexts = ax.pie(
        risk["sum"].values, labels=risk_order, colors=[LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE][:len(risk)],
        autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
        wedgeprops=dict(width=0.44, edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT, fontsize=9.5),
    )
    ax.text(0, 0, "风险偏好\n金额结构", ha="center", va="center", fontsize=10, color=TEXT)
    ax.set_title("风险偏好结构", loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)

    ax = fig.add_subplot(gs[1, 2:4])
    top = cat.head(7).sort_values()
    bars = ax.barh(top.index, top.values / 10000, height=0.6,
                   color=[LIGHT_YELLOW if i % 2 == 0 else LIGHT_BLUE for i in range(len(top))],
                   edgecolor="white")
    for b, v in zip(bars, top.values / 10000):
        ax.text(b.get_width() + cat.max() / 10000 * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,.0f} 万", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax, "品类销售 TOP7", "销售额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, cat.max() / 10000 * 1.22)

    ax = fig.add_subplot(gs[2, 0:2])
    age_order = ["18-30岁", "31-40岁", "41-50岁", "50-60岁", "61岁以上"]
    age = sales["年龄"].value_counts().reindex(age_order).fillna(0)
    bars = ax.bar(age.index, age.values, width=0.56, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, age.values):
        ax.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom",
                fontsize=9, color=TEXT)
    style_axes(ax, "客户年龄分布", None, "客户数")
    ax.set_ylim(0, age.max() * 1.18)

    ax = fig.add_subplot(gs[2, 2:4])
    edu = sales["学历名称"].value_counts()
    bars = ax.barh(edu.index[::-1], edu.values[::-1], height=0.58, color=LIGHT_BLUE,
                   edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, edu.values[::-1]):
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, int(v), va="center",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "客户学历分布", "客户数", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, edu.max() * 1.18)

    fig.suptitle("银行理财业务看板：目标进度 · 客户结构 · 产品结构", fontsize=17,
                 fontweight="bold", color=TEXT, y=0.98)
    save(fig, "01_银行理财业务看板.png")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    ax = axes[0, 0]
    gender = sales["性别"].value_counts()
    ax.pie(gender.values, labels=gender.index, colors=[LIGHT_BLUE, LIGHT_YELLOW],
           autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           textprops=dict(color=TEXT, fontsize=10))
    ax.text(0, 0, "性别", ha="center", va="center", fontsize=11, color=TEXT)
    ax.set_title("客户性别", loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)

    ax = axes[0, 1]
    term = sales["期限"].value_counts().sort_index()
    bars = ax.bar([f"{int(i)} 年" for i in term.index], term.values, width=0.56,
                  color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, term.values):
        ax.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom",
                fontsize=9, color=TEXT)
    style_axes(ax, "产品期限偏好", None, "客户数")
    ax.set_ylim(0, term.max() * 1.18)

    ax = axes[1, 0]
    img = wordcloud_image(sales["购买目的"])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("购买目的词云", loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)

    ax = axes[1, 1]
    risk_colors = {"S1 低风险低收益": LIGHT_BLUE, "S2 中风险中收益": LIGHT_YELLOW,
                   "S3 高风险高收益": MID_BLUE}
    for name, part in sales.groupby("风险偏好"):
        ax.scatter(part["购买金额"] / 10000, part["收益"] / 10000, s=38,
                   color=risk_colors.get(name, MID_BLUE), edgecolors="white",
                   linewidths=0.6, alpha=0.8, label=name)
    style_axes(ax, "购买金额与收益（按风险偏好）", "购买金额（万元）", "收益（万元）")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("客户画像：谁在买、买多久、为什么买", fontsize=16,
                 fontweight="bold", color=TEXT, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "02_银行客户画像.png")

    print("\n  经营建议：")
    print(f"  - 目标完成 {completion:.1%}，剩余 {gap:,.1f} 万，需优先推动高金额品类与高净值客群；")
    print(f"  - {risk.index[0]} 客户贡献最多，产品配置应与风险偏好分层匹配；")
    print(f"  - 31-50 岁客户占 {age.loc[['31-40岁', '41-50岁']].sum() / len(sales):.1%}，"
          f"是主攻人群；本科学历客户最多，沟通材料应专业化。")


# ============================================================================
# 二、WPS 归因分析
# ============================================================================
def attribution_analysis():
    print("\n" + "=" * 78)
    print("二、WPS 归因分析：5 月贷款申请量环比 -14.5% 的来源")
    print("=" * 78)
    df = clean_columns(pd.read_excel(ATTR))
    df["申请日期"] = pd.to_datetime(df["申请日期"])
    df["月份"] = df["申请日期"].dt.to_period("M").astype(str)
    monthly = df.groupby("月份").size().sort_index()
    mom = monthly.pct_change()
    print("  月度申请量：")
    print(pd.DataFrame({"申请量": monthly, "环比": mom.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")})
          .to_string())

    cur, prev = monthly.index[-1], monthly.index[-2]
    channel = pd.crosstab(df["月份"], df["申请渠道"]).sort_index()
    delta = channel.loc[cur] - channel.loc[prev]
    total_delta = delta.sum()
    contrib = (delta / abs(total_delta)).sort_values()
    print(f"\n  {cur} 较 {prev}：{monthly.loc[cur]:,} - {monthly.loc[prev]:,} = {total_delta:+,} 单"
          f"（{mom.loc[cur]:+.1%}）")
    print("  渠道贡献拆解（变化量 / 对总变化的贡献率）：")
    print(pd.DataFrame({
        "上月": channel.loc[prev],
        "本月": channel.loc[cur],
        "变化量": delta,
        "贡献率": delta / abs(total_delta),
        "本月占比": channel.loc[cur] / channel.loc[cur].sum(),
        "上月占比": channel.loc[prev] / channel.loc[prev].sum(),
    }).assign(贡献率=lambda d: d["贡献率"].map(lambda v: f"{v:+.1%}"),
              本月占比=lambda d: d["本月占比"].map(lambda v: f"{v:.1%}"),
              上月占比=lambda d: d["上月占比"].map(lambda v: f"{v:.1%}")).to_string())
    print(f"\n  结论：下降主要由「{contrib.index[0]}」贡献 {abs(contrib.iloc[0]):.0%}，"
          f"其次「{contrib.index[1]}」贡献 {abs(contrib.iloc[1]):.0%}；"
          f"「{contrib.index[-1]}」变化 {delta.loc[contrib.index[-1]]:+d} 单，起到缓冲作用。")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    ax = axes[0, 0]
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2.6, color=DEEP_BLUE,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.fill_between(monthly.index, monthly.values, color=LIGHT_BLUE, alpha=0.35)
    for x, y in zip(monthly.index, monthly.values):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=TEXT)
    style_axes(ax, "月度贷款申请量", None, "申请笔数")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.set_ylim(monthly.min() * 0.78, monthly.max() * 1.14)

    ax = axes[0, 1]
    colors = [RED if v < 0 else MID_BLUE for v in mom.dropna().values]
    bars = ax.bar(mom.dropna().index, mom.dropna().values, width=0.58, color=colors, edgecolor=colors)
    ax.axhline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, mom.dropna().values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1%}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=TEXT)
    style_axes(ax, "月度环比变化", None, "环比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.set_ylim(mom.min() * 1.35, max(mom.max() * 1.3, 0.05))

    ax = axes[1, 0]
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE]
    bottom = np.zeros(len(channel))
    for i, col in enumerate(channel.columns):
        ax.bar(channel.index, channel[col], bottom=bottom, width=0.58, label=col,
               color=colors[i % len(colors)], edgecolor="white", linewidth=1.0)
        bottom += channel[col].to_numpy()
    for i, total in enumerate(channel.sum(axis=1).values):
        ax.text(i, total + 5, f"{int(total)}", ha="center", fontsize=9, color=TEXT)
    style_axes(ax, "分渠道月度申请量", None, "申请笔数")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
    ax.set_ylim(0, channel.sum(axis=1).max() * 1.16)

    ax = axes[1, 1]
    order = delta.sort_values()
    colors = [RED if v < 0 else MID_BLUE for v in order.values]
    bars = ax.barh(order.index, order.values, height=0.5, color=colors, edgecolor=colors)
    ax.axvline(0, color=GREY, linewidth=1.2)
    for b, v in zip(bars, order.values):
        ax.text(v + (0.6 if v >= 0 else -0.6), b.get_y() + b.get_height() / 2,
                f"{v:+d} 单（{v / abs(total_delta):+.0%}）", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
    style_axes(ax, f"{cur} 较 {prev} 的渠道贡献", "变化量（单）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(order.min() * 1.9, max(order.max() * 1.9, 5))

    fig.suptitle("归因分析：环比 -14.5% 到底是谁造成的", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.55, wspace=0.24)
    save(fig, "03_归因分析_渠道拆解.png")

    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    share_prev = (channel.loc[prev] / channel.loc[prev].sum()).reindex(channel.columns)
    share_cur = (channel.loc[cur] / channel.loc[cur].sum()).reindex(channel.columns)
    x = np.arange(len(channel.columns))
    ax.bar(x - 0.18, share_prev.values, width=0.34, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, label=prev)
    ax.bar(x + 0.18, share_cur.values, width=0.34, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW, label=cur)
    for xi, (a, b) in enumerate(zip(share_prev.values, share_cur.values)):
        ax.text(xi - 0.18, a + 0.01, f"{a:.1%}", ha="center", fontsize=9, color=TEXT)
        ax.text(xi + 0.18, b + 0.01, f"{b:.1%}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(channel.columns)
    style_axes(ax, "渠道结构占比变化：数量下降之外，结构也在变", None, "占比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylim(0, max(share_prev.max(), share_cur.max()) * 1.2)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "04_归因分析_渠道结构变化.png")

    return delta, monthly


def main():
    set_style()
    bank_dashboard()
    attribution_analysis()
    print("\n第 5 章全部案例完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()