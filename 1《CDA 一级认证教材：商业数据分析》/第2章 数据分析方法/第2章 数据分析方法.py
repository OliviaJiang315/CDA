"""第 2 章 数据分析方法：Excel / Power BI 操作的 Python 迁移

教材结构与本脚本对应关系：
1. EXCEL 基础操作：常用函数、查找引用函数 -> pandas / numpy 等价实现；
2. 统计制图原理：双轴图误用与正确表达 -> matplotlib 对比图；
3. 由基础范式引申出的六类分析方法：趋势、对比、构成、分布、关系、桑基；
4. 基础分析范式（模型）：波士顿矩阵、RFM、用户忠诚度、同期群、漏斗、相关分析；
5. Power BI 基本操作：ETL、DAX 度量值、星型数据模型、零售销售看板。

所有图表沿用教材的淡蓝 / 淡黄配色，柱形之间保留间隙，图例与标注框使用圆角。
"""

from pathlib import Path
import warnings
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)

D1 = BASE / "1、分析的基础范式" / "data-有答案"
D2 = BASE / "2、由基础分析范式引申出的分析方法" / "data-有答案"
D2N = BASE / "2、由基础分析范式引申出的分析方法" / "data-无答案"
EXCEL_DIR = BASE / "4、EXCEL基础操作"
PBI = BASE / "5、PowerBI基本操作"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
LIGHT_GREY = "#D9D9D9"
GREY = "#A6A6A6"
TEXT = "#404040"
GRID = "#E3E3E3"
PALETTE = ["#A8CCE8", "#FFE699", "#7FA8CF", "#F5C46B", "#C9DAEA", "#E8D9A8"]


def set_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#BFBFBF"


def style_axes(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if title:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def rounded_box(ax, x, y, w, h, text, fc, fs=11, ec=None, lw=1.4, alpha=1.0, transform=None):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.025",
        linewidth=lw, facecolor=fc, edgecolor=ec or fc, alpha=alpha,
        transform=transform or ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            linespacing=1.5, transform=transform or ax.transAxes, zorder=4)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图表] {path.name}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    return df


def read_csv_smart(path, **kwargs):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def fmt_wan(v):
    return f"{v / 10000:,.1f}"


# ============================================================================
# 一、EXCEL 基础操作迁移
# ============================================================================
def excel_function_demo():
    print("\n" + "=" * 78)
    print("一、EXCEL 函数 -> Python 等价实现")
    print("=" * 78)

    stu_file = PBI / "函数练习文件" / "学生考试案例"
    students = clean_columns(pd.read_excel(stu_file / "学生及老师信息.xlsx", sheet_name="学生信息"))
    scores = clean_columns(pd.read_excel(stu_file / "考试记录.xlsx", sheet_name="考试记录"))
    teachers = clean_columns(pd.read_excel(stu_file / "学生及老师信息.xlsx", sheet_name="教师信息"))

    df = students.merge(scores, on="学号", how="left")
    df["总分"] = df[["语文", "数学", "英语"]].sum(axis=1)
    df["平均分"] = df[["语文", "数学", "英语"]].mean(axis=1).round(1)
    df["等级"] = np.select(
        [df["总分"] >= 270, df["总分"] >= 240, df["总分"] >= 210],
        ["优秀", "良好", "及格"], default="待提升",
    )
    df["性别"] = df["性别(1男生,2女生)"].map({1: "男", 2: "女"})
    df["班级排名"] = df.groupby("班级")["总分"].rank(ascending=False, method="min").astype(int)
    df["班主任"] = df.merge(teachers, on="班级", how="left")["班主任姓名"]
    df["出生日期"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["出生日期"], unit="D")
    df["年龄"] = ((pd.Timestamp("2025-01-01") - df["出生日期"]).dt.days // 365.25).astype(int)

    print("  学生成绩（VLOOKUP 班主任、IF 等级、RANK 班级排名、日期序列号还原）：")
    print(df[["学号", "姓名", "班级", "班主任", "总分", "平均分", "等级", "班级排名", "年龄"]]
          .head(10).to_string(index=False))

    hr = clean_columns(pd.read_excel(
        PBI / "函数练习文件" / "人力案例" / "员工信息.xlsx", sheet_name="员工信息"))
    hr["出生日期"] = pd.to_datetime(hr["出生日期"])
    hr["入职日期"] = pd.to_datetime(hr["入职日期"])
    hr["离职日期"] = pd.to_datetime(hr["离职日期"])
    ref = pd.Timestamp("2025-01-01")
    hr["年龄"] = ((ref - hr["出生日期"]).dt.days // 365.25).astype(int)
    hr["工龄年"] = ((ref - hr["入职日期"]).dt.days / 365.25).round(1)
    hr["司龄段"] = pd.cut(hr["工龄年"], bins=[-1, 1, 3, 5, 100],
                       labels=["1年以内", "1-3年", "3-5年", "5年以上"])
    hr["是否离职"] = np.where(hr["员工状态"] == "离职", 1, 0)

    print("\n  人力资源（DATEDIF 工龄、IF 状态、分组统计）：")
    print(hr[["员工工号", "姓名", "部门", "职级", "入职日期", "工龄年", "司龄段", "员工状态"]]
          .head(8).to_string(index=False))
    print(f"\n  在职 {int((hr['员工状态'] == '在职').sum())} 人，"
          f"离职 {int(hr['是否离职'].sum())} 人，整体离职率 {hr['是否离职'].mean():.1%}")
    print("  各部门离职率：")
    print(hr.groupby("部门")["是否离职"].mean().sort_values(ascending=False)
          .map(lambda v: f"{v:.1%}").to_string())

    print("\n  常用文本/日期函数等价写法：")
    demo = pd.DataFrame({"手机号": ["13907767111", "13577601234"], "日期": pd.to_datetime(["2024-02-29", "2023-12-31"])})
    demo["脱敏手机号"] = demo["手机号"].str[:3] + "****" + demo["手机号"].str[-4:]
    demo["年份"] = demo["日期"].dt.year
    demo["月份"] = demo["日期"].dt.month
    demo["是否闰年"] = demo["日期"].dt.is_leap_year
    demo["取整INT"] = np.floor(pd.Series([3.345, -3.4]))
    demo["取余MOD"] = np.mod(pd.Series([10, 100]), 3)
    print(demo.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.0))
    ax = axes[0]
    dept = hr.groupby("部门").size().sort_values()
    ax.barh(dept.index, dept.values, height=0.58, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for i, v in enumerate(dept.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9.5, color=TEXT)
    style_axes(ax, "人力资源：部门人数", "人数", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, dept.max() * 1.18)

    ax = axes[1]
    band = hr["司龄段"].value_counts().reindex(["1年以内", "1-3年", "3-5年", "5年以上"]).fillna(0)
    bars = ax.bar(band.index, band.values, width=0.6, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, band.values):
        ax.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom", fontsize=9.5, color=TEXT)
    style_axes(ax, "人力资源：司龄结构", None, "人数")
    ax.set_ylim(0, band.max() * 1.18)
    fig.suptitle("Excel 公式迁移为 Python：一次计算，整表复用", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "01_Excel函数迁移_人力资源.png")


# ============================================================================
# 二、统计制图原理
# ============================================================================
def chart_principle_demo():
    print("\n" + "=" * 78)
    print("二、统计制图原理：双轴图的误导与正确表达")
    print("=" * 78)
    sale = read_csv_smart(BASE / "3、统计制图原理" / "sale.csv")
    wide = sale.pivot_table(index="year", columns="market", values="sale", aggfunc="sum")
    total = wide.sum(axis=1)
    profit_rate = sale.groupby("year").apply(lambda d: d["profit"].sum() / d["sale"].sum())
    print("  年度销售额与利润率：")
    print(pd.DataFrame({"销售额": total, "利润率": profit_rate.map(lambda v: f"{v:.2%}")}).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.0))
    ax = axes[0]
    ax.bar(total.index, total.values / 10000, width=0.58, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    ax.set_ylim(0, total.max() / 10000 * 1.3)
    style_axes(ax, "反面示例：双轴放大微小波动", "年份", "销售额（万元）")
    ax2 = ax.twinx()
    ax2.plot(total.index, profit_rate.values * 100, color="#C0504D", marker="o", linewidth=2.4)
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("利润率（%）", color="#C0504D")
    ax2.tick_params(colors="#C0504D", length=0)
    ax2.spines["top"].set_visible(False)
    ax.text(0.02, 0.96, "两条轴刻度可随意拉伸，\n视觉上制造了不存在的强相关",
            transform=ax.transAxes, fontsize=9.5, color="#C0504D", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FBEAEA", edgecolor="#C0504D"))

    ax = axes[1]
    idx_sale = total / total.iloc[0] * 100
    idx_rate = profit_rate / profit_rate.iloc[0] * 100
    ax.plot(idx_sale.index, idx_sale.values, marker="o", linewidth=2.4, color=LIGHT_BLUE, label="销售额指数")
    ax.plot(idx_rate.index, idx_rate.values, marker="s", linewidth=2.4, color=LIGHT_YELLOW, label="利润率指数")
    ax.axhline(100, color=GREY, linestyle="--", linewidth=1.2)
    style_axes(ax, "正面示例：统一为同基数指数", "年份", "2010 年 = 100")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_xticks(idx_sale.index)
    fig.suptitle("统计制图第一原则：一张图只讲一个尺度", fontsize=15, fontweight="bold", color=TEXT, y=1.0)
    save(fig, "02_统计制图原理_双轴图对比.png")


# ============================================================================
# 三、六类分析方法
# ============================================================================
def six_methods_demo():
    print("\n" + "=" * 78)
    print("三、六类分析方法：趋势 / 对比 / 构成 / 分布 / 关系 / 桑基")
    print("=" * 78)

    trend_raw = pd.read_excel(D2 / "1趋势分析-参考.xlsx", sheet_name="趋势图", header=None)
    trend = pd.DataFrame({"年份": trend_raw.iloc[2:13, 1].astype(int).to_numpy(),
                          "订单量": trend_raw.iloc[2:13, 2].astype(int).to_numpy()})
    print("  趋势分析：2014-2024 年订单量")
    print(trend.to_string(index=False))

    compare_raw = pd.read_excel(D2 / "2对比分析-参考.xlsx", sheet_name="对比分析", header=None)
    compare = pd.DataFrame({"产品": compare_raw.iloc[2:8, 1].to_numpy(),
                            "订单量": compare_raw.iloc[2:8, 2].astype(int).to_numpy()})
    print("\n  对比分析：各产品订单量")
    print(compare.to_string(index=False))

    comp_raw = pd.read_excel(D2 / "4构成分析-参考.xlsx", sheet_name="构成分析", header=None)
    comp = pd.DataFrame({"产品": comp_raw.iloc[4:8, 1].to_numpy(),
                         "销量": comp_raw.iloc[4:8, 2].astype(int).to_numpy()})
    comp["占比"] = comp["销量"] / comp["销量"].sum()
    print("\n  构成分析：产品销量结构")
    print(comp.assign(占比=comp["占比"].map(lambda v: f"{v:.1%}")).to_string(index=False))

    year_comp = pd.DataFrame({
        "年份": comp_raw.iloc[17:22, 1].astype(int).to_numpy(),
        "产品1": comp_raw.iloc[17:22, 2].astype(int).to_numpy(),
        "产品2": comp_raw.iloc[17:22, 3].astype(int).to_numpy(),
        "产品3": comp_raw.iloc[17:22, 4].astype(int).to_numpy(),
    })

    dist_raw = pd.read_excel(D2 / "3分布分析-参考.xlsx", sheet_name="分布分析", header=None)
    traffic = pd.to_numeric(dist_raw.iloc[2:340, 2], errors="coerce").dropna()
    ages = pd.to_numeric(dist_raw.iloc[2:12, 11], errors="coerce").dropna()
    q1, q2, q3 = np.percentile(ages, [25, 50, 75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    print(f"\n  分布分析：门店客流量 {len(traffic)} 个半小时观测；"
          f"用户年龄 n={len(ages)}，Q1={q1}，中位数={q2}，Q3={q3}，IQR={iqr}，上限={upper}")

    rel_raw = pd.read_excel(D2 / "5关系分析-参考.xlsx", sheet_name="关系图", header=None)
    campaign = pd.DataFrame({
        "活动": rel_raw.iloc[3:8, 1].to_numpy(),
        "下单用户": rel_raw.iloc[3:8, 2].astype(int).to_numpy(),
        "未下单用户": rel_raw.iloc[3:8, 3].astype(int).to_numpy(),
    })
    invest = pd.DataFrame({
        "投入成本": pd.to_numeric(rel_raw.iloc[1:21, 25], errors="coerce"),
        "产出收益": pd.to_numeric(rel_raw.iloc[1:21, 24], errors="coerce"),
    }).dropna()
    r = invest["投入成本"].corr(invest["产出收益"])
    print(f"\n  关系分析：营销活动下单结构；投入成本与产出收益相关系数 r={r:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(17.6, 10.2))
    ax = axes[0, 0]
    ax.plot(trend["年份"], trend["订单量"], marker="o", linewidth=2.4, color=DEEP_BLUE,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.fill_between(trend["年份"], trend["订单量"], color=LIGHT_BLUE, alpha=0.35)
    style_axes(ax, "趋势分析：订单量 2014-2024", "年份", "订单量（个）")
    ax.set_xticks(trend["年份"])
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    order = compare.sort_values("订单量")
    ax.barh(order["产品"], order["订单量"], height=0.58, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    ax.axvline(compare["订单量"].mean(), color="#C0504D", linestyle="--", linewidth=1.3)
    for i, v in enumerate(order["订单量"].values):
        ax.text(v + 6, i, str(v), va="center", fontsize=9.5, color=TEXT)
    style_axes(ax, "对比分析：产品订单量（虚线为均值）", "订单量（个）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, order["订单量"].max() * 1.16)

    ax = axes[0, 2]
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE, "#F5C46B"]
    wedges, texts, autotexts = ax.pie(
        comp["销量"], labels=comp["产品"], colors=colors,
        autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT, fontsize=10),
    )
    for t in autotexts:
        t.set_fontsize(9.5)
    ax.text(0, 0, f"合计\n{comp['销量'].sum():,}", ha="center", va="center", fontsize=11, color=TEXT)
    ax.set_title("构成分析：产品销量结构", loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=14)

    ax = axes[1, 0]
    ax.hist(traffic, bins=28, color=LIGHT_BLUE, edgecolor="white", linewidth=1.0)
    ax.axvline(traffic.mean(), color=DEEP_BLUE, linestyle="--", linewidth=1.6, label=f"均值 {traffic.mean():.0f}")
    style_axes(ax, "分布分析：门店客流量分布", "客流量", "频数")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    bp = ax.boxplot([ages], widths=0.32, patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW, linewidth=1.6),
                    medianprops=dict(color="#C0504D", linewidth=2.2),
                    whiskerprops=dict(color=GREY, linewidth=1.4),
                    capprops=dict(color=GREY, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=5, markerfacecolor=LIGHT_BLUE,
                                    markeredgecolor=LIGHT_BLUE))
    ax.set_xticks([1])
    ax.set_xticklabels(["用户年龄"])
    ax.text(1.22, q3, f"Q3 = {q3:g}", fontsize=9, color=TEXT, va="center")
    ax.text(1.22, q2, f"中位数 = {q2:g}", fontsize=9, color=TEXT, va="center")
    ax.text(1.22, q1, f"Q1 = {q1:g}", fontsize=9, color=TEXT, va="center")
    ax.text(1.22, upper, f"上限 = {upper:g}", fontsize=9, color="#C0504D", va="center")
    style_axes(ax, "分布分析：用户年龄箱线图", None, "年龄（岁）")
    ax.set_xlim(0.6, 1.7)

    ax = axes[1, 2]
    bottom = np.zeros(len(campaign))
    ax.bar(campaign["活动"], campaign["下单用户"], width=0.58, label="下单用户",
           color=LIGHT_BLUE, edgecolor="white")
    ax.bar(campaign["活动"], campaign["未下单用户"], bottom=campaign["下单用户"], width=0.58,
           label="未下单用户", color=LIGHT_YELLOW, edgecolor="white")
    style_axes(ax, "关系分析：营销活动下单结构", None, "用户数")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("六类分析方法：先用图形看清结构，再用数字确认结论", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "03_六类分析方法.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6))
    ax = axes[0]
    ax.stackplot(year_comp["年份"], [year_comp["产品1"], year_comp["产品2"], year_comp["产品3"]],
                 labels=["产品 1", "产品 2", "产品 3"], colors=[LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE],
                 alpha=0.9, edgecolor="white", linewidth=1.2)
    style_axes(ax, "构成分析（面积图）：各年产品销售结构", "年份", "销量")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_xticks(year_comp["年份"])

    ax = axes[1]
    share = comp.sort_values("销量", ascending=False).copy()
    share["累计占比"] = share["销量"].cumsum() / share["销量"].sum()
    ax.bar(share["产品"], share["销量"], width=0.58, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    ax2 = ax.twinx()
    ax2.plot(share["产品"], share["累计占比"], color="#C0504D", marker="o", linewidth=2.0)
    ax2.axhline(0.8, color=GREY, linestyle="--", linewidth=1.2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("累计占比", color="#C0504D")
    ax2.tick_params(colors="#C0504D", length=0)
    ax2.spines["top"].set_visible(False)
    for x, v in zip(share["产品"], share["累计占比"]):
        ax2.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=9, color="#C0504D")
    style_axes(ax, "构成分析（帕累托）：销量与累计占比", "产品", "销量")
    fig.suptitle("构成分析的两种追问：结构随时间怎样变化？谁贡献了大部分？", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.02)
    save(fig, "04_构成分析_面积图与帕累托.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8))
    ax = axes[0]
    ax.scatter(invest["投入成本"], invest["产出收益"], s=70, c=LIGHT_BLUE,
               edgecolors=DEEP_BLUE, linewidths=1.2, alpha=0.9)
    k, b = np.polyfit(invest["投入成本"], invest["产出收益"], 1)
    xs = np.linspace(invest["投入成本"].min(), invest["投入成本"].max(), 100)
    ax.plot(xs, k * xs + b, color="#C0504D", linewidth=2.2, linestyle="--")
    ax.text(0.04, 0.95, f"y = {k:.3f}x + {b:.1f}\nr = {r:.3f}", transform=ax.transAxes,
            fontsize=10, va="top", color=TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F7FC", edgecolor=LIGHT_BLUE))
    style_axes(ax, "关系分析（连续变量）：投入成本与产出收益", "投入成本", "产出收益")

    ax = axes[1]
    x = np.arange(len(campaign))
    w = 0.36
    ax.bar(x - w / 2, campaign["下单用户"] / (campaign["下单用户"] + campaign["未下单用户"]),
           width=w, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, label="下单转化率")
    ax.set_xticks(x)
    ax.set_xticklabels(campaign["活动"], rotation=20)
    for xi, (a, b2) in zip(x, zip(campaign["下单用户"], campaign["未下单用户"])):
        ax.text(xi - w / 2, a / (a + b2) + 0.02, f"{a / (a + b2):.0%}", ha="center", fontsize=9, color=TEXT)
    style_axes(ax, "关系分析（分类变量）：活动转化率", None, "下单转化率")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    fig.suptitle("关系分析：先问「是否相关」，再问「相关多强、能否用于预测」", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.02)
    save(fig, "05_关系分析_散点与转化率.png")

    sankey_raw = pd.read_excel(D2N / "6桑吉图.xlsx", header=0)
    sankey_raw = clean_columns(sankey_raw)
    print("\n  桑基图：页面流量路径")
    print(sankey_raw.to_string(index=False))
    draw_sankey(sankey_raw)


def draw_sankey(data):
    levels = []
    for _, row in data.iterrows():
        for node in (row["来源"], row["目标"]):
            if node not in levels:
                levels.append(node)

    depth = {}
    for _, row in data.iterrows():
        depth.setdefault(row["来源"], 0)
        depth[row["目标"]] = max(depth.get(row["目标"], 0), depth[row["来源"]] + 1)

    columns = {}
    for node, d in depth.items():
        columns.setdefault(d, []).append(node)

    throughput = {}
    for node in levels:
        inbound = data.loc[data["目标"] == node, "流量"].sum()
        outbound = data.loc[data["来源"] == node, "流量"].sum()
        throughput[node] = max(inbound, outbound, 1)

    total = max(sum(throughput[n] for n in columns[c]) for c in columns)
    col_total = {c: sum(throughput[n] for n in nodes) for c, nodes in columns.items()}
    gap = total * 0.035

    node_pos = {}
    for c, nodes in columns.items():
        nodes = sorted(nodes, key=lambda n: -throughput[n])
        height = col_total[c] + gap * (len(nodes) - 1)
        y = (total + height) / 2
        for n in nodes:
            h = throughput[n]
            node_pos[n] = dict(x=c, y0=y - h, y1=y, h=h)
            y -= h + gap

    max_col = max(columns)
    fig, ax = plt.subplots(figsize=(14.2, 8.2))
    ax.set_xlim(-0.12, max_col + 0.42)
    ax.set_ylim(-total * 0.03, total * 1.02)
    ax.axis("off")

    for _, row in data.iterrows():
        s, t, flow = row["来源"], row["目标"], row["流量"]
        sp, tp = node_pos[s], node_pos[t]
        s_used = data.loc[data["来源"] == s].index.get_loc(_)
        s_offset = data.loc[data["来源"] == s, "流量"].iloc[:s_used].sum()
        t_used = data.loc[(data["目标"] == t)].index.get_loc(_)
        t_offset = data.loc[data["目标"] == t, "流量"].iloc[:t_used].sum()
        y0s, y1s = sp["y0"] + s_offset, sp["y0"] + s_offset + flow
        y0t, y1t = tp["y0"] + t_offset, tp["y0"] + t_offset + flow
        x0, x1 = sp["x"] + 0.05, tp["x"] - 0.02
        mid = (x0 + x1) / 2
        verts = [
            (x0, y1s), (mid, y1s), (mid, y1t), (x1, y1t),
            (x1, y0t), (mid, y0t), (mid, y0s), (x0, y0s), (x0, y1s),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        color = LIGHT_BLUE if depth[s] == 0 else LIGHT_YELLOW
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="white",
                               alpha=0.55, linewidth=0.8))

    for node, p in node_pos.items():
        ax.add_patch(FancyBboxPatch(
            (p["x"] - 0.025, p["y0"]), 0.05, p["h"],
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=DEEP_BLUE if p["x"] == 0 else MID_BLUE, edgecolor="white", linewidth=1.0,
        ))
        label_x = p["x"] - 0.06 if p["x"] == 0 else p["x"] + 0.06
        ha = "right" if p["x"] == 0 else "left"
        if p["x"] == max_col:
            label_x, ha = p["x"] + 0.05, "left"
        ax.text(label_x, (p["y0"] + p["y1"]) / 2, node, ha=ha, va="center",
                fontsize=9, color=TEXT)

    ax.set_title("桑基图：首页流量如何流向最终页面", loc="left", fontsize=15,
                 fontweight="bold", color=TEXT, pad=16)
    ax.text(0.0, 1.0, f"总入口流量 {int(total):,}", transform=ax.transAxes,
            fontsize=10, color=GREY, va="bottom")
    save(fig, "06_桑基图_页面流量路径.png")


# ============================================================================
# 四、基础分析范式（四大模型 + 漏斗 + 相关）
# ============================================================================
def boston_matrix_demo():
    print("\n" + "=" * 78)
    print("四、基础分析范式：波士顿矩阵（市场占有率 × 销售增长率）")
    print("=" * 78)
    sales = clean_columns(pd.read_excel(D1 / "1-1波士顿矩阵-参考.xlsx", sheet_name="销售表"))
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    sales["月份"] = sales["销售日期"].dt.month

    pivot = sales.pivot_table(index="品牌", columns="月份", values="订单金额",
                              aggfunc="sum", fill_value=0)
    if 7 not in pivot.columns or 8 not in pivot.columns:
        raise ValueError("波士顿矩阵数据需要 7 月与 8 月销售额")
    total = pivot[[7, 8]].sum().sum()
    df = pd.DataFrame({
        "7月销售额": pivot[7],
        "8月销售额": pivot[8],
        "销售额": pivot[[7, 8]].sum(axis=1),
    })
    df["市场占有率"] = df["销售额"] / total
    df["销售增长率"] = np.where(df["7月销售额"] > 0, df["8月销售额"] / df["7月销售额"] - 1, np.nan)
    df = df.dropna(subset=["销售增长率"])

    market_growth = pivot[8].sum() / pivot[7].sum() - 1
    share_line = df["市场占有率"].mean()
    print(f"  总市场环比增长率 = {market_growth:.2%}（教材参考 2.17%）")
    print(f"  品牌平均市场占有率 = {share_line:.2%}（教材参考 7.14%）")

    def quadrant(row):
        high_share = row["市场占有率"] >= share_line
        high_growth = row["销售增长率"] >= market_growth
        if high_share and high_growth:
            return "明星品牌"
        if not high_share and high_growth:
            return "问题品牌"
        if high_share and not high_growth:
            return "现金牛品牌"
        return "瘦狗品牌"

    df["战略类型"] = df.apply(quadrant, axis=1)
    print(df.assign(市场占有率=df["市场占有率"].map(lambda v: f"{v:.2%}"),
                    销售增长率=df["销售增长率"].map(lambda v: f"{v:+.2%}"))
          .sort_values("销售额", ascending=False).head(12).to_string())

    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    colors = {"明星品牌": MID_BLUE, "问题品牌": LIGHT_YELLOW,
              "现金牛品牌": LIGHT_BLUE, "瘦狗品牌": "#D9D9D9"}
    for kind, part in df.groupby("战略类型"):
        ax.scatter(part["市场占有率"], part["销售增长率"],
                   s=part["销售额"] / df["销售额"].max() * 2100 + 120,
                   color=colors[kind], edgecolors="white", linewidths=1.6,
                   alpha=0.95, label=kind, zorder=3)
    ax.axvline(share_line, color=GREY, linestyle="--", linewidth=1.3)
    ax.axhline(market_growth, color=GREY, linestyle="--", linewidth=1.3)

    offsets = {"A": (10, 8), "K": (10, 8), "M": (10, -16), "C": (-12, 10), "H": (10, 8)}
    for brand, row in df.iterrows():
        dx, dy = offsets.get(str(brand), (9, 9))
        ax.annotate(str(brand), (row["市场占有率"], row["销售增长率"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10,
                    color=TEXT, fontweight="bold")

    xmax = df["市场占有率"].max() * 1.35
    ax.text(xmax * 0.98, df["销售增长率"].max() * 1.02, "明星：高份额 · 高增长\n加大投入",
            ha="right", va="top", fontsize=9.5, color=DEEP_BLUE)
    ax.text(0.0005, df["销售增长率"].max() * 1.02, "问题：低份额 · 高增长\n评估是否值得投入",
            ha="left", va="top", fontsize=9.5, color="#B8860B")
    ax.text(xmax * 0.98, min(df["销售增长率"].min() * 1.1, -0.02), "现金牛：高份额 · 低增长\n收获现金流",
            ha="right", va="bottom", fontsize=9.5, color=DEEP_BLUE)
    ax.text(0.0005, min(df["销售增长率"].min() * 1.1, -0.02), "瘦狗：低份额 · 低增长\n收缩或退出",
            ha="left", va="bottom", fontsize=9.5, color=GREY)

    ax.axvspan(0, share_line, ymin=0, ymax=1, color="#F7F7F7", zorder=0)
    ax.axhline(0, color="#EDEDED", linewidth=1, zorder=0)
    style_axes(ax, "波士顿矩阵：品牌组合诊断", "市场占有率（销售额占比）", "销售增长率（8 月环比）")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=4, loc="upper left",
              bbox_to_anchor=(0, 1.0), markerscale=0.35, fontsize=10)
    save(fig, "07_波士顿矩阵.png")


def rfm_model_demo():
    print("\n" + "=" * 78)
    print("五、RFM 模型：用最近一次消费、频次、金额做客户分层")
    print("=" * 78)
    raw = clean_columns(pd.read_excel(D1 / "1-2RFM模型-参考.xlsx", sheet_name="原始数据"))
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw["销售日期"] = pd.to_datetime(raw["销售日期"])
    ref = pd.Timestamp("2019-08-03")

    rfm = raw.groupby("用户编号").agg(
        最近消费日期=("销售日期", "max"),
        消费频次=("流水号", "nunique"),
        消费金额=("销售金额", "sum"),
    ).reset_index()
    rfm["R"] = (ref - rfm["最近消费日期"]).dt.days
    print(f"  数据 {len(raw):,} 行 / {len(rfm):,} 位用户；分析时点 {ref:%Y-%m-%d}")
    print(f"  R 均值 {rfm['R'].mean():.2f}（教材 82.74）  "
          f"F 均值 {rfm['消费频次'].mean():.2f}（教材 3.72）  "
          f"M 均值 {rfm['消费金额'].mean():.2f}（教材 743.41）")

    rfm["R程度"] = np.where(rfm["R"] <= rfm["R"].mean(), "高", "低")
    rfm["F程度"] = np.where(rfm["消费频次"] >= rfm["消费频次"].mean(), "高", "低")
    rfm["M程度"] = np.where(rfm["消费金额"] >= rfm["消费金额"].mean(), "高", "低")
    label_map = {
        ("高", "高", "高"): "重要价值用户", ("高", "低", "高"): "重要发展用户",
        ("低", "高", "高"): "重要保持用户", ("低", "低", "高"): "重要挽留用户",
        ("高", "高", "低"): "一般价值用户", ("高", "低", "低"): "一般发展用户",
        ("低", "高", "低"): "一般保持用户", ("低", "低", "低"): "一般挽留用户",
    }
    rfm["用户类别"] = [label_map[(r, f, m)] for r, f, m in
                     zip(rfm["R程度"], rfm["F程度"], rfm["M程度"])]

    summary = (rfm.groupby("用户类别")
               .agg(用户数=("用户编号", "count"), 消费金额=("消费金额", "sum"))
               .sort_values("消费金额", ascending=False))
    summary["人数占比"] = summary["用户数"] / summary["用户数"].sum()
    summary["金额占比"] = summary["消费金额"] / summary["消费金额"].sum()
    print(summary.assign(消费金额=summary["消费金额"].map(lambda v: f"{v:,.0f}"),
                         人数占比=summary["人数占比"].map(lambda v: f"{v:.1%}"),
                         金额占比=summary["金额占比"].map(lambda v: f"{v:.1%}")).to_string())

    pareto = rfm.sort_values("消费金额", ascending=False).reset_index(drop=True)
    pareto["累计金额占比"] = pareto["消费金额"].cumsum() / pareto["消费金额"].sum()
    n80 = int((pareto["累计金额占比"] < 0.8).sum()) + 1
    print(f"\n  帕累托：前 {n80:,} 位用户（{n80 / len(pareto):.1%}）贡献了 80% 的消费金额")

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.0))
    ax = axes[0]
    x = np.arange(len(pareto))
    ax.bar(x, pareto["消费金额"], width=1.0, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    ax.set_xlim(-5, len(pareto) * 1.005)
    ax2 = ax.twinx()
    ax2.plot(x, pareto["累计金额占比"], color="#C0504D", linewidth=2.2)
    ax2.axhline(0.8, color=GREY, linestyle="--", linewidth=1.3)
    ax2.axvline(n80, color=GREY, linestyle=":", linewidth=1.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("累计金额占比", color="#C0504D")
    ax2.tick_params(colors="#C0504D", length=0)
    ax2.spines["top"].set_visible(False)
    ax2.text(n80, 0.82, f" 80% 分位：{n80:,} 人", color="#C0504D", fontsize=9.5)
    style_axes(ax, "RFM 帕累托：消费金额高度集中", "用户（按消费金额降序）", "消费金额")
    ax.set_xticks([])

    ax = axes[1]
    order = summary.index[::-1]
    bars = ax.barh(order, summary.loc[order, "消费金额"] / 10000, height=0.6,
                   color=[LIGHT_BLUE if i % 2 == 0 else LIGHT_YELLOW for i in range(len(order))],
                   edgecolor="white", linewidth=1.0)
    for b, (name, row) in zip(bars, summary.loc[order].iterrows()):
        ax.text(b.get_width() + summary["消费金额"].max() / 10000 * 0.015,
                b.get_y() + b.get_height() / 2,
                f"{row['用户数']:,} 人 · {row['金额占比']:.0%}",
                va="center", fontsize=9, color=TEXT)
    style_axes(ax, "RFM 八类用户：人数少但贡献大", "消费金额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, summary["消费金额"].max() / 10000 * 1.48)
    fig.suptitle("RFM：从「所有用户」到「值得被区别对待的用户」", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    fig.subplots_adjust(wspace=0.36)
    save(fig, "08_RFM模型_帕累托与八类用户.png")
    return rfm


def loyalty_model_demo():
    print("\n" + "=" * 78)
    print("六、用户忠诚度模型：潜在用户与易流失用户的识别")
    print("=" * 78)
    flow = clean_columns(pd.read_excel(D1 / "1-3用户忠诚度模型-参考.xlsx", sheet_name="销售流水表"))
    flow["消费日期"] = pd.to_datetime(flow["消费日期"])
    ref = pd.Timestamp("2021-01-01")
    cutoff = pd.Timestamp("2020-10-02")

    user = flow.groupby("会员编号").agg(
        消费次数=("订单编号", "nunique"),
        消费额=("消费金额", "sum"),
        首次消费=("消费日期", "min"),
        最近消费=("消费日期", "max"),
    ).reset_index()
    user["活跃度天数"] = (ref - user["最近消费"]).dt.days
    before = flow[flow["消费日期"] < cutoff].groupby("会员编号").size()
    after = flow[flow["消费日期"] >= cutoff].groupby("会员编号").size()
    user["窗口前次数"] = user["会员编号"].map(before).fillna(0)
    user["窗口后次数"] = user["会员编号"].map(after).fillna(0)

    def classify(row):
        if row["窗口前次数"] == 0 and row["窗口后次数"] > 0:
            return "潜在用户"
        if row["窗口后次数"] == 0:
            return "易流失用户"
        return "活跃用户"

    user["用户类别"] = user.apply(classify, axis=1)
    q25_f, q75_f = user["消费次数"].quantile([0.25, 0.75])
    q25_m, q75_m = user["消费额"].quantile([0.25, 0.75])
    print(f"  数据 {len(flow):,} 行 / {len(user):,} 位会员")
    print(f"  消费次数分位：25%={q25_f:.0f}，75%={q75_f:.0f}（教材 4 / 7）")
    print(f"  消费额分位：25%={q25_m:,.0f}，75%={q75_m:,.0f}（教材 3,198 / 5,724）")
    print(user.groupby("用户类别").agg(人数=("会员编号", "count"),
                                        消费额=("消费额", "sum")).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    ax = axes[0]
    colors = {"潜在用户": LIGHT_YELLOW, "易流失用户": "#D9D9D9", "活跃用户": LIGHT_BLUE}
    for kind, part in user.groupby("用户类别"):
        ax.scatter(part["消费次数"], part["消费额"], s=26, alpha=0.75,
                   color=colors[kind], edgecolors="white", linewidths=0.5, label=kind)
    ax.set_yscale("log")
    ax.axvline(q25_f, color=GREY, linestyle=":", linewidth=1.2)
    ax.axvline(q75_f, color=GREY, linestyle=":", linewidth=1.2)
    ax.axhline(q25_m, color=GREY, linestyle=":", linewidth=1.2)
    ax.axhline(q75_m, color=GREY, linestyle=":", linewidth=1.2)
    style_axes(ax, "忠诚度分层：消费频次 × 消费金额", "消费次数", "消费金额（对数轴）")
    ax.legend(frameon=False, loc="lower right", markerscale=1.6)
    style_axes(ax, "忠诚度分层：消费频次 × 消费金额", "消费次数", "消费金额（对数轴）")

    ax = axes[1]
    stat = user.groupby("用户类别").agg(人数=("会员编号", "count"), 消费额=("消费额", "sum"))
    stat = stat.reindex(["活跃用户", "潜在用户", "易流失用户"]).fillna(0)
    x = np.arange(len(stat))
    ax.bar(x - 0.17, stat["人数"] / stat["人数"].sum(), width=0.34,
           color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, label="人数占比")
    ax.bar(x + 0.17, stat["消费额"] / stat["消费额"].sum(), width=0.34,
           color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW, label="消费额占比")
    for xi, (p, m) in enumerate(zip(stat["人数"] / stat["人数"].sum(),
                                     stat["消费额"] / stat["消费额"].sum())):
        ax.text(xi - 0.17, p + 0.015, f"{p:.1%}", ha="center", fontsize=9, color=TEXT)
        ax.text(xi + 0.17, m + 0.015, f"{m:.1%}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(stat.index)
    style_axes(ax, "三类用户：人数与消费额贡献", None, "占比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle("忠诚度模型：潜在用户值得拉新，易流失用户值得召回", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "09_用户忠诚度模型.png")
    return user


def cohort_demo():
    print("\n" + "=" * 78)
    print("七、同期群分析：同一批注册用户，后续留存如何衰减")
    print("=" * 78)
    log = clean_columns(pd.read_excel(D1 / "1-4同期群分析-参考.xlsx", sheet_name="用户登录明细"))
    log["注册日期"] = pd.to_datetime(log["注册日期"])
    log["登录日期"] = pd.to_datetime(log["登录日期"])
    log["注册日"] = log["注册日期"].dt.normalize()
    log["留存天数"] = (log["登录日期"].dt.normalize() - log["注册日"]).dt.days
    log = log[log["留存天数"] >= 0]

    reg = log.groupby("注册日")["用户编号"].nunique().sort_index()
    login = log.groupby("登录日期")["用户编号"].nunique().sort_index()
    print(f"  注册用户 {log['用户编号'].nunique():,} 人，注册日期 "
          f"{reg.index.min():%Y-%m-%d} ~ {reg.index.max():%Y-%m-%d}")

    max_day = int(log["留存天数"].max())
    matrix = []
    cohorts = reg.index[: min(14, len(reg))]
    for c in cohorts:
        sub = log[log["注册日"] == c]
        base = sub["用户编号"].nunique()
        rows = []
        for d in range(0, 11):
            alive = sub.loc[sub["留存天数"] == d, "用户编号"].nunique()
            rows.append(alive / base if base else np.nan)
        matrix.append(rows)
    mat = pd.DataFrame(matrix, index=[c.strftime("%m-%d") for c in cohorts],
                       columns=[f"第{d}日" for d in range(11)])

    print("  留存矩阵（行=注册日同期群，列=注册后第 N 日，值=留存率）：")
    print(mat.map(lambda v: f"{v:.0%}" if pd.notna(v) else "").to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "retention", ["#F4F8FC", LIGHT_BLUE, MID_BLUE, "#3F6F9E"])
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("同期群留存热力图", loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=14)
    ax.set_xlabel("注册后第 N 日", color=TEXT, fontsize=10)
    ax.set_ylabel("注册同期群", color=TEXT, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)

    ax = axes[1]
    ax.plot(reg.index, reg.values, color=LIGHT_BLUE, marker="o", linewidth=2.2,
            markersize=4.5, label="注册用户数")
    ax.plot(login.index, login.values, color=LIGHT_YELLOW, marker="s", linewidth=2.2,
            markersize=4.5, label="登录用户数")
    style_axes(ax, "注册与登录用户日趋势", None, "用户数")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("同期群：把「增长」拆成拉新与留存两件事", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "10_同期群分析_留存热力图.png")


def funnel_demo():
    print("\n" + "=" * 78)
    print("八、漏斗分析：找到转化率最低的那一段")
    print("=" * 78)
    raw = pd.read_excel(D1 / "2-1漏斗图-参考.xlsx", sheet_name="Sheet1", header=None)
    funnel = pd.DataFrame({
        "阶段": raw.iloc[3:9, 0].to_numpy(),
        "用户数": raw.iloc[3:9, 2].astype(int).to_numpy(),
    })
    funnel["环节转化率"] = funnel["用户数"] / funnel["用户数"].shift(1)
    funnel["总体转化率"] = funnel["用户数"] / funnel["用户数"].iloc[0]
    print(funnel.assign(环节转化率=funnel["环节转化率"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
                        总体转化率=funnel["总体转化率"].map(lambda v: f"{v:.1%}")).to_string(index=False))
    worst = funnel.loc[1:].assign(率=funnel["环节转化率"].iloc[1:]).sort_values("率").iloc[0]
    print(f"  最大流失环节：{worst['阶段']} 环节，环节转化率 {worst['率']:.1%}")

    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    n = len(funnel)
    colors = ["#C9DAEA", "#B4D0E8", "#A8CCE8", "#D8DEA8", "#E8DFA8", LIGHT_YELLOW]
    for i, row in funnel.iterrows():
        bar = FancyBboxPatch(
            (0, n - i - 0.34), row["用户数"], 0.58,
            boxstyle="round,pad=0.004,rounding_size=0.14",
            facecolor=colors[i], edgecolor=colors[i], linewidth=1.0,
        )
        ax.add_patch(bar)
        ax.text(row["用户数"] + 2, n - i - 0.05, f"{row['用户数']} 人",
                va="center", fontsize=10, color=TEXT)
        label = str(row["阶段"])
        if i > 0:
            label += f"\n环节转化 {row['环节转化率']:.1%}"
            is_worst = row["环节转化率"] == funnel["环节转化率"].iloc[1:].min()
        else:
            is_worst = False
        ax.text(-6, n - i - 0.05, label, va="center", ha="right", fontsize=9.5,
                color="#C0504D" if is_worst else TEXT, linespacing=1.4)
    ax.set_xlim(-34, 120)
    ax.set_ylim(0, n)
    ax.axis("off")
    ax.set_title("销售漏斗：每一层的真实转化率", loc="left", fontsize=15,
                 fontweight="bold", color=TEXT, pad=18)
    save(fig, "11_销售漏斗.png")


def correlation_demo():
    print("\n" + "=" * 78)
    print("九、相关分析：投入与产出的线性关系")
    print("=" * 78)
    raw = clean_columns(pd.read_excel(D1 / "3-1相关分析-参考.xlsx", sheet_name="原始数据"))
    raw = raw.loc[:, ["y", "x"]].apply(pd.to_numeric, errors="coerce").dropna()
    r = raw["x"].corr(raw["y"])
    k, b = np.polyfit(raw["x"], raw["y"], 1)
    r2 = r ** 2
    print(f"  n={len(raw)}，相关系数 r={r:.4f}，R²={r2:.4f}，回归式 y={k:.4f}x+{b:.2f}")
    print("  业务口径：r 衡量线性相关强度，R² 表示 x 能解释 y 波动的比例，二者都不能证明因果。")

    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.scatter(raw["x"], raw["y"], s=78, color=LIGHT_BLUE, edgecolors=DEEP_BLUE,
               linewidths=1.3, alpha=0.9, zorder=3)
    xs = np.linspace(raw["x"].min(), raw["x"].max(), 200)
    ax.plot(xs, k * xs + b, color="#C0504D", linewidth=2.2, linestyle="--", zorder=2)
    ax.fill_between(xs, k * xs + b - raw["y"].std() * 0.5, k * xs + b + raw["y"].std() * 0.5,
                    color=LIGHT_YELLOW, alpha=0.25, zorder=1)
    ax.text(0.04, 0.95, f"r = {r:.3f}\nR² = {r2:.3f}\ny = {k:.3f}x + {b:.1f}",
            transform=ax.transAxes, va="top", fontsize=10.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#F2F7FC", edgecolor=LIGHT_BLUE))
    style_axes(ax, "相关分析：投入成本与产出收益", "投入成本", "产出收益")
    save(fig, "12_相关分析_回归散点.png")


# ============================================================================
# 五、Power BI 基本操作迁移
# ============================================================================
def etl_demo():
    print("\n" + "=" * 78)
    print("十、Power BI ETL -> pandas 数据清洗")
    print("=" * 78)
    etl_dir = PBI / "ETL练习文件"
    split = clean_columns(pd.read_excel(etl_dir / "ETL演示数据" / "拆分列示例数据.xlsx", sheet_name="Sheet2"))
    exploded = split.assign(订单号=split["订单号"].str.split(",")).explode("订单号")
    exploded["订单号"] = exploded["订单号"].str.strip()
    exploded[["省份", "城市", "区县"]] = exploded["收货地址"].str.extract(
        r"(?P<省份>.+?省)(?P<城市>.+?市)(?P<区县>.+?[区县])")
    exploded["数量"] = exploded["数量"].str.extract(r"(\d+)").astype(float).astype("Int64")
    print("  拆分列：订单号按逗号拆分、地址拆省市区、数量去单位")
    print(exploded[["订单号", "省份", "城市", "区县", "商品名称", "数量"]].head(6).to_string(index=False))

    raw = pd.read_excel(etl_dir / "ETL演示数据" / "数据清洗示例数据.xlsx", sheet_name="sheet1", header=1)
    raw = clean_columns(raw)
    raw = raw.dropna(how="all")
    before = {
        "缺失单元格占比": raw.isna().sum().sum() / raw.size,
        "重复订单占比": raw.duplicated(subset=["订单号"]).sum() / max(len(raw), 1),
        "金额异常占比": (pd.to_numeric(raw["实付金额"], errors="coerce").fillna(-1) <= 0).sum() / max(len(raw), 1),
    }
    cleaned = raw.copy()
    cleaned["实付金额"] = pd.to_numeric(cleaned["实付金额"], errors="coerce")
    cleaned["下单数量"] = pd.to_numeric(cleaned["下单数量"], errors="coerce").fillna(1).astype(int)
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    cleaned["下单时间"] = pd.to_datetime(
        cleaned["order_date"].dt.strftime("%Y-%m-%d") + " " + cleaned["order_time"].astype(str),
        errors="coerce")
    cleaned = cleaned.drop_duplicates(subset=["订单号"]).dropna(subset=["order_date"])
    cleaned["实付金额"] = cleaned["实付金额"].fillna(cleaned["实付金额"].median())
    cleaned["商品名称"] = cleaned["商品名称"].astype(str).str.strip()
    after = {
        "缺失单元格占比": cleaned.isna().sum().sum() / cleaned.size,
        "重复订单占比": cleaned.duplicated(subset=["订单号"]).sum() / max(len(cleaned), 1),
        "金额异常占比": (cleaned["实付金额"] <= 0).sum() / max(len(cleaned), 1),
    }
    print(f"\n  数据清洗：{len(raw)} 行 -> {len(cleaned)} 行")
    print(pd.DataFrame({"清洗前": pd.Series(before), "清洗后": pd.Series(after)})
          .map(lambda v: f"{v:.2%}").to_string())

    course = clean_columns(pd.read_excel(etl_dir / "合并演示数据" / "BI横向合并练习文件 .xlsx", sheet_name="课程表"))
    student = clean_columns(pd.read_excel(etl_dir / "合并演示数据" / "BI横向合并练习文件 .xlsx", sheet_name="学生表"))
    elective = clean_columns(pd.read_excel(etl_dir / "合并演示数据" / "BI横向合并练习文件 .xlsx", sheet_name="选课表"))
    wide = (elective.merge(student, on="学号", how="left")
            .merge(course, on="课程号", how="left"))
    wide["出生日期"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(wide["出生日期"], unit="D")
    print("\n  横向合并（学生 + 选课 + 课程）：")
    print(wide[["学号", "姓名", "课程名", "学分", "成绩", "出生日期"]].head(6).to_string(index=False))

    q1 = clean_columns(pd.read_excel(etl_dir / "合并演示数据" / "纵向合并演示-字段名称不一致.xlsx", sheet_name="第一季度"))
    q2 = clean_columns(pd.read_excel(etl_dir / "合并演示数据" / "纵向合并演示-字段名称不一致.xlsx", sheet_name="第二季度"))
    q2 = q2.rename(columns={"金额": "销售额"})
    q1["季度"] = "第一季度"
    q2["季度"] = "第二季度"
    vertical = pd.concat([q1, q2], ignore_index=True)
    vertical["门店"] = vertical["门店"].str.rstrip("1").replace({"A": "A", "B": "B"})
    print("\n  纵向合并（字段名不一致先对齐，再追加行）：")
    print(vertical.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))
    ax = axes[0]
    metrics = list(before.keys())
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, [before[m] for m in metrics], width=0.34, color=LIGHT_YELLOW,
           edgecolor=LIGHT_YELLOW, label="清洗前")
    ax.bar(x + 0.18, [after[m] for m in metrics], width=0.34, color=LIGHT_BLUE,
           edgecolor=LIGHT_BLUE, label="清洗后")
    for xi, m in enumerate(metrics):
        ax.text(xi - 0.18, before[m] + 0.004, f"{before[m]:.1%}", ha="center", fontsize=8.5, color=TEXT)
        ax.text(xi + 0.18, after[m] + 0.004, f"{after[m]:.1%}", ha="center", fontsize=8.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12)
    style_axes(ax, "ETL 前后数据质量对比", None, "比例")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.1%}")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(before.values()) * 1.35)

    ax = axes[1]
    q = vertical.groupby("季度")[["销售数量", "销售额"]].sum()
    ax.bar(q.index, q["销售额"], width=0.48, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for i, v in enumerate(q["销售额"].values):
        ax.text(i, v + 4, f"{v:,.0f}", ha="center", fontsize=9.5, color=TEXT)
    style_axes(ax, "纵向合并结果：季度销售额", None, "销售额")
    ax.set_ylim(0, q["销售额"].max() * 1.18)
    fig.suptitle("ETL 三件事：拆分、清洗、合并", fontsize=15, fontweight="bold", color=TEXT, y=1.0)
    save(fig, "13_ETL_拆分清洗合并.png")


def dax_demo():
    print("\n" + "=" * 78)
    print("十一、DAX 度量值 -> pandas 聚合")
    print("=" * 78)
    retail = clean_columns(pd.read_excel(PBI / "零售订单表.xlsx", sheet_name="sheet1"))
    retail["订单时间"] = pd.to_datetime(retail["订单时间"])
    retail["月份"] = retail["订单时间"].dt.to_period("M").astype(str)

    gmv = retail["实际支付"].sum()
    orders = retail["订单号"].nunique()
    aov = gmv / orders
    print(f"  总GMV = SUM(实际支付) = {gmv:,.0f}")
    print(f"  订单数 = DISTINCTCOUNT(订单号) = {orders:,}")
    print(f"  客单价 = GMV / 订单数 = {aov:,.1f}")

    monthly = retail.groupby("月份").agg(GMV=("实际支付", "sum"),
                                         订单数=("订单号", "nunique"))
    monthly["客单价"] = monthly["GMV"] / monthly["订单数"]
    monthly["环比"] = monthly["GMV"].pct_change()
    print("\n  月度 GMV 与环比（DAX 的 SUM + DATEADD 等价写法）：")
    print(monthly.assign(GMV=monthly["GMV"].map(lambda v: f"{v:,.0f}"),
                         客单价=monthly["客单价"].map(lambda v: f"{v:,.0f}"),
                         环比=monthly["环比"].map(lambda v: f"{v:+.1%}")).to_string())

    category = retail.groupby("商品类别")["实际支付"].sum().sort_values(ascending=False)
    category_share = category / gmv
    city = retail.groupby("城市")["实际支付"].sum().sort_values(ascending=False)
    print("\n  类别占比 = DIVIDE(SUM(实际支付), CALCULATE(SUM(实际支付), ALL(类别)))：")
    print(category_share.map(lambda v: f"{v:.1%}").to_string())
    print("\n  城市排名 TOP5（RANKX）：")
    print(city.head(5).map(lambda v: f"{v:,.0f}").to_string())

    daily = retail.groupby(retail["订单时间"].dt.date)["实际支付"].sum()
    daily.index = pd.to_datetime(daily.index)
    cum = daily.cumsum()

    fig = plt.figure(figsize=(17.0, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(monthly.index, monthly["GMV"] / 10000, width=0.55, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    ax1.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}万")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["环比"], color="#C0504D", marker="o", linewidth=2.0)
    ax2.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax2.tick_params(colors="#C0504D", length=0)
    ax2.spines["top"].set_visible(False)
    style_axes(ax1, "月度 GMV 与环比（柱 + 线）", None, "GMV")
    ax1.tick_params(axis="x", rotation=35)

    ax3 = fig.add_subplot(gs[0, 1])
    bars = ax3.barh(category.index[::-1], category.values[::-1] / 10000, height=0.55,
                    color=[LIGHT_BLUE if i % 2 == 0 else LIGHT_YELLOW
                           for i in range(len(category))], edgecolor="white")
    for b, v in zip(bars, category.values[::-1] / 10000):
        ax3.text(b.get_width() + category.max() / 10000 * 0.015,
                 b.get_y() + b.get_height() / 2, f"{v:,.0f}万", va="center", fontsize=9, color=TEXT)
    style_axes(ax3, "商品类别销售额（ALL + DIVIDE）", "GMV（万元）", None, grid_axis="x")
    ax3.yaxis.grid(False)
    ax3.set_xlim(0, category.max() / 10000 * 1.22)

    ax4 = fig.add_subplot(gs[1, 0])
    top = city.head(8)
    bars = ax4.bar(top.index, top.values / 10000, width=0.58, color=LIGHT_YELLOW,
                   edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, top.values / 10000):
        ax4.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=9, color=TEXT)
    style_axes(ax4, "城市 GMV TOP8（RANKX）", None, "GMV（万元）")
    ax4.set_ylim(0, top.max() / 10000 * 1.16)
    ax4.tick_params(axis="x", rotation=25)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(cum.index, cum.values / 10000, color=LIGHT_BLUE, alpha=0.6)
    ax5.plot(cum.index, cum.values / 10000, color=DEEP_BLUE, linewidth=2.2)
    style_axes(ax5, "累计 GMV（TOTALYTD / 累计求和）", None, "累计 GMV（万元）")
    ax5.tick_params(axis="x", rotation=30)
    fig.suptitle("DAX 度量值迁移：SUM / DISTINCTCOUNT / DIVIDE / RANKX / 时间智能",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)
    save(fig, "14_DAX度量值迁移_零售订单.png")


def data_model_demo():
    print("\n" + "=" * 78)
    print("十二、数据模型：把四张 Excel 表连成星型模型")
    print("=" * 78)
    path = PBI / "创建数据模型示例.xlsx"
    fact = clean_columns(pd.read_excel(path, sheet_name="销售业绩表"))
    customer = clean_columns(pd.read_excel(path, sheet_name="客户信息表"))
    city = clean_columns(pd.read_excel(path, sheet_name="城市信息表"))
    product = clean_columns(pd.read_excel(path, sheet_name="产品信息表"))

    model = (fact.merge(customer, on="客户ID", how="left")
             .merge(city, on="城市编号", how="left")
             .merge(product, on="产品编号", how="left"))
    revenue_col = [c for c in model.columns if "金额" in c][0]
    print(f"  事实表 {len(fact):,} 行，模型宽表 {len(model):,} 行，金额字段={revenue_col}")
    region = model.groupby("区域名")[revenue_col].sum().sort_values(ascending=False)
    print("\n  按区域汇总销售额（百万）：")
    print(region.map(lambda v: f"{v:,.1f}").to_string())
    print("\n  产品系列 × 客户规模：")
    print(pd.crosstab(model["产品系列"], model["企业规模"], values=model[revenue_col],
                      aggfunc="sum").fillna(0).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))
    ax = axes[0]
    bars = ax.bar(region.index, region.values, width=0.56, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, region.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=9.5, color=TEXT)
    style_axes(ax, "数据模型：区域销售额", "区域", "销售额（百万）")
    ax.set_ylim(0, region.max() * 1.16)

    ax = axes[1]
    cross = pd.crosstab(model["产品系列"], model["企业规模"], values=model[revenue_col], aggfunc="sum").fillna(0)
    bottom = np.zeros(len(cross))
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE, "#F5C46B"]
    for i, col in enumerate(cross.columns):
        ax.bar(cross.index.astype(str), cross[col], bottom=bottom, width=0.56,
               label=col, color=colors[i % len(colors)], edgecolor="white", linewidth=1.0)
        bottom += cross[col].to_numpy()
    style_axes(ax, "数据模型：产品系列 × 企业规模", "产品系列", "销售额（百万）")
    ax.legend(frameon=False, ncol=len(cross.columns), loc="upper left",
              bbox_to_anchor=(0, 1.0), fontsize=9)
    fig.suptitle("数据模型的价值：维度自由组合，度量只写一次", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "15_数据模型_区域与产品分析.png")


def pbi_case1_demo():
    print("\n" + "=" * 78)
    print("十三、Power BI 案例 1：分省销售分析")
    print("=" * 78)
    d = PBI / "案例1"
    orders = clean_columns(read_csv_smart(d / "订单表.csv"))
    detail = clean_columns(read_csv_smart(d / "订单表明细表.csv"))
    detail = detail.rename(columns={"产口ID": "产品ID"})
    goods = clean_columns(read_csv_smart(d / "商品表.csv"))
    category = clean_columns(read_csv_smart(d / "商品类型表.csv"))
    employee = clean_columns(read_csv_smart(d / "雇员表.csv"))

    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])
    orders["月份"] = orders["OrderDate"].dt.to_period("M").astype(str)
    sales = (detail.merge(orders, left_on="订单ID", right_on="OrderID", how="left")
             .merge(goods, left_on="产品ID", right_on="ProductID", how="left")
             .merge(category, on="CategoryID", how="left")
             .merge(employee, on="EmployeeID", how="left"))
    sales["毛利"] = sales["销售金额"] - sales["销售成本"]
    print(f"  明细 {len(detail):,} 行 -> 宽表 {len(sales):,} 行；"
          f"销售额 {sales['销售金额'].sum():,.0f}，毛利率 {sales['毛利'].sum() / sales['销售金额'].sum():.1%}")

    province = sales.groupby("ShipProvince")["销售金额"].sum().sort_values(ascending=False).head(10)
    cat = sales.groupby("CategoryName")["销售金额"].sum().sort_values(ascending=False)
    month = sales.groupby("月份")["销售金额"].sum()
    emp = sales.groupby("FullName")["销售金额"].sum().sort_values(ascending=False).head(8)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))
    ax = axes[0, 0]
    bars = ax.barh(province.index[::-1], province.values[::-1] / 10000, height=0.58,
                   color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, province.values[::-1] / 10000):
        ax.text(b.get_width() + province.max() / 10000 * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,.1f}", va="center", fontsize=9, color=TEXT)
    style_axes(ax, "省份销售额 TOP10", "销售额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, province.max() / 10000 * 1.2)

    ax = axes[0, 1]
    bars = ax.bar(cat.index, cat.values / 10000, width=0.58, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, cat.values / 10000):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=9, color=TEXT)
    style_axes(ax, "商品类别销售额", None, "销售额（万元）")
    ax.set_ylim(0, cat.max() / 10000 * 1.16)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    ax.plot(month.index, month.values / 10000, marker="o", linewidth=2.4, color=DEEP_BLUE,
            markerfacecolor="white", markeredgewidth=1.6)
    ax.fill_between(month.index, month.values / 10000, color=LIGHT_BLUE, alpha=0.45)
    style_axes(ax, "月度销售额趋势", None, "销售额（万元）")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1, 1]
    bars = ax.bar(emp.index, emp.values / 10000, width=0.58, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, emp.values / 10000):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                fontsize=9, color=TEXT)
    style_axes(ax, "雇员销售额 TOP8", None, "销售额（万元）")
    ax.set_ylim(0, emp.max() / 10000 * 1.16)
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Power BI 案例 1：分省销售分析看板（Python 复刻）", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "16_PowerBI案例1_分省销售看板.png")


def pbi_case2_demo():
    print("\n" + "=" * 78)
    print("十四、Power BI 案例 2：零售服装代理商销售看板")
    print("=" * 78)
    retail = clean_columns(pd.read_excel(PBI / "零售订单表.xlsx", sheet_name="sheet1"))
    retail["订单时间"] = pd.to_datetime(retail["订单时间"])
    retail["月份"] = retail["订单时间"].dt.to_period("M").astype(str)

    gmv = retail["实际支付"].sum()
    orders = retail["订单号"].nunique()
    aov = gmv / orders
    cities = retail["城市"].nunique()
    print(f"  GMV={gmv:,.0f}，订单={orders:,}，客单价={aov:,.1f}，覆盖城市={cities}")
    print("  业务解读：GMV 由订单数与客单价共同决定；零售看板要同时监控规模、结构与效率。")

    monthly = retail.groupby("月份")["实际支付"].sum()
    cat = retail.groupby("商品类别")["实际支付"].sum().sort_values(ascending=False)
    city = retail.groupby("城市")["实际支付"].sum().sort_values(ascending=False).head(10)
    qty = retail["数量"].sum()

    fig = plt.figure(figsize=(16.4, 9.8))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.55)
    kpis = [("GMV", f"{gmv / 10000:,.1f} 万", LIGHT_BLUE),
            ("订单数", f"{orders:,}", LIGHT_YELLOW),
            ("客单价", f"{aov:,.0f}", LIGHT_BLUE),
            ("销售件数", f"{qty:,}", LIGHT_YELLOW)]
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i * 1 + (i // 2): i * 1 + (i // 2) + 2])
        ax.set_axis_off()
        rounded_box(ax, 0.5, 0.58, 0.86, 0.52, "", color, alpha=0.9)
        ax.text(0.5, 0.70, label, ha="center", fontsize=10, color=TEXT, transform=ax.transAxes)
        ax.text(0.5, 0.40, value, ha="center", fontsize=20, fontweight="bold",
                color=TEXT, transform=ax.transAxes)

    ax1 = fig.add_subplot(gs[1, 0:3])
    ax1.plot(monthly.index, monthly.values / 10000, marker="o", linewidth=2.4,
             color=DEEP_BLUE, markerfacecolor="white", markeredgewidth=1.6)
    ax1.fill_between(monthly.index, monthly.values / 10000, color=LIGHT_BLUE, alpha=0.4)
    style_axes(ax1, "月度 GMV 趋势", None, "GMV（万元）")
    ax1.tick_params(axis="x", rotation=40)

    ax2 = fig.add_subplot(gs[1, 3:6])
    bars = ax2.barh(city.index[::-1], city.values[::-1] / 10000, height=0.58,
                    color=[LIGHT_BLUE if i % 2 == 0 else LIGHT_YELLOW for i in range(len(city))],
                    edgecolor="white")
    for b, v in zip(bars, city.values[::-1] / 10000):
        ax2.text(b.get_width() + city.max() / 10000 * 0.012, b.get_y() + b.get_height() / 2,
                 f"{v:.0f}", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax2, "城市 GMV TOP10", "GMV（万元）", None, grid_axis="x")
    ax2.yaxis.grid(False)
    ax2.set_xlim(0, city.max() / 10000 * 1.18)

    fig.suptitle("零售服装代理商销售看板：规模 · 结构 · 趋势", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    save(fig, "17_PowerBI案例2_零售销售看板.png")
    print("\n  类别结构：")
    print((cat / gmv).map(lambda v: f"{v:.1%}").to_string())


def main():
    set_style()
    excel_function_demo()
    chart_principle_demo()
    six_methods_demo()
    boston_matrix_demo()
    rfm_model_demo()
    loyalty_model_demo()
    cohort_demo()
    funnel_demo()
    correlation_demo()
    etl_demo()
    dax_demo()
    data_model_demo()
    pbi_case1_demo()
    pbi_case2_demo()
    print("\n第 2 章全部演示完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()