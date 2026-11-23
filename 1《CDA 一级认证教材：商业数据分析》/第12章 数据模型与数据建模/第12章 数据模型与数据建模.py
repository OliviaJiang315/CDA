"""第 12 章 数据模型与数据建模：从 ODS 到 ADS 的分层建模

教材参考阿里云数据中台零售数据模型白皮书，核心是：
- 维度建模四步：选择业务过程 -> 声明粒度 -> 确认维度 -> 确认事实；
- 数仓分层：ODS（原始）-> DWD（明细）-> DWS（汇总）-> ADS（应用）；
- 维度表演进：缓慢变化维 SCD2（拉链表）。

本脚本用「商场销售数据表」完整走一遍：
6 张 Excel 原始表 -> 清洗后的明细事实 -> 品牌月度汇总 -> 品牌经营报表，
并实现通用的 SCD2 拉链生成逻辑（本数据价格稳定，另附示例调价事件演示版本演进）。
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
CH1 = BASE.parent / "第1章 数据分析思维"
DIM_FILE = CH1 / "维度模型演示" / "商场销售数据表.xlsx"

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


def rounded_box(ax, x, y, w, h, text, fc, fs=9.5):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.028",
                         facecolor=fc, edgecolor=fc, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            zorder=4, linespacing=1.55)


# ============================================================================
# 一、分层建模
# ============================================================================
def build_layers():
    print("\n" + "=" * 78)
    print("一、数仓分层：ODS -> DWD -> DWS -> ADS")
    print("=" * 78)
    xl = pd.ExcelFile(DIM_FILE)
    ods = {sheet: clean_columns(xl.parse(sheet)) for sheet in xl.sheet_names}
    print("  ODS 原始层：")
    for name, df in ods.items():
        print(f"    {name:<8} {len(df):>7,} 行 × {len(df.columns)} 列")

    fact = ods["产品销售数据表"].rename(columns={"销售单价": "成交单价", "销售额": "销售金额"})
    dim_product = ods["产品信息表"]
    dim_date = ods["日期表"]
    dim_city = ods["商铺城市"]
    dim_brand = ods["品牌"]
    dim_category = ods["类别"]

    fact["产品编号"] = fact["产品编号"].astype(str).str.upper()
    dim_product["产品编号"] = dim_product["产品编号"].astype(str).str.upper()
    fact["订单日期"] = pd.to_datetime(fact["订单日期"])
    dim_date["日期"] = pd.to_datetime(dim_date["日期"])
    for col in ("销售数量", "成交单价", "销售金额"):
        fact[col] = pd.to_numeric(fact[col], errors="coerce")

    dwd = (fact
           .merge(dim_product[["产品编号", "采购单价"]], on="产品编号", how="left")
           .merge(dim_date[["日期", "年度", "季度", "月"]], left_on="订单日期",
                  right_on="日期", how="left")
           .merge(dim_brand, on="品牌名称", how="left")
           .merge(dim_category, on="产品类别", how="left"))
    dwd["销售成本"] = dwd["采购单价"].fillna(0) * dwd["销售数量"]
    dwd["毛利"] = dwd["销售金额"] - dwd["销售成本"]
    dwd["毛利率"] = np.where(dwd["销售金额"] > 0, dwd["毛利"] / dwd["销售金额"], 0)
    dwd = dwd.drop_duplicates()
    dwd["订单月份"] = dwd["订单日期"].dt.to_period("M").astype(str)

    print(f"\n  DWD 明细层：{len(dwd):,} 行 × {len(dwd.columns)} 列；"
          f"重复行 {int(dwd.duplicated().sum())}，"
          f"产品维度匹配率 {(dwd['采购单价'].notna()).mean():.1%}")

    dws = (dwd.groupby(["品牌名称", "订单月份"])
           .agg(销售额=("销售金额", "sum"), 销量=("销售数量", "sum"),
                订单数=("销售金额", "count"), 毛利=("毛利", "sum"))
           .reset_index())
    dws["毛利率"] = dws["毛利"] / dws["销售额"]
    print(f"  DWS 汇总层：{len(dws):,} 行（品牌 × 月）")

    ads = (dws.groupby("品牌名称")
           .agg(销售额=("销售额", "sum"), 销量=("销量", "sum"),
                毛利=("毛利", "sum"), 月份数=("订单月份", "nunique"))
           .reset_index())
    ads["市场份额"] = ads["销售额"] / ads["销售额"].sum()
    ads["毛利率"] = ads["毛利"] / ads["销售额"]
    ads["排名"] = ads["销售额"].rank(ascending=False, method="min").astype(int)
    ads = ads.sort_values("排名")
    print(f"  ADS 应用层：{len(ads)} 行品牌经营汇总")
    print("\n  ADS 品牌经营报表：")
    print(ads.assign(销售额=ads["销售额"].map(lambda v: f"{v:,.0f}"),
                     销量=ads["销量"].map(lambda v: f"{v:,.0f}"),
                     毛利=ads["毛利"].map(lambda v: f"{v:,.0f}"),
                     市场份额=ads["市场份额"].map(lambda v: f"{v:.1%}"),
                     毛利率=ads["毛利率"].map(lambda v: f"{v:.1%}")).to_string(index=False))

    print("\n  维度建模四步：")
    print("  1) 选择业务过程：商场产品零售销售")
    print("  2) 声明粒度：一行 = 一个订单日期 × 一个产品 的销售记录")
    print("  3) 确认维度：日期、产品、品牌、类别、城市")
    print("  4) 确认事实：销售数量、销售金额、成本、毛利（可加性度量）")

    ads.to_csv(BASE / "ADS_品牌经营汇总.csv", index=False, encoding="utf-8-sig")
    return ods, dwd, dws, ads, dim_product, dim_date, dim_city, dim_brand, dim_category


def build_scd2():
    print("\n" + "=" * 78)
    print("二、缓慢变化维 SCD2：价格变化如何保留历史")
    print("=" * 78)
    events = pd.DataFrame([
        {"产品编号": "A001", "生效日期": "2018-01-01", "销售单价": 4669},
        {"产品编号": "A001", "生效日期": "2020-01-01", "销售单价": 4899},
        {"产品编号": "A001", "生效日期": "2021-01-01", "销售单价": 5099},
        {"产品编号": "B001", "生效日期": "2018-01-01", "销售单价": 4199},
        {"产品编号": "B001", "生效日期": "2020-07-01", "销售单价": 4399},
    ])
    events["生效日期"] = pd.to_datetime(events["生效日期"])

    rows = []
    for pid, part in events.sort_values(["产品编号", "生效日期"]).groupby("产品编号"):
        part = part.reset_index(drop=True)
        for i, row in part.iterrows():
            start = row["生效日期"]
            end = part.loc[i + 1, "生效日期"] - pd.Timedelta(days=1) if i + 1 < len(part) else pd.Timestamp("2022-12-31")
            rows.append({
                "产品编号": pid, "版本号": i + 1, "销售单价": row["销售单价"],
                "生效日期": start, "失效日期": end,
                "是否当前版本": i + 1 == len(part),
            })
    scd = pd.DataFrame(rows)
    print("  本数据源价格在观测期内稳定（每产品仅 1 个价格），以下用示例调价事件演示 SCD2：")
    print(scd.assign(生效日期=scd["生效日期"].dt.strftime("%Y-%m-%d"),
                     失效日期=scd["失效日期"].dt.strftime("%Y-%m-%d")).to_string(index=False))
    print("  拉链表用法：事实表按「订单日期 BETWEEN 生效日期 AND 失效日期」关联维度，"
          "即可还原当时价格，避免历史报表被新价格篡改。")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for pid, color in [("A001", DEEP_BLUE), ("B001", MID_BLUE)]:
        part = scd[scd["产品编号"] == pid]
        xs = [part["生效日期"].iloc[0]]
        ys = [part["销售单价"].iloc[0]]
        for _, row in part.iterrows():
            xs.extend([row["生效日期"], row["失效日期"]])
            ys.extend([row["销售单价"], row["销售单价"]])
        ax.step(xs, ys, where="post", color=color, linewidth=2.4, label=pid)
        ax.scatter(part["生效日期"], part["销售单价"], color=color, s=42, zorder=4)
    style_axes(ax, "SCD2：价格版本随生效日期阶梯变化", "日期", "销售单价（元）")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(pd.Timestamp("2017-10-01"), pd.Timestamp("2021-12-31"))

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    part = scd[scd["产品编号"] == "A001"].reset_index(drop=True)
    for i, row in part.iterrows():
        y = 0.75 - i * 0.22
        rounded_box(ax, 0.5, y, 0.8, 0.13,
                    f"版本 {row['版本号']}  {row['销售单价']} 元\n"
                    f"{row['生效日期']:%Y-%m-%d} ~ {row['失效日期']:%Y-%m-%d}"
                    + ("（当前）" if row["是否当前版本"] else ""),
                    LIGHT_BLUE if row["是否当前版本"] else LIGHT_YELLOW, fs=9)
    ax.set_title("A001 拉链版本", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)
    fig.suptitle("SCD2：维度不是一张静态表，而是一段可追溯的历史", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.02)
    save(fig, "03_SCD2价格拉链.png")
    return scd


# ============================================================================
# 二、架构与结果图
# ============================================================================
def draw_architecture(ods, dwd, dws, ads, dims):
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [
        ("ODS 原始层", f"{len(ods['产品销售数据表']):,} 行销售明细\n6 张原始 Excel 表", LIGHT_YELLOW),
        ("DWD 明细层", f"{len(dwd):,} 行清洗明细\n维度关联 + 金额校验", LIGHT_BLUE),
        ("DWS 汇总层", f"{len(dws):,} 行品牌×月\n销售额/销量/毛利", MID_BLUE),
        ("ADS 应用层", f"{len(ads)} 行品牌报表\n排名/份额/毛利率", "#DCE9F6"),
    ]
    ys = [0.78, 0.56, 0.34, 0.12]
    for (name, body, color), y in zip(layers, ys):
        rounded_box(ax, 0.5, y, 0.56, 0.15, f"{name}\n{body}", color, fs=10)
        if y > 0.15:
            ax.annotate("", xy=(0.5, y - 0.085), xytext=(0.5, y - 0.065),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))
    ax.text(0.87, 0.78, "贴源", fontsize=9, color=GREY)
    ax.text(0.87, 0.56, "清洗/标准化", fontsize=9, color=GREY)
    ax.text(0.87, 0.34, "聚合", fontsize=9, color=GREY)
    ax.text(0.87, 0.12, "面向应用", fontsize=9, color=GREY)
    ax.set_title("数仓分层架构", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    dims_show = [("日期维度", 0.5, 0.86), ("产品维度", 0.16, 0.66),
                 ("品牌维度", 0.84, 0.66), ("城市维度", 0.16, 0.26),
                 ("类别维度", 0.84, 0.26)]
    rounded_box(ax, 0.5, 0.47, 0.34, 0.18, "事实表\n销售明细\n粒度：日 × 产品", LIGHT_BLUE, fs=9.5)
    for name, x, y in dims_show:
        rounded_box(ax, x, y, 0.26, 0.13, name, LIGHT_YELLOW, fs=9)
        ax.plot([x, 0.5], [y, 0.47], color=GREY, linewidth=1.3, zorder=0)
    ax.set_title("星型模型（5 个维度 + 1 个事实）", loc="left", fontsize=13.5,
                 fontweight="bold", color=TEXT, pad=12)
    fig.suptitle("零售数据模型：先分层，再星型，最后才是报表", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "01_数仓分层与星型模型.png")


def draw_results(dws, ads):
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8))
    ax = axes[0]
    dws["月份"] = pd.to_datetime(dws["订单月份"] + "-01")
    for brand, color in zip(ads["品牌名称"], [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE]):
        part = dws[dws["品牌名称"] == brand].sort_values("月份")
        ax.plot(part["月份"], part["销售额"] / 10000, marker="o", markersize=4,
                linewidth=2.2, color=color, label=brand)
    style_axes(ax, "DWS：品牌月度销售额", None, "销售额（万元）")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    order = ads.sort_values("销售额")
    bars = ax.barh(order["品牌名称"], order["销售额"] / 10000, height=0.58,
                   color=[LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE][:len(order)], edgecolor="white")
    for b, row in zip(bars, order.itertuples()):
        ax.text(b.get_width() + ads["销售额"].max() / 10000 * 0.015,
                b.get_y() + b.get_height() / 2,
                f"{row.销售额 / 10000:,.0f} 万 · {row.市场份额:.0%} · No.{row.排名}",
                va="center", fontsize=9, color=TEXT)
    style_axes(ax, "ADS：品牌经营排名", "销售额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, ads["销售额"].max() / 10000 * 1.45)
    fig.suptitle("分层建模的产出：明细可追溯，汇方可复用，报表可解释", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "02_DWS与ADS结果.png")


def main():
    set_style()
    ods, dwd, dws, ads, dim_product, dim_date, dim_city, dim_brand, dim_category = build_layers()
    scd = build_scd2()
    draw_architecture(ods, dwd, dws, ads,
                      (dim_product, dim_date, dim_city, dim_brand, dim_category))
    draw_results(dws, ads)
    print("\n第 12 章数据建模完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()