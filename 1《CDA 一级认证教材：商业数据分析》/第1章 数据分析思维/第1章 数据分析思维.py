"""第 1 章 数据分析思维：关系模型与维度模型的 Python 复现

教材中用 Power BI 演示了两种建模思维：
1. 关系模型（第三范式）：数据拆成多张业务表，通过主外键层层关联，查询时需要多跳 JOIN；
2. 维度模型（星型模型）：以事实表为中心、维度表环绕，面向分析而不是面向业务操作。

本脚本把这两个演示完整迁移到 Python：
- 用 pandas 复现关系模型的多表关联与"连接膨胀"问题；
- 用 pandas 构建星型模型（事实表 + 5 张维度表）并完成品牌、城市、月度分析；
- 用 matplotlib 复刻教材图表，配色采用淡蓝 / 淡黄，图形间距与圆角保持统一。
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

REL_DIR = BASE / "关系模型演示-不建议用poweBI操作关系型数据模型"
DIM_FILE = BASE / "维度模型演示" / "商场销售数据表.xlsx"
SALE_FILE = BASE / "sale.csv"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
GREY = "#A6A6A6"
TEXT = "#404040"
GRID = "#D9D9D9"


def set_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#BFBFBF"


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFBFBF")
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-", alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def rounded_box(ax, x, y, w, h, text, fc, ec=None, fs=11, lw=1.6):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=lw, facecolor=fc, edgecolor=ec or fc, alpha=1.0,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            transform=ax.transAxes, linespacing=1.6)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图表] 已保存 {path.name}")


def read_csv_smart(path, **kwargs):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    return df


def load_relational():
    tables = {}
    for name in ("clients", "disp", "accounts", "loans", "card", "order", "district"):
        path = REL_DIR / f"{name}.csv"
        if not path.exists():
            continue
        tables[name] = read_csv_smart(path)
    return tables


def relational_model_demo():
    print("\n" + "=" * 78)
    print("一、关系模型：多张业务表 + 主外键，查询靠多跳 JOIN")
    print("=" * 78)

    t = load_relational()
    for name, df in t.items():
        print(f"  {name:<9} 行数={len(df):>7,}  字段={list(df.columns)}")

    clients, disp = t["clients"], t["disp"]
    accounts, loans = t["accounts"], t["loans"]

    client_account = clients.merge(disp, on="client_id", how="inner")
    print(f"\n  clients JOIN disp           -> {len(client_account):,} 行"
          f"（客户-账户关系，一人可有多个账户）")
    account_loan = client_account.merge(accounts, on="account_id", how="inner")
    print(f"  ... JOIN accounts           -> {len(account_loan):,} 行")
    loan_detail = account_loan.merge(loans, on="account_id", how="inner")
    print(f"  ... JOIN loans              -> {len(loan_detail):,} 行")

    fact_rows = len(loan_detail)
    unique_clients = loan_detail["client_id"].nunique()
    print(f"  结果粒度：{fact_rows:,} 条贷款记录，覆盖 {unique_clients:,} 位客户")
    print("  结论：关系模型字段不冗余、一致性最好，但每做一次分析都要重复这段数仓式的 JOIN，"
          "而且一对多连接会让行数膨胀，聚合口径稍不留神就会重复计数。")

    agg = (loan_detail.groupby("client_id")
           .agg(贷款笔数=("loan_id", "count"), 贷款总额=("amount", "sum"))
           .sort_values("贷款总额", ascending=False)
           .head(10))
    print("\n  贷款总额 TOP10 客户（多跳 JOIN 后的结果）：")
    print(agg.to_string())

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    labels = [f"客户 {int(i)}" for i in agg.index]
    vals = agg["贷款总额"].to_numpy() / 10000
    bars = ax.bar(labels, vals, width=0.62, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}", ha="center", va="bottom",
                fontsize=9, color=DEEP_BLUE)
    style_axes(ax, "关系模型多跳 JOIN：客户贷款总额 TOP10", "客户编号", "贷款总额（万元）")
    ax.set_ylim(0, vals.max() * 1.16)
    save(fig, "01_关系模型多跳JOIN客户贷款TOP10.png")

    return t, loan_detail


def model_shape_demo():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    ax1, ax2 = axes
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")

    ax1.text(0.5, 0.96, "关系模型（第三范式）", ha="center", fontsize=15, fontweight="bold", color=TEXT)
    rel_nodes = [
        (0.5, 0.80, "clients\n客户"),
        (0.5, 0.58, "disp\n客户-账户"),
        (0.5, 0.36, "accounts\n账户"),
        (0.5, 0.14, "loans / orders\n贷款 / 转账"),
    ]
    for i, (x, y, txt) in enumerate(rel_nodes):
        rounded_box(ax1, x, y, 0.46, 0.13, txt, LIGHT_YELLOW, fs=11)
        if i < len(rel_nodes) - 1:
            ax1.annotate("", xy=(x, y - 0.065), xytext=(x, y - 0.075),
                         arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
            ax1.plot([x, x], [y - 0.065, y - 0.075], color=GREY, lw=1.6)
    ax1.text(0.5, 0.015, "查一次指标 = 串 4 张表 JOIN", ha="center", fontsize=10, color=GREY)

    ax2.text(0.5, 0.96, "维度模型（星型模型）", ha="center", fontsize=15, fontweight="bold", color=TEXT)
    rounded_box(ax2, 0.5, 0.48, 0.34, 0.17, "事实表\n销售明细\n（度量 + 外键）", LIGHT_BLUE, fs=11)
    dims = [
        (0.50, 0.83, "日期维度"),
        (0.15, 0.63, "产品维度"),
        (0.85, 0.63, "品牌维度"),
        (0.15, 0.28, "城市维度"),
        (0.85, 0.28, "类别维度"),
    ]
    for x, y, txt in dims:
        rounded_box(ax2, x, y, 0.27, 0.13, txt, LIGHT_YELLOW, fs=10)
        ax2.plot([x, 0.5], [y, 0.48], color=GREY, lw=1.4, zorder=0)
        ax2.plot([x, 0.5], [y, 0.48], color=GREY, lw=1.4, zorder=0)
    ax2.scatter([0.5] * 5, [0.83, 0.63, 0.63, 0.28, 0.28], s=0, color="white")
    ax2.text(0.5, 0.015, "维度环绕事实，聚合路径最短", ha="center", fontsize=10, color=GREY)

    fig.suptitle("两种数据建模思维对比：为「记录业务」还是为「分析业务」", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.99)
    save(fig, "02_关系模型与星型模型对比.png")


def build_star_schema():
    print("\n" + "=" * 78)
    print("二、维度模型：从 Excel 多张表构建星型模型")
    print("=" * 78)

    xl = pd.ExcelFile(DIM_FILE)
    print("  教材工作表：", " | ".join(xl.sheet_names))

    dim_brand = clean_columns(xl.parse("品牌"))
    dim_category = clean_columns(xl.parse("类别"))
    dim_city = clean_columns(xl.parse("商铺城市"))
    dim_product = clean_columns(xl.parse("产品信息表"))
    fact_sales = clean_columns(xl.parse("产品销售数据表"))
    dim_date = clean_columns(xl.parse("日期表"))

    for name, df in [("dim_brand", dim_brand), ("dim_category", dim_category),
                     ("dim_city", dim_city), ("dim_product", dim_product),
                     ("fact_sales", fact_sales), ("dim_date", dim_date)]:
        print(f"  {name:<13} 行数={len(df):>7,}  列={list(df.columns)}")

    fact = fact_sales.rename(columns={"销售单价": "销售单价_明细", "销售额": "销售金额"})
    dim_product = dim_product.rename(columns={"销售单价": "销售单价_标准"})
    fact["产品编号"] = fact["产品编号"].astype(str).str.upper()
    dim_product["产品编号"] = dim_product["产品编号"].astype(str).str.upper()
    fact["订单日期"] = pd.to_datetime(fact["订单日期"])
    dim_date["日期"] = pd.to_datetime(dim_date["日期"])
    dim_city["城市编码"] = dim_city["城市编码"].astype(str)

    city_map = dim_city.set_index("商铺城市")["城市编码"]
    fact["城市编码"] = fact["商铺城市"].map(city_map)

    star = (fact
            .merge(dim_product[["产品编号", "采购单价", "销售单价_标准"]], on="产品编号",
                   how="left")
            .merge(dim_date[["日期", "年度", "季度", "月"]], left_on="订单日期",
                   right_on="日期", how="left")
            .merge(dim_brand, on="品牌名称", how="left")
            .merge(dim_category, on="产品类别", how="left"))

    star["销售成本"] = star["采购单价"].fillna(0) * star["销售数量"]
    star["毛利"] = star["销售金额"] - star["销售成本"]
    star["毛利率"] = np.where(star["销售金额"] > 0, star["毛利"] / star["销售金额"], 0)

    match_rate = star["采购单价"].notna().mean()
    print(f"\n  事实表 {len(star):,} 行，产品维度匹配率 {match_rate:.2%}")
    print(f"  时间范围：{star['订单日期'].min():%Y-%m-%d} ~ {star['订单日期'].max():%Y-%m-%d}")
    print(f"  销售总额：{star['销售金额'].sum():,.0f} 元，整体毛利率 {star['毛利'].sum() / star['销售金额'].sum():.2%}")

    return star, dim_product, dim_date, dim_city


def star_schema_charts(star):
    brand = (star.groupby("品牌名称")
             .agg(销售额=("销售金额", "sum"), 毛利=("毛利", "sum"))
             .sort_values("销售额", ascending=False))
    brand["毛利率"] = brand["毛利"] / brand["销售额"]

    print("\n  品牌维度汇总：")
    print(brand.assign(销售额=brand["销售额"].map(lambda v: f"{v:,.0f}"),
                       毛利=brand["毛利"].map(lambda v: f"{v:,.0f}"),
                       毛利率=brand["毛利率"].map(lambda v: f"{v:.2%}")).to_string())

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    vals = brand["销售额"] / 10000
    bars = ax.barh(brand.index[::-1], vals[::-1], height=0.58,
                   color=LIGHT_BLUE, edgecolor=LIGHT_BLUE, linewidth=1.2)
    for b, (name, row) in zip(bars, brand.iloc[::-1].iterrows()):
        ax.text(b.get_width() + vals.max() * 0.012, b.get_y() + b.get_height() / 2,
                f"{row['销售额']:,.0f} 万 · 毛利率 {row['毛利率']:.1%}",
                va="center", fontsize=9.5, color=TEXT)
    style_axes(ax, "维度模型分析：各品牌销售额与毛利率", "销售额（万元）", None)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlim(0, vals.max() * 1.42)
    save(fig, "03_品牌销售额与毛利率.png")

    city = (star.groupby("商铺城市")["销售金额"].sum()
            .sort_values(ascending=False).head(8) / 10000)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.bar(city.index, city.values, width=0.62, color=LIGHT_YELLOW,
                  edgecolor=LIGHT_YELLOW, linewidth=1.2)
    for b, v in zip(bars, city.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=9.5, color=TEXT)
    style_axes(ax, "维度模型分析：商铺城市销售额 TOP8", "商铺城市", "销售额（万元）")
    ax.set_ylim(0, city.max() * 1.15)
    save(fig, "04_城市销售额TOP8.png")

    month = star.groupby(["年度", "月"])["销售金额"].sum().reset_index()
    month["年月"] = pd.to_datetime(dict(year=month["年度"], month=month["月"], day=1))
    month = month.sort_values("年月")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.fill_between(month["年月"], month["销售金额"] / 10000, color=LIGHT_BLUE, alpha=0.55)
    ax.plot(month["年月"], month["销售金额"] / 10000, color=DEEP_BLUE, linewidth=2.4,
            marker="o", markersize=4.5, markerfacecolor="white", markeredgecolor=DEEP_BLUE)
    style_axes(ax, "维度模型分析：月度销售额趋势", None, "销售额（万元）")
    ax.margins(x=0.01)
    save(fig, "05_月度销售额趋势.png")

    quarter = (star.groupby(["季度", "品牌名称"])["销售金额"].sum()
               .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE]
    bottom = np.zeros(len(quarter))
    for i, col in enumerate(quarter.columns):
        ax.bar(quarter.index.astype(str), quarter[col] / 10000, bottom=bottom,
               width=0.58, label=col, color=colors[i % len(colors)],
               edgecolor="white", linewidth=1.0)
        bottom += quarter[col].to_numpy() / 10000
    style_axes(ax, "维度模型分析：季度 × 品牌销售额构成", "季度", "销售额（万元）")
    ax.legend(frameon=False, ncol=len(quarter.columns), loc="upper left",
              bbox_to_anchor=(0, 1.0), fontsize=10)
    save(fig, "06_季度品牌销售构成.png")


def wide_long_demo():
    print("\n" + "=" * 78)
    print("三、事实表的宽表 / 长表思维：同一份 sale.csv 的两种形态")
    print("=" * 78)
    sale = read_csv_smart(SALE_FILE)
    print(sale.head(8).to_string(index=False))

    wide = sale.pivot_table(index="year", columns="market", values="sale", aggfunc="sum")
    print("\n  宽表（pivot：年份 × 市场，便于横向比较）：")
    print(wide.to_string())
    print("\n  长表（melt 还原，便于透视与建模）：")
    print(wide.reset_index().melt(id_vars="year", var_name="market", value_name="sale").head(8).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = {"东": LIGHT_BLUE, "南": LIGHT_YELLOW, "西": MID_BLUE, "北": GREY}
    for market in wide.columns:
        ax.plot(wide.index, wide[market] / 10000, marker="o", markersize=5,
                linewidth=2.2, color=colors.get(market, DEEP_BLUE), label=market,
                markerfacecolor="white", markeredgewidth=1.6)
    style_axes(ax, "长表转宽表：各市场销售额年度走势", "年份", "销售额（万元）")
    ax.set_xticks(wide.index)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
    save(fig, "07_各市场销售额年度走势.png")


def main():
    set_style()
    relational_model_demo()
    model_shape_demo()
    star, _, _, _ = build_star_schema()
    star_schema_charts(star)
    wide_long_demo()
    print("\n第 1 章全部演示完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()