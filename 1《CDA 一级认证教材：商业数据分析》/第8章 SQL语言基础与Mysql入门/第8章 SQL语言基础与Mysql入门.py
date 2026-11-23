"""第 8 章 SQL 语言基础与 MySQL 入门：用 SQLite 复现教材 SQL 语法

教材的 SQL 基础语法文件覆盖：查询、去重、过滤、排序、计算列、CASE 重编码、
分组汇总、多表连接、子查询，以及 MySQL 8.0 的开窗函数。

本脚本用 Python 内置 SQLite 在内存中建库，逐条执行等价 SQL，并给出 pandas 对照写法。
数据来源：
- sale 表：教材统计制图用的年度 × 市场销售表；
- accounts / clients / disp / loans：第 1 章关系模型演示的银行数据。
"""

from pathlib import Path
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)
CH1 = BASE.parent / "第1章 数据分析思维"
SALE_FILE = CH1 / "sale.csv"
REL_DIR = CH1 / "关系模型演示-不建议用poweBI操作关系型数据模型"

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


def read_csv_smart(path, **kwargs):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def show(conn, title, sql):
    print("\n--- " + title)
    print("SQL:", " ".join(sql.strip().split()))
    df = pd.read_sql_query(sql, conn)
    print(df.head(12).to_string(index=False))
    return df


def build_database():
    conn = sqlite3.connect(":memory:")
    sale = read_csv_smart(SALE_FILE)
    sale.to_sql("sale", conn, index=False)

    tables = {}
    for name in ("accounts", "clients", "disp", "loans"):
        df = read_csv_smart(REL_DIR / f"{name}.csv")
        tables[name] = df
        df.to_sql(name, conn, index=False)
    conn.execute("CREATE INDEX idx_loans_account ON loans(account_id)")
    conn.execute("CREATE INDEX idx_disp_client ON disp(client_id)")
    conn.commit()
    print(f"  已建立内存库：sale {len(sale)} 行，"
          f"accounts {len(tables['accounts']):,} 行，clients {len(tables['clients']):,} 行，"
          f"disp {len(tables['disp']):,} 行，loans {len(tables['loans']):,} 行")
    return conn, sale


def sql_basics(conn):
    print("\n" + "=" * 78)
    print("一、单表查询：选择列 / 去重 / 过滤 / 排序")
    print("=" * 78)
    show(conn, "选择指定列", "SELECT year, market, sale, profit FROM sale")
    show(conn, "去重年份（DISTINCT）", "SELECT DISTINCT year FROM sale ORDER BY year")
    show(conn, "条件过滤（WHERE）", "SELECT * FROM sale WHERE year = 2012")
    show(conn, "按销售额降序（ORDER BY）", "SELECT market, sale, profit FROM sale ORDER BY sale DESC")


def sql_derived(conn):
    print("\n" + "=" * 78)
    print("二、表达式与重编码：新列 / CASE WHEN")
    print("=" * 78)
    show(conn, "创建利润率新列",
         "SELECT year, market, sale, profit, ROUND(1.0 * profit / sale, 4) AS rate FROM sale")
    show(conn, "用新列过滤",
         "SELECT year, market, profit / sale AS rate FROM sale WHERE 1.0 * profit / sale < 0.07")
    show(conn, "CASE 重编码：按市场划分区域",
         """SELECT year, market, sale,
                   CASE WHEN market IN ('东', '南') THEN '南方'
                        WHEN market IN ('西', '北') THEN '北方'
                        ELSE '未知' END AS region
            FROM sale""")
    show(conn, "CASE 分档：利润率等级",
         """SELECT year, market, profit / sale AS rate,
                   CASE WHEN profit / sale >= 0.09 THEN '高'
                        WHEN profit / sale >= 0.07 THEN '中'
                        ELSE '低' END AS level
            FROM sale ORDER BY rate DESC""")


def sql_aggregate(conn):
    print("\n" + "=" * 78)
    print("三、分组与汇总：GROUP BY / HAVING")
    print("=" * 78)
    df = show(conn, "按年份汇总销售额",
              """SELECT year, COUNT(*) AS 记录数, SUM(sale) AS 销售额,
                        AVG(profit) AS 平均利润, MAX(profit) AS 最大利润
                 FROM sale GROUP BY year ORDER BY year""")
    show(conn, "HAVING 过滤分组",
         """SELECT year, SUM(sale) AS total_sale
            FROM sale GROUP BY year HAVING SUM(sale) > 400000""")
    show(conn, "分组内排序：年份 + 销售额",
         "SELECT year, market, sale FROM sale ORDER BY year, sale DESC")

    print("\n  pandas 对照：")
    p = sale_pivot()
    print(p.to_string())


def sale_pivot():
    sale = read_csv_smart(SALE_FILE)
    return sale.pivot_table(index="year", columns="market", values="sale", aggfunc="sum")


def sql_join(conn):
    print("\n" + "=" * 78)
    print("四、多表连接：笛卡尔积 / 内连接 / 左连接")
    print("=" * 78)
    show(conn, "内连接：客户 -> 账户 -> 贷款（三表）",
         """SELECT c.client_id, d.type AS 账户类型, SUM(l.amount) AS 贷款总额
            FROM clients c
            JOIN disp d ON c.client_id = d.client_id
            JOIN loans l ON d.account_id = l.account_id
            GROUP BY c.client_id, d.type
            ORDER BY 贷款总额 DESC
            LIMIT 10""")
    show(conn, "左连接：保留没有贷款的账户",
         """SELECT a.account_id, a.frequency, COUNT(l.loan_id) AS 贷款笔数
            FROM accounts a
            LEFT JOIN loans l ON a.account_id = l.account_id
            GROUP BY a.account_id, a.frequency
            HAVING COUNT(l.loan_id) = 0
            LIMIT 10""")
    show(conn, "连接后计数膨胀的检查",
         """SELECT COUNT(*) AS joined_rows,
                   COUNT(DISTINCT c.client_id) AS clients
            FROM clients c
            JOIN disp d ON c.client_id = d.client_id""")


def sql_subquery(conn):
    print("\n" + "=" * 78)
    print("五、子查询：先算总体，再筛个体")
    print("=" * 78)
    show(conn, "贷款金额高于全表平均值的贷款",
         """SELECT loan_id, account_id, amount
            FROM loans
            WHERE amount > (SELECT AVG(amount) FROM loans)
            ORDER BY amount DESC LIMIT 10""")
    show(conn, "每个账户的最新一笔贷款（相关子查询）",
         """SELECT l.account_id, l.date, l.amount
            FROM loans l
            WHERE l.date = (SELECT MAX(l2.date) FROM loans l2 WHERE l2.account_id = l.account_id)
            LIMIT 10""")


def sql_window(conn):
    print("\n" + "=" * 78)
    print("六、MySQL 8.0 开窗函数：排名 / 累计 / 环比")
    print("=" * 78)
    rank = show(conn, "ROW_NUMBER / RANK / DENSE_RANK（每年内按销售额排名）",
                """SELECT year, market, sale,
                          ROW_NUMBER() OVER (PARTITION BY year ORDER BY sale DESC) AS rn,
                          RANK()       OVER (PARTITION BY year ORDER BY sale DESC) AS rnk,
                          DENSE_RANK() OVER (PARTITION BY year ORDER BY sale DESC) AS dense_rnk
                   FROM sale ORDER BY year, rn""")

    cum = show(conn, "累计销售额与累计占比（SUM OVER）",
               """SELECT year, market, sale,
                         SUM(sale) OVER (PARTITION BY year ORDER BY sale DESC
                                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_sale,
                         ROUND(1.0 * SUM(sale) OVER (PARTITION BY year ORDER BY sale DESC
                                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                               / SUM(sale) OVER (PARTITION BY year), 4) AS cum_share
                  FROM sale ORDER BY year, sale DESC""")

    lag = show(conn, "LAG：同一市场的年度环比",
               """SELECT year, market, sale,
                         LAG(sale) OVER (PARTITION BY market ORDER BY year) AS prev_sale,
                         ROUND(1.0 * sale / LAG(sale) OVER (PARTITION BY market ORDER BY year) - 1, 4) AS yoy
                  FROM sale ORDER BY market, year""")

    print("\n  pandas 对照：")
    sale = read_csv_smart(SALE_FILE).sort_values(["year", "sale"], ascending=[True, False])
    sale["rn"] = sale.groupby("year")["sale"].rank(ascending=False, method="first").astype(int)
    sale["rnk"] = sale.groupby("year")["sale"].rank(ascending=False, method="min").astype(int)
    sale["cum_share"] = sale.groupby("year")["sale"].cumsum() / sale.groupby("year")["sale"].transform("sum")
    print(sale.head(8).to_string(index=False))
    return rank, cum, lag


def sql_charts(sale):
    pivot = sale.pivot_table(index="year", columns="market", values="sale", aggfunc="sum")
    y2012 = pivot.loc[2012].sort_values(ascending=False)
    cum = y2012.cumsum() / y2012.sum()

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    ax = axes[0, 0]
    bars = ax.bar(y2012.index, y2012.values / 10000, width=0.56, color=LIGHT_BLUE,
                  edgecolor=LIGHT_BLUE)
    for i, (b, v) in enumerate(zip(bars, y2012.values / 10000)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\nNo.{i + 1}", ha="center",
                va="bottom", fontsize=9, color=TEXT)
    style_axes(ax, "ROW_NUMBER：2012 年各市场销售额排名", None, "销售额（万元）")
    ax.set_ylim(0, y2012.max() / 10000 * 1.2)

    ax = axes[0, 1]
    ax.bar(y2012.index, y2012.values / 10000, width=0.56, color=LIGHT_YELLOW,
           edgecolor=LIGHT_YELLOW)
    ax2 = ax.twinx()
    ax2.plot(y2012.index, cum.values, color=RED, marker="o", linewidth=2.2)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax2.tick_params(colors=RED, length=0)
    ax2.spines["top"].set_visible(False)
    for x, v in zip(y2012.index, cum.values):
        ax2.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8.5, color=RED)
    style_axes(ax, "SUM OVER：累计销售额占比", None, "销售额（万元）")

    ax = axes[1, 0]
    for market, color in zip(pivot.columns, [LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE, GREY]):
        yoy = pivot[market].pct_change()
        ax.plot(yoy.index, yoy.values, marker="o", linewidth=2.0, color=color, label=market)
    ax.axhline(0, color=GREY, linewidth=1.1)
    style_axes(ax, "LAG：各市场销售额年度环比", "年份", "环比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=9)
    ax.set_xticks(pivot.index)

    ax = axes[1, 1]
    bottom = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        ax.bar(pivot.index.astype(str), pivot[col] / 10000, bottom=bottom, width=0.56,
               label=col, color=[LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE, GREY][i],
               edgecolor="white", linewidth=1.0)
        bottom += pivot[col].to_numpy() / 10000
    style_axes(ax, "GROUP BY：年度 × 市场销售结构", "年份", "销售额（万元）")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.suptitle("SQL 查询结果可视化：SQL 负责取数，图表负责表达", fontsize=15,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "01_SQL查询结果可视化.png")


def main():
    set_style()
    conn, sale = build_database()
    sql_basics(conn)
    sql_derived(conn)
    sql_aggregate(conn)
    sql_join(conn)
    sql_subquery(conn)
    rank, cum, lag = sql_window(conn)
    sql_charts(sale)
    conn.close()
    print("\n第 8 章 SQL 演示完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()