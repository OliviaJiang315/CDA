# -*- coding: utf-8 -*-
"""
第5章 数据可视化与描述性统计

对应教材：
    第5章-数据可视化/第5章 python数据可视化.ipynb
    第5章-数据可视化/第5章 描述性统计分析与绘图.ipynb

内容：
1. matplotlib 基础图形：线图、饼图、条形图/柱形图、直方图、散点图、箱线图；
2. seaborn 统计图形：带核密度估计的直方图、二元分布图、热图；
3. 连续变量的描述性统计：集中趋势、离散程度、偏度与峰度、相关系数。

说明：
    教材中图表以 Excel 为参照，这里全部用 Python 复刻；
    图形会依次弹出，关闭当前窗口后继续下一幅。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "第5章-数据可视化")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第5章 数据可视化")
os.makedirs(FIG_DIR, exist_ok=True)

_fig_no = [0]
_orig_show = plt.show


def _save_figure():
    """保存当前图形到本章图表目录。"""
    fig = plt.gcf()
    name = ""
    if fig._suptitle is not None:
        name = fig._suptitle.get_text()
    if not name:
        for ax in fig.axes:
            if ax.get_title():
                name = ax.get_title()
                break
    _fig_no[0] += 1
    safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "图"
    path = os.path.join(FIG_DIR, "%02d_%s.png" % (_fig_no[0], safe[:40]))
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    print("已保存图表：", path)


def _show_and_save(*args, **kwargs):
    _save_figure()
    _orig_show(*args, **kwargs)
    plt.close("all")


plt.show = _show_and_save

# 中文显示与负号显示设置
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def data_path(name):
    return os.path.join(DATA_DIR, name)


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ======================================================================
# 第一部分  matplotlib 基础图形
# ======================================================================
title("2.1.1 线图：股价走势")

df = pd.read_excel(data_path("2017股价.xlsx"))
print(df)

plt.figure(figsize=(10, 4))
plt.plot(range(15), df.loc[:, "收盘价"], "-*", label="收盘价")
plt.plot(range(15), df.loc[:, "开盘价"], "-o", label="开盘价")
plt.plot(range(15), df.loc[:, "最高价"], "-v", label="最高价")
plt.plot(range(15), df.loc[:, "最低价"], "-^", label="最低价")
plt.title("股价走势图")
plt.xlabel("时间")
plt.ylabel("股价")
plt.xticks(range(15), df["日期"].tolist(), rotation=30)
plt.legend(["收盘价", "开盘价", "最高价", "最低价"])
plt.tight_layout()
plt.show()


title("2.1.2 饼图：方案投票占比")

np.random.seed(1)
df_vote = pd.DataFrame(
    {
        "方案": ["plan A", "plan B", "plan C", "plan D", "plan E"],
        "投票人数": np.random.randint(10, 100, 5),
    }
)
print(df_vote)

plt.figure(figsize=(6, 6))
plt.pie(
    df_vote["投票人数"],          # 饼图数据源
    labels=df_vote["方案"],       # 每个扇形对应的标签
    autopct="%.2f%%",             # 百分比精度
    textprops={"fontsize": 13},   # 饼图文字大小
)
plt.title("方案投票占比饼图", fontsize=14)
plt.show()


title("2.1.3 条形图与柱形图：各方案投票人数")

# 竖向柱形图
plt.figure(figsize=(8, 4))
plt.bar(df_vote["方案"], df_vote["投票人数"], width=0.6)
plt.title("各方案投票人数柱形图", fontsize=14)
plt.ylabel("投票人数", fontsize=13)
plt.xlabel("方案", fontsize=13)
plt.tight_layout()
plt.show()

# 横向条形图（先按人数排序，阅读更直观）
df_sorted = df_vote.sort_values("投票人数")
plt.figure(figsize=(8, 4))
plt.barh(df_sorted["方案"], df_sorted["投票人数"], height=0.6)
plt.xlabel("投票人数", fontsize=13)
plt.ylabel("方案", fontsize=13)
plt.title("各方案投票人数条形图", fontsize=14)
plt.tight_layout()
plt.show()


title("2.1.4 直方图：成绩分布")

np.random.seed(1)
data = np.random.normal(loc=80, scale=20, size=200).astype(int)
bins = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
plt.hist(data, bins)
plt.xticks(bins)
plt.xlabel("分数")
plt.ylabel("频数")
plt.title("成绩分布直方图")
plt.show()


title("2.1.5 散点图：降水量与产量")

df_yield = pd.DataFrame(
    {
        "产量": [2200, 3300, 4400, 5500, 5700, 7300, 8000],
        "降水量": [22, 39, 55, 66, 105, 100, 122],
    }
)
print(df_yield)

plt.figure(figsize=(6, 4))
plt.scatter(df_yield["降水量"], df_yield["产量"])
plt.xlabel("降水量")
plt.ylabel("产量")
plt.title("降水量与产量散点图")
plt.show()


title("2.1.6 箱线图：三科成绩")

np.random.seed(1)
df_score = pd.DataFrame(
    {
        "语文": np.random.randint(60, 150, 10),
        "英语": np.random.randint(40, 150, 10),
        "数学": np.random.randint(20, 150, 10),
    }
)
print(df_score)

plt.figure(figsize=(8, 6))
plt.boxplot(
    [df_score["语文"], df_score["英语"], df_score["数学"]],
    labels=["语文", "英语", "数学"],
)
plt.title("三科成绩箱线图")
plt.show()


# ======================================================================
# 第二部分  seaborn 统计图形
# ======================================================================
title("2.2.1 带核密度估计的直方图")

np.random.seed(1)
data = np.random.normal(loc=80, scale=20, size=200).astype(int)
sns.displot(
    data,
    bins=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    kde=True,  # kde=True 显示核密度估计曲线
)
plt.title("带核密度曲线的成绩分布")
plt.show()


title("2.2.2 二元分布图")

np.random.seed(0)
mean, cov = [0, 1], [(1, 0.5), (0.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df_biv = pd.DataFrame(data, columns=["x", "y"])
print(df_biv.head())

# 联合分布：散点 + 边缘直方图
g = sns.jointplot(x="x", y="y", data=df_biv)
g.figure.suptitle("二元联合分布散点图")
plt.show()

# 关系图
g = sns.relplot(x="x", y="y", data=df_biv)
g.figure.suptitle("二元关系图")
plt.show()

# 联合分布：二维核密度
g = sns.jointplot(x="x", y="y", data=df_biv, kind="kde")
g.figure.suptitle("二元联合分布核密度图")
plt.show()

# 二维密度图
g = sns.displot(data=df_biv, x="x", y="y", kind="kde")
g.figure.suptitle("二元核密度估计图")
plt.show()


title("2.2.3 热图：相关系数/矩阵可视化")

np.random.seed(0)
df0 = pd.DataFrame(
    np.random.uniform(-1, 1, (7, 7)),
    index=list("abcdefg"),
    columns=list("ABCDEFG"),
)
print(df0)

plt.figure(figsize=(8, 6))
sns.set_theme(
    font_scale=1.5,
    rc={"font.sans-serif": ["SimHei", "Microsoft YaHei"], "axes.unicode_minus": False},
)
sns.heatmap(df0, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("随机矩阵热图")
plt.show()
sns.set_theme(
    font_scale=1.0,
    rc={"font.sans-serif": ["SimHei", "Microsoft YaHei"], "axes.unicode_minus": False},
)


# ======================================================================
# 第三部分  描述性统计分析与绘图
# ======================================================================
title("3.1.3 连续变量的分布与集中趋势")

data = pd.read_csv(data_path("sndHsPr.csv"))
print(data.head())
print("样本量：", len(data))

print("均价、中位数、标准差：\n", data.price.agg(["mean", "median", "std"]))
print("四分位数：\n", data.price.quantile([0.25, 0.5, 0.75]))

plt.figure(figsize=(7, 4))
data.price.hist(bins=20)
plt.title("房屋价格分布")
plt.xlabel("price")
plt.ylabel("频数")
plt.show()


title("3.1.4 连续变量的离散程度")

print("极差：", data.price.max() - data.price.min())
print("方差：", data.price.var())
print("标准差：", data.price.std())


title("3.1.5 数据分布的对称与高矮（偏度、峰度）")

normal = pd.Series(np.random.randn(1000), name="normal")
print("随机正态序列偏度：", normal.skew())
print("随机正态序列峰度：", normal.kurtosis())
print("房价偏度：", data.price.skew())
print("房价峰度：", data.price.kurtosis())


title("3.1.5 相关系数：AREA 与 price")

print("pearson：\n", data[["AREA", "price"]].corr("pearson"))
print("spearman：\n", data[["AREA", "price"]].corr("spearman"))
print("kendall：\n", data[["AREA", "price"]].corr("kendall"))

# 相关系数热图，便于直观比较
plt.figure(figsize=(6, 4))
sns.heatmap(
    data[["AREA", "price"]].corr("pearson"),
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
)
plt.title("AREA 与 price 的 Pearson 相关系数")
plt.show()

print("\n第5章全部演示执行完毕。")