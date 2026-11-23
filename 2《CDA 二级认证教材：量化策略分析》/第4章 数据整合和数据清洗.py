# -*- coding: utf-8 -*-
"""
第4章 数据整合和数据清洗（pandas）

对应教材：第4章- 数据处理(pandas)/第4章  数据整合和数据清洗.ipynb

本章围绕 pandas 的两大核心工作展开：
1. 数据整合：行列操作、条件查询、横向连接、纵向合并、排序、分组汇总、拆分表；
2. 数据清洗：重复值、缺失值、异常值的识别与处理，以及数值替换、条件赋值、分箱离散化。

运行方式：
    python "第4章 数据整合和数据清洗.py"
数据目录：
    第4章- 数据处理(pandas)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 中文字体设置（pandas 的 hist 也复用这套 rcParams）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "第4章- 数据处理(pandas)")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第4章 数据整合和数据清洗")
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


def data_path(*parts):
    """按章节数据目录拼接路径，保证脚本可在任意工作目录下运行。"""
    return os.path.join(DATA_DIR, *parts)


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1.1.1 读取数据
# ----------------------------------------------------------------------
title("1.1.1 读取数据")

csv = pd.read_csv(data_path("sample.csv"))
print(csv)

# 只读取 id、name 两列，且只读取前两行
csv_part = pd.read_csv(
    data_path("sample.csv"),
    usecols=["id", "name"],
    nrows=2,
)
print(csv_part)

# 分块读取大文件：每次读取 900 行
csvs = pd.read_csv(data_path("data", "big.csv"), chunksize=900)
for chunk in csvs:
    print("分块大小：", chunk.shape)

# 将所有分块纵向拼接成一个完整数据框
chunks = pd.read_csv(data_path("data", "big.csv"), chunksize=900)
dat = pd.concat(chunks, ignore_index=True)
print("拼接后形状：", dat.shape)

# 将 99999 识别为缺失值
csv_na = pd.read_csv(data_path("sample.csv"), na_values="99999")
print(csv_na)

# 指定文件编码
csv_utf8 = pd.read_csv(data_path("sample.csv"), encoding="utf-8")
print(csv_utf8)

# 写出文件：不写行索引
csv_utf8.to_csv(data_path("write.csv"), encoding="utf-8", index=False)


# ----------------------------------------------------------------------
# 1.2.1 行列操作
# ----------------------------------------------------------------------
title("1.2.1 行列操作")

np.random.seed(0)  # 固定随机数种子，保证每次运行结果一致
sample = pd.DataFrame(np.random.randn(4, 5), columns=list("abcde"))
print(sample)

print("取单列（返回 Series）：\n", sample["a"])
print("按位置取第一列：\n", sample.iloc[:, 0])
print("按标签取 a 列：\n", sample.loc[:, "a"])

# 用列表取列，返回 DataFrame
print("取单列（返回 DataFrame）：\n", sample[["a"]])

# 取第 0~1 行、a~b 列
print("loc 标签切片：\n", sample.loc[0:1, "a":"b"])
print("iloc 位置切片：\n", sample.iloc[0:2, 0:2])

# 新增列
sample["new_col"] = sample["a"] - sample["b"]
print(sample)

# assign 可以一次性新增多列，且不修改原数据
print(
    sample.assign(
        new_col2=sample["a"] - sample["b"],
        new_col3=sample["a"] + sample["b"],
    )
)

# 删除列
print("删除 a 列：\n", sample.drop(columns="a"))
print("删除 a、b 两列：\n", sample.drop(columns=["a", "b"]))


# ----------------------------------------------------------------------
# 1.2.2 条件查询
# ----------------------------------------------------------------------
title("1.2.2 条件查询")

sample = pd.DataFrame(
    {
        "group": [1, 1, 1, 2, 1, 2],
        "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"],
        "score": [98, 78, 87, 77, 65, 67],
    }
)
print(sample)

print("布尔序列：\n", sample.score > 70)
print("分数大于 70：\n", sample[sample.score > 70])
print("分数大于 70 且组别为 1：\n", sample[(sample.score > 70) & (sample.group == 1)])
print("组别不为 1：\n", sample[~(sample.group == 1)])
print("组别为 1 或 2：\n", sample[(sample.group == 2) | (sample.group == 1)])

# query 方式，语法更接近 SQL
print("query 分数大于 90：\n", sample.query("score > 90"))
print("query 组别为 1 或 2：\n", sample.query("(group == 2) | (group == 1)"))

# between 做区间判断（pandas 1.3 起使用 inclusive='both' 等字符串）
print(
    "分数在 70~80 之间：\n",
    sample[sample["score"].between(70, 80, inclusive="both")],
)

# isin 判断取值是否在给定集合中
print("姓名为 Bob 或 Lindy：\n", sample[sample["name"].isin(["Bob", "Lindy"])])

# 字符串正则匹配：姓名中包含 M
print("姓名含 M：\n", sample[sample["name"].str.contains("[M]+")])

# 注意：pandas 2.x 的 .loc 行切片按标签解释，RangeIndex 与位置一致
print("loc 标签切片 0~1 行：\n", sample.loc[0:1])


# ----------------------------------------------------------------------
# 1.2.3 横向连接
# ----------------------------------------------------------------------
title("1.2.3 横向连接")

df1 = pd.DataFrame({"col1": ["a", "b", "c"], "id": [1, 2, 3]})
df2 = pd.DataFrame({"col2": ["d", "e"], "id": [4, 3]})
print(df1)
print(df2)

print("内连接：\n", df1.merge(df2, how="inner", on="id"))
print("内连接（左右键名相同）：\n", df1.merge(df2, how="inner", left_on="id", right_on="id"))
print("左连接：\n", df1.merge(df2, how="left", on="id"))
print("右连接：\n", df1.merge(df2, how="right", on="id"))
print("外连接：\n", df1.merge(df2, how="outer", on="id"))

# 按索引对齐的横向拼接
df1 = pd.DataFrame({"id1": [1, 2, 3], "col1": ["a", "b", "c"]}, index=[1, 2, 3])
df2 = pd.DataFrame({"id2": [1, 2, 3], "col2": ["aa", "bb", "cc"]}, index=[1, 3, 2])
print(df1)
print(df2)
print("concat 按索引对齐：\n", pd.concat([df1, df2], axis=1))
print("join 按索引对齐：\n", df1.join(df2))


# ----------------------------------------------------------------------
# 1.2.4 纵向合并
# ----------------------------------------------------------------------
title("1.2.4 纵向合并")

df1 = pd.DataFrame({"id": [1, 1, 1, 2, 3, 4, 6], "col": ["a", "a", "b", "c", "v", "e", "q"]})
df2 = pd.DataFrame({"id": [1, 2, 3, 3, 5], "col": ["x", "y", "z", "v", "w"]})
print(df1)
print(df2)

print("纵向拼接并重建索引：\n", pd.concat([df1, df2], ignore_index=True, axis=0))
print("拼接后去重：\n", pd.concat([df1, df2], ignore_index=True).drop_duplicates())

# 列名不一致时，先统一列名再纵向合并
df3 = df1.rename(columns={"col": "new_col"})
print(df3)
print("统一列名后拼接去重：\n", pd.concat([df1, df3], ignore_index=True).drop_duplicates())


# ----------------------------------------------------------------------
# 1.2.5 排序
# ----------------------------------------------------------------------
title("1.2.5 排序")

sample = pd.DataFrame(
    {
        "group": [1, 1, 1, 2, 1, 2],
        "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"],
        "score": [98, 78, 87, 77, 77, np.nan],
    }
)
print(sample)
print("按分数降序，缺失值放最后：\n", sample.sort_values("score", ascending=False, na_position="last"))
print("先按组别、再按分数升序：\n", sample.sort_values(["group", "score"]))


# ----------------------------------------------------------------------
# 1.2.6 分组汇总
# ----------------------------------------------------------------------
title("1.2.6 分组汇总")

sample = pd.DataFrame(
    {
        "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"],
        "class": [1, 1, 1, 2, 1, 2],
        "grade": [1, 1, 1, 2, 2, 2],
        "chinese": [88, 78, 86, 56, 77, 54],
        "math": [98, 78, 87, 77, 77, np.nan],
    }
)
print(sample)

print("按年级求数学最高分：\n", sample.groupby("grade")[["math"]].max())
print("按年级、班级求数学平均分：\n", sample.groupby(["grade", "class"])[["math"]].mean())
print("按年级求数学、语文平均分：\n", sample.groupby(["grade"])[["math", "chinese"]].mean())

# agg 一次计算多个统计量
print("按班级对数学聚合：\n", sample.groupby("class")["math"].agg(["mean", "min", "max"]))

df = sample.groupby(["grade", "class"])[["math", "chinese"]].agg(["min", "max"])
print("多列多统计量：\n", df)
print("取 math 的 min 列（Series 方式）：\n", df["math"]["min"])
print("取 math 的 min 列（loc 方式）：\n", df.loc[:, ("math", "min")])
print("取 (2, 2) 组的 math min：\n", df.loc[(2, 2), ("math", "min")])


# ----------------------------------------------------------------------
# 1.2.7 拆分表（透视表）
# ----------------------------------------------------------------------
title("1.2.7 拆分表（透视表）")

table = pd.DataFrame(
    {
        "Monetary": [3608, 420, 1894, 3503, 4567],
        "cust_id": [10001, 10001, 10002, 10002, 10003],
        "type": ["Normal", "Special_offer", "Normal", "Special_offer", "Special_offer"],
    }
)
print(table)

print("以客户为行、交易类型为列：\n", pd.pivot_table(table, index="cust_id", columns="type", values="Monetary"))
print(
    "缺失补 0、求和：\n",
    pd.pivot_table(
        table,
        index="cust_id",
        columns="type",
        values="Monetary",
        fill_value=0,
        aggfunc="sum",
    ),
)


# ----------------------------------------------------------------------
# 1.2.8 赋值与条件赋值
# ----------------------------------------------------------------------
title("1.2.8 赋值与条件赋值")

sample = pd.DataFrame(
    {
        "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"],
        "score": [99, 78, 999, 77, 77, np.nan],
        "group": [1, 1, 1, 2, 1, 2],
    }
)
print(sample)

# 把 999 替换成缺失值
print("单列替换：\n", sample.score.replace(999, np.nan))
print(
    "多列按字典替换：\n",
    sample.replace({"score": {999: np.nan}, "name": {"Bob": np.nan}}),
)
print("注意：replace 默认返回新对象，原数据不变：\n", sample)


def transform(row):
    """按行把 group 映射为类别名称。"""
    if row["group"] == 1:
        return "class1"
    elif row["group"] == 2:
        return "class2"
    return np.nan


print("apply 按行变换：\n", sample.apply(transform, axis=1))
print("把变换结果新增为列：\n", sample.assign(class_n=sample.apply(transform, axis=1)))

# loc 条件赋值
sample_copy = sample.copy()
sample_copy.loc[sample_copy.group == 1, "class_n"] = "class1"
sample_copy.loc[sample_copy.group == 2, "class_n"] = "class2"
print(sample_copy)


# ----------------------------------------------------------------------
# 1.3.1 重复值处理
# ----------------------------------------------------------------------
title("1.3.1 重复值处理")

sample = pd.DataFrame(
    {
        "group": [1, 1, 1, 2, 1, 2],
        "id": [1, 1, 1, 3, 4, 5],
        "name": ["Bob", "Bob", "Mark", "Miki", "Sully", "Rose"],
        "score": [99, 99, 87, 77, 77, np.nan],
    }
)
print(sample)
print("查看重复行：\n", sample[sample.duplicated()])
print("删除完全重复的行：\n", sample.drop_duplicates())
print("按 id 去重（保留首次出现）：\n", sample.drop_duplicates("id"))


# ----------------------------------------------------------------------
# 1.3.2 缺失值处理
# ----------------------------------------------------------------------
title("1.3.2 缺失值处理")

sample = pd.DataFrame(
    {
        "group": [1, 1, np.nan, 2, 1, np.nan],
        "id": [1, 1, 1, 3, 4, np.nan],
        "name": ["Bob", "Bob", "Mark", "Miki", "Sully", np.nan],
        "score": [99, np.nan, 87, 77, 77, np.nan],
    }
)
print(sample)

print("各列缺失比例：\n", sample.apply(lambda col: sum(col.isnull()) / col.size))
print("用均值填补 score：\n", sample.score.fillna(sample.score.mean()))
print("用中位数填补 score：\n", sample.score.fillna(sample.score.median()))
print("缺失值标记：\n", sample.score.isnull())
print("缺失值转 0/1：\n", sample.score.isnull().apply(int))


# ----------------------------------------------------------------------
# 1.3.3 噪声值处理：盖帽法
# ----------------------------------------------------------------------
title("1.3.3 噪声值处理（盖帽法）")


def cap(x, quantile=(0.01, 0.99)):
    """盖帽法处理异常值。

    参数：
        x：pd.Series，连续变量；
        quantile：盖帽的下、上分位数。
    返回：
        截断后的 Series。
    """
    # 生成分位数
    q01, q99 = x.quantile(quantile).values.tolist()

    # 低于下分位数的值替换为下分位数
    if q01 > x.min():
        x = x.copy()
        x.loc[x < q01] = q01

    # 高于上分位数的值替换为上分位数
    if q99 < x.max():
        x = x.copy()
        x.loc[x > q99] = q99

    return x


sample = pd.DataFrame({"normal": np.random.randn(1000)})
sample.hist(bins=50, figsize=(6, 4))
plt.title("盖帽法处理前分布")
plt.show()
print("盖帽处理前统计：\n", sample.normal.describe())

new = sample.apply(cap, quantile=(0.01, 0.99))
new.hist(bins=50, figsize=(6, 4))
plt.title("盖帽法处理后分布")
plt.show()
print("盖帽处理后统计：\n", new.normal.describe())


# ----------------------------------------------------------------------
# 1.3.3 分箱离散化
# ----------------------------------------------------------------------
title("1.3.3 分箱离散化（cut）")

np.random.seed(0)
sample = pd.DataFrame({"normal": np.random.randn(10)})
print(sample)

print("等宽分 5 箱：\n", pd.cut(sample.normal, 5))
print("等宽分 5 箱并打标签：\n", pd.cut(sample.normal, bins=5, labels=[1, 2, 3, 4, 5]))
print("等宽分 2 箱：\n", pd.cut(sample.normal, bins=2, labels=["bad", "good"]))
print("分位数：\n", sample.normal.quantile([0, 0.5, 1]))
print(
    "按分位数分箱：\n",
    pd.cut(sample.normal, bins=sample.normal.quantile([0, 0.5, 1]), include_lowest=True),
)
print(
    "按分位数分箱并打标签：\n",
    pd.cut(
        sample.normal,
        bins=sample.normal.quantile([0, 0.5, 1]),
        labels=["bad", "good"],
        include_lowest=True,
    ),
)

print("\n第4章全部演示执行完毕。")