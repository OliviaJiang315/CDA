# -*- coding: utf-8 -*-
"""
第7章 特征降维方法（主成分分析与因子分析）

对应教材：
    第7章-特征降维方法/在 Python 中实现主成分分析.ipynb
    第7章-特征降维方法/在 Python 中实现因子分析.ipynb

内容：
1. 主成分分析（PCA）的标准化、拟合、解释方差、成分载荷与降维打分；
2. 用累计解释方差与特征值判断保留主成分的个数；
3. 因子分析：主成分法提取因子、最大方差（varimax）旋转、因子得分计算与命名。

说明：
    教材使用第三方包 fa_kit 完成因子分析。本书代码改为纯 scikit-learn +
    numpy 实现（PCA 提取 + varimax 旋转），不依赖 fa_kit，结果等价且可直接运行。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "第7章-特征降维方法")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第7章 特征降维方法")
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


def data_path(name):
    return os.path.join(DATA_DIR, name)


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """最大方差法（varimax）旋转因子载荷矩阵。

    参数：
        loadings：形状为 (变量数, 因子数) 的载荷矩阵；
        gamma：旋转参数，1.0 为标准 varimax；
        max_iter：最大迭代次数；
        tol：收敛阈值。
    返回：
        旋转后的载荷矩阵，形状不变。
    """
    p, k = loadings.shape
    rotation = np.eye(k)
    objective = 0.0

    for _ in range(max_iter):
        objective_old = objective
        rotated = loadings @ rotation

        # Varimax 的解析解：对载荷三次方做 SVD
        transformed = loadings.T @ (
            rotated ** 3
            - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
        )
        u, s, vh = np.linalg.svd(transformed)
        rotation = u @ vh
        objective = s.sum()

        if objective_old != 0 and objective / objective_old < 1 + tol:
            break

    return loadings @ rotation


# ======================================================================
# 第一部分  主成分分析（案例一：电信客户业务使用情况）
# ======================================================================
title("7.1 主成分分析：电信客户分群")

profile = pd.read_csv(data_path("profile_telecom.csv"))
print(profile.head())

# 选取业务使用量字段
data_tel = profile.loc[:, "cnt_call":]
print("变量：", data_tel.columns.tolist())
print("相关系数矩阵：\n", data_tel.corr(method="pearson"))

# 主成分分析前必须中心标准化
data_scaled = scale(data_tel)

pca = PCA(n_components=2, whiten=True).fit(data_scaled)
print("前两个主成分解释方差比例：", pca.explained_variance_ratio_)
print("累计解释比例：", pca.explained_variance_ratio_.sum())
print("主成分载荷（每一行是一个主成分）：\n", pca.components_)

# 降维后的得分（白化后各主成分方差为 1）
scores = pca.transform(data_scaled)
print("主成分得分前 5 行：\n", scores[:5])

plt.figure(figsize=(6, 4))
plt.scatter(scores[:, 0], scores[:, 1], s=12, alpha=0.6)
plt.xlabel("主成分 1")
plt.ylabel("主成分 2")
plt.title("电信客户主成分得分分布")
plt.show()


# ======================================================================
# 第二部分  主成分分析（案例二：10 个城市经济指标）
# ======================================================================
title("7.2 主成分个数选择：10 城市经济指标")

# cities_10.csv 为 GBK 编码
model_data = pd.read_csv(data_path("cities_10.csv"), encoding="gbk")
print(model_data.head())

data = model_data.loc[:, "X1":]
print("参与分析的指标：", data.columns.tolist())

# 中心标准化
data_std = scale(data)
print("标准化后形状：", data_std.shape)

# 先放入全部 9 个主成分，观察解释方差
pca_full = PCA(n_components=9)
pca_full.fit(data_std)

print("各主成分特征值（解释方差）：\n", pca_full.explained_variance_)
print("各主成分解释方差比例：\n", pca_full.explained_variance_ratio_)
print("累计解释方差比例：\n", np.cumsum(pca_full.explained_variance_ratio_))

# 选取标准：
# 1. 累计解释方差比例 > 0.85；
# 2. 最后一个保留主成分的特征值 > 1（标准化数据）。
n_keep = 2
pca = PCA(n_components=n_keep).fit(data_std)
print("保留 %d 个主成分的特征值：%s" % (n_keep, pca.explained_variance_))
print("保留 %d 个主成分的载荷：\n%s" % (n_keep, pca.components_))

# 主成分得分
pc_scores = pca.transform(data_std)
print("主成分得分前 5 行：\n", pc_scores[:5])

# 碎石图：辅助判断保留个数
plt.figure(figsize=(6, 4))
plt.plot(range(1, 10), pca_full.explained_variance_, "o-")
plt.axhline(1.0, color="r", linestyle="--", label="特征值 = 1")
plt.xlabel("主成分序号")
plt.ylabel("特征值")
plt.title("碎石图")
plt.legend()
plt.show()


# ======================================================================
# 第三部分  因子分析：主成分法 + 最大方差旋转
# ======================================================================
title("7.3 因子分析：城市综合实力评价")

# 因子分析沿用标准化后的 9 个指标
# 步骤 1：提取因子（主成分法），暂不旋转
pca_fa = PCA(n_components=n_keep)
pca_fa.fit(data_std)

# 初始载荷矩阵：变量数 × 因子数
loadings = pca_fa.components_.T

# 步骤 2：最大方差旋转，使因子含义更清晰
rotated = varimax(loadings)
print("旋转前载荷：\n", pd.DataFrame(loadings, index=data.columns))
print("旋转后载荷（每列为一个因子）：\n", pd.DataFrame(rotated, index=data.columns))
print("旋转后载荷列平方和（列范数）：\n", (rotated ** 2).sum(axis=0))
print("保留因子的解释方差：\n", pca_fa.explained_variance_)

# 步骤 3：计算因子得分
# 因子得分 = 标准化原始数据(n×p) @ 旋转后载荷(p×k)
fa_score = pd.DataFrame(
    data_std @ rotated,
    columns=["Factor1", "Factor2"],
    index=model_data.index,
)
print("因子得分前 5 行：\n", fa_score.head())

# 步骤 4：因子命名并拼回原始数据
# 依据载荷大小命名：综合经济规模因子 / 平均发展水平因子
fa_score_named = fa_score.rename(columns={"Factor1": "Gross", "Factor2": "Avg"})
citi10_fa = model_data.join(fa_score_named)
print(citi10_fa)

# 因子得分散点图，并标注城市名称
plt.figure(figsize=(7, 5))
x = citi10_fa["Gross"]
y = citi10_fa["Avg"]
labels = citi10_fa["AREA"]

plt.scatter(x, y, s=40)
for a, b, label in zip(x, y, labels):
    plt.text(a, b + 0.1, "%s." % label, ha="center", va="bottom", fontsize=10)

plt.xlabel("Gross（因子 1）")
plt.ylabel("Avg（因子 2）")
plt.title("10 个城市因子得分分布")
plt.tight_layout()
plt.show()

print("\n第7章全部演示执行完毕。")