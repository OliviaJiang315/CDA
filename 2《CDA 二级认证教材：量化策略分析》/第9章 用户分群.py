# -*- coding: utf-8 -*-
"""
第9章 用户分群（K-Means 聚类与层次聚类）

对应教材：
    第9章-用户分群/k-means聚类/Kmeans 的应用 - 用户细分.ipynb
    第9章-用户分群/层次聚类/hc_city10.ipynb

内容：
1. 银行交易行为数据的因子分析降维，再对客户做 K-Means 聚类；
2. 变量正态变换后重新聚类，并用决策树解释各群的业务特征；
3. 城市经济指标的因子分析 + 层次聚类（Ward 法）与树状图。

说明：
    教材使用 fa_kit 做因子分析，本文件用 scikit-learn PCA + varimax 旋转实现，
    并用 sklearn.tree.plot_tree 替代 pydotplus，避免额外依赖。
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import tree
from scipy.cluster import hierarchy as sch

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.join(BASE_DIR, "第9章-用户分群")
KMEANS_DIR = os.path.join(CHAPTER_DIR, "k-means聚类")
HC_DIR = os.path.join(CHAPTER_DIR, "层次聚类")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第9章 用户分群")
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


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """最大方差法旋转因子载荷矩阵（变量数 × 因子数）。"""
    p, k = loadings.shape
    rotation = np.eye(k)
    objective = 0.0

    for _ in range(max_iter):
        objective_old = objective
        rotated = loadings @ rotation
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
# 第一部分  K-Means 用户细分
# ======================================================================
title("9.1 银行客户交易行为聚类")

# 四个特征是客户去柜台、使用 ATM、刷 POS 机、使用信用卡的次数
model_data = pd.read_csv(os.path.join(KMEANS_DIR, "profile_bank.csv"))
data = model_data.loc[:, "CNT_TBM":"CNT_CSC"]
print(model_data.head())
print("参与分析的变量：", data.columns.tolist())

# 1. 相关系数矩阵：判断变量降维的必要性
corr_matrix = data.corr(method="pearson")
print("相关系数矩阵：\n", corr_matrix)

# 2. 中心标准化
data_scaled = preprocessing.scale(data)

# 3. PCA 判断保留主成分个数
pca = PCA(n_components=3)
pca.fit(data_scaled)
print("主成分特征值：\n", pca.explained_variance_)
print("主成分解释方差比例：\n", pca.explained_variance_ratio_)
print("累计解释方差比例：\n", np.cumsum(pca.explained_variance_ratio_))

# 4. 因子分析：3 个因子 + varimax 旋转
#    分子得分用于聚类，降低变量间共线性的影响
n_factors = 3
pca_fa = PCA(n_components=n_factors).fit(data_scaled)
loadings = varimax(pca_fa.components_.T)
print("旋转后因子载荷：\n", pd.DataFrame(loadings, index=data.columns))

fa_scores = pd.DataFrame(data_scaled @ loadings)
fa_scores = fa_scores.rename(columns={0: "ATM_POS", 1: "TBM", 2: "CSC"})
print("因子得分前 5 行：\n", fa_scores.head())

# 5. 识别异常：查看因子偏度
var = ["ATM_POS", "TBM", "CSC"]
skew_var = {i: abs(fa_scores[i].skew()) for i in var}
skew = pd.Series(skew_var).sort_values(ascending=False)
print("各因子偏度绝对值：\n", skew)

# 6. K-Means 聚类
kmeans = KMeans(n_clusters=3, n_init=10, random_state=12345)
result = kmeans.fit(fa_scores)
print("各群样本量：\n", pd.Series(result.labels_).value_counts().sort_index())
print("各群中心：\n", result.cluster_centers_)

# 7. 分类结果解读：拼回原始数据并画饼图
model_data_l = model_data.join(pd.DataFrame(result.labels_, columns=["clustor"]))
print(model_data_l.head())

plt.figure(figsize=(5, 5))
model_data_l["clustor"].value_counts().sort_index().plot(kind="pie", autopct="%.1f%%")
plt.title("三类客户占比")
plt.ylabel("")
plt.show()


title("9.2 变量正态变换后重新聚类")

# 1. 变量分布的正态转换
quantile_transformer = preprocessing.QuantileTransformer(
    output_distribution="normal", random_state=0
)
fa_scores_trans = quantile_transformer.fit_transform(fa_scores)
fa_scores_trans = pd.DataFrame(fa_scores_trans, columns=["ATM_POS", "TBM", "CSC"])

# 查看正态变换后的偏度
skew_var = {i: abs(fa_scores_trans[i].skew()) for i in var}
print("正态变换后偏度绝对值：\n", pd.Series(skew_var).sort_values(ascending=False))

# 2. 重新聚类
kmeans2 = KMeans(n_clusters=3, n_init=10, random_state=12345)
result2 = kmeans2.fit(fa_scores_trans)

model_data_l2 = model_data.join(pd.DataFrame(result2.labels_, columns=["clustor"]))
print(model_data_l2["clustor"].value_counts().sort_index())
print("各群均值：\n", model_data_l2.groupby("clustor").mean())

# 3. 聚类事后分析：用决策树解释分群规则
clf = tree.DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=100,
    min_samples_leaf=100,
    random_state=12345,
)
clf.fit(model_data, result2.labels_)

plt.figure(figsize=(14, 7))
tree.plot_tree(
    clf,
    feature_names=list(model_data.columns),
    class_names=["0", "1", "2"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("客户分群决策树")
plt.show()

print("决策树在训练集上的准确率：%.4f" % clf.score(model_data, result2.labels_))


# ======================================================================
# 第二部分  层次聚类
# ======================================================================
title("9.3 城市经济指标的因子分析与层次聚类")

df = pd.read_csv(os.path.join(HC_DIR, "cities_10.csv"), encoding="gbk")
model_data_city = df.loc[:, "X1":]
print(df.head())

# 1. 相关系数矩阵：判断是否适合降维
print("相关系数矩阵：\n", model_data_city.corr())

# 2. 标准化
scaler = preprocessing.StandardScaler()
scaler_model_data = scaler.fit_transform(model_data_city)

# 3. 主成分分析：判断保留个数
pca_city = PCA(n_components=3)
pca_city.fit(scaler_model_data)
np.set_printoptions(suppress=True)
print("主成分可解释的变异：\n", pca_city.explained_variance_)
print("主成分可解释的变异比例：\n", pca_city.explained_variance_ratio_)
print("前两个主成分累计解释比例：%.4f" % np.cumsum(pca_city.explained_variance_ratio_)[1])

# 4. 因子分析：保留 2 个因子并做 varimax 旋转
pca_fa2 = PCA(n_components=2).fit(scaler_model_data)
rotated_city = varimax(pca_fa2.components_.T)
print("旋转后因子载荷：\n", pd.DataFrame(rotated_city, index=model_data_city.columns))

fa_scores_city = pd.DataFrame(
    scaler_model_data @ rotated_city,
    columns=["Gross", "Avg"],
    index=df.index,
)
cities_fa_scores = df.join(fa_scores_city)
print(cities_fa_scores)

# 5. 因子得分散点图，肉眼观察城市分布
plt.figure(figsize=(7, 5))
plt.scatter(cities_fa_scores["Gross"], cities_fa_scores["Avg"], s=40)
for xx, yy, label in zip(
    cities_fa_scores["Gross"],
    cities_fa_scores["Avg"],
    cities_fa_scores["AREA"],
):
    plt.text(xx, yy, "%s." % label, ha="center", va="bottom", fontsize=10)
plt.xlabel("Gross（因子 1：经济规模）")
plt.ylabel("Avg（因子 2：人均水平）")
plt.title("10 个城市因子得分分布")
plt.tight_layout()
plt.show()

# 6. 层次聚类
# 计算欧氏距离矩阵
dis_mat = sch.distance.pdist(cities_fa_scores[["Gross", "Avg"]], metric="euclidean")

# Ward 法连接类与类
Z = sch.linkage(dis_mat, method="ward")
print("层次聚类连接矩阵：\n", Z)

# 绘制树状图
plt.figure(figsize=(9, 5))
sch.dendrogram(Z, labels=cities_fa_scores["AREA"].tolist(), leaf_rotation=45)
plt.title("10 个城市层次聚类树状图（Ward 法）")
plt.ylabel("距离")
plt.tight_layout()
plt.show()

print("\n第9章全部演示执行完毕。")