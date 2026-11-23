# -*- coding: utf-8 -*-
"""
2.3 用户画像（主成分、因子分析、变量聚类、层次聚类、KMeans 聚类）

项目背景
--------
用户画像的核心是把多维特征压缩为少数几个可解释的维度，再基于这些维度对客户分群，
最终用业务语言描述每一类客户。本脚本把教材中 5 个笔记本的内容整合为一条完整流水线：

    2.3.1-2.3.2  经济发展数据的 PCA / SparsePCA / 因子分析 / 层次聚类
    2.3.1        贷款申请数据的 PCA 打分
    2.3.3        基于 SparsePCA 的变量聚类与变量筛选
    2.3.4        经济发展数据的层次聚类（复用 2.3.1-2.3.2 的因子得分流程）
    2.3.5        客户细分：特征工程 + 因子分析 + KMeans + 决策树画像

数据文件
--------
    cities_10.csv        10 个城市的 9 项经济发展指标
    Loan_aply.csv        贷款申请数据（变量 X1 起）
    data_clust.csv       客户人口统计、资产、兴趣等画像数据

依赖
----
    必装：numpy pandas matplotlib scikit-learn scipy
    选装：fa_kit（因子分析）、pydotplus（决策树导出）
    未安装 fa_kit 时，脚本自动退化为 sklearn PCA 因子得分，保证流程可跑通。
"""

from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '2.3用户画像'
FIG_DIR = BASE_DIR / "figures"
_fig_counter = [0]


def save_show(*_args, **_kwargs):
    """保存当前图形到 figures/ 目录后再显示。"""
    FIG_DIR.mkdir(exist_ok=True)
    _fig_counter[0] += 1
    fig_path = FIG_DIR / ("%s_%02d.png" % (FIG_PREFIX, _fig_counter[0]))
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")
    return fig_path

pd.set_option("display.max_columns", 8)
pd.set_option("display.max_column", 8)

# 中文字体设置（Windows 下通常可用 SimHei）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def scatter_with_labels(x, y, labels, **kw):
    plt.figure(figsize=(6, 4))
    plt.xlabel("第一维度得分", fontsize=13)
    plt.ylabel("第二维度得分", fontsize=13)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.scatter(x, y)
    for xx, yy, label in zip(x, y, labels):
        plt.text(xx, yy, "%s." % label, ha="center", va="bottom", **kw)
    save_show()


# ============================================================
# 2.3.1-2.3.2 经济发展数据：PCA / SparsePCA / 因子分析 / 层次聚类
# ============================================================
cities = pd.read_csv(BASE_DIR / "cities_10.csv", encoding="gbk")
print(cities.head())

print(cities.loc[:, "X1":].corr(method="pearson").head())


# ---------- 主成分分析 ----------
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler

n_components = 2
scaler = StandardScaler()
scale_cities = scaler.fit_transform(cities.loc[:, "X1":])

pca = PCA(n_components=n_components)
pca_scores = pca.fit_transform(scale_cities)
print("Cumsum of explained variance ratio: ", pca.explained_variance_ratio_.cumsum())

pca_scores = pd.DataFrame(pca_scores, columns=["Gross", "Avg"])
scale_pca_cities = cities.join(pca_scores)
print(scale_pca_cities.head())

print("主成分载荷矩阵（用于业务解释）:\n", pca.components_)

scatter_with_labels(
    scale_pca_cities["Gross"],
    scale_pca_cities["Avg"],
    scale_pca_cities["AREA"],
    fontsize=15,
)


# ---------- 稀疏主成分：让载荷更易解释 ----------
alpha = 0.5
spca = SparsePCA(n_components=n_components, alpha=alpha)
spca.fit(scale_cities)
print(pd.DataFrame(spca.components_))

n_cols = scale_cities.shape[1]
spca_with_alpha = []
for i in np.arange(0.1, 1, 0.05):
    spca = SparsePCA(n_components=n_components, alpha=i)
    spca.fit(scale_cities)
    components_ = pd.DataFrame(spca.components_).T
    diff = np.count_nonzero(components_) - n_cols
    spca_with_alpha.append((i, diff))
print(pd.DataFrame(spca_with_alpha).T)

spca_ = SparsePCA(n_components=n_components, alpha=0.65).fit(scale_cities)
print(pd.DataFrame(spca_.components_))

scale_spca_cities = pd.DataFrame(
    spca_.transform(scale_cities), columns=["Gross", "Avg"]
)
print(scale_spca_cities)
scatter_with_labels(
    scale_spca_cities["Gross"],
    scale_spca_cities["Avg"],
    cities["AREA"],
)


# ---------- 因子分析（fa_kit；缺失时退化为 PCA） ----------
def factor_scores_or_pca(data_matrix, n_factors):
    """优先使用 fa_kit 做最大方差旋转因子分析，否则用 PCA 得分近似。"""
    try:
        from fa_kit import FactorAnalysis

        fa = FactorAnalysis.load_data_samples(
            data_matrix, preproc_demean=True, preproc_scale=True
        )
        fa.extract_components()
        fa.find_comps_to_retain(method="top_n", num_keep=n_factors)
        fa.rotate_components(method="varimax")
        print(fa.comps["rot"])
        return fa.get_component_scores(data_matrix), fa
    except ImportError:
        print("未安装 fa_kit，使用 sklearn PCA 因子得分近似。")
        pca_fallback = PCA(n_components=n_factors)
        return pca_fallback.fit_transform(data_matrix), pca_fallback


try:
    from fa_kit import plotting as fa_plotting

    fa = None
    fa_scores, fa = factor_scores_or_pca(scale_cities, 2)
    if fa is not None and hasattr(fa, "comps"):
        try:
            plt.rcParams["axes.titlesize"] = 20
            plt.rcParams["xtick.labelsize"] = 15
            plt.rcParams["ytick.labelsize"] = 15
            fa_plotting.graph_summary(fa)
            save_show()
        except Exception as exc:
            print("fa_kit 绘图失败，跳过：", exc)
except ImportError:
    fa_scores, _ = factor_scores_or_pca(scale_cities, 2)

cities_factor_analysis = cities.join(
    pd.DataFrame(fa_scores, columns=["Gross", "Avg"])
)
cities_factor_analysis.to_csv(BASE_DIR / "cities_FA_score.csv", index=False)
print(cities_factor_analysis.head())

scatter_with_labels(
    cities_factor_analysis["Gross"],
    cities_factor_analysis["Avg"],
    cities_factor_analysis["AREA"],
    fontsize=15,
)


# ---------- 层次聚类 ----------
import scipy.cluster.hierarchy as sch

distMat = sch.distance.pdist(cities_factor_analysis[["Gross", "Avg"]], "euclidean")
Z = sch.linkage(distMat, method="ward")

fig, ax = plt.subplots(figsize=(6, 4))
sch.dendrogram(
    Z,
    ax=ax,
    labels=cities_factor_analysis["AREA"].tolist(),
    leaf_font_size=15,
    leaf_rotation=90,
)
ax.set_title("城市经济发展层次聚类树状图", fontsize=14)
plt.savefig(BASE_DIR / "plot_dendrogram1.png")
save_show()

labels = sch.cut_tree(Z, n_clusters=2)
cities_factor_analysis["cluster"] = labels
print(
    cities_factor_analysis.loc[:, "X1":"X9"]
    .groupby(cities_factor_analysis["cluster"])
    .mean()
    .T
)


# ============================================================
# 2.3.1 贷款申请数据：主成分打分
# ============================================================
raw_data = pd.read_csv(BASE_DIR / "Loan_aply.csv", encoding="gbk")
print(raw_data.head())

data = raw_data.loc[:, "X1":]
print(data.corr(method="pearson"))

scaled_data = StandardScaler().fit_transform(data)
pca5 = PCA(n_components=5)
pca5.fit(scaled_data)
print("Explained variance: \n", pca5.explained_variance_)
print("Explained variance ratio:\n", pca5.explained_variance_ratio_)
print("Principal Components: \n", pca5.components_)
print(scaled_data[0])
print(raw_data[["ID"]].join(pd.DataFrame(pca5.transform(scaled_data))))

print(
    'PCA_score of "ID=1":  %.4f' % np.dot(scaled_data[0], pca5.components_[0])
)

pca5.set_params(n_components=1)
pca5.fit(scaled_data)
data_pca = raw_data.join(
    pd.DataFrame(pca5.transform(scaled_data), columns=["PCAScore"])
)
print(data_pca.head())


# ============================================================
# 2.3.3 变量聚类：用 SparsePCA 从高度相关的变量中挑选代表变量
# ============================================================
def var_select(orgdata, k, max_alpha=10, alpha_step=0.2):
    """
    基于 SparsePCA 的变量聚类筛选。

    orgdata   需要压缩的数据框
    k         期望保留的最大变量个数
    max_alpha SparsePCA 惩罚项上限
    alpha_step 惩罚项递增步长
    """
    if len(orgdata) > 5000:
        data_sel = orgdata.sample(5000)
    else:
        data_sel = orgdata

    data_sel = StandardScaler().fit_transform(data_sel)
    n_components_sel = k

    best_alpha = None
    last_alpha = 0.1
    for i in np.arange(0.1, max_alpha, alpha_step):
        last_alpha = i
        pca_model = SparsePCA(n_components=n_components_sel, alpha=i)
        pca_model.fit(data_sel)
        pca_components = pd.DataFrame(pca_model.components_).T
        n = data_sel.shape[1] - sum(sum(np.array(pca_components != 0)))
        if n == 0:
            best_alpha = i
            break
    if best_alpha is None:
        best_alpha = last_alpha

    pca_model = SparsePCA(n_components=n_components_sel, alpha=best_alpha)
    pca_model.fit(data_sel)
    pca_components = pd.DataFrame(pca_model.components_).T
    data_sel = pd.DataFrame(data_sel)
    score = pd.DataFrame(pca_model.fit_transform(data_sel))

    r = []
    R_square = []
    for xk in range(data_sel.shape[1]):
        for paj in range(n_components_sel):
            r.append(abs(np.corrcoef(data_sel.iloc[:, xk], score.iloc[:, paj])[0, 1]))
            r_max1 = max(r)
            r.remove(r_max1)
            r.append(-2)
            r_max2 = max(r)
            R_square.append((1 - r_max1 ** 2) / (1 - r_max2 ** 2))

    R_square = abs(
        pd.DataFrame(
            np.array(R_square).reshape((data_sel.shape[1], n_components_sel))
        )
    )
    var_list = []
    for i in range(n_components_sel):
        vmin = R_square[i].min()
        var_list.append(R_square[R_square[i] == vmin][i].index)

    news_ids = []
    for idx in var_list:
        if idx not in news_ids:
            news_ids.append(idx)
    print(news_ids)
    return orgdata.iloc[:, np.array(news_ids).reshape(len(news_ids))]


model_data = pd.read_csv(BASE_DIR / "cities_10.csv", encoding="gbk")
print(model_data.head())
selected_data = var_select(model_data.loc[:, "X1":], k=2)
print(selected_data)


# ============================================================
# 2.3.4 经济发展数据的层次聚类
# （与 2.3.1-2.3.2 使用同一套因子得分 + Ward 层次聚类流程，此处复用函数）
# ============================================================
def hierarchical_cluster_on_cities(cities_df, n_factors=2, n_clusters=2):
    scale_cities_df = StandardScaler().fit_transform(cities_df.loc[:, "X1":])
    scores, _ = factor_scores_or_pca(scale_cities_df, n_factors)
    factor_df = cities_df.join(pd.DataFrame(scores, columns=["Gross", "Avg"]))
    factor_df.to_csv(BASE_DIR / "cities_FA_score.csv", index=False)

    dist = sch.distance.pdist(factor_df[["Gross", "Avg"]], "euclidean")
    linkage = sch.linkage(dist, method="ward")

    fig_, ax_ = plt.subplots(figsize=(6, 4))
    sch.dendrogram(
        linkage,
        ax=ax_,
        labels=factor_df["AREA"].tolist(),
        leaf_font_size=15,
        leaf_rotation=90,
    )
    ax_.set_title("城市经济发展层次聚类树状图", fontsize=14)
    plt.savefig(BASE_DIR / "plot_dendrogram1.png")
    save_show()

    factor_df["cluster"] = sch.cut_tree(linkage, n_clusters=n_clusters)
    print(
        factor_df.loc[:, "X1":"X9"].groupby(factor_df["cluster"]).mean().T
    )
    return factor_df


cities_for_hclust = pd.read_csv(BASE_DIR / "cities_10.csv", encoding="gbk")
hierarchical_cluster_on_cities(cities_for_hclust)


# ============================================================
# 2.3.5 客户细分：特征工程 + 因子分析 + KMeans + 决策树画像
# ============================================================
customer = pd.read_csv(BASE_DIR / "data_clust.csv", skipinitialspace=True)
print(customer.describe(include="all"))
print(customer.head())

# 缺失值检查
for col in customer.columns:
    if customer[col].hasnans:
        print("含缺失值的变量:", col)

fill_cols = ["interested_travel", "computer_owner", "HH_adults_num"]
fill_values = {col: customer[col].mode()[0] for col in fill_cols}
customer = customer.fillna(fill_values)

print(customer["interested_reading"].value_counts(dropna=False))

customer["HH_has_children"] = customer["HH_has_children"].replace(
    {"N": 0, "Y": 1, np.nan: 0}
)
customer["interested_reading"] = (
    customer["interested_reading"].replace({".": "0"}).astype("int")
)

# 类别变量之间的卡方独立性检验：p>0.05 说明两变量关联不显著
from itertools import combinations

from scipy import stats

_cols = [
    "interested_travel",
    "computer_owner",
    "marital",
    "interested_golf",
    "interested_gambling",
    "HH_has_children",
    "interested_reading",
]
sample = customer[_cols].sample(3000, random_state=12345)
for colA, colB in combinations(_cols, 2):
    crosstab = pd.crosstab(sample[colA], sample[colB])
    pval = stats.chi2_contingency(crosstab)[1]
    if pval > 0.05:
        print('p-value = %0.3f between "%s" and "%s"' % (pval, colA, colB))

# 二值化并构造“兴趣偏好填充率”综合指标
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=1.5)
customer["interested_reading"] = binarizer.fit_transform(
    customer[["interested_reading"]]
)

interest = [
    "interested_travel",
    "computer_owner",
    "interested_golf",
    "interested_gambling",
    "interested_reading",
]
n_ = len(interest)
customer = customer.drop(interest, axis=1).assign(
    interest=customer[interest].sum(axis=1) / n_
)
print(customer.head())

continuous_cols = [
    "age",
    "home_value",
    "risk_score",
    "interested_sport",
    "HH_dieting",
    "auto_member",
    "HH_grandparent",
    "HH_head_age",
    "loan_ratio",
]
categorical_cols = ["marital", "interest", "HH_adults_num"]
discreate_cols = ["HH_has_children"]

# 绘图时把变量名映射为中文，便于阅读
continuous_cols_cn = {
    "age": "年龄",
    "home_value": "房产价值",
    "risk_score": "风险评分",
    "interested_sport": "运动兴趣",
    "HH_dieting": "家庭饮食",
    "auto_member": "汽车会员",
    "HH_grandparent": "家庭祖辈",
    "HH_head_age": "户主年龄",
    "loan_ratio": "贷款比率",
}

customer[continuous_cols].rename(columns=continuous_cols_cn).hist(
    bins=25, figsize=[8, 8]
)
plt.suptitle("连续变量分布（原始）", fontsize=14)
save_show()

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")
qt_data = qt.fit_transform(customer[continuous_cols])
pd.DataFrame(qt_data, columns=continuous_cols).rename(
    columns=continuous_cols_cn
).hist(bins=25, figsize=[8, 8])
plt.suptitle("连续变量分布（分位数正态变换后）", fontsize=14)
save_show()

scale_data = StandardScaler().fit_transform(customer[categorical_cols])

data = np.hstack([qt_data, scale_data, customer[discreate_cols]])
data = pd.DataFrame(
    data, columns=continuous_cols + categorical_cols + discreate_cols
)
print(data.head())

household = [
    "age",
    "marital",
    "HH_adults_num",
    "home_value",
    "risk_score",
    "HH_grandparent",
    "HH_head_age",
    "loan_ratio",
]
hobby = ["HH_dieting", "auto_member", "interest", "interested_sport"]

print(data[hobby].corr())

pca_hh = PCA().fit(data[household])
print("家庭维度 PCA 累计解释方差:\n", pca_hh.explained_variance_ratio_.cumsum())

hh_scores, _ = factor_scores_or_pca(data[household].to_numpy(), 4)
data_hh = pd.DataFrame(
    hh_scores,
    columns=["life_circle", "finance", "HH_size", "risk"],
)
print(data_hh.head())

hb_scores, _ = factor_scores_or_pca(data[hobby].to_numpy(), 3)
data_hb = pd.DataFrame(
    hb_scores,
    columns=["sports", "health", "leisure"],
)
print(data_hb.head())


# ---------- KMeans 聚类与 k 值选择 ----------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_plot(data_, k_range=range(2, 12), n_init=5, sample_size=2000,
                 figsize=[9, 2]):
    """遍历候选 k，绘制簇内离差平方和（SSD）与轮廓系数，返回各 k 的模型。"""
    scores = []
    models = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=n_init)
        kmeans.fit(data_)
        models[k] = kmeans
        sil = silhouette_score(
            data_, kmeans.labels_, sample_size=sample_size
        )
        scores.append([k, kmeans.inertia_, sil])

    scores_df = pd.DataFrame(scores, columns=["k", "sum_square_dist", "sil"])
    plt.figure(figsize=figsize)
    plt.subplot(121, xlabel="聚类簇数", ylabel="簇内离差平方和")
    plt.plot(scores_df.k, scores_df.sum_square_dist)
    plt.subplot(122, xlabel="聚类簇数", ylabel="轮廓系数")
    plt.plot(scores_df.k, scores_df.sil)
    plt.suptitle("KMeans 聚类效果随簇数的变化", fontsize=14)
    save_show()
    return models


hh_scaler = StandardScaler()
scale_data_hh = hh_scaler.fit_transform(data_hh)
hb_scaler = StandardScaler()
scale_data_hb = hb_scaler.fit_transform(data_hb)

models_hh = cluster_plot(scale_data_hh)
print("-" * 80)
models_hb = cluster_plot(scale_data_hb)

hh_labels = pd.DataFrame(models_hh[3].labels_, columns=["hh"])
hb_labels = pd.DataFrame(models_hb[2].labels_, columns=["hb"])
clusters = customer.join(hh_labels).join(hb_labels)
print(clusters.head())


# ---------- 用决策树解释每个簇的特征 ----------
from sklearn.tree import DecisionTreeClassifier, plot_tree

household_cn = [
    "年龄", "婚姻状况", "家庭成人数", "房产价值",
    "风险评分", "家庭祖辈", "户主年龄", "贷款比率",
]
hobby_cn = ["家庭饮食", "汽车会员", "综合兴趣", "运动兴趣"]


def localize_tree_annotations(ax=None):
    """把 sklearn 决策树节点里的英文指标名替换为中文。"""
    ax = ax if ax is not None else plt.gca()
    replace_map = [
        ("gini", "基尼系数"),
        ("samples", "样本数"),
        ("value", "分布"),
        ("class", "类别"),
        ("True", "真"),
        ("False", "假"),
    ]
    for text in ax.texts:
        s = text.get_text()
        for en, zh in replace_map:
            s = s.replace(en, zh)
        text.set_text(s)


tree_hh = DecisionTreeClassifier()
tree_hb = DecisionTreeClassifier()
tree_hh.fit(clusters[household], clusters["hh"])
tree_hb.fit(clusters[hobby], clusters["hb"])

plt.figure(figsize=(12, 8))
plot_tree(
    tree_hh,
    feature_names=household_cn,
    class_names=[str(c) for c in tree_hh.classes_],
    max_depth=2,
    filled=True,
)
localize_tree_annotations()
plt.title("家庭维度聚类决策树")
save_show()

plt.figure(figsize=(12, 8))
plot_tree(
    tree_hb,
    feature_names=hobby_cn,
    class_names=[str(c) for c in tree_hb.classes_],
    max_depth=2,
    filled=True,
)
localize_tree_annotations()
plt.title("兴趣维度聚类决策树")
save_show()

cluster_pt = pd.pivot_table(clusters, index="hh", columns="hb", aggfunc="mean").T
print(cluster_pt.swaplevel("hb", 0).sort_index(level=0))
