# -*- coding: utf-8 -*-
"""
3.2 违规识别（孤立森林、局部异常因子）

项目背景
--------
信用卡违规/欺诈识别往往面临“坏样本极少、无标注”的场景。教材使用无监督异常检测：
    1. 孤立森林（IsolationForest）：用随机切分把稀疏异常点“孤立”出来
    2. 局部异常因子（LOF）：用局部密度偏离程度衡量异常

数据文件
--------
    creditcard.csv    V1~V28 为匿名化特征，Class 为是否欺诈（仅用于事后评估）

依赖
----
    numpy pandas matplotlib scikit-learn scipy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 中文字体设置，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '3.2违规识别'
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


# ============================================================
# 一、数据读取与类别分布
# ============================================================
data = pd.read_csv(BASE_DIR / "creditcard.csv")
print(data.head())

data["Class"].value_counts().plot(kind="bar", colormap="cool")
plt.title("欺诈类别分布", fontsize=12)
plt.xlabel("类别（0=正常，1=欺诈）", fontsize=12)
plt.ylabel("样本数", fontsize=12)
save_show()

print(data["Class"].value_counts())

# 只看 V1~V28 特征，计算峰度（异常点通常位于重尾分布）
train = data.loc[:, "V1":"V28"].copy()
ax = train.kurtosis().sort_values().plot()
ax.set_title("各特征峰度排序", fontsize=12)
ax.set_xlabel("特征变量", fontsize=12)
ax.set_ylabel("峰度", fontsize=12)
save_show()

# 少量抽样便于快速查看（教材未抽样，这里保留全量以便复现）
rng = np.random.RandomState(42)


# ============================================================
# 二、孤立森林
# ============================================================
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

iso = IsolationForest(
    contamination=0.02,
    n_estimators=100,
    max_samples=0.6,
    max_features=0.6,
    random_state=rng,
    n_jobs=-1,
)
iso_outlier = iso.fit_predict(train)
print("CrossTable: \n", pd.crosstab(data["Class"], iso_outlier))
print("ARI: ", adjusted_rand_score(data["Class"], iso_outlier))


# ============================================================
# 三、孤立森林随机搜索调参
# ============================================================
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

n_iter = 10
param_distributions = {
    "contamination": uniform(0.015, 0.01),
    "n_estimators": randint(80, 121),
    "max_samples": uniform(0.5, 0.2),
    "max_features": uniform(0.5, 0.2),
}
iso = IsolationForest(random_state=rng)
iso_cv = RandomizedSearchCV(
    estimator=iso,
    param_distributions=param_distributions,
    scoring="adjusted_rand_score",
    cv=3,
    n_iter=n_iter,
    n_jobs=-1,
)
iso_cv.fit(train, data["Class"])
print("Best_params: ", iso_cv.best_params_)
print("Best_score: ", iso_cv.best_score_)

best_iso = iso_cv.best_estimator_
best_iso_outlier = best_iso.predict(train)
print("CrossTable: \n", pd.crosstab(data["Class"], best_iso_outlier))
print("-" * 50)
print("ARI: ", adjusted_rand_score(data["Class"], best_iso_outlier))


# ============================================================
# 四、模型持久化（教材使用 pickle）
# ============================================================
import pickle

with open(BASE_DIR / "M_IsolationForest.model", "wb") as f:
    pickle.dump(best_iso, f)
with open(BASE_DIR / "M_IsolationForest.model", "rb") as f:
    clf_load = pickle.load(f)
print("模型已保存并重新加载:", type(clf_load))


# ============================================================
# 五、局部异常因子 LOF
# ============================================================
import time

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=500,
    contamination=0.02,
    novelty=True,
    n_jobs=-1,
)
start = time.time()
lof.fit(train)
lof_outlier = lof.predict(train)
print("LOF 拟合耗时: %.2f 秒" % (time.time() - start))
print("CrossTable: \n", pd.crosstab(data["Class"], lof_outlier))
print("ARI: ", adjusted_rand_score(data["Class"], lof_outlier))


# ============================================================
# 六、LOF 调参（教材中已注释，计算量大，按需开启）
# ============================================================
# n_iter = 2
# param_distributions = {
#     "n_neighbors": randint(100, 500),
#     "contamination": uniform(0.02, 0.04),
# }
# lof = LocalOutlierFactor(algorithm="kd_tree", novelty=True)
# lof_cv = RandomizedSearchCV(
#     estimator=lof,
#     param_distributions=param_distributions,
#     scoring="adjusted_rand_score",
#     cv=3,
#     n_iter=n_iter,
#     n_jobs=-1,
# )
# start = time.time()
# lof_cv.fit(train, data["Class"])
# print(time.time() - start)
# print("Best_params: ", lof_cv.best_params_)
# print("Best_score: ", lof_cv.best_score_)
# best_lof = lof_cv.best_estimator_
# best_lof_outlier = best_lof.predict(train)
# print("CrossTable: \n", pd.crosstab(data["Class"], best_lof_outlier))
# print("ARI: ", adjusted_rand_score(data["Class"], best_lof_outlier))
