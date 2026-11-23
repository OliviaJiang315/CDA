# -*- coding: utf-8 -*-
"""
7 类别不平衡问题

项目背景
--------
欺诈、违约、流失等业务的目标变量天然不平衡（坏样本通常不足 1%）。教材系统演示：

    1. 不平衡数据可视化（make_blobs 构造 1000:100 的两类样本）
    2. 过采样：SMOTE
    3. 欠采样：RandomUnderSampler、TomekLinks、EditedNearestNeighbours
    4. 组合采样：SMOTETomek、SMOTEENN
    5. 真实数据建模：XGBoost 基线
    6. SMOTE / SMOTENC 过采样后再建模
    7. 代价敏感学习：scale_pos_weight、max_delta_step

数据文件
--------
    train_set.csv   ID, age, job, ..., y（y 为不平衡标签）

依赖
----
    必装：numpy pandas matplotlib scikit-learn
    选装：imbalanced-learn（imblearn）、xgboost
    未安装选装包时对应示例会跳过，数据读取与切分仍会执行。
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
FIG_PREFIX = '7 类别不平衡问题'
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


# ============================================================
# 一、构造不平衡数据并可视化
# ============================================================
from sklearn.datasets import make_blobs

class_1 = 1000
class_2 = 100
centers = [[0, 0], [2.0, 2.0]]
clusters_std = [1.5, 1]
X, y = make_blobs(
    n_samples=[class_1, class_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=42,
    shuffle=False,
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", s=10)
plt.title("不平衡数据集")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
save_show()
print(pd.DataFrame(y).value_counts())


# ============================================================
# 二、采样方法对比（imbalanced-learn）
# ============================================================
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import (
        EditedNearestNeighbours,
        RandomUnderSampler,
        TomekLinks,
    )

    sampling_methods = {
        "SMOTE 过采样": SMOTE(random_state=42, sampling_strategy=0.5),
        "随机欠采样": RandomUnderSampler(
            sampling_strategy=0.3, random_state=42
        ),
        "Tomek Links 欠采样": TomekLinks(),
        "编辑近邻欠采样（ENN）": EditedNearestNeighbours(n_neighbors=20),
        "SMOTE + Tomek 组合采样": SMOTETomek(random_state=42),
        "SMOTE + ENN 组合采样": SMOTEENN(random_state=42),
    }

    for name, sampler in sampling_methods.items():
        X_res, y_res = sampler.fit_resample(X, y)
        print("=" * 50)
        print(name, "采样后类别分布:")
        print(pd.DataFrame(y_res).value_counts())
        plt.scatter(X_res[:, 0], X_res[:, 1], c=y_res, cmap="rainbow", s=10)
        plt.title(name + " 采样结果")
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        save_show()
except ImportError:
    print("未安装 imbalanced-learn，跳过采样方法演示（可 pip install imbalanced-learn）。")


# ============================================================
# 三、真实数据准备
# ============================================================
raw_data = pd.read_csv(BASE_DIR / "train_set.csv")
raw_data.info()

df_features = raw_data.drop(["ID"], axis=1)
df = df_features.dropna().reset_index(drop=True)
labels = df.pop("y")

cat_cols = df.select_dtypes(include=["object"])
num_cols = df.select_dtypes(exclude=["object"])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
cat_trans = encoder.fit_transform(cat_cols)
df_cat = pd.DataFrame(cat_trans, columns=cat_cols.columns)
data = pd.concat([df_cat, num_cols], axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)
print("训练集类别分布:\n", Ytrain.value_counts())
print("测试集类别分布:\n", Ytest.value_counts())


# ============================================================
# 四、XGBoost 基线
# ============================================================
try:
    import xgboost as xgb
    from sklearn.metrics import classification_report, roc_auc_score

    train = xgb.DMatrix(data=Xtrain, label=Ytrain)
    XGB = xgb.train(params={}, dtrain=train)
    test = xgb.DMatrix(data=Xtest)
    print(classification_report(Ytest, XGB.predict(test) > 0.5))
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))
except ImportError:
    print("未安装 xgboost，跳过后面的 XGBoost 建模示例（可 pip install xgboost）。")
    xgb = None


# ============================================================
# 五、SMOTE / SMOTENC 过采样后再建模
# ============================================================
if xgb is not None:
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC

        smote = SMOTE(random_state=42)
        X_sm, Y_sm = smote.fit_resample(Xtrain, Ytrain)
        print("SMOTE 采样后类别分布:\n", pd.DataFrame(Y_sm).value_counts())

        train_sm = xgb.DMatrix(data=X_sm, label=Y_sm)
        XGB_sm = xgb.train(params={}, dtrain=train_sm)
        test_sm = xgb.DMatrix(data=Xtest)
        print(classification_report(Ytest, XGB_sm.predict(test_sm) > 0.5))
        print("SMOTE AUC: ", roc_auc_score(Ytest, XGB_sm.predict(test_sm)))

        # SMOTENC 需要传入类别特征所在列的下标
        cat_idx = [data.columns.get_loc(c) for c in cat_cols.columns]
        smote_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
        X_smnc, Y_smnc = smote_nc.fit_resample(Xtrain, Ytrain)
        print("SMOTENC 采样后类别分布:\n", pd.DataFrame(Y_smnc).value_counts())

        train_smnc = xgb.DMatrix(data=X_smnc, label=Y_smnc)
        XGB_smnc = xgb.train(params={}, dtrain=train_smnc)
        print(classification_report(Ytest, XGB_smnc.predict(test_sm) > 0.5))
        print("SMOTENC AUC: ", roc_auc_score(Ytest, XGB_smnc.predict(test_sm)))
    except ImportError:
        print("未安装 imbalanced-learn，跳过 SMOTE / SMOTENC 建模示例。")


# ============================================================
# 六、代价敏感学习：类别权重与 max_delta_step
# ============================================================
if xgb is not None:
    weight = Ytrain.value_counts()[0] / Ytrain.value_counts()[1]
    params = {"scale_pos_weight": weight, "eval_metric": "auc"}
    XGB_weight = xgb.train(params=params, dtrain=train)
    print(classification_report(Ytest, XGB_weight.predict(test) > 0.5))
    print("scale_pos_weight AUC: ", roc_auc_score(Ytest, XGB_weight.predict(test)))

    params = {"max_delta_step": 0.6, "eval_metric": "auc"}
    XGB_delta = xgb.train(params=params, dtrain=train)
    print(classification_report(Ytest, XGB_delta.predict(test) > 0.5))
    print("max_delta_step AUC: ", roc_auc_score(Ytest, XGB_delta.predict(test)))
