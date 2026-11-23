# -*- coding: utf-8 -*-
"""
3.1 申请反欺诈（决策树系列等分类算法）

项目背景
--------
消费金融在放款前需要识别欺诈申请。本案例使用申请人在关系网络中的结构特征
（net_size、degree、average_neighbor_degree、percentage）预测其是否为欺诈客户
（flag=1）。教材覆盖了从单模型到集成模型、再到模型解释的完整体系：

    3.1.1-3.1.2  决策树、随机森林、AdaBoost、XGBoost、LightGBM
    3.1.3-3.1.6  KNN、朴素贝叶斯、SVM、多层感知机（MLP）
    进阶专题      XGBoost 高级特性、LightGBM 高级特性、模型解释性
                  （特征重要性、SHAP、PDP、决策路径、规则提取、反事实解释）

数据文件
--------
    loan_net.csv                             反欺诈建模数据
    进阶内容_组合算法最近特性和模型解释/保险数据_第一期.csv   保险响应数据

依赖
----
    必装：numpy pandas matplotlib scikit-learn scipy statsmodels joblib
    选装：xgboost lightgbm shap dice-ml pydotplus
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '3.1申请反欺诈'
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
ADV_DIR = BASE_DIR / "进阶内容_组合算法最近特性和模型解释"

pd.set_option("display.max_columns", None)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metrics_curve(test_target, train_target, test_proba, train_proba):
    """同时绘制 ROC、PR、KS 三条评估曲线，并打印 AUC / AP。"""
    fpr_test, tpr_test, th_test = roc_curve(test_target, test_proba)
    fpr_train, tpr_train, th_train = roc_curve(train_target, train_proba)
    precision_test, recall_test, _ = precision_recall_curve(
        test_target, test_proba
    )
    precision_train, recall_train, _ = precision_recall_curve(
        train_target, train_proba
    )

    plt.figure(figsize=[6, 6])
    plt.subplot(221)
    plt.plot(fpr_test, tpr_test, "b--")
    plt.plot(fpr_train, tpr_train, "r-")
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.title("ROC 曲线（测试集为虚线）")

    plt.subplot(222)
    plt.plot(recall_test, precision_test, "b--")
    plt.plot(recall_train, precision_train, "r-")
    plt.xlabel("召回率")
    plt.ylabel("精确率")
    plt.title("PR 曲线（测试集为虚线）")

    plt.subplot(212)
    plt.plot(1 - th_test, tpr_test, "b--", label="真正率")
    plt.plot(1 - th_test, fpr_test, "b:", label="假正率")
    plt.plot(1 - th_test, tpr_test - fpr_test, "r-", label="KS 值（真正率 - 假正率）")
    plt.xlabel("评分")
    plt.title("KS 曲线  KS= %0.2f" % max(tpr_test - fpr_test))
    plt.legend(loc="upper left", shadow=True, fontsize="x-large")

    plt.tight_layout()
    save_show()

    print(
        "TestAUC=%.4f," % auc(fpr_test, tpr_test),
        "TrainAUC=%.4f" % auc(fpr_train, tpr_train),
    )
    print(
        "TestAP=%.4f," % average_precision_score(test_target, test_proba),
        "TrainAP=%.4f" % average_precision_score(train_target, train_proba),
    )


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


def tree_to_code(tree, feature_names):
    """把 sklearn 决策树转换成等价的 if-elif-else Python 代码并打印。"""
    from sklearn.tree import _tree

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


# ============================================================
# 3.1.1-3.1.2 数据准备
# ============================================================
data = pd.read_csv(BASE_DIR / "loan_net.csv", skipinitialspace=True)
print(data.head())

target = data["flag"].copy()
features = data.iloc[:, :4].copy()
feature_cn_names = ["网络规模", "节点度", "平均邻居度", "占比"]

train_data, test_data, train_target, test_target = train_test_split(
    features, target, test_size=0.4, train_size=0.6, random_state=1234
)
print("TrainSet size: ", len(train_data))
print("TestSet size: ", len(test_data))


# ============================================================
# 3.1.1-3.1.2 决策树（网格搜索）
# ============================================================
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "criterion": ["entropy", "gini"],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [2, 4, 6, 8, 10],
    "class_weight": [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}],
}
dtree = DecisionTreeClassifier(random_state=1234)
dtree_cv = GridSearchCV(
    estimator=dtree, param_grid=param_grid, scoring="roc_auc", cv=4, n_jobs=-1
)
dtree_cv.fit(train_data, train_target)
print("Best parameters generated by grid search: \n", dtree_cv.best_params_)

best_tree = dtree_cv.best_estimator_

# 决策树可视化：优先写 PNG，缺少 pydotplus 时用 sklearn.plot_tree
try:
    import pydotplus
    from sklearn import tree as sklearn_tree

    dot_data = sklearn_tree.export_graphviz(
        best_tree,
        out_file=None,
        feature_names=feature_cn_names,
        max_depth=3,
        class_names=True,
        filled=True,
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(str(BASE_DIR / "fraud_decision_tree.png"))
except ImportError:
    from sklearn.tree import plot_tree

    plt.figure(figsize=(14, 8))
    plot_tree(
        best_tree,
        feature_names=feature_cn_names,
        class_names=[str(c) for c in best_tree.classes_],
        max_depth=3,
        filled=True,
    )
    localize_tree_annotations()
    plt.title("欺诈识别决策树")
    save_show()

# 评估
test_est = best_tree.predict(test_data)
train_est = best_tree.predict(train_data)
print("Classification report on test:")
print(classification_report(test_target, test_est, digits=3))

test_proba = best_tree.predict_proba(test_data)[:, 1]
train_proba = best_tree.predict_proba(train_data)[:, 1]
plot_metrics_curve(test_target, train_target, test_proba, train_proba)

# 模型持久化
model_dir = BASE_DIR / "model"
model_dir.mkdir(exist_ok=True)
import joblib

save_path = model_dir / "fraud_detection_dtree.model"
joblib.dump(best_tree, save_path)
best_tree_load = joblib.load(save_path)

# 人工复现决策树逻辑的示例（与训练出的树结构对应）
def tree(net_size, degree, average_neighbor_degree, percentage):
    if net_size <= 16.0:
        if degree <= 1.5:
            if percentage <= 0.2916666716337204:
                if net_size <= 2.5:
                    return [[11802.0, 31.0]]
                return [[285.0, 0.0]]
            return [[70.0, 1.0]]
        if average_neighbor_degree <= 1.1666666865348816:
            if net_size <= 3.5:
                if percentage <= 0.1666666716337204:
                    return [[4082.0, 3.0]]
                return [[3.0, 0.0]]
            return [[4.0, 0.0]]
        return [[102.0, 0.0]]
    return [[0.0, 9.0]]


print(tree(2, 1, 1.0, 0.0))

# 把最优决策树转成 if-then 代码
tree_to_code(best_tree, features.columns)


# ============================================================
# 3.1.2 随机森林
# ============================================================
import sklearn.ensemble as ensemble

param_grid = {
    "criterion": ["gini"],
    "max_depth": [8],
    "n_estimators": [13],
    "max_features": [0.5, 0.6, 0.7],
    "min_samples_split": [2, 3, 4],
}
rfc = ensemble.RandomForestClassifier(random_state=1234)
rfc_cv = GridSearchCV(
    estimator=rfc, param_grid=param_grid, scoring="f1", cv=4, n_jobs=-1
)
rfc_cv.fit(train_data, train_target)
print("Best parameters : \n", rfc_cv.best_params_)

best_rf = rfc_cv.best_estimator_
train_proba = best_rf.predict_proba(train_data)[:, 1]
test_proba = best_rf.predict_proba(test_data)[:, 1]
plot_metrics_curve(test_target, train_target, test_proba, train_proba)


# ============================================================
# 3.1.2 AdaBoost
# ============================================================
from sklearn.ensemble import AdaBoostClassifier

param_grid = {
    "n_estimators": [15, 20, 25],
    "learning_rate": [0.5, 1.0, 1.5],
}
adaboost = AdaBoostClassifier(random_state=1234)
adaboost_cv = GridSearchCV(
    estimator=adaboost, param_grid=param_grid, scoring="recall", cv=4, n_jobs=-1
)
adaboost_cv.fit(train_data, train_target)
best_ada = adaboost_cv.best_estimator_

train_proba = best_ada.predict_proba(train_data)[:, 1]
test_proba = best_ada.predict_proba(test_data)[:, 1]
plot_metrics_curve(test_target, train_target, test_proba, train_proba)


# ============================================================
# 3.1.2 XGBoost
# ============================================================
try:
    from sklearn.model_selection import GridSearchCV as _Grid
    from xgboost import plot_importance
    from xgboost.sklearn import XGBClassifier

    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        eval_metric="auc",
        early_stopping_rounds=3,
        random_state=1234,
    )
    xgb.fit(
        train_data, train_target,
        eval_set=[[test_data, test_target]],
        verbose=False,
    )
    print("best_score:", xgb.best_score)
    print("best_iteration:", xgb.best_iteration)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "reg_lambda": np.arange(0.01, 1, 0.2),
        "max_depth": [2, 3, 5],
        "learning_rate": [0.5, 1],
    }
    xgb = XGBClassifier(
        objective="binary:logistic",
        verbosity=0,
        use_label_encoder=False,
        random_state=1234,
    )
    xgb_cv = _Grid(
        estimator=xgb, param_grid=param_grid, cv=4, scoring="recall", n_jobs=-1
    )
    xgb_cv.fit(train_data, train_target)
    print("best_params:", xgb_cv.best_params_)
    print("best_score:", xgb_cv.best_score_)

    best_xgb = xgb_cv.best_estimator_
    train_proba = best_xgb.predict_proba(train_data)[:, 1]
    test_proba = best_xgb.predict_proba(test_data)[:, 1]
    plot_metrics_curve(test_target, train_target, test_proba, train_proba)

    plot_importance(best_xgb)
    imp_ax = plt.gca()
    imp_name_map = {
        "net_size": "网络规模",
        "degree": "节点度",
        "average_neighbor_degree": "平均邻居度",
        "percentage": "占比",
    }
    imp_ax.set_yticklabels(
        [imp_name_map.get(t.get_text(), t.get_text()) for t in imp_ax.get_yticklabels()]
    )
    plt.xlabel("重要性得分")
    plt.ylabel("特征")
    plt.title("XGBoost 特征重要性")
    save_show()
except ImportError:
    print("未安装 xgboost，跳过 XGBoost 示例（可 pip install xgboost）。")


# ============================================================
# 3.1.2 LightGBM
# ============================================================
try:
    import lightgbm as lgb
    from scipy.stats import randint, uniform
    from sklearn.model_selection import RandomizedSearchCV

    param_distributions = {
        "num_leaves": randint(5, 50),
        "max_depth": randint(2, 5),
        "learning_rate": [0.04, 0.08],
        "n_estimators": randint(30, 100),
        "min_child_samples": [10, 20],
        "subsample": [0.5],
        "colsample_bytree": [0.5],
        "reg_alpha": uniform(1, 5),
        "reg_lambda": uniform(1, 5),
        "cat_smooth": uniform(3, 10),
    }
    n_iter = 300
    gbm = lgb.LGBMClassifier()
    gbm_cv = RandomizedSearchCV(
        gbm,
        param_distributions=param_distributions,
        scoring="recall",
        n_iter=n_iter,
        cv=4,
        random_state=12345,
        verbose=0,
        n_jobs=-1,
    )
    gbm_cv.fit(train_data, train_target)
    print("best_params:", gbm_cv.best_params_)
    print("best_score:", gbm_cv.best_score_)

    best_gbm = gbm_cv.best_estimator_
    train_proba = best_gbm.predict_proba(train_data)[:, 1]
    test_proba = best_gbm.predict_proba(test_data)[:, 1]
    plot_metrics_curve(test_target, train_target, test_proba, train_proba)

    joblib.dump(best_gbm, model_dir / "fraud_detection_lightGBM.model")
    gbm_load = joblib.load(model_dir / "fraud_detection_lightGBM.model")
except ImportError:
    print("未安装 lightgbm，跳过 LightGBM 示例（可 pip install lightgbm）。")


# ============================================================
# 3.1.3 数据准备（KNN / NB / SVM / MLP 共用）
# ============================================================
data_cls = pd.read_csv(BASE_DIR / "loan_net.csv", skipinitialspace=True)
target_cls = data_cls["flag"].copy()
features_cls = data_cls.iloc[:, :4].copy()
print(data_cls.describe(include="all"))

train_cls, test_cls, train_y, test_y = train_test_split(
    features_cls, target_cls, test_size=0.4, train_size=0.6, random_state=1234
)

scaler = StandardScaler()
scaler.fit(train_cls)
train_cls_scale = scaler.transform(train_cls)
test_cls_scale = scaler.transform(test_cls)


# ============================================================
# 3.1.3 KNN
# ============================================================
from sklearn.neighbors import KNeighborsClassifier

param_grid = {"n_neighbors": list(range(1, 19, 2))}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(
    estimator=knn, param_grid=param_grid, scoring="roc_auc", cv=4, n_jobs=-1
)
knn_cv.fit(train_cls_scale, train_y.values.flatten())
print("Best parameters generated by grid search: \n", knn_cv.best_params_)

best_knn = knn_cv.best_estimator_
test_proba = best_knn.predict_proba(test_cls_scale)[:, 1]
train_proba = best_knn.predict_proba(train_cls_scale)[:, 1]
plot_metrics_curve(test_y, train_y, test_proba, train_proba)


# ============================================================
# 3.1.4 朴素贝叶斯
# ============================================================
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

nb = MultinomialNB()
nb.fit(train_cls, train_y.values.flatten())
test_proba = nb.predict_proba(test_cls)[:, 1]
train_proba = nb.predict_proba(train_cls)[:, 1]
plot_metrics_curve(test_y, train_y, test_proba, train_proba)


# ============================================================
# 3.1.5 SVM
# ============================================================
from sklearn.svm import SVC

param_grid = {
    "gamma": np.arange(0.01, 1, 0.1),
    "kernel": ("linear", "rbf"),
    "C": np.arange(0.01, 1.0, 0.1),
}
svc = SVC(probability=True)
svc_cv = GridSearchCV(
    estimator=svc, param_grid=param_grid, scoring="roc_auc", cv=4, n_jobs=-1
)
svc_cv.fit(train_cls_scale, train_y)
print("Best parameters is :\n{}".format(svc_cv.best_params_))

test_proba = svc_cv.predict_proba(test_cls_scale)[:, 1]
train_proba = svc_cv.predict_proba(train_cls_scale)[:, 1]
plot_metrics_curve(test_y, train_y, test_proba, train_proba)


# ============================================================
# 3.1.6 多层感知机 MLP
# ============================================================
try:
    from scipy.stats import randint, uniform
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neural_network import MLPClassifier

    n_iter = 20
    param_distributions = {
        "hidden_layer_sizes": [(10,), (15,), (5, 5)],
        "activation": ["logistic", "relu"],
        "alpha": uniform(0.01, 10),
    }
    mlp = MLPClassifier(max_iter=1000, random_state=1234)
    mlp_cv = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=4,
        n_jobs=-1,
    )
    mlp_cv.fit(train_cls_scale, train_y)
    print("Best parameters is :\n{}".format(mlp_cv.best_params_))

    best_mlp = mlp_cv.best_estimator_
    test_proba = best_mlp.predict_proba(test_cls_scale)[:, 1]
    train_proba = best_mlp.predict_proba(train_cls_scale)[:, 1]
    plot_metrics_curve(test_y, train_y, test_proba, train_proba)
except ImportError:
    print("scikit-learn MLP 不可用，跳过 MLP 示例。")


# ============================================================
# 进阶专题一：XGBoost 高级特性（保险数据）
# ============================================================
def load_insurance():
    """读取保险数据，做缺失值处理、类别编码与训练测试切分。"""
    raw = pd.read_csv(ADV_DIR / "保险数据_第一期.csv")
    df = raw.dropna().reset_index(drop=True)
    labels = df.pop("resp_flag")
    cat_cols = df.select_dtypes(include=["object"])
    num_cols = df.select_dtypes(exclude=["object"])
    encoder = OrdinalEncoder()
    cat_trans = encoder.fit_transform(cat_cols)
    df_cat = pd.DataFrame(cat_trans, columns=cat_cols.columns)
    data_ = pd.concat([df_cat, num_cols], axis=1)
    return train_test_split(
        data_, labels, test_size=0.3, stratify=labels, random_state=42
    )


Xtrain, Xtest, Ytrain, Ytest = load_insurance()

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from xgboost.sklearn import XGBClassifier

    # sklearn 接口
    XGB = XGBClassifier(random_state=42)
    XGB.fit(Xtrain, Ytrain)
    print("AUC: ", roc_auc_score(Ytest, XGB.predict_proba(Xtest)[:, 1]))

    # 原生接口需要 DMatrix
    train_dm = xgb.DMatrix(data=Xtrain, label=Ytrain)
    XGB = xgb.train(params={}, dtrain=train_dm)
    test_dm = xgb.DMatrix(data=Xtest)
    print(classification_report(Ytest, XGB.predict(test_dm) > 0.5))
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_dm)))

    # 设置参数与评估指标
    param = {"max_depth": 5, "eta": 0.4}
    param["eval_metric"] = "auc"
    XGB = xgb.train(
        params=param, dtrain=train_dm, evals=[(train_dm, "train")],
        num_boost_round=15,
    )
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_dm)))

    # evals：计算评估指标的数据集
    # early_stopping_rounds：连续多少次指标没有改进则停止
    Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(
        Xtrain, Ytrain, test_size=0.3, stratify=Ytrain, random_state=42
    )
    train_e = xgb.DMatrix(data=Xtrain1, label=Ytrain1)
    test_e = xgb.DMatrix(data=Xtest1, label=Ytest1)
    eval_list = [(train_e, "train_e"), (test_e, "test_e")]
    XGB = xgb.train(
        params=param, dtrain=train_e, evals=eval_list,
        num_boost_round=15, early_stopping_rounds=3,
    )

    # 单调约束
    params_monotonic = param.copy()
    params_monotonic["tree_method"] = "exact"
    params_monotonic["monotone_constraints"] = (0, 1, -1)
    XGB = xgb.train(
        params=params_monotonic, dtrain=train_dm, evals=[(train_dm, "train")]
    )
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_dm)))

    # 交互约束
    params_interaction = param.copy()
    params_interaction["interaction_constraints"] = "[[0, 2], [1, 3, 4], [5, 6]]"
    XGB = xgb.train(
        params=params_interaction, dtrain=train_dm, evals=[(train_dm, "train")]
    )
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_dm)))

    # 用 XGBoost 实现随机森林
    params_rf = {
        "colsample_bynode": 0.8,
        "learning_rate": 1,
        "num_parallel_tree": 100,
        "subsample": 0.8,
    }
    XGB = xgb.train(
        params=params_rf, dtrain=train_dm, evals=[(train_dm, "train")],
        num_boost_round=1,
    )
    print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_dm)))
except ImportError:
    print("未安装 xgboost，跳过 XGBoost 高级特性示例。")


# ============================================================
# 进阶专题二：LightGBM 高级特性（保险数据）
# ============================================================
try:
    import lightgbm
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split as _split
    from lightgbm import LGBMClassifier

    # sklearn 的 HistGradientBoosting 可视为 LightGBM 的轻量替代
    HGBC = HistGradientBoostingClassifier()
    HGBC.fit(Xtrain, Ytrain)
    print("HistGB AUC: ", roc_auc_score(Ytest, HGBC.predict_proba(Xtest)[:, 1]))

    # LightGBM sklearn 接口
    lgb_sk = LGBMClassifier()
    lgb_sk.fit(Xtrain, Ytrain)
    print("AUC: ", roc_auc_score(Ytest, lgb_sk.predict_proba(Xtest)[:, 1]))

    # 原生接口
    train_lgb = lightgbm.Dataset(Xtrain, label=Ytrain)
    lgb_native = lightgbm.train(params={}, train_set=train_lgb)
    print("AUC: ", roc_auc_score(Ytest, lgb_native.predict(Xtest)))

    # 二进制数据集可加快加载
    train_lgb.save_binary(str(BASE_DIR / "train.bin"))
    train_bin = lightgbm.Dataset(str(BASE_DIR / "train.bin"))
    lgb_bin = lightgbm.train(params={}, train_set=train_bin)

    Xtrain1, Xtest1, Ytrain1, Ytest1 = _split(
        Xtrain, Ytrain, test_size=0.3, stratify=Ytrain, random_state=42
    )
    train_ds = lightgbm.Dataset(
        data=Xtrain1,
        label=Ytrain1,
        categorical_feature=["GEND", "AART"],
        params={"max_bin": 255},
        free_raw_data=False,
    )
    test_ds = lightgbm.Dataset(data=Xtest1, label=Ytest1, reference=train_ds)

    params = {"num_leaves": 31, "objective": "binary"}
    params["metric"] = "auc"
    params["verbosity"] = 3
    lgb_model = lightgbm.train(
        params=params,
        train_set=train_ds,
        valid_sets=test_ds,
        num_boost_round=200,
        callbacks=[lightgbm.early_stopping(stopping_rounds=5, min_delta=0)],
    )
    print("AUC: ", roc_auc_score(Ytest, lgb_model.predict(Xtest)))
    print("best_iteration:", lgb_model.best_iteration)
    print("feature_importance:", lgb_model.feature_importance())
    lgb_model.save_model(str(BASE_DIR / "model.txt"))

    # 单调约束：必须给所有特征传入约束标识
    my_list = [1, 0, -1] + [0] * 46
    params = {"metric": "auc", "objective": "binary"}
    params["monotone_constraints"] = my_list
    params["monotone_constraints_method"] = "advanced"
    lgb_model = lightgbm.train(params=params, train_set=train_ds, valid_sets=test_ds)
    print("AUC: ", roc_auc_score(Ytest, lgb_model.predict(Xtest)))

    # 交互约束
    params = {"metric": "auc", "objective": "binary"}
    params["interaction_constraints"] = [[1, 3, 4], [5, 6]]
    lgb_model = lightgbm.train(params=params, train_set=train_ds, valid_sets=test_ds)
    print("AUC: ", roc_auc_score(Ytest, lgb_model.predict(Xtest)))
except ImportError:
    print("未安装 lightgbm，跳过 LightGBM 高级特性示例。")


# ============================================================
# 进阶专题三：模型解释性
# ============================================================
# 1) 逻辑回归系数
import statsmodels.api as sm

Xtrain_lr = Xtrain.copy()
Xtrain_lr["const"] = 1
LR = sm.Logit(Ytrain, Xtrain_lr).fit()
print(LR.summary())

# 2) 树模型的特征重要性
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xtrain, Ytrain)
print("决策树特征重要性:", dt.feature_importances_)

rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(Xtrain, Ytrain)
print("随机森林特征重要性:", rf.feature_importances_)

try:
    import xgboost as xgb

    train_xgb = xgb.DMatrix(data=Xtrain, label=Ytrain)
    XGB_model = xgb.train(params={}, dtrain=train_xgb)
    print("XGBoost 重要性(weight):", XGB_model.get_score(importance_type="weight"))
except ImportError:
    XGB_model = None
    print("未安装 xgboost，跳过 XGBoost 特征重要性。")

try:
    import lightgbm

    train_lgb_imp = lightgbm.Dataset(Xtrain, label=Ytrain)
    lgb_imp = lightgbm.train(params={}, train_set=train_lgb_imp)
    print("LightGBM 重要性(split):", lgb_imp.feature_importance(importance_type="split"))
except ImportError:
    print("未安装 lightgbm，跳过 LightGBM 特征重要性。")

# 3) SHAP 解释
try:
    import shap

    if XGB_model is not None:
        explainer = shap.Explainer(XGB_model)
        shap_values = explainer(Xtrain)
        shap.plots.bar(shap_values, max_display=10)
        shap.plots.bar(shap_values[1], max_display=10)
        shap.plots.beeswarm(shap_values)
except ImportError:
    print("未安装 shap，跳过 SHAP 解释（可 pip install shap）。")

# 4) 偏依赖图（PDP）
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

pdp = PartialDependenceDisplay.from_estimator(
    estimator=rf,
    X=Xtrain,
    features=["GEND", "BANK", "age", "meda"],
    subsample=1000,
)
pdp.figure_.suptitle("偏依赖图", fontsize=14)
feature_cn = {"GEND": "性别", "BANK": "开户行", "age": "年龄", "meda": "医疗支出"}
for ax_, feat_ in zip(pdp.axes.ravel(), ["GEND", "BANK", "age", "meda"]):
    ax_.set_title(feature_cn.get(feat_, feat_))
    ax_.set_ylabel("偏依赖值")
save_show()

print(
    partial_dependence(
        estimator=rf,
        X=Xtrain,
        features=["age"],
    )
)

# 5) 决策树可视化与决策路径
from sklearn.datasets import load_iris

iris = load_iris()
clf = DecisionTreeClassifier(random_state=42)
clf.fit(iris.data, iris.target)

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"],
    class_names=["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"],
    filled=True,
)
localize_tree_annotations()
plt.title("鸢尾花决策树示例")
save_show()

sample = np.array(Xtest.iloc[1])
node_indicator = dt.decision_path(sample.reshape(1, -1))
node_index = node_indicator.indices
feature = dt.tree_.feature
threshold = dt.tree_.threshold

print("决策路径:")
for node_id in node_index:
    if feature[node_id] != -2:
        feature_name = dt.feature_names_in_[feature[node_id]]
        threshold_value = threshold[node_id]
        sample_value = sample[feature[node_id]]
        print(
            f"节点 {node_id}: 如果 {feature_name} <= {threshold_value:.2f} "
            f"(样本值: {sample_value:.2f})"
        )
    else:
        print(
            f"叶节点 {node_id}: 预测结果 = "
            f"{dt.classes_[dt.tree_.value[node_id].argmax()]}"
        )

# 6) 随机森林中每棵树的决策路径
rf_clf = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=42)
rf_clf.fit(Xtrain, Ytrain)
for i, tree_obj in enumerate(rf_clf.estimators_):
    print(f"第 {i + 1} 棵树的决策路径:")
    node_indicator = tree_obj.decision_path(sample.reshape(1, -1))
    node_index = node_indicator.indices
    feature = tree_obj.tree_.feature
    threshold = tree_obj.tree_.threshold
    for node_id in node_index:
        if feature[node_id] != -2:
            feature_name = rf_clf.feature_names_in_[feature[node_id]]
            threshold_value = threshold[node_id]
            sample_value = sample[feature[node_id]]
            print(
                f"  节点 {node_id}: 如果 {feature_name} <= {threshold_value:.2f} "
                f"(样本值: {sample_value:.2f})"
            )
        else:
            leaf_value = tree_obj.tree_.value[node_id]
            print(
                f"  叶节点 {node_id}: 分类结果 = {np.argmax(leaf_value)}, "
                f"节点值 = {leaf_value}\n"
            )

# 7) 取重要性前 5 的特征重新训练浅树并提取规则
feature_importance = pd.DataFrame(
    {"Feature": dt.feature_names_in_, "Importance": dt.feature_importances_}
)
feature_sort = feature_importance.sort_values(by="Importance", ascending=False)
Xtrain_rule = Xtrain[feature_sort.head(5)["Feature"]]
dt_rule = DecisionTreeClassifier(max_depth=5, random_state=42).fit(
    Xtrain_rule, Ytrain
)
rules_text = export_text(
    dt_rule, feature_names=dt_rule.feature_names_in_.tolist()
)
print(rules_text)

plt.figure(figsize=(12, 8))
plot_tree(
    dt_rule,
    feature_names=dt_rule.feature_names_in_.tolist(),
    class_names=["0（正常）", "1（欺诈）"],
    filled=True,
)
localize_tree_annotations()
plt.title("重要特征决策树规则")
save_show()

# 8) 反事实解释（dice-ml）
try:
    import dice_ml

    train_dataset = pd.concat([Xtrain, Ytrain], axis=1)
    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=pd.Series(Xtrain.columns).tolist(),
        outcome_name="resp_flag",
    )
    m = dice_ml.Model(model=dt, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    e1 = exp.generate_counterfactuals(
        Xtest[0:1], total_CFs=4, desired_class="opposite", random_seed=42
    )
    e1.visualize_as_dataframe(show_only_changes=True)
except ImportError:
    print("未安装 dice-ml，跳过反事实解释（可 pip install dice-ml）。")
