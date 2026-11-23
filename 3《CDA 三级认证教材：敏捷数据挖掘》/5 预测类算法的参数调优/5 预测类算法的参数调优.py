# -*- coding: utf-8 -*-
"""
5 预测类算法的参数调优

项目背景
--------
同一个模型在不同超参数下效果差异很大。教材以保险响应数据为例，系统演示四类调参方法：

    5.1  网格搜索 GridSearchCV
         减半网格搜索 HalvingGridSearchCV
         随机搜索 RandomizedSearchCV
         贝叶斯优化 Optuna(TPE)

    5.2&3  用 Pipeline + ColumnTransformer 把“缺失值填充、编码、建模”串成一体，
           再对整条流水线调参，最后持久化模型并预测新进用户。

数据文件
--------
    保险数据_第一期.csv     建模数据（resp_flag 为标签）
    保险案例_新进用户.csv   待预测的新用户

依赖
----
    必装：numpy pandas scikit-learn scipy joblib
    选装：xgboost lightgbm optuna
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 一、数据准备
# ============================================================
def load_insurance_data():
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split

    raw_data = pd.read_csv(BASE_DIR / "保险数据_第一期.csv")
    df = raw_data.dropna().reset_index(drop=True)
    labels = df.pop("resp_flag")

    cat_cols = df.select_dtypes(include=["object"])
    num_cols = df.select_dtypes(exclude=["object"])

    encoder = OrdinalEncoder()
    cat_trans = encoder.fit_transform(cat_cols)
    df_cat = pd.DataFrame(cat_trans, columns=cat_cols.columns)
    data = pd.concat([df_cat, num_cols], axis=1)

    return train_test_split(
        data, labels, test_size=0.3, stratify=labels, random_state=42
    )


x_train, x_test, y_train, y_test = load_insurance_data()
print("训练集:", x_train.shape, "测试集:", x_test.shape)


# ============================================================
# 二、网格搜索（XGBoost）
# ============================================================
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    param = {
        "n_estimators": np.arange(100, 150, 20),
        "max_depth": np.arange(3, 6, 2),
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, verbosity=0)
    grid_search_xgb = GridSearchCV(xgb, param, cv=3, n_jobs=-1)
    grid_search_xgb.fit(x_train, y_train)

    print("AUC: ", roc_auc_score(y_test, grid_search_xgb.predict_proba(x_test)[:, 1]))
    print("最优参数:", grid_search_xgb.best_params_)
except ImportError:
    print("未安装 xgboost，跳过网格搜索示例（可 pip install xgboost）。")


# ============================================================
# 三、减半网格搜索 HalvingGridSearchCV
# ============================================================
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    param = {
        "n_estimators": np.arange(100, 150, 1),
        "max_depth": np.arange(3, 6, 1),
        "subsample": np.arange(0.5, 1, 0.1),
        "colsample_bytree": np.arange(0.5, 1, 0.1),
    }
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, verbosity=0)
    halving_search_xgb = HalvingGridSearchCV(
        xgb, param, factor=3, cv=3, verbose=1, n_jobs=-1
    )
    halving_search_xgb.fit(x_train, y_train)

    print(
        "AUC: ",
        roc_auc_score(y_test, halving_search_xgb.predict_proba(x_test)[:, 1]),
    )
    print("最优参数:", halving_search_xgb.best_params_)
except ImportError:
    print("xgboost 或 HalvingGridSearchCV 不可用，跳过减半搜索示例。")


# ============================================================
# 四、随机搜索 RandomizedSearchCV
# ============================================================
try:
    from scipy.stats import uniform
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier

    param = dict(
        n_estimators=np.arange(100, 150, 1),
        max_depth=np.arange(3, 6, 1),
        subsample=uniform(loc=0.5, scale=0.5),
        colsample_bytree=uniform(loc=0.5, scale=0.5),
    )
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, verbosity=0)
    random_search_xgb = RandomizedSearchCV(
        xgb, param, cv=3, n_iter=50, verbose=1, n_jobs=-1, random_state=42
    )
    random_search_xgb.fit(x_train, y_train)

    print(
        "AUC: ",
        roc_auc_score(y_test, random_search_xgb.predict_proba(x_test)[:, 1]),
    )
    print("最优参数:", random_search_xgb.best_params_)
except ImportError:
    print("xgboost 不可用，跳过随机搜索示例。")


# ============================================================
# 五、贝叶斯优化（Optuna + LightGBM）
# ============================================================
try:
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        param = {
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "verbosity": -1,
        }
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_test, label=y_test)
        model = lgb.train(params=param, train_set=d_train, valid_sets=d_valid)
        return roc_auc_score(y_test, model.predict(x_test))

    sampler = TPESampler()
    lgb_opt = optuna.create_study(sampler=sampler, direction="maximize")
    lgb_opt.optimize(
        func=objective,
        n_trials=100,
        show_progress_bar=True,
        n_jobs=-1,
    )
    print("Best_params: ", lgb_opt.best_params)
    print("Best score: ", lgb_opt.best_value)
except ImportError:
    print("未安装 lightgbm 或 optuna，跳过贝叶斯优化示例。")


# ============================================================
# 六、Pipeline + ColumnTransformer：保险案例
# ============================================================
df = pd.read_csv(BASE_DIR / "保险数据_第一期.csv")
labels = df.pop("resp_flag")

cat_cols = df.select_dtypes(include=["object"])
num_cols = df.select_dtypes(exclude=["object"])
print("分类变量:", cat_cols.columns.tolist())
print("数值变量个数:", len(num_cols.columns))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

cat_imp = SimpleImputer(strategy="most_frequent")
cat_encode = OrdinalEncoder()
cat_pipe = Pipeline(steps=[("cat_imp", cat_imp), ("cat_encode", cat_encode)])

num_imp = SimpleImputer(strategy="mean")

col_trans = ColumnTransformer(
    transformers=[
        ("cat_pipe", cat_pipe, cat_cols.columns),
        ("num_imp", num_imp, num_cols.columns),
    ]
)
print("转换后的数据维度:", col_trans.fit_transform(df).shape)

lr = LogisticRegression(max_iter=1000)
lr_pipe = Pipeline(steps=[("col_trans", col_trans), ("lr", lr)])

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    df, labels, test_size=0.3, stratify=labels, random_state=42
)
lr_pipe.fit(Xtrain, Ytrain)
print("逻辑回归流水线准确率:", lr_pipe.score(Xtest, Ytest))

# 使用 <estimator>__<parameter> 访问流水线内部参数
from sklearn.model_selection import GridSearchCV

param = dict(
    lr__C=np.arange(1, 2, 0.1),
    lr__class_weight=[None, "balanced"],
    col_trans__num_imp__strategy=["mean", "median"],
)
grid_lr = GridSearchCV(lr_pipe, param, cv=3, n_jobs=-1)
grid_lr.fit(Xtrain, Ytrain)
print("最优参数:", grid_lr.best_params_)

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(Ytest, grid_lr.predict(Xtest)))
print("AUC: ", roc_auc_score(Ytest, grid_lr.predict_proba(Xtest)[:, 1]))

print(classification_report(Ytrain, grid_lr.predict(Xtrain)))
print("AUC: ", roc_auc_score(Ytrain, grid_lr.predict_proba(Xtrain)[:, 1]))

# 保存模型并预测新进用户
import joblib

joblib.dump(grid_lr, BASE_DIR / "model_20241222.joblib")
df_new = pd.read_csv(BASE_DIR / "保险案例_新进用户.csv")
print(df_new)
print("新用户预测结果:", grid_lr.predict(df_new))