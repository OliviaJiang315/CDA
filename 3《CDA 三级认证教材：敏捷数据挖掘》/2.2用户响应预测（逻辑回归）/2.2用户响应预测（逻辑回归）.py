# -*- coding: utf-8 -*-
"""
2.2 用户响应预测（逻辑回归）

项目背景
--------
基于信用卡客户的人口统计、收入与行为特征，预测客户对营销活动的响应概率（Acc），
并对响应概率排序，辅助营销名单筛选。

建模流程
--------
    1. 数据读取、变量划分与训练/测试集切分
    2. statsmodels 逻辑回归与不同阈值下的混淆矩阵评估
    3. 连续变量标准化
    4. sklearn 逻辑回归
    5. 递归特征消除（RFECV）筛选变量
    6. 正则化逻辑回归（交叉验证选择 C）
    7. 分类报告、ROC 曲线与 AUC
    8. 样本外预测、结果落盘与模型持久化

数据文件
--------
    creditcard_exp.csv                    训练样本（含响应标签 Acc）
    creditcard_exp_out_of_sample.csv      样本外客户

依赖
----
    numpy pandas scipy matplotlib statsmodels scikit-learn joblib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '2.2用户响应预测'
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

# 中文字体设置，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 一、数据读取
# ============================================================
data = pd.read_csv(BASE_DIR / "creditcard_exp.csv", skipinitialspace=True)
print(data.head())

oos_data = pd.read_csv(
    BASE_DIR / "creditcard_exp_out_of_sample.csv", skipinitialspace=True
)
print(oos_data.head())

data = data.drop("avg_exp", axis=1).drop("avg_exp_ln", axis=1).drop("age2", axis=1)
oos_data = oos_data.drop("age2", axis=1)
oos_data["edu_class"].replace({0: 1}, inplace=False)


# ============================================================
# 二、训练集 / 测试集切分
# ============================================================
train = data.sample(frac=0.7, random_state=1234).copy()
test = data[~data.index.isin(train.index)].copy()
print(" 训练集样本量: %i \n 测试集样本量: %i" % (len(train), len(test)))


# ============================================================
# 三、statsmodels 逻辑回归：Acc ~ Income
# ============================================================
lr_model = smf.logit(formula="Acc ~ Income", data=train).fit()
print(lr_model.summary())

test["Pred_Acc"] = lr_model.predict(test)
print(test["Pred_Acc"].head())


# ============================================================
# 四、不同阈值下的分类评估
# ============================================================
for i in np.arange(0.1, 1, 0.2):
    prediction = (test["Pred_Acc"] > i).astype("int")
    confusion_matrix = pd.crosstab(prediction, test["Acc"], margins=True)
    precision = confusion_matrix.iloc[0, 0] / confusion_matrix.loc["All", 0]
    recall = confusion_matrix.iloc[0, 0] / confusion_matrix.loc[0, "All"]
    specificity = confusion_matrix.iloc[1, 1] / confusion_matrix.loc[1, "All"]
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Threshold: %.2f" % i)
    print(
        "Precision: %.2f, Recall:%.2f ,Specificity:%.2f ,f1_score:%.2f \n"
        % (precision, recall, specificity, f1_score)
    )


# ============================================================
# 五、连续变量标准化
# ============================================================
y_col = "Acc"
x_col_continuous = ["Age", "Income", "dist_home_val", "dist_avg_income", "high_avg"]
x_col_discrete = ["gender", "Ownrent", "Selfempl", "edu_class"]
x_col_all = x_col_continuous + x_col_discrete

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[x_col_continuous] = scaler.fit(train[x_col_continuous]).transform(
    train[x_col_continuous]
)
test[x_col_continuous] = scaler.transform(test[x_col_continuous])
oos_data[x_col_continuous] = scaler.transform(oos_data[x_col_continuous])


# ============================================================
# 六、sklearn 逻辑回归
# ============================================================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train[x_col_all], train[y_col])

print("Prediction of test: ")
print(lr.predict(test[x_col_all]))
print("Probability of test: ")
print(lr.predict_proba(test[x_col_all]))


# ============================================================
# 七、递归特征消除（RFECV）
# ============================================================
from sklearn.feature_selection import RFECV

rfecv = RFECV(lr, cv=3, scoring="accuracy", n_jobs=-1)
rfecv.fit(train[x_col_all], train[y_col])

selected_cols = np.array(x_col_all)[rfecv.support_]
print("Ranking of columns: ", rfecv.ranking_)
print("Selected columns: ", selected_cols)


# ============================================================
# 八、正则化逻辑回归（交叉验证选择 C）
# ============================================================
from sklearn.linear_model import LogisticRegressionCV

lrcv = LogisticRegressionCV(Cs=10, cv=4, n_jobs=-1)
lrcv.fit(train[selected_cols], train[y_col])
print("Regulation strength: ", lrcv.C_)


# ============================================================
# 九、预测与评估
# ============================================================
train_prediction = lrcv.predict(train[selected_cols])
test_prediction = lrcv.predict(test[selected_cols])
train_proba = lrcv.predict_proba(train[selected_cols])[:, 1]
test_proba = lrcv.predict_proba(test[selected_cols])[:, 1]

from sklearn.metrics import auc, classification_report, roc_curve

print(classification_report(test["Acc"], test_prediction))

fpr_test, tpr_test, th_test = roc_curve(test["Acc"], test_proba)
fpr_train, tpr_train, th_train = roc_curve(train["Acc"], train_proba)

fig, ax = plt.subplots(figsize=[5, 5])
ax.set_xlabel("假正率", fontsize=15)
ax.set_ylabel("真正率", fontsize=15)
ax.plot(fpr_test, tpr_test, "b-", label="测试集")
ax.plot(fpr_train, tpr_train, "r-", label="训练集")
ax.legend(loc="lower right", fontsize=12)
plt.title("ROC 曲线", fontsize=20)
save_show()

print("AUC on test = %6.4f" % auc(fpr_test, tpr_test))


# ============================================================
# 十、样本外预测与结果落盘
# ============================================================
print(oos_data.head())

output_data_dir = BASE_DIR / "data_processed"
output_data_dir.mkdir(exist_ok=True)
output_data_path = output_data_dir / "creditcard_exp_logistic_regression.csv"

oos_data["Pred_Acc"] = lrcv.predict_proba(oos_data[selected_cols])[:, 1]
oos_data.to_csv(output_data_path, index=False)
print("样本外预测结果已保存至:", output_data_path)


# ============================================================
# 十一、模型持久化
# ============================================================
import joblib

output_dir = BASE_DIR / "model"
output_dir.mkdir(exist_ok=True)
save_path = output_dir / "creditcard_logisticRegression.model"

joblib.dump(lrcv, save_path)
model_reload = joblib.load(save_path)
print(model_reload.predict(oos_data[selected_cols]))
