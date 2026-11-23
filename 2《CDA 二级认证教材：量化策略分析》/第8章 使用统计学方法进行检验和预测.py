# -*- coding: utf-8 -*-
"""
第8章 使用统计学方法进行检验和预测

对应教材：
    第8章-使用统计学方法进行检验和预测/假设检验与方差分析/方差分析.ipynb
    第8章-使用统计学方法进行检验和预测/列联表检验/列联表分析.ipynb
    第8章-使用统计学方法进行检验和预测/线性回归/线性回归.ipynb
    第8章-使用统计学方法进行检验和预测/逻辑回归/逻辑回归.ipynb

内容：
1. 方差分析：单因素、双因素及交互作用；
2. 列联表分析：交叉表、行百分比、卡方独立性检验；
3. 线性回归：数据清洗、编码、哑变量、相关分析、模型拟合、共线性/同方差/
   正态性诊断、高次项与交互项；
4. 逻辑回归：交叉表与卡方、一元/多元逻辑回归、混淆矩阵、ROC/AUC、
   逐步回归特征选择、VIF 共线性诊断。

依赖：pandas、numpy、scipy、statsmodels、scikit-learn、matplotlib、seaborn。
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import sklearn.metrics as metrics

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.join(BASE_DIR, "第8章-使用统计学方法进行检验和预测")
ANOVA_DIR = os.path.join(CHAPTER_DIR, "假设检验与方差分析")
CAT_DIR = os.path.join(CHAPTER_DIR, "列联表检验")
LR_DIR = os.path.join(CHAPTER_DIR, "线性回归")
LOGIT_DIR = os.path.join(CHAPTER_DIR, "逻辑回归")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第8章 使用统计学方法进行检验和预测")
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


# ======================================================================
# 一、方差分析
# ======================================================================
def run_anova():
    title("8.1 方差分析")

    data = pd.read_csv(os.path.join(ANOVA_DIR, "creditcard_exp.csv"))
    df = data[["avg_exp", "edu_class", "gender"]].dropna(how="any", axis=0)
    df["edu_class"] = df["edu_class"].astype(str)
    df["gender"] = df["gender"].astype(str)

    # 均值支出字段中的空格视作 0
    df["avg_exp"] = df["avg_exp"].apply(lambda x: 0 if x == " " else x)
    df["avg_exp"] = df["avg_exp"].astype("float")

    print("样本量：", len(df))
    print(df.groupby("edu_class")["avg_exp"].agg(["count", "mean", "std"]))

    # 单因素方差分析：不同教育水平的平均支出是否有显著差异
    model_one = ols("avg_exp ~ C(edu_class)", data=df).fit()
    print("\n单因素方差分析：")
    print(anova_lm(model_one))

    # 双因素方差分析：教育水平 + 性别
    model_two = ols("avg_exp ~ C(edu_class) + C(gender)", data=df).fit()
    print("\n双因素方差分析：")
    print(anova_lm(model_two, typ=2))

    # 含交互作用的双因素方差分析
    model_inter = ols("avg_exp ~ C(edu_class) * C(gender)", data=df).fit()
    print("\n含交互作用的方差分析：")
    print(anova_lm(model_inter, typ=2))

    # 分组箱线图，直观比较各组支出
    plt.figure(figsize=(9, 5))
    sns.boxplot(x="edu_class", y="avg_exp", hue="gender", data=df)
    plt.title("不同教育水平、性别的平均支出")
    plt.tight_layout()
    plt.show()


# ======================================================================
# 二、列联表检验
# ======================================================================
def run_contingency():
    title("8.2 列联表分析与卡方检验")

    accepts = pd.read_csv(os.path.join(CAT_DIR, "accepts.csv"))

    cross_table = pd.crosstab(
        accepts["bankruptcy_ind"], accepts["bad_ind"], margins=True
    )
    print("交叉表（含边际）：\n", cross_table)

    # 行百分比：每一行除以该行合计
    cross_table_rowpct = cross_table.div(cross_table["All"], axis=0)
    print("行百分比：\n", cross_table_rowpct)

    # 卡方检验必须排除 margins 的合计行列
    chi2, p_value, dof, expected = stats.chi2_contingency(cross_table.iloc[:2, :2])
    print(
        "chisq = %6.4f\np-value = %6.4f\ndof = %d\nexpected_freq = %s"
        % (chi2, p_value, dof, expected)
    )
    print("结论：p-value < 0.05，破产标识与坏客户标识存在显著相关关系。" if p_value < 0.05
          else "结论：p-value >= 0.05，不能拒绝独立性假设。")


# ======================================================================
# 三、线性回归
# ======================================================================
def run_linear_regression():
    title("8.3 线性回归：平均支出的影响因素")

    # ---------- 3.1 读取数据 ----------
    data_raw = pd.read_excel(os.path.join(LR_DIR, "LR_practice.xlsx"))
    print(data_raw.head())
    print(data_raw.info())

    # ---------- 3.2 数据清洗 ----------
    # 删除无用变量
    data_raw.drop(["id", "Acc", "edad2"], axis=1, inplace=True)

    # 删除重复值
    data_raw.drop_duplicates(inplace=True)

    # 缺失值：avg_exp 用均值填补
    print("缺失情况：\n", data_raw.isnull().sum())
    data_raw["avg_exp"] = data_raw["avg_exp"].fillna(data_raw["avg_exp"].mean())

    # 性别编码：Male=1，Female=0
    data_raw["gender"] = data_raw["gender"].map({"Male": 1, "Female": 0})

    # 教育水平编码：缺失值作为独立类别参与编码，保证信息不丢失
    label = data_raw["edu_class"].astype(str).unique().tolist()
    print("教育水平类别：", label)
    data_raw["edu_class"] = data_raw["edu_class"].astype(str).apply(lambda x: label.index(x))
    print(data_raw["edu_class"].value_counts())

    # 异常值：Age = 999 用其余样本均值填补
    z = stats.zscore(data_raw["Age"])
    print("Age 的 3 倍标准差异常值索引：", z[(z > 3) | (z < -3)])
    print("异常值原值：", data_raw["Age"].iloc[40])
    data_raw["Age"].iloc[40] = data_raw["Age"].drop(index=40).mean()

    # 箱线图查看连续变量异常值
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_raw["dist_avg_income"])
    plt.title("dist_avg_income 箱线图")
    plt.show()

    # 哑变量转换：drop_first=True 避免虚拟变量陷阱
    dummy = pd.get_dummies(data=data_raw["edu_class"], prefix="edu", drop_first=True)
    print("哑变量：\n", dummy.head())
    data = pd.concat([data_raw, dummy], axis=1)
    print("合并后数据列：", data.columns.tolist())

    # ---------- 3.3 相关分析 ----------
    print("\nPearson 相关系数：\n", data.corr())
    print("\nKendall 相关系数：\n", data.corr(method="kendall"))

    plt.figure(figsize=(6, 4))
    plt.scatter(data["Income"], data["avg_exp"])
    plt.xlabel("Income")
    plt.ylabel("avg_exp")
    plt.title("收入与平均支出散点图")
    plt.show()

    # ---------- 3.4 回归模型 ----------
    edu_cols = [c for c in data.columns if c.startswith("edu_")]
    print("教育哑变量：", edu_cols)

    formula = "avg_exp ~ gender + Age + Income + Ownrent + Selfempl + dist_home_val + dist_avg_income + " + " + ".join(edu_cols)
    print("回归公式：", formula)
    model_1 = ols(formula, data).fit()
    print(model_1.summary())
    print("残差前 5 个：", model_1.resid.head().values)

    # ---------- 3.5 模型调优 ----------
    # 共线性：方差膨胀因子 VIF，经验上 VIF > 10 提示严重共线性
    def vif(df, y_col):
        x_cols = list(df.columns)
        x_cols.remove(y_col)
        formula_vif = y_col + "~" + "+".join(x_cols)
        rsquared = ols(formula_vif, df).fit().rsquared
        return 1.0 / (1.0 - rsquared)

    exog = data[
        [
            "gender",
            "Age",
            "Income",
            "Ownrent",
            "Selfempl",
            "dist_home_val",
            "dist_avg_income",
            "edu_class",
        ]
    ]
    print("\n各变量 VIF：")
    for y_col in exog.columns:
        print("%-16s\t%.4f" % (y_col, vif(df=exog, y_col=y_col)))

    # 同方差：残差图（横坐标为拟合值，纵坐标为残差）
    plt.figure(figsize=(6, 4))
    plt.scatter(model_1.predict(data), model_1.resid)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("拟合值")
    plt.ylabel("残差")
    plt.title("残差图（同方差检验）")
    plt.show()

    # 正态性：Q-Q 图
    plt.figure(figsize=(6, 4))
    stats.probplot(model_1.resid, plot=plt)
    plt.title("残差 Q-Q 图")
    plt.show()

    # 因变量取对数后再拟合
    data["ln_avg_exp"] = np.log(data["avg_exp"])
    formula_ln = "ln_avg_exp ~ gender + Age + Income + Ownrent + Selfempl + dist_home_val + dist_avg_income + " + " + ".join(edu_cols)
    model_ln = ols(formula_ln, data).fit()
    print("\n对数模型：")
    print(model_ln.summary())

    plt.figure(figsize=(6, 4))
    stats.probplot(model_ln.resid, plot=plt)
    plt.title("对数模型残差 Q-Q 图")
    plt.show()

    # 高次项：加入年龄平方项，捕捉非线性关系
    data["Age_sq"] = data["Age"] ** 2
    formula_sq = formula_ln + " + Age_sq"
    model_sq = ols(formula_sq, data).fit()
    print("\n加入 Age^2：")
    print(model_sq.summary())

    # 交互项：年龄 × 性别
    data["Age_gender"] = data["Age"] * data["gender"]
    formula_inter = formula_sq + " + Age_gender"
    model_inter = ols(formula_inter, data).fit()
    print("\n加入 Age×gender 交互项：")
    print(model_inter.summary())

    print("\n模型对比（AIC 越小越好）：")
    for name, m in [("原始", model_1), ("对数", model_ln), ("对数+平方", model_sq), ("对数+平方+交互", model_inter)]:
        print("%-14s AIC=%.2f  R2=%.4f" % (name, m.aic, m.rsquared))


# ======================================================================
# 四、逻辑回归
# ======================================================================
def run_logistic_regression():
    title("8.4 逻辑回归：电信客户流失预测")

    churn = pd.read_csv(os.path.join(LOGIT_DIR, "telecom_churn.csv"))
    print(churn.head())
    print(churn.info())

    # ---------- 4.1 分类变量与流失的相关关系 ----------
    cross_table = pd.crosstab(churn["posTrend"], churn["churn"], margins=True)
    print("posTrend 与 churn 交叉表：\n", cross_table)

    def percConvert(ser):
        return ser / float(ser[-1])

    print("行百分比：\n", cross_table.apply(percConvert, axis=1))

    # 卡方独立性检验（排除 margins 合计行列）
    chi2, p_value, dof, expected = stats.chi2_contingency(cross_table.iloc[:2, :2])
    print("chisq = %.4f, p-value = %.4g, dof = %d" % (chi2, p_value, dof))
    print("结论：通话趋势与流失显著相关。" if p_value < 0.05 else "结论：不显著。")

    # ---------- 4.2 一元逻辑回归 ----------
    plt.figure(figsize=(6, 4))
    churn.plot(x="duration", y="churn", kind="scatter", alpha=0.3)
    plt.title("在网时长与流失散点图")
    plt.show()

    # 随机划分训练集 / 测试集
    train = churn.sample(frac=0.7, random_state=666)
    test = churn[~churn.index.isin(train.index)]
    print("训练集样本量：%i，测试集样本量：%i" % (len(train), len(test)))

    lg = smf.logit("churn ~ duration", train).fit()
    print(lg.summary())

    train["proba"] = lg.predict(train)
    test["proba"] = lg.predict(test)
    print(test[["churn", "proba"]].head())

    # ---------- 4.3 模型评估（阈值 0.3） ----------
    test["prediction"] = (test["proba"] > 0.3).astype("int")
    print("混淆矩阵：\n", pd.crosstab(test["churn"], test["prediction"], margins=True))

    acc = sum(test["prediction"] == test["churn"]) / len(test)
    print("准确率：%.2f%%" % (acc * 100))

    # 不同阈值下的 precision / recall / specificity / F1
    print("\n阈值扫描：")
    for i in np.arange(0.1, 0.9, 0.1):
        prediction = (test["proba"] > i).astype("int")
        confusion_matrix = pd.crosstab(prediction, test["churn"], margins=True)
        precision = confusion_matrix.loc[0, 0] / confusion_matrix.loc["All", 0]
        recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, "All"]
        specificity = confusion_matrix.loc[1, 1] / confusion_matrix.loc[1, "All"]
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(
            "threshold: %.1f, precision: %.2f, recall: %.2f, specificity: %.2f, f1: %.2f"
            % (i, precision, recall, specificity, f1_score)
        )

    # ROC 曲线与 AUC
    fpr_test, tpr_test, th_test = metrics.roc_curve(test["churn"], test["proba"])
    fpr_train, tpr_train, th_train = metrics.roc_curve(train["churn"], train["proba"])

    plt.figure(figsize=(4, 4))
    plt.plot(fpr_test, tpr_test, "b--", label="test")
    plt.plot(fpr_train, tpr_train, "r-", label="train")
    plt.plot([0, 1], [0, 1], "k:", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()

    print("AUC = %.4f" % metrics.auc(fpr_test, tpr_test))
    print("约登指数最优阈值：%.4f" % th_test[(tpr_test - fpr_test).argmax()])

    # ---------- 4.4 多元逻辑回归：连续变量逐步回归 ----------
    def forward_select(data, response):
        """基于 AIC 的前向逐步回归，返回最终模型。"""
        remaining = set(data.columns)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = float("inf"), float("inf")

        while remaining:
            aic_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {}".format(response, " + ".join(selected + [candidate]))
                aic = smf.logit(formula=formula, data=data).fit(disp=0).aic
                aic_with_candidates.append((aic, candidate))

            aic_with_candidates.sort(reverse=True)
            best_new_score, best_candidate = aic_with_candidates.pop()

            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
                print("aic is %.2f, continuing!" % current_score)
            else:
                print("forward selection over!")
                break

        formula = "{} ~ {} ".format(response, " + ".join(selected))
        print("final formula is %s" % formula)
        return smf.logit(formula=formula, data=data).fit()

    candidates = [
        "churn",
        "duration",
        "AGE",
        "incomeCode",
        "nrProm",
        "peakMinAv",
        "peakMinDiff",
        "call_10086",
    ]
    data_for_select = train[candidates]
    lg_m1 = forward_select(data=data_for_select, response="churn")
    print(lg_m1.summary())
    print("原有 %d 个变量，筛选后保留 %d 个参数（含截距）。" % (len(candidates) - 1, len(lg_m1.params)))

    # 分类变量筛选：逐个做卡方检验
    class_col = [
        "posTrend", "prom", "edu_class", "feton", "curPlan", "avgplan",
        "gender", "negTrend", "planChange", "posPlanChange", "negPlanChange",
    ]
    print("\n分类变量卡方检验：")
    for i in class_col:
        tab = pd.crosstab(churn[i], churn["churn"])
        print("%-14s p-value = %.6f" % (i, stats.chi2_contingency(tab)[1]))

    # ---------- 4.5 多元逻辑回归模型 ----------
    # curPlan 与 avgplan 的交叉表中约 99% 的样本取值相同，同时放入会产生完全共线性，
    # 导致部分类别系数无法估计（标准误为 nan）。因此只保留 curPlan。
    formula = (
        "churn ~ duration + AGE + incomeCode + peakMinDiff + call_10086 "
        "+ C(edu_class) + C(feton) + C(posTrend) + C(negTrend) "
        "+ C(curPlan) + C(gender)"
    )
    lg_m = smf.logit(formula=formula, data=train).fit()
    print(lg_m.summary())

    train["proba"] = lg_m.predict(train)
    test["proba"] = lg_m.predict(test)
    print(test[["churn", "proba"]].head(10))

    # ---------- 4.6 多元模型评估 ----------
    test["prediction"] = (test["proba"] > 0.5).astype("int")
    print("混淆矩阵（阈值 0.5）：\n", pd.crosstab(test["churn"], test["prediction"], margins=True))

    acc_m = sum(test["prediction"] == test["churn"]) / len(test)
    print("准确率：%.2f%%" % (acc_m * 100))

    fpr_test, tpr_test, th_test = metrics.roc_curve(test["churn"], test["proba"])
    fpr_train, tpr_train, th_train = metrics.roc_curve(train["churn"], train["proba"])

    plt.figure(figsize=(4, 4))
    plt.plot(fpr_test, tpr_test, "b--", label="test")
    plt.plot(fpr_train, tpr_train, "r-", label="train")
    plt.plot([0, 1], [0, 1], "k:", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("多元逻辑回归 ROC curve")
    plt.legend()
    plt.show()

    print("AUC = %.4f" % metrics.auc(fpr_test, tpr_test))
    opt_threshold = th_test[(tpr_test - fpr_test).argmax()]
    print("约登指数最优阈值：%.4f" % opt_threshold)

    test["prediction"] = (test["proba"] > opt_threshold).astype("int")
    print("最优阈值下混淆矩阵：\n", pd.crosstab(test["churn"], test["prediction"], margins=True))

    # ---------- 4.7 共线性诊断 ----------
    def vif(df, col_i):
        cols = list(df.columns)
        cols.remove(col_i)
        formula_vif = col_i + " ~ " + " + ".join(cols)
        r2 = ols(formula_vif, df).fit().rsquared
        return 1.0 / (1.0 - r2)

    exog = train[candidates].drop(["churn"], axis=1)
    print("\n连续变量 VIF：")
    for i in exog.columns:
        print("%-16s VIF = %.4f" % (i, vif(df=exog, col_i=i)))


# ======================================================================
# 主流程
# ======================================================================
if __name__ == "__main__":
    run_anova()
    run_contingency()
    run_linear_regression()
    run_logistic_regression()
    print("\n第8章全部演示执行完毕。")