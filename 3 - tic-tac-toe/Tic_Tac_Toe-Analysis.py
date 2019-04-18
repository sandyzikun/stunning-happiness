#!/usr/bin/env Python
# -*- coding: utf-8 -*-
PLOTTING_SWITCH = False
################################################################################
# 数据导入
import pandas as pd
TICTACTOE = pd.read_csv(
    "C:\\tic-tac-toe.csv",
    names = [
        "top-left-square", "top-middle-square", "top-right-square",
        "middle-left-square", "middle-middle-square", "middle-right-square",
        "bottom-left-square", "bottom-middle-square", "bottom-right-square",
        "class"
        ]
    )
print(
    "数据基本分布状况信息:",
    TICTACTOE.describe(),
    sep = "\n", end = "\n\n\n"
    )
################################################################################
# 数据清洗
TTT = []
for i in range(len(TICTACTOE.values)):
    TTT.append([])
    for j in range(len(TICTACTOE.values[i])):
        if TICTACTOE.values[i][j] == "x":
            TTT[-1].append(1)
        elif TICTACTOE.values[i][j] == "o":
            TTT[-1].append(-1)
        elif TICTACTOE.values[i][j] == "b":
            TTT[-1].append(0)
        else:
            TTT[-1].append(TICTACTOE.values[i][j])
import numpy as np
TTT = pd.DataFrame(
    TTT,
    columns = [
        "top-left-square", "top-middle-square", "top-right-square",
        "middle-left-square", "middle-middle-square", "middle-right-square",
        "bottom-left-square", "bottom-middle-square", "bottom-right-square",
        "class"
        ]
    )
print(
    "数据基本分布状况信息:",
    TTT.describe(),
    sep = "\n", end = "\n\n\n"
    )
################################################################################
# 可视化成像
import matplotlib, matplotlib.pyplot as plt
matplotlib.style.use("seaborn")
if PLOTTING_SWITCH == True:
############################################################
# 直方图
    TTT.hist(
        sharex = False,
        sharey = False,
        xlabelsize = 1,
        ylabelsize = 1
        )
    plt.show()
############################################################
# 箱线图
    TTT.plot(
        kind = "box",
        subplots = True,
        layout = (3, 3),
        sharex = False,
        sharey = False,
        fontsize = 8
        )
    plt.show()
############################################################
# 密度图
    TTT.plot(
        kind = "density",
        subplots = True,
        layout = (3, 3),
        sharex = False,
        sharey = False,
        fontsize = 8
        )
    plt.show()
############################################################
# 散点矩阵
    pd.plotting.scatter_matrix(frame = TTT)
    plt.show()
############################################################
# 关系矩阵
    _Fig = plt.figure()
    _Ax = _Fig.add_subplot(111)
    _Fig.colorbar(
        mappable = _Ax.matshow(
            Z = TTT.corr(),
            vmin = -1,
            vmax = 1,
            interpolation = "none"
            )
        )
    plt.show()
############################################################
# 数据分布折线图
    _TICTACTOE_LINE_X = np.arange(
        start = 1,
        stop = len(TTT.values) + 1,
        step = 1
        )
    _TICTACTOE_LINE_Y = []
    for x in _TICTACTOE_LINE_X:
        y = []
        for i in range(len(TTT.values[x - 1]) - 1):
            y.append(TTT.values[x - 1][i])
        _TICTACTOE_LINE_Y.append(y)
        pass
    plt.plot(
        _TICTACTOE_LINE_X,
        _TICTACTOE_LINE_Y
        )
    plt.show()
################################################################################
else:
    pass
# 数据训练评估集分离
from sklearn import model_selection
_X_TRAIN, _X_VAL, _Y_TRAIN, _Y_VAL = model_selection.train_test_split(
    TTT.values[ : , 0 : (-1) ],
    TTT.values[ : , (-1) ],
    test_size = 0.2,
    random_state = 7
    )
################################################################################
# 导入算法模型
############################################################
# 导入分类算法
# LR : sklearn.linear_model.logistic.LogisticRegression
from sklearn.linear_model.logistic import LogisticRegression as LR
# LDA : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# KNC : sklearn.neighbors.classification.KNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier as KNC
# GNB : sklearn.naive_bayes.GaussianNB
from sklearn.naive_bayes import GaussianNB as GNB
# DTC : sklearn.tree.tree.DecisionTreeClassifier
from sklearn.tree.tree import DecisionTreeClassifier as DTC
# SVC : sklearn.svm.classes.SVC
from sklearn.svm.classes import SVC
############################################################
# 导入集成算法
# ABC : sklearn.ensemble.weight_boosting.AdaBoostClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier as ABC
# GBC : sklearn.ensemble.gradient_boosting.GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GBC
# RFC : sklearn.ensemble.forest.RandomForestClassifier
# ETsC : sklearn.ensemble.forest.ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier as RFC
from sklearn.ensemble.forest import ExtraTreesClassifier as ETsC
################################################################################
# 算法审查比较
_MODELS = {
    "LR" : LR(),
    "LDA" : LDA(),
    "KNC" : KNC(),
    "GNB" : GNB(),
    "DTC" : DTC(),
    "SVC" : SVC()
    }
############################################################
# 审查结果比较
_ALGORITHM_CMP_RESULTS = []
_ALGORITHM_CMP_RESULT_LIST = []
for _Each_Model in _MODELS:
    _CV_Results = model_selection.cross_val_score(
        estimator = _MODELS[_Each_Model],
        X = _X_TRAIN,
        y = _Y_TRAIN,
        cv = model_selection.KFold(
            n_splits = 10,
            random_state = 7
            ),
        scoring = "accuracy"
        )
    _ALGORITHM_CMP_RESULT_LIST.append(_CV_Results)
    _ALGORITHM_CMP_RESULTS.append(
        {
            "Algorithm Name" : _Each_Model,
            "CV Results" : _CV_Results,
            "Model" : _MODELS[_Each_Model]
            }
        )
    print(
        _Each_Model, "\n",
        " " * 4, "MEAN:   ", _CV_Results.mean(), "\n",
        " " * 4, "STD:    ", _CV_Results.std(), "\n",
        sep = "", end = "\n"
        )
print()
# 当前最优 : KNC-K近邻算法, SVC-资瓷矢量机
############################################################
# 审查结果可视化
_Fig = plt.figure()
_Fig.suptitle(t = "ALGORITHM COMPARISION")
_Ax = _Fig.add_subplot(111)
plt.boxplot(x = _ALGORITHM_CMP_RESULT_LIST)
_Ax.set_xticklabels(labels = _MODELS.keys())
plt.show()
################################################################################
# 预测程序启动...
from sklearn import metrics
############################################################
# K近邻算法预测
_KNC_MODEL = KNC()
_KNC_MODEL.fit(
    X = _X_TRAIN,
    y = _Y_TRAIN
    )
_KNC_PREDICTIONS = _KNC_MODEL.predict(
    X = _X_VAL
    )
print(
    "KNC-K近邻算法预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _KNC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
############################################################
# 资瓷矢量机预测
_SVC_MODEL = SVC()
_SVC_MODEL.fit(
    X = _X_TRAIN,
    y = _Y_TRAIN
    )
_SVC_PREDICTIONS = _SVC_MODEL.predict(
    X = _X_VAL
    )
print(
    "SVC-资瓷矢量机预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _SVC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
################################################################################
# 数据正态化重新预测
from sklearn import preprocessing, pipeline
_STANDARDSCALED_MODELS = {
    "LR" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "LR", LR() ) # STEP2
            ]
        ),
    "LDA" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "LDA", LDA() ) # STEP2
            ]
        ),
    "KNC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "KNC", KNC() ) # STEP2
            ]
        ),
    "GNB" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "GNB", GNB() ) # STEP2
            ]
        ),
    "DTC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "DTC", DTC() ) # STEP2
            ]
        ),
    "SVC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "SVC", SVC() ) # STEP2
            ]
        ),
    }
############################################################
# 数据正态化后审查结果比较
_STANDARDSCALED_ALGORITHM_CMP_RESULTS = []
_STANDARDSCALED_ALGORITHM_CMP_RESULT_LIST = []
for _Each_Model in _STANDARDSCALED_MODELS:
    _CV_Results = model_selection.cross_val_score(
        estimator = _STANDARDSCALED_MODELS[_Each_Model],
        X = preprocessing.StandardScaler().fit(X = _X_TRAIN).transform(X = _X_TRAIN),
        y = _Y_TRAIN, 
        cv = model_selection.KFold(
            n_splits = 10,
            random_state = 7
            ),
        scoring = "accuracy"
        )
    _STANDARDSCALED_ALGORITHM_CMP_RESULT_LIST.append(_CV_Results)
    _STANDARDSCALED_ALGORITHM_CMP_RESULTS.append(
        {
            "Algorithm Name" : _Each_Model,
            "CV Results" : _CV_Results,
            "Model" : _STANDARDSCALED_MODELS[_Each_Model]
            }
        )
    print(
        _Each_Model, "\n",
        " " * 4, "MEAN:   ", _CV_Results.mean(), "\n",
        " " * 4, "STD:    ", _CV_Results.std(), "\n",
        sep = "", end = "\n"
        )
print()
# 当前最优 : KNC-K近邻算法, SVC-资瓷矢量机
############################################################
# 数据正态化后审查结果可视化
_Fig = plt.figure()
_Fig.suptitle(t = "STANDARDSCALED ALGORITHM COMPARISION")
_Ax = _Fig.add_subplot(111)
plt.boxplot(x = _STANDARDSCALED_ALGORITHM_CMP_RESULT_LIST)
_Ax.set_xticklabels(labels = _STANDARDSCALED_MODELS.keys())
plt.show()
################################################################################
# 数据正态化后预测程序启动...
############################################################
# 数据正态化后K近邻算法预测
_STANDARDSCALED_KNC_MODEL = KNC()
_STANDARDSCALED_KNC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_STANDARDSCALED_KNC_MODEL.fit(
    X = _STANDARDSCALED_KNC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_STANDARDSCALED_KNC_PREDICTIONS = _STANDARDSCALED_KNC_MODEL.predict(
    X = _STANDARDSCALED_KNC_SCALER.transform(X = _X_VAL)
    )
print(
    "KNC-K近邻算法数据正态化后预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
############################################################
# 数据正态化后资瓷矢量机预测
_STANDARDSCALED_SVC_MODEL = SVC()
_STANDARDSCALED_SVC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_STANDARDSCALED_SVC_MODEL.fit(
    X = _STANDARDSCALED_SVC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_STANDARDSCALED_SVC_PREDICTIONS = _STANDARDSCALED_SVC_MODEL.predict(
    X = _X_VAL
    )
print(
    "SVC-资瓷矢量机数据正态化后预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
################################################################################
# 数据正态化后参数调整
############################################################
# K近邻算法参数调整
_CHANGEDPARA_STANDARDSCALED_KNC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_CHANGEDPARA_STANDARDSCALED_KNC_PARAM_GRID = {
    "n_neighbors" : [
        1, 3, 5, 7, 9,
        11, 13, 15, 17, 19,
        21
        ]
    }
_CHANGEDPARA_STANDARDSCALED_KNC_MODEL = KNC()
_CHANGEDPARA_STANDARDSCALED_KNC_GRID = model_selection.GridSearchCV(
    estimator = _CHANGEDPARA_STANDARDSCALED_KNC_MODEL,
    param_grid = _CHANGEDPARA_STANDARDSCALED_KNC_PARAM_GRID,
    scoring = "accuracy",
    cv = model_selection.KFold(
        n_splits = 10,
        random_state = 7
        )
    )
_CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT = _CHANGEDPARA_STANDARDSCALED_KNC_GRID.fit(
    X = _CHANGEDPARA_STANDARDSCALED_KNC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_CHANGEDPARA_STANDARDSCALED_KNC_CV_RESULTS = zip(
    _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.cv_results_["mean_test_score"],
    _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.cv_results_["std_test_score"],
    _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.cv_results_["params"]
    )
for _Each_Mean, _Each_Std, _Each_Param in _CHANGEDPARA_STANDARDSCALED_KNC_CV_RESULTS:
    print(
        "MEAN:\n",
        " " * 4, _Each_Mean, "\n",
        "STD:\n",
        " " * 4, _Each_Std, "\n",
        "PARAM:\n",
        " " * 4, _Each_Param, "\n",
        sep = "", end = "\n"
        )
print(
    "最优:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.best_score_, "\n",
    "使用:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.best_params_, "\n",
    sep = "", end = "\n"
    )
print()
########################################
# 参数调整后K近邻算法预测
_BESTPARA_STANDARDSCALED_KNC_MODEL = KNC(
    n_neighbors = _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.best_params_["n_neighbors"]
    )
_BESTPARA_STANDARDSCALED_KNC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_BESTPARA_STANDARDSCALED_KNC_MODEL.fit(
    X = _BESTPARA_STANDARDSCALED_KNC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_BESTPARA_STANDARDSCALED_KNC_PREDICTIONS = _BESTPARA_STANDARDSCALED_KNC_MODEL.predict(
    X = _BESTPARA_STANDARDSCALED_KNC_SCALER.transform(X = _X_VAL)
    )
print(
    "KNC-K近邻算法数据正态化后最优参数调整预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_KNC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
############################################################
# 资瓷矢量机参数调整
"""
_CHANGEDPARA_STANDARDSCALED_SVC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_CHANGEDPARA_STANDARDSCALED_SVC_PARAM_GRID = {
    "C" : [
        0.1, 0.3, 0.5, 0.7, 0.9,
        1.1, 1.3, 1.5, 1.7, 1.9
        ],
    "kernel" : [ "linear", "poly", "rbf", "sigmoid", "precomputed" ]
    }
_CHANGEDPARA_STANDARDSCALED_SVC_MODEL = SVC()
_CHANGEDPARA_STANDARDSCALED_SVC_GRID = model_selection.GridSearchCV(
    estimator = _CHANGEDPARA_STANDARDSCALED_SVC_MODEL,
    param_grid = _CHANGEDPARA_STANDARDSCALED_SVC_PARAM_GRID,
    scoring = "accuracy",
    cv = model_selection.KFold(
        n_splits = 10,
        random_state = 7
        )
    )
_CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT = _CHANGEDPARA_STANDARDSCALED_SVC_GRID.fit(
    X = _CHANGEDPARA_STANDARDSCALED_SVC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_CHANGEDPARA_STANDARDSCALED_SVC_CV_RESULTS = zip(
    _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.cv_results_["mean_test_score"],
    _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.cv_results_["std_test_score"],
    _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.cv_results_["params"]
    )
for _Each_Mean, _Each_Std, _Each_Param in _CHANGEDPARA_STANDARDSCALED_SVC_CV_RESULTS:
    print(
        "MEAN:\n",
        " " * 4, _Each_Mean, "\n",
        "STD:\n",
        " " * 4, _Each_Std, "\n",
        "PARAM:\n",
        " " * 4, _Each_Param, "\n",
        sep = "", end = "\n"
        )
print(
    "最优:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.best_score_, "\n",
    "使用:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.best_params_, "\n",
    sep = "", end = "\n"
    )
print()
########################################
# 参数调整后资瓷矢量机预测
_BESTPARA_STANDARDSCALED_SVC_MODEL = SVC(
    C = _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.best_params_["C"],
    kernel = _CHANGEDPARA_STANDARDSCALED_SVC_GRID_RESULT.best_params_["kernel"]
    )
_BESTPARA_STANDARDSCALED_SVC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_BESTPARA_STANDARDSCALED_SVC_MODEL.fit(
    X = _BESTPARA_STANDARDSCALED_SVC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_BESTPARA_STANDARDSCALED_SVC_PREDICTIONS = _BESTPARA_STANDARDSCALED_SVC_MODEL.predict(
    X = _BESTPARA_STANDARDSCALED_SVC_SCALER.transform(X = _X_VAL)
    )
print(
    "SVC-资瓷矢量机数据正态化后最优参数调整预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_SVC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
############################################################
print()
#"""
################################################################################
# 集成算法测试
_ENSEMBLED_MODELS = {
    "ABC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "ABC", ABC() ) # STEP2
            ]
        ),
    "GBC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "GBC", GBC() ) # STEP2
            ]
        ),
    "RFC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "RFC", RFC() ) # STEP2
            ]
        ),
    "ETC" : pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ), # STEP1
            ( "ETC", ETsC() ) # STEP2
            ]
        )
    }
############################################################
# 集成算法结果比较
_ENSEMBLED_ALGORITHM_CMP_RESULTS = []
_ENSEMBLED_ALGORITHM_CMP_RESULT_LIST = []
for _Each_Model in _ENSEMBLED_MODELS:
    _CV_Results = model_selection.cross_val_score(
        estimator = _ENSEMBLED_MODELS[_Each_Model],
        X = preprocessing.StandardScaler().fit(X = _X_TRAIN).transform(X = _X_TRAIN),
        y = _Y_TRAIN,
        cv = model_selection.KFold(
            n_splits = 10,
            random_state = 7
            ),
        scoring = "accuracy"
        )
    _ENSEMBLED_ALGORITHM_CMP_RESULT_LIST.append(_CV_Results)
    _ENSEMBLED_ALGORITHM_CMP_RESULTS.append(
        {
            "Algorithm Name" : _Each_Model,
            "CV Results" : _CV_Results,
            "Model" : _ENSEMBLED_MODELS[_Each_Model]
            }
        )
    print(
        _Each_Model, "\n",
        " " * 4, "MEAN:   ", _CV_Results.mean(), "\n",
        " " * 4, "STD:    ", _CV_Results.std(), "\n",
        sep = "", end = "\n"
        )
print()
# 当前最优 : GBC-随机梯度上升
############################################################
# 数据正态化后审查结果可视化
_Fig = plt.figure()
_Fig.suptitle(t = "STANDARDSCALED ALGORITHM COMPARISION")
_Ax = _Fig.add_subplot(111)
plt.boxplot(x = _STANDARDSCALED_ALGORITHM_CMP_RESULT_LIST)
_Ax.set_xticklabels(labels = _STANDARDSCALED_MODELS.keys())
plt.show()
################################################################################
# 数据正态化后随机梯度上升预测
_STANDARDSCALED_GBC_MODEL = GBC()
_STANDARDSCALED_GBC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_STANDARDSCALED_GBC_MODEL.fit(
    X = _STANDARDSCALED_GBC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_STANDARDSCALED_GBC_PREDICTIONS = _STANDARDSCALED_GBC_MODEL.predict(
    X = _STANDARDSCALED_GBC_SCALER.transform(X = _X_VAL)
    )
print(
    "GBC-随机梯度上升数据正态化后预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
################################################################################
# 集成算法调参
_CHANGEDPARA_STANDARDSCALED_GBC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_CHANGEDPARA_STANDARDSCALED_GBC_PARAM_GRID = {
    "n_estimators" : [
        10, 50, 100,
        200, 300, 400, 500,
        600, 700, 800, 900
        ]
    }
_CHANGEDPARA_STANDARDSCALED_GBC_MODEL = GBC()
_CHANGEDPARA_STANDARDSCALED_GBC_GRID = model_selection.GridSearchCV(
    estimator = _CHANGEDPARA_STANDARDSCALED_GBC_MODEL,
    param_grid = _CHANGEDPARA_STANDARDSCALED_GBC_PARAM_GRID,
    scoring = "accuracy",
    cv = model_selection.KFold(
        n_splits = 10,
        random_state = 7
        )
    )
_CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT = _CHANGEDPARA_STANDARDSCALED_GBC_GRID.fit(
    X = _CHANGEDPARA_STANDARDSCALED_GBC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_CHANGEDPARA_STANDARDSCALED_GBC_CV_RESULTS = zip(
    _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.cv_results_["mean_test_score"],
    _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.cv_results_["std_test_score"],
    _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.cv_results_["params"]
    )
for _Each_Mean, _Each_Std, _Each_Param in _CHANGEDPARA_STANDARDSCALED_GBC_CV_RESULTS:
    print(
        "MEAN:\n",
        " " * 4, _Each_Mean, "\n",
        "STD:\n",
        " " * 4, _Each_Std, "\n",
        "PARAM:\n",
        " " * 4, _Each_Param, "\n",
        sep = "", end = "\n"
        )
print(
    "最优:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.best_score_, "\n",
    "使用:\n",
    " " * 4, _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.best_params_, "\n",
    sep = "", end = "\n"
    )
print()
########################################
# 参数调整后随机梯度上升算法预测
_BESTPARA_STANDARDSCALED_GBC_MODEL = GBC(
    n_estimators = _CHANGEDPARA_STANDARDSCALED_GBC_GRID_RESULT.best_params_["n_estimators"]
    )
_BESTPARA_STANDARDSCALED_GBC_SCALER = preprocessing.StandardScaler().fit(X = _X_TRAIN)
_BESTPARA_STANDARDSCALED_GBC_MODEL.fit(
    X = _BESTPARA_STANDARDSCALED_GBC_SCALER.transform(X = _X_TRAIN),
    y = _Y_TRAIN
    )
_BESTPARA_STANDARDSCALED_GBC_PREDICTIONS = _BESTPARA_STANDARDSCALED_GBC_MODEL.predict(
    X = _BESTPARA_STANDARDSCALED_GBC_SCALER.transform(X = _X_VAL)
    )
print(
    "GBC-随机梯度上升数据正态化后最优参数调整预测结果:\n",
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _BESTPARA_STANDARDSCALED_GBC_PREDICTIONS
        ),
    "\n",
#
    sep = "", end = "\n"
    )
print()
################################################################################
# 最后裁判 - K近邻算法预测
_MODEL = KNC(
    n_neighbors = _CHANGEDPARA_STANDARDSCALED_KNC_GRID_RESULT.best_params_["n_neighbors"]
    ) # 准备模型
_SCALER = preprocessing.StandardScaler().fit(
    X = _X_TRAIN
    ) # 准备数据转化器
_TRANSFORMED_X_TRAIN = _SCALER.transform(
    X = _X_TRAIN
    ) # 转化训练数据
_TRANSFORMED_X_VAL = _SCALER.transform(
    X = _X_VAL
    ) # 转化检测数据
_MODEL.fit(
    X = _TRANSFORMED_X_TRAIN,
    y = _Y_TRAIN
    ) # 训(tiao)练(jiao)模型
_Y_PRED = _MODEL.predict(
    X = _TRANSFORMED_X_VAL
    ) # 模型预测
print(
    "最终预测结果:\n",
    "使用算法: K最近邻(K Nearest Neighbors)\n"
#
    " " * 4, "ACCURACY_SCORE:\n",
    " " * 8,
    metrics.accuracy_score(
        y_true = _Y_VAL,
        y_pred = _Y_PRED
        ),
    "\n",
#
    " " * 4, "CONFUSION_MATRIX:\n",
    metrics.confusion_matrix(
        y_true = _Y_VAL,
        y_pred = _Y_PRED
        ),
    "\n",
#
    " " * 4, "CLASSIFICATION_REPORT:\n",
    metrics.classification_report(
        y_true = _Y_VAL,
        y_pred = _Y_PRED
        ),
    "\n",
#
    sep = "", end = "\n"
    )
################################################################################
print("-*- END -*-")
del _Each_Model, _Each_Mean, _Each_Std, _Each_Param, _Fig, _Ax, PLOTTING_SWITCH
# -*- END -*- #
