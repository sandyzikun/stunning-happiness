#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import read_csv
names = []
for i in range(60):
    names.append( "VAR" + str(i) )
del i
names.append("class")
sonar = read_csv(
    "C:\\sonar.all-data.csv",
    # header = None
    names = names
    )
_Result_File = open("C:\\Users\\admin\\Desktop\\_Sonar_Result.txt", mode = "wt")
# 查看数据分布
print("显示数据分布状况及其可视化图像...:")
_Result_File.write( str(sonar.describe()) + "\n\n" )
print(sonar.describe())
print()
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use("seaborn")
sonar.hist(
    sharex = False,
    sharey = False,
    xlabelsize = 1,
    ylabelsize = 1
    ) # 直方图
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_Hist.jpeg")
plt.show()
sonar.plot(
    kind = "box",
    subplots = True,
    layout = (6, 10),
    sharex = False,
    sharey = False,
    fontsize = 8
    ) # 箱线图
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_Box.jpeg")
plt.show()
sonar.plot(
    kind = "density",
    subplots = True,
    layout = (6, 10),
    sharex = False,
    legend = False,
    fontsize = 8
    ) # 密度图
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_Density.jpeg")
plt.show()
# 关系矩阵图
_Fig = plt.figure()
_Ax = _Fig.add_subplot(111)
_Cax = _Ax.matshow(
    sonar.corr(),
    vmin = -1,
    vmax = 1,
    interpolation = "none"
    )
_Fig.colorbar(_Cax)
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_ConnectionMatrix.jpeg")
plt.show()
"""
pd.plotting.scatter_matrix(sonar) # 多变量图表 - 散点矩阵图
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_ScatterMatrix.jpeg")
plt.show() # 这个图片太大了, 恐怕画不出来, 别让它画了
print("分布情况:")
print(sonar.groupby(60).size()) # WTH?
"""
# 分离 训练 & 评估 数据集
from sklearn import model_selection
_X_Train, _X_Val, _Y_Train, _Y_Val = model_selection.train_test_split(
    sonar.values[ : , 0 : (-1) ].astype(float),
    sonar.values[ : , (-1) ],
    test_size = 0.2,
    random_state = 7
    )
# 算法审查
# LR : sklearn.linear_model.logistic.LogisticRegression
# LRCV : sklearn.linear_model.logistic.LogisticRegressionCV
from sklearn.linear_model.logistic import LogisticRegression as LR, LogisticRegressionCV as LRCV
# LDA : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
# QDA : sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
# KNC : sklearn.neighbors.classification.KNeighborsClassifier
# RNC : sklearn.neighbors.classification.RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier as KNC, RadiusNeighborsClassifier as RNC
# DTC : sklearn.tree.tree.DecisionTreeClassifier
# ETC : sklearn.tree.tree.ExtraTreeClassifier
# 有时也称为 "分类与回归树", CART (= Classifier And Regressor Tree)
from sklearn.tree.tree import DecisionTreeClassifier as DTC, ExtraTreeClassifier as ETC
# GNB : sklearn.naive_bayes.GaussianNB
# BDNB : sklearn.naive_bayes.BaseDiscreteNB
# MNB : sklearn.naive_bayes.MultinomialNB
# BNB : sklearn.naive_bayes.BernoulliNB
from sklearn.naive_bayes import GaussianNB as GNB, BaseDiscreteNB as BDNB, MultinomialNB as MNB, BernoulliNB as BNB
# LSVC : sklearn.svm.classes.LinearSVC
# SVC : sklearn.svm.classes.SVC
# NSVC : sklearn.svm.classes.NuSVC
# OCSVM : sklearn.svm.classes.OneClassSVM
from sklearn.svm.classes import LinearSVC as LSVC, SVC, NuSVC as NSVC, OneClassSVM as OCSVM
# ABC : sklearn.ensemble.weight_boosting.AdaBoostClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier as ABC
# GBC : sklearn.ensemble.gradient_boosting.GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GBC
# RFC : sklearn.ensemble.forest.RandomForestClassifier
# ETsC : sklearn.ensemble.forest.ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier as RFC
from sklearn.ensemble.forest import ExtraTreesClassifier as ETsC
_Models = {
    "LR" : LR(),
    "LRCV" : LRCV(),
    "LDA" : LDA(),
    "QDA" : QDA(),
    "KNC" : KNC(),
    # "RNC" : RNC(),
    "DTC" : DTC(),
    "ETC" : ETC(),
    "GNB" : GNB(),
    # "BDNB" : BDNB(),
    "MNB" : MNB(),
    "BNB" : BNB(),
    "LSVC" : LSVC(),
    "SVC" : SVC(),
    "NSVC" : NSVC(),
    # "OCSVM" : OCSVM()
    }
# 审查结果比较
print("审查结果比较及其可视化...:")
_Algorithm_CMP_Results = []
_Algorithm_CMP_Result_List = []
_Result_File.write("模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)\n")
print("模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)")
for _Each in _Models:
    cv_results = model_selection.cross_val_score(
        _Models[_Each],
        X = _X_Train,
        y = _Y_Train,
        cv = model_selection.KFold(n_splits = 10, random_state = 7),
        scoring = "accuracy"
    )
    _Result_File.write( _Each + " : \t\t" + str(cv_results.mean()) + " \t\t" + str(cv_results.std()) + "\n" )
    print("%s : \t\t%f \t\t%f" % (_Each, cv_results.mean(), cv_results.std()))
    _Algorithm_CMP_Results.append(
        {
            "Algorithm Name" : _Each,
            "CV Results" : cv_results,
            "Model" : _Models[_Each]
            }
        )
    _Algorithm_CMP_Result_List.append(cv_results)
del cv_results, _Each
print()
# 比较结果可视化
_Fig = plt.figure()
_Fig.suptitle(
    "Algorithm Comparision",
    fontsize = 14
    )
_Ax = _Fig.add_subplot(111)
plt.boxplot(_Algorithm_CMP_Result_List)
_Ax.set_xticklabels(_Models.keys())
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_AlgoCmp.jpeg")
plt.show()
del _Algorithm_CMP_Result_List
# 筛选至强模型
_Best_CVResults = _Algorithm_CMP_Results[0]
for _Each in _Algorithm_CMP_Results:
    if _Best_CVResults["CV Results"].mean() < _Each["CV Results"].mean():
        _Best_CVResults = _Each
    elif _Best_CVResults["CV Results"].mean() == _Each["CV Results"].mean() and _Best_CVResults["CV Results"].std() > _Each["CV Results"].std():
        _Best_CVResults = _Each
_Result_File.write(("\n选择的算法是: %s\n\n" % _Best_CVResults["Algorithm Name"]))
print("选择的算法是: %s" % _Best_CVResults["Algorithm Name"])
_Result_File.write("MEAN(精确率): %f,        STD(应该是标准差): %f\n\n" % (_Best_CVResults["CV Results"].mean(), _Best_CVResults["CV Results"].std()))
print("MEAN(精确率): %f,        STD(应该是标准差): %f" % (_Best_CVResults["CV Results"].mean(), _Best_CVResults["CV Results"].std()))
print()
################################################################################
#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
################################################################################
# 开始预测
_Best_Model = _Best_CVResults["Model"]
_Best_Model.fit(
    X = _X_Train,
    y = _Y_Train
    )
_Predictions = _Best_Model.predict(_X_Val)
# 读出预测结果
import sklearn.metrics.classification
_Result_File.write("Accuracy分数...:\n")
_Result_File.write( str(sklearn.metrics.classification.accuracy_score(y_true = _Y_Val, y_pred = _Predictions)) + "\n\n" )
_Result_File.write("冲突矩阵...:\n")
_Result_File.write( str(sklearn.metrics.classification.confusion_matrix(y_true = _Y_Val, y_pred = _Predictions)) + "\n\n" )
_Result_File.write("分类报告...:\n")
_Result_File.write("\t\t精确率\t召回率\tF1数值\t支持情况\n")
_Result_File.write( str(sklearn.metrics.classification.classification_report(y_true = _Y_Val, y_pred = _Predictions)) + "\n\n" )
print(
    "Accuracy分数...:",
    sklearn.metrics.classification.accuracy_score(y_true = _Y_Val, y_pred = _Predictions),
    "\n冲突矩阵...:",
    sklearn.metrics.classification.confusion_matrix(y_true = _Y_Val, y_pred = _Predictions),
    "\n分类报告...:",
    "\t\t精确率\t召回率\tF1数值\t支持情况",
    sklearn.metrics.classification.classification_report(y_true = _Y_Val, y_pred = _Predictions),
    sep = "\n"
    )
# -*- END...?
# 刚才的一系列操作并不完美, 也不尽善尽美,
# 数据分布的多样性导致SVM不够准确,
# 接下来将对所有数据进行正态化, 然后再次评估,
# 采用 Pipeline 开始流程化处理!
_Result_File.write("正态化模型重新处理:...\n\n")
print("正态化模型重新处理...:\n")
# 正态化数据处理
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
_Pipeline_Models = {
    "Scaler-LR" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "LR", LR() )
            ]
        ),
    "Scaler-LRCV" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "LRCV", LRCV() )
            ]
        ),
    "Scaler-LDA" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "LDA", LDA() )
            ]
        ),
    "Scaler-QDA" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "QDA", QDA() )
            ],
        ),
    "Scaler-KNC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "KNC", KNC() )
            ]
        ),
    # "Scaler-RNC" : Pipeline(
        # [
            # ( "Scaler", StandardScaler() ),
            # ( "RNC", RNC() )
            # ]
        # ),
    "Scaler-DTC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "DTC", DTC() )
            ]
        ),
    "Scaler-ETC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "ETC", ETC() )
            ]
        ),
    "Scaler-GNB" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "GNB", GNB() )
            ]
        ),
    # "Scaler-BDNB" : Pipeline(
        # [
            # ( "Scaler", StandardScaler() ),
            # ( "BDNB", BDNB() )
            # ]
        # ),
    # "Scaler-MNB" : Pipeline(
        # [
            # ( "Scaler", StandardScaler() ),
            # ( "MNB", MNB() )
            # ]
        # ),
    "Scaler-BNB" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "BNB", BNB() )
            ]
        ),
    "Scaler-LSVC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "LSVC", LSVC() )
            ]
        ),
    "Scaler-SVC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "SVC", SVC() )
            ]
        ),
    "Scaler-NSVC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "NSVC", NSVC() )
            ]
        ),
    # "Scaler-OCSVM" : Pipeline(
        # [
            # ( "Scaler", StandardScaler() ),
            # ( "OCSVM", OCSVM() )
            # ]
        # ),
    }
print("(正态化后的)审查结果比较及其可视化...:")
_StandardScaled_Algorithm_CMP_Results = []
_StandardScaled_Algorithm_CMP_Result_List = []
_Result_File.write( "正态化后...:\n" + "模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)\n" )
print("模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)")
for _Each in _Pipeline_Models:
    cv_results = model_selection.cross_val_score(
        _Pipeline_Models[_Each],
        X = _X_Train,
        y = _Y_Train,
        cv = model_selection.KFold(n_splits = 10, random_state = 7),
        scoring = "accuracy"
        )
    _Result_File.write( "%s : \t\t%f \t\t%f" % (_Each, cv_results.mean(), cv_results.std()) + "\n" )
    print("%s : \t\t%f \t\t%f" % (_Each, cv_results.mean(), cv_results.std()))
    _StandardScaled_Algorithm_CMP_Results.append(
        {
            "Algorithm Name" : _Each,
            "CV Results" : cv_results,
            "Model" : _Pipeline_Models[_Each]
            }
        )
    _StandardScaled_Algorithm_CMP_Result_List.append(cv_results)
del cv_results, _Each
_Result_File.write("\n")
print()
# 比较结果可视化
_Fig = plt.figure()
_Fig.suptitle(
    "Scaled Algorithm Comparision",
    fontsize = 14
    )
_Ax = _Fig.add_subplot(111)
plt.boxplot(_StandardScaled_Algorithm_CMP_Result_List)
_Ax.set_xticklabels(_Pipeline_Models.keys())
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_Scaled_AlgoCmp.jpeg")
plt.show()
del _StandardScaled_Algorithm_CMP_Result_List
# K近邻参数调整
_Result_File.write("K近邻参数调整...:\n")
print("开始K近邻参数调整...:")
_KNC_Scaler = StandardScaler().fit(_X_Train)
_KNC_Rescaled_X = _KNC_Scaler.transform(_X_Train)
_KNC_Param_Grid = {
    "n_neighbors" : [
        1, 3, 5, 7, 9,
        11, 13, 15, 17, 19,
        21
        ]
    }
_KNC_Model = KNC()
_KNC_KFold = model_selection.KFold(
    n_splits = 10,
    random_state = 7
    )
_KNC_Grid = model_selection.GridSearchCV(
    estimator = _KNC_Model,
    param_grid = _KNC_Param_Grid,
    scoring = "accuracy",
    cv = _KNC_KFold
    )
_KNC_Grid_Result = _KNC_Grid.fit(
    X = _KNC_Rescaled_X,
    y = _Y_Train
    )
_Result_File.write( "最优: %s, 使用: %s" % ( _KNC_Grid_Result.best_score_, _KNC_Grid_Result.best_params_ ) + "\n\n" )
print( "最优: %s, 使用: %s" % ( _KNC_Grid_Result.best_score_, _KNC_Grid_Result.best_params_ ) )
_KNC_CV_Results = zip(
    _KNC_Grid_Result.cv_results_["mean_test_score"],
    _KNC_Grid_Result.cv_results_["std_test_score"],
    _KNC_Grid_Result.cv_results_["params"]
    )
_Result_File.write( "_KNC_MEAN \t" + "_KNC_STD \t" + "_KNC_PARAM" + "\n" )
print( "_KNC_MEAN \t" + "_KNC_STD \t" + "_KNC_PARAM" )
for _KNC_MEAN, _KNC_STD, _KNC_PARAM in _KNC_CV_Results:
    _Result_File.write( ( "%f \t" + "%f \t" + "%r" ) % ( _KNC_MEAN, _KNC_STD, _KNC_PARAM ) + "\n" )
    print( ( "%f \t" + "%f \t" + "%r" ) % ( _KNC_MEAN, _KNC_STD, _KNC_PARAM ) )
_Result_File.write("\n")
print()
# 资瓷矢量机参数调整
# 破参数, 劳资不调了!
"""
_Result_File.write("资瓷矢量机参数调整...:\n")
print("开始资瓷矢量机参数调整...:")
_SVC_Scaler = StandardScaler().fit(_X_Train)
_SVC_Rescaled_X = _SVC_Scaler.transform(_X_Train).astype(float)
_SVC_Rescaled_X_Val = _SVC_Scaler.transform(_X_Val)
_SVC_Param_Grid = {}
_SVC_Param_Grid["C"] = [
    0.1, 0.3, 0.5, 0.7, 0.9,
    1.0, 1.3, 1.5, 1.7, 2,0
    ]
_SVC_Param_Grid["kernel"] = [ "linear", "poly", "rbf", "sigmoid", "precomputed" ]
_SVC_Model = SVC()
_SVC_KFold = model_selection.KFold(
    n_splits = 10,
    random_state = 7
    )
_SVC_Grid = model_selection.GridSearchCV(
    estimator = _SVC_Model,
    param_grid = _SVC_Param_Grid,
    scoring = "accuracy",
    cv = _SVC_KFold
    )
_SVC_Grid_Result = _SVC_Grid.fit(
    X = _SVC_Rescaled_X,
    y = _Y_Train
    )
_Result_File.write( "最优: %s, 使用: %s" % ( _SVC_Grid_Result.best_score_, _SVC_Grid_Result.best_params_ ) + "\n\n" )
print( "最优: %s, 使用: %s" % ( _SVC_Grid_Result.best_score_, _SVC_Grid_Result.best_params_ ) )
_SVC_CV_Results = zip(
    _SVC_Grid_Result.cv_results_["mean_test_score"],
    _SVC_Grid_Result.cv_results_["std_test_score"],
    _SVC_Grid_Result.cv_results_["params"]
    )
_Result_File.write( "_SVC_MEAN \t" + "_SVC_STD \t" + "_SVC_PARAM" + "\n" )
print( "_SVC_MEAN \t" + "_SVC_STD \t" + "_SVC_PARAM" )
for _SVC_MEAN, _SVC_STD, _SVC_PARAM in _SVC_CV_Results:
    _Result_File.write( ( "%f \t" + "%f \t" + "%r" ) % ( _SVC_MEAN, _SVC_STD, _SVC_PARAM ) + "\n" )
    print( ( "%f \t" + "%f \t" + "%r" ) % ( _SVC_MEAN, _SVC_STD, _SVC_PARAM ) )
_Result_File.write("\n")
print()
# """
# 集成算法
_Result_File.write("集成算法处理:...\n\n")
print("集成算法处理...:\n")
_Ensemble_Models = {
    # 袋装算法
    "Scaled-ABC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "ABC", ABC() )
            ]
        ),
    "Scaled-GBC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "GBC", GBC() )
            ]
        ),
    # 提升算法
    "Scaled-RFC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "RFC", RFC() )
            ]
        ),
    "Scaled-ETsC" : Pipeline(
        [
            ( "Scaler", StandardScaler() ),
            ( "ETsC", ETsC() )
            ]
        ),
    }
print("集成算法审查结果比较及其可视化...:")
_Ensembled_Algorithm_CMP_Results = []
_Ensembled_Algorithm_CMP_Result_List = []
_Result_File.write( "集成算法...:\n" + "模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)\n" )
print( "模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)" )
for _Each in _Ensemble_Models:
    cv_results = model_selection.cross_val_score(
        _Ensemble_Models[_Each],
        X = _X_Train,
        y = _Y_Train,
        cv = model_selection.KFold(n_splits = 10, random_state = 7),
        scoring = "accuracy"
    )
    _Result_File.write( "%s : \t\t%f \t\t%f" % (_Each, cv_results.mean(), cv_results.std()) + "\n" )
    print("%s : \t\t%f \t\t%f" % (_Each, cv_results.mean(), cv_results.std()))
    _Ensembled_Algorithm_CMP_Results.append(
        {
            "Algorithm Name" : _Each,
            "CV Results" : cv_results,
            "Model" : _Ensemble_Models[_Each]
            }
        )
    _Ensembled_Algorithm_CMP_Result_List.append(cv_results)
del cv_results, _Each
_Result_File.write("\n")
print()
# 比较结果可视化
_Fig = plt.figure()
_Fig.suptitle(
    "Ensembled Algorithm Comparision",
    fontsize = 14
    )
_Ax = _Fig.add_subplot(111)
plt.boxplot(_Ensembled_Algorithm_CMP_Result_List)
_Ax.set_xticklabels(_Ensemble_Models.keys())
plt.savefig("C:\\Users\\admin\\Desktop\\_Sonar_ResultFig_Ensembled_AlgoCmp.jpeg")
plt.show()
del _Ensembled_Algorithm_CMP_Result_List
# 随机梯度上升算法调参
_Result_File.write("随机梯度上升算法参数调整...:\n")
print("开始随机梯度上升算法参数调整...:")
_GBC_Scaler = StandardScaler().fit(_X_Train)
_GBC_Rescaled_X = _GBC_Scaler.transform(_X_Train)
_GBC_Param_Grid = {
    "n_estimators" : [
        10, 50, 100, 200,
        300, 400, 500, 600,
        700, 800, 900
        ]
    }
_GBC_Model = GBC()
_GBC_KFold = model_selection.KFold(
    n_splits = 10,
    random_state = 7
    )
_GBC_Grid = model_selection.GridSearchCV(
    estimator = _GBC_Model,
    param_grid = _GBC_Param_Grid,
    scoring = "accuracy",
    cv = _GBC_KFold
    )
_GBC_Grid_Result = _GBC_Grid.fit(
    X = _GBC_Rescaled_X,
    y = _Y_Train
    )
_Result_File.write( "最优: %s, 使用: %s" % ( _GBC_Grid_Result.best_score_, _GBC_Grid_Result.best_params_ ) + "\n\n" )
print( "最优: %s, 使用: %s" % ( _GBC_Grid_Result.best_score_, _GBC_Grid_Result.best_params_ ) )
_GBC_CV_Results = zip(
    _GBC_Grid_Result.cv_results_["mean_test_score"],
    _GBC_Grid_Result.cv_results_["std_test_score"],
    _GBC_Grid_Result.cv_results_["params"]
    )
_Result_File.write( "_GBC_MEAN \t" + "_GBC_STD \t" + "_GBC_PARAM" + "\n" )
print( "_GBC_MEAN \t" + "_GBC_STD \t" + "_GBC_PARAM" )
for _GBC_MEAN, _GBC_STD, _GBC_PARAM in _GBC_CV_Results:
    _Result_File.write( ( "%f \t" + "%f \t" + "%r" ) % ( _GBC_MEAN, _GBC_STD, _GBC_PARAM ) + "\n" )
    print( ( "%f \t" + "%f \t" + "%r" ) % ( _GBC_MEAN, _GBC_STD, _GBC_PARAM ) )
_Result_File.write("\n")
print()
# Final Trail - 终结性裁判
# Final Model - 模型终结化
_FINAL_Scaler = StandardScaler().fit(_X_Train)
_FINAL_Rescaled_X = _FINAL_Scaler.transform(_X_Train)
_FINAL_Model = SVC(
    C = 1.5,
    kernel = "rbf"
    )
_FINAL_Model.fit(
    X = _FINAL_Rescaled_X,
    y = _Y_Train
    )
_FINAL_Rescaled_X_Val = _FINAL_Scaler.transform(_X_Val)
# 终结性模型评估启动
_FINAL_Predictions = _FINAL_Model.predict(_FINAL_Rescaled_X_Val)
_Result_File.write("终结性裁判启动...:\n\n")
_Result_File.write("Accuracy分数...:\n")
_Result_File.write( str(sklearn.metrics.classification.accuracy_score(y_true = _Y_Val, y_pred = _FINAL_Predictions)) + "\n\n" )
_Result_File.write("冲突矩阵...:\n")
_Result_File.write( str(sklearn.metrics.classification.confusion_matrix(y_true = _Y_Val, y_pred = _FINAL_Predictions)) + "\n\n" )
_Result_File.write("分类报告...:\n")
_Result_File.write("\t\t精确率\t召回率\tF1数值\t支持情况\n")
_Result_File.write( str(sklearn.metrics.classification.classification_report(y_true = _Y_Val, y_pred = _FINAL_Predictions)) + "\n\n" )
print(
    "Accuracy分数...:",
    sklearn.metrics.classification.accuracy_score(y_true = _Y_Val, y_pred = _FINAL_Predictions),
    "\n冲突矩阵...:",
    sklearn.metrics.classification.confusion_matrix(y_true = _Y_Val, y_pred = _FINAL_Predictions),
    "\n分类报告...:",
    "\t\t精确率\t召回率\tF1数值\t支持情况",
    sklearn.metrics.classification.classification_report(y_true = _Y_Val, y_pred = _FINAL_Predictions),
    sep = "\n"
    )
