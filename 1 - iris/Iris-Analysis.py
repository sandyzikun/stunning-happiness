#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn import model_selection
from random import randint as rand
names = [ "sepal-length", "sepal-width", "petal-length", "petal-width", "class" ]
iris = read_csv(
    "C:\\iris.csv",
    names = names
    )
_Result_File = open("C:\\Users\\admin\\Desktop\\_Iris_Result.txt", mode = "wt")
# 查看数据分布
print("显示数据分布状况及其可视化图像...:")
_Result_File.write( str(iris.describe()) + "\n" )
print(iris.describe())
matplotlib.style.use("seaborn")
_IRIS_Line_X = np.arange( 1, len(iris.values) + 1, 1 )
_IRIS_Line_Y = []
for _Each in iris.values:
    _IRIS_Line_Y.append([ _Each[0], _Each[1], _Each[2], _Each[3] ])
_IRIS_Line_Y = np.array(_IRIS_Line_Y)
plt.plot( _IRIS_Line_X, _IRIS_Line_Y )
plt.savefig("C:\\Users\\admin\\Desktop\\_Iris_ResultFig_Line.jpeg")
plt.show()
iris.plot(
    kind = "box",
    subplots = True,
    layout = (2, 2),
    sharex = False,
    sharey = False,
    fontsize = 8
    ) # 箱线图
plt.savefig("C:\\Users\\admin\\Desktop\\_Iris_ResultFig_Box.jpeg")
plt.show()
iris.hist() # 直方图
plt.savefig("C:\\Users\\admin\\Desktop\\_Iris_ResultFig_Hist.jpeg")
plt.show()
pd.plotting.scatter_matrix(iris) # 多变量图表 - 散点矩阵图
plt.savefig("C:\\Users\\admin\\Desktop\\_Iris_ResultFig_ScatterMatrix.jpeg")
plt.show()
# 分离 训练 & 评估 数据集
_X_Train, _X_Val, _Y_Train, _Y_Val = model_selection.train_test_split(
    iris.values[ : , 0 : (-1) ],
    iris.values[ : , (-1) ],
    test_size = 0.2, # 测试数据集占总数的 20%
    random_state = 7
    )
# 算法审查
# LR : sklearn.linear_model.logistic.LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
# LDA : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# KNN : sklearn.neighbors.classification.KNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier as KNN
# DTC : sklearn.tree.tree.DecisionTreeClassifier
# 有时也称为 "分类与回归树", CART (= Classifier And Regressor Tree)
from sklearn.tree.tree import DecisionTreeClassifier as DTC
# NB : sklearn.naive_bayes.GaussianNB
from sklearn.naive_bayes import GaussianNB as NB
# SVM : sklearn.svm.classes.SVC
from sklearn.svm.classes import SVC as SVM
_Models = {
    "LR" : LR(),
    "LDA" : LDA(),
    "KNN" : KNN(),
    "DTC" : DTC(),
    "NB" : NB(),
    "SVM" : SVM()
    }
# 审查结果比较
print("审查结果比较及其可视化...:")
_Algorithm_CMP_Results = []
_Algorithm_CMP_Result_List = []
_Result_File.write("模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)\n")
print("模型名称" + " " * 6 + "MEAN(准确度)" + " " * 12 + "STD(应该是标准差)")
for _Each in _Models:
    # KFold : K折叠 : sklearn.model_selection._split.KFold(n_splits = 10, random_state = 7)
    # LeaveOneOut : 弃一 : sklearn.model_selection._split.LeaveOneOut()
    # KFold 与 LeaveOneOut 为两种不同的数据集分割策略方案
    cv_results = model_selection.cross_val_score( # cross_val_score : 交叉验证
        _Models[_Each],
        X = _X_Train,
        y = _Y_Train,
        cv = model_selection.KFold(n_splits = 10, random_state = 7),
        # cv = model_selection.LeaveOneOut(),
        scoring = "accuracy" # 不知道这个参数起什么作用, 似乎可以不要
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
# 比较结果可视化
_Fig = plt.figure()
_Fig.suptitle(
    "Algorithm Comparision",
    fontsize = 14
    )
_Ax = _Fig.add_subplot(111)
plt.boxplot(_Algorithm_CMP_Result_List)
_Ax.set_xticklabels(_Models.keys())
plt.savefig("C:\\Users\\admin\\Desktop\\_Iris_ResultFig_AlgoCmp.jpeg")
plt.show()
del _Algorithm_CMP_Result_List
# 筛选至强模型
_Best_CVResults = _Algorithm_CMP_Results[0]
for _Each in _Algorithm_CMP_Results:
    if _Best_CVResults["CV Results"].mean() < _Each["CV Results"].mean():
        _Best_CVResults = _Each
    elif _Best_CVResults["CV Results"].mean() == _Each["CV Results"].mean() and _Best_CVResults["CV Results"].std() > _Each["CV Results"].std():
        _Best_CVResults = _Each
_Result_File.write( "\n选择的算法是: " + _Best_CVResults["Algorithm Name"] + "\n" )
print("选择的算法是: %s" % _Best_CVResults["Algorithm Name"])
_Result_File.write( "MEAN(精确率): " + str(_Best_CVResults["CV Results"]) + ",\nSTD(应该是标准差): " + str(_Best_CVResults["CV Results"].std()) + "\n\n" )
print("MEAN(精确率): %f,\nSTD(应该是标准差): %f" % (_Best_CVResults["CV Results"].mean(), _Best_CVResults["CV Results"].std()))
print()
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
_Result_File.write( str(sklearn.metrics.classification.accuracy_score(y_true = _Y_Val, y_pred = _Predictions)) + "\n" )
_Result_File.write("\n冲突矩阵...:\n")
_Result_File.write( str(sklearn.metrics.classification.confusion_matrix(y_true = _Y_Val, y_pred = _Predictions)) + "\n" )
_Result_File.write("\n分类报告...:\n")
_Result_File.write("\t\t精确率\t召回率\tF1数值\t支持情况\n")
_Result_File.write(str(sklearn.metrics.classification.classification_report(y_true = _Y_Val, y_pred = _Predictions)))
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
_Result_File.close()
input("Press ENTER(or RETURN) To Continue......")
# -*- END -*- #
