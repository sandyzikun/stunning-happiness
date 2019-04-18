#!/usr/bin/env Python
# -*- coding: utf-8 -*-
_Plotting_Switch = False
################################################################################
# 导入数据
import numpy as np, pandas as pd
CANCER = pd.read_csv(
    filepath_or_buffer = "C:\\breast-CANCER-wisconsin.csv",
    names = [
        "Sample code number",
        "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
        "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
        "Bland Chromatin", "Normal Nucleoli", "Mitoses", 
        "class"
        ]
    )
print(CANCER.describe(), "\n\n\n", sep = "", end = "")
################################################################################
# 数据清洗
CANCER = list(CANCER.values[ : , 1 : ])
for i in range(len(CANCER)):
    CANCER[i] = list(CANCER[i])
    if CANCER[i][5] == "?":
        CANCER[i][5] = 0
    else:
        CANCER[i][5] = int(CANCER[i][5])
CANCER = pd.DataFrame(
    data = CANCER,
    columns = [
        "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
        "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
        "Bland Chromatin", "Normal Nucleoli", "Mitoses", 
        "class"
        ]
    )
print(CANCER.describe(), "\n\n\n", sep = "", end = "")
################################################################################
# 数据可视化
import matplotlib, matplotlib.pyplot as plt
matplotlib.style.use("seaborn")
if _Plotting_Switch == True:
# 直方图
    CANCER.hist(
        sharex = False,
        sharey = False,
        xlabelsize = 1,
        ylabelsize = 1
        )
    plt.show()
# 箱线图
    CANCER.plot(
        kind = "box",
        subplots = True,
        layout = (3, 4),
        sharex = False,
        sharey = False,
        fontsize = 8
        )
    plt.show()
else:
    pass
################################################################################
if True:
    # 装备算法模型 - 分类算法
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
    # 装备算法模型 - 集成算法
    # ABC : sklearn.ensemble.weight_boosting.AdaBoostClassifier
    from sklearn.ensemble.weight_boosting import AdaBoostClassifier as ABC
    # GBC : sklearn.ensemble.gradient_boosting.GradientBoostingClassifier
    from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GBC
    # RFC : sklearn.ensemble.forest.RandomForestClassifier
    from sklearn.ensemble.forest import RandomForestClassifier as RFC
    # ETC : sklearn.ensemble.forest.ExtraTreesClassifier
    from sklearn.ensemble.forest import ExtraTreesClassifier as ETC
else:
    pass
################################################################################
# 数据集分离
from sklearn import model_selection
_X_TRAIN, _X_TEST, _Y_TRAIN, _Y_TEST = model_selection.train_test_split(
    CANCER.values[ : , : (-1) ],
    CANCER.values[ : , (-1) ],
    test_size = 0.2,
    random_state = 7,
    )
################################################################################
# 审查函数定义
from sklearn import preprocessing, pipeline
def _Models_Cmp( _Models, X = _X_TRAIN, y = _Y_TRAIN, _Scale = False, scoring = "accuracy", _Figure_Title = "ALGORITHM COMPARISON", _Cross_Val_Type = "KFold", n_splits = 10, random_state = 7, test_size = 0.2 ):
    _Algorithm_Cmp_Results, _Algorithm_Cmp_Result_List = [], []
    if _Cross_Val_Type == "KFold":
        _Cross_Val = model_selection.KFold(
            n_splits = n_splits,
            random_state = random_state
            )
    elif _Cross_Val_Type == "LeaveOneOut":
        _Cross_Val = model_selection.LeaveOneOut()
    elif _Cross_Val_Type == "ShuffleSplit":
        _Cross_Val = model_selection.ShuffleSplit(
            n_splits = n_splits,
            test_size = test_size,
            random_state = random_state
            )
    else:
        raise Exception()
    if _Scale == True:
        _Scaler = preprocessing.StandardScaler().fit(X = X)
        X = _Scaler.transform(X = X)
    else:
        pass
    for _Each_Model in _Models:
        _Cross_Val_Results = model_selection.cross_val_score(
            estimator = _Models[_Each_Model],
            X = X,
            y = y,
            cv = _Cross_Val,
            scoring = scoring
            )
        _Algorithm_Cmp_Result_List.append(_Cross_Val_Results)
        _Algorithm_Cmp_Results.append(
            {
                "Algorithm Name" : _Each_Model,
                "CV Results" : _Algorithm_Cmp_Result_List[-1],
                "Model" : _Models[_Each_Model]
                }
            )
        print(
            _Each_Model, "\n",
            " " * 4, "MEAN:   ", _Cross_Val_Results.mean(), "\n",
            " " * 4, "STD:    ", _Cross_Val_Results.std(), "\n",
            sep = "", end = "\n"
            )
    print()
    if _Plotting_Switch == True:
        _Fig = plt.figure()
        _Fig.suptitle(t = _Figure_Title)
        _Ax = _Fig.add_subplot(111)
        plt.boxplot(x = _Algorithm_Cmp_Result_List)
        _Ax.set_xticklabels(labels = _Models.keys())
        plt.show()
    else:
        pass
    return _Algorithm_Cmp_Results
################################################################################
# 运行函数定义
from sklearn import metrics
def _Model_Run( _Model, X = _X_TRAIN, y = _Y_TRAIN, X_Test = _X_TEST, y_True = _Y_TEST, _Scale = False, _Report_Title = "" ):
    if _Scale == True:
        _Scaler = preprocessing.StandardScaler().fit(X = X)
        X = _Scaler.transform(X = X)
        X_Test = _Scaler.transform(X = X_Test)
    else:
        pass
    _Model.fit( X = X, y = y )
    y_Pred = _Model.predict( X = X_Test )
    print(
        _Report_Title + "预测结果:\n",
        " " * 4, "ACCURACY_SCORE:\n",
        " " * 8,
        metrics.accuracy_score(
            y_true = y_True,
            y_pred = y_Pred
            ),
        "\n",
        " " * 4, "CONFUSION_MATRIX:\n",
        metrics.confusion_matrix(
            y_true = y_True,
            y_pred = y_Pred
            ),
        "\n",
        " " * 4, "CLASSIFICATION_REPORT:\n",
        metrics.classification_report(
            y_true = y_True,
            y_pred = y_Pred
            ),
        "\n",
        sep = "", end = "\n"
        )
    print()
################################################################################
# 参数调整函数定义部分
def _Param_Optimalize( _Model, X = _X_TRAIN, y = _Y_TRAIN, _Scale = False, param_grid = {}, scoring = "accuracy", _Report_Title = "", _Cross_Val_Type = "KFold", n_splits = 10, random_state = 7, test_size = 0.2 ):
    if _Cross_Val_Type == "KFold":
        _Cross_Val = model_selection.KFold(
            n_splits = n_splits,
            random_state = random_state
            )
    elif _Cross_Val_Type == "LeaveOneOut":
        _Cross_Val = model_selection.LeaveOneOut()
    elif _Cross_Val_Type == "ShuffleSplit":
        _Cross_Val = model_selection.ShuffleSplit(
            n_splits = n_splits,
            test_size = test_size,
            random_state = random_state
            )
    else:
        raise Exception()
    if _Scale == True:
        _Scaler = preprocessing.StandardScaler().fit(X = X)
        X = _Scaler.transform(X = X)
    else:
        pass
    _Grid = model_selection.GridSearchCV(
        estimator = _Model,
        param_grid = param_grid,
        scoring = scoring,
        cv = _Cross_Val
        )
    _Grid_Result = _Grid.fit(
        X = X,
        y = y
        )
    _Cross_Val_Results = zip(
        _Grid_Result.cv_results_["mean_test_score"],
        _Grid_Result.cv_results_["std_test_score"],
        _Grid_Result.cv_results_["params"]
        )
    print( _Report_Title + "参数调整结果:" )
    for _Each_Mean, _Each_Std, _Each_Param in _Cross_Val_Results:
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
        " " * 4, _Grid_Result.best_score_, "\n",
        "使用:\n",
        " " * 4, _Grid_Result.best_params_, "\n",
        sep = "", end = "\n"
        )
    print()
    return _Grid_Result
################################################################################
# 分类算法审查
_ORDINARY_MODELS = {
    "LR" : LR(),
    "LDA" : LDA(),
    "KNC" : KNC(),
    "GNB" : GNB(),
    "DTC" : DTC(),
    "SVC" : SVC()
    }
_ORDINARY_ALGORITHM_CMP_RESULTS = _Models_Cmp( _Models = _ORDINARY_MODELS, _Figure_Title = "ALGORITHM COMPARISON" ) # Best: KNC
_Model_Run( _Model = _ORDINARY_MODELS["KNC"], _Report_Title = "KNC-K近邻算法" )
################################################################################
# 数据正态化后重审
_SCALED_MODELS = {
    "LR" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "LR", LR() ) ] ),
    "LDA" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "LDA", LDA() ) ] ),
    "KNC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "KNC", KNC() ) ] ),
    "GNB" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "GNB", GNB() ) ] ),
    "DTC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "DTC", DTC() ) ] ),
    "SVC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "SVC", SVC() ) ] )
    }
_SCALED_ALGORITHM_CMP_RESULTS = _Models_Cmp( _Models = _SCALED_MODELS, _Scale = True, _Figure_Title = "SCALED ALGORITHM COMPARISON" ) # Best: SVC
_Model_Run( _Model = _SCALED_MODELS["SVC"], _Scale = True, _Report_Title = "SVC-资瓷矢量机" )
################################################################################
# 集成算法审查
_ENSEMBLED_MODELS = {
    "ABC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "ABC", ABC() ) ] ),
    "GBC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "GBC", GBC() ) ] ),
    "RFC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "RFC", RFC() ) ] ),
    "ETC" : pipeline.Pipeline( [ ( "Scaler", preprocessing.StandardScaler() ), ( "ETC", ETC() ) ] )
    }
_ENSEMBLED_ALGORITHM_CMP_RESULTS = _Models_Cmp( _Models = _ENSEMBLED_MODELS, _Scale = True, _Figure_Title = "ENSEMBLED ALGORITHM COMPARISON" ) # Best: ETC
#_Model_Run( _Model = ABC(), _Scale = True, _Report_Title = "ABC-AdaBoost算法" )
#_Model_Run( _Model = GBC(), _Scale = True, _Report_Title = "GBC-随机梯度算法" )
#_Model_Run( _Model = RFC(), _Scale = True, _Report_Title = "RFC-随机森林算法" )
#_Model_Run( _Model = ETC(), _Scale = True, _Report_Title = "ETC-极端随机树" )
################################################################################
# 参数调整后进入最终预测
# 莫名报错, 不调了
"""
_Param_Grid = {
    "n_neighbors" : [
        1, 3, 5, 7,
        9, 11, 13, 15,
        17, 19, 21
        ]
    }
_KNC_GRID_RESULT =_Param_Optimalize( _Model = _SCALED_MODELS["KNC"], _Scale = True, _Report_Title = "KNC-K近邻算法", param_grid = _Param_Grid )
_Model_Run(
    _Model = pipeline.Pipeline(
        steps = [
            ( "Scaler", preprocessing.StandardScaler() ),
            ( "KNC", KNC(n_neighbors = _KNC_GRID_RESULT.best_params_["n_neighbors"]) )
            ]
        ),
    _Scale = True,
    _Report_Title = "KNC-K近邻算法-最终"
    )
"""
################################################################################
print("-*- END -*-")
# -*- END -*- #
