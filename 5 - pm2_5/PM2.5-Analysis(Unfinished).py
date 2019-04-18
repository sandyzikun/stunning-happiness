#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
pm25 = pd.read_csv("C:\\PRSA_data_2010.1.1-2014.12.31.csv")
from sklearn import model_selection
XTrain, XTest, YTrain, YTest = model_selection.train_test_split(
    pm25.values[ : , : (-1) ].astype(float),
    pm25.values[ : , (-1) ],
    test_size = 0.2,
    random_state = 7
    )
