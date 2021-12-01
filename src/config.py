#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:28:05 2021

@author: doeringe
"""
import pickle
import matplotlib
import numpy as np
# plotting
# matplotlib config
cm = matplotlib.cm.get_cmap('PuOr')
cm_ADNI = cm(0.2)
cm_OASIS = cm(0.8)
cm_all = np.array([cm_ADNI, cm_OASIS])
pickle.dump(cm_all, open('../data/config/plotting_config.p', 'wb'))

random_state = 42
# hyperparameters
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
degree = [2, 3]
cs = [0.001, 0.01, 0.1, 1, 10, 100, 500]
# hyperparameters gb
loss = ['squared_error', 'absolute_error']
n_estimators = [10, 100, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
max_depth = [2, 3, 4, 5, 10, 15]
# hyperparameter gpr
alpha = [1e-10, 1e-5]

model_params = [{'rvr__C': cs,
                 'rvr__degree': degree,
                 'rvr__kernel': kernels},
                {'svm__C': cs,
                 'svm__degree': degree,
                 'svm__kernel': kernels},
                {'gradientboost__n_estimators': n_estimators,
                 'gradientboost__learning_rate': learning_rate,
                 'gradientboost__max_depth': max_depth,
                 'random_state': [random_state]}]

pickle.dump(model_params, open("../data/config/hyperparams_allmodels.p", "wb"))
