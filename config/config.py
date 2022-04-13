#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:28:05 2021

@author: doeringe
"""
import pickle
import numpy as np
from matplotlib import cm
from nilearn.datasets import fetch_atlas_schaefer_2018
# plotting
# matplotlib config
cm_ = cm.get_cmap('magma')
black = cm.get_cmap('Greys')
cm_ADNI = cm_(0.1)
cm_OASIS = cm_(0.9)
cm_all = np.array([cm_ADNI, cm_OASIS])
pickle.dump(cm_all, open('plotting_config_main.p', 'wb'))

oranges = cm.get_cmap('Oranges')
blues = cm.get_cmap('Blues')
cm_young = blues(0.9)
cm_zero = black(0.9)
cm_old = oranges(0.4)

# neuropathology/neuropsychology
cm_neurop_MRI = np.array([cm_zero, cm_old, cm_young])
pickle.dump(cm_neurop_MRI, open('plotting_config_np_MRI.p', 'wb'))
cm_neurop_PET = np.array([cm_old, cm_young, cm_zero])
pickle.dump(cm_neurop_PET, open('plotting_config_np_PET.p', 'wb'))
cm_neurop_mci_MRI = np.array([cm_zero, cm_old, cm_young])
pickle.dump(cm_neurop_mci_MRI, open('plotting_config_np_mci_MRI.p', 'wb'))
cm_neurop_mci_PET = np.array([cm_zero, cm_old, cm_young])
pickle.dump(cm_neurop_mci_PET, open('plotting_config_np_mci_PET.p', 'wb'))

# gender

cm_gender = cm.get_cmap('BrBG')
male = cm_gender(0.1)
female = cm_gender(0.9)
cm_gender_CN = np.array([female, male])
pickle.dump(cm_gender_CN, open('plotting_config_gender_CN.p', 'wb'))
cm_gender_MCI = np.array([male, female])
pickle.dump(cm_gender_MCI, open('plotting_config_gender_MCI.p', 'wb'))

random_state = 0
# hyperparameters
kernels = ['linear', 'rbf', 'poly']
degree = [2, 3]
cs = [0.001, 0.01, 0.1, 1, 10, 100, 500]
# hyperparameters gb
loss = ['squared_error', 'absolute_error']
n_estimators = [10, 100, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
max_depth = [2, 3, 4, 5, 10, 15]

model_params = [{'rvr__C': cs,
                 'rvr__degree': degree,
                 'rvr__kernel': kernels},
                {'svm__C': cs,
                 'svm__degree': degree,
                 'svm__kernel': kernels}]

pickle.dump(model_params, open("hyperparams_allmodels.p", "wb"))

# create labels text file for plotting feature importance
schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
schaefer_labels = schaefer['labels']
tian_labels = open('../data/Tian_Subcortex_S1_3T_label.txt')
labels = open('../data/composite_atlas_labels.txt', 'w+')
for label in schaefer_labels:
    labels.write(label.decode('ascii') + '\n')
for label in tian_labels.read().split('\n'):
    labels.write(label + '\n')
labels.close()
