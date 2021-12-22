#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""
import steps_of_analysis
import plots

from skrvm import RVR

import pickle
import pandas as pd
# %%
RAND_SEED = 42
df_mri = pd.read_csv('../data/main/MRI_parcels_all.csv')
df_pet = pd.read_csv('../data/main/PET_parcels_all.csv')
col = df_mri.columns[3:-1].tolist()
steps_of_analysis.split_data(df_mri, df_pet, col,
                             rand_seed=RAND_SEED)

# LOAD DATA
# load and inspect data, set modality
modality = 'MRI'
database = "1_CN_ADNI_OASIS"
mode = "train"
plotting = True
df = pd.read_csv('../data/main/test_train_' + modality +
                 '_' + str(RAND_SEED) + '.csv')
df = df[df['AGE_CHECK'] & df['IQR']]
df_train = df[df['train']]
df_train = df_train.reset_index(drop=True)

if plotting:
    plots.plot_hist(df_train, mode, modality, df_train['Dataset'], y='age')

# CROSS-VALIDATE MODELS
# define models and model names (some are already included in julearn)
models = [RVR(), 'svm']
model_names = ['rvr', 'svm']
SPLITS = 5
SCORING = ['r2', 'neg_mean_absolute_error']

model_params = pickle.load(open("../data/config/hyperparams_allmodels.p",
                                "rb"))

model_results, scores = steps_of_analysis.cross_validate(df_train, col,
                                                         models, model_params,
                                                         splits=SPLITS,
                                                         rand_seed=RAND_SEED,
                                                         scoring=SCORING,
                                                         y='age')
# %%
# BIAS CORRECTION
final_model, pred_param = steps_of_analysis.bias_correct(df_train, col,
                                                         model_results,
                                                         model_names,
                                                         modality, database,
                                                         plotting=plotting)

# %%

slope_ = pred_param[final_model + "_slope"]
intercept_ = pred_param[final_model + "_intercept"]
model_ = model_results[model_names.index(final_model)]

# TEST
# How well does the model perform on unseen data?
df_test = df[~df['train']]
df_test = df_test.reset_index(drop=True)
mode = "test"

if plotting:
    plots.plot_hist(df_test, mode, modality, df_test['Dataset'], y='age')

pred = steps_of_analysis.predict(df_test, col, model_, final_model,
                                 slope_, intercept_, modality, mode, database,
                                 plotting=plotting)
