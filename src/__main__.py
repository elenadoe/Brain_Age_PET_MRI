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

RAND_SEED = 42
df_mri = pd.read_csv('../data/merged/MRI_parcels_all.csv')
df_pet = pd.read_csv('../data/merged/PET_parcels_all.csv')
col = df_mri.columns[3:-1].tolist()
steps_of_analysis.split_data(df_mri, df_pet, col, 
                             rand_seed=RAND_SEED)

# %%
# LOAD DATA
# load and inspect data, set modality
modality = 'MRI'
database = "1_CN_ADNI_OASIS"
mode = "train"
df = pd.read_csv('../data/merged/test_train_' + modality +
                 '_' + str(RAND_SEED) + '.csv')
df = df[df['AGE_CHECK'] & df['IQR']]
df_train = df[df['train']]
df_train = df_train.reset_index(drop=True)

plots.plot_hist(df_train, mode, modality, df_train['Dataset'], y='age')

# CROSS-VALIDATE MODELS
# define models and model names (some are already included in julearn)
models = [RVR(), 'svm', 'gradientboost']
model_names = ['rvr', 'svm', 'gradientboost']
SPLITS = 5

model_params = pickle.load(open("../data/config/hyperparams_allmodels.p",
                                "rb"))

steps_of_analysis.cross_validate(df_train, col, models, model_params,
                                 splits=SPLITS, rand_seed=RAND_SEED,
                                 scoring=['r2', 'neg_mean_absolute_error'],
                                 y='age')
