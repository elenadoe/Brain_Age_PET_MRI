#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""
import steps_of_analysis
import plots

from skrvm import RVR
from tqdm import tqdm

import pickle
import pandas as pd


def main(modality, rand_seed=42, imp='main', plotting=True):
    df_mri = pd.read_csv('../data/main/MRI_parcels_all.csv')
    df_pet = pd.read_csv('../data/main/PET_parcels_all.csv')
    col = df_mri.columns[3:-1].tolist()
    n_outliers = steps_of_analysis.split_data(df_mri, df_pet, col, imp=imp,
                                              plotting=plotting,
                                              rand_seed=rand_seed)

    # LOAD DATA
    # load and inspect data, set modality
    database = "CN"
    mode = "train"
    df = pd.read_csv('../data/main/test_train_' + modality +
                     '_' + str(rand_seed) + '.csv')
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
                                                             models,
                                                             model_params,
                                                             splits=SPLITS,
                                                             rand_seed=rand_seed,
                                                             scoring=SCORING,
                                                             y='age')

    # BIAS CORRECTION
    final_model, pred_param = steps_of_analysis.bias_correct(df_train, col,
                                                             model_results,
                                                             model_names,
                                                             modality,
                                                             database,
                                                             plotting=plotting)

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

    pred, mae, r2 = steps_of_analysis.predict(df_test, col, model_,
                                              final_model,
                                              slope_, intercept_,
                                              modality, mode,
                                              database, plotting=plotting)

    return n_outliers, pred, mae, r2


if __name__ == "__main__":
    mae_range = []
    r2_range = []
    n_outliers_range = []
    for i in tqdm(range(50)):
        n_outliers, pred, mae, r2 = main('MRI', rand_seed=i, plotting=False)
        n_outliers_range.append(n_outliers)
        mae_range.append(mae)
        r2_range.append(r2)
    # main('MRI')
