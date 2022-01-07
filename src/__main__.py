#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""
import steps_of_analysis
import plots

from skrvm import RVR
from collections import Counter
from tqdm import tqdm

import pickle
import pandas as pd
import numpy as np

def brain_age(modality, rand_seed=0, imp='main', info=True):
    """
    Execute brain age prediction pipeline.

    Main function uses functions defined in steps_of_analysis
    to run all steps: (1) train-test split, (2) hyperparameter
    tuning using cross-validation, (3) bias correction using
    cross-validated predictions, (4) prediction of test set.

    Parameters
    ----------
    modality : str
        MRI or PET
    rand_seed : int, optional
        Random seed to use throughout pipeline. The default is 42.
    imp : str, optional
        Main analysis with one random seed or validation
        with several random seeds. The default is 'main'.
    info : boolean, optional
        Whether to print intermediate info. Recommended to set
        to False for validation_random_seeds. The default is True.

    Returns
    -------
    n_outliers : int
        Number of outliers excluded prior to splitting.
    pred : list
        Predicted & bias-corrected age.
    mae : float
        Mean absolute error of pred.
    r2 : float
        R squared of chronological age and pred.

    """
    df_mri = pd.read_csv('../data/main/MRI_parcels_all.csv')
    df_pet = pd.read_csv('../data/main/PET_parcels_all.csv')
    col = df_mri.columns[3:-1].tolist()
    n_outliers = steps_of_analysis.split_data(df_mri, df_pet, col, imp=imp,
                                              info=info,
                                              rand_seed=rand_seed)

    # LOAD DATA
    database = "CN"
    mode = "train"
    df = pd.read_csv('../data/main/test_train_' + modality +
                     '_' + str(rand_seed) + '.csv')
    df = df[df['AGE_CHECK'] & df['IQR']]
    df_train = df[df['train']]
    df_train = df_train.reset_index(drop=True)

    if info:
        plots.plot_hist(df_train, mode, modality, df_train['Dataset'], y='age')

    # CROSS-VALIDATE MODELS
    # define models and model names (some are already included in julearn)
    models = [RVR(), 'svm']
    model_names = ['rvr', 'svm']
    SPLITS = 5
    SCORING = ['r2']
    model_params = pickle.load(open("../data/config/hyperparams_allmodels.p",
                                    "rb"))

    model_results, scores = steps_of_analysis.cross_validate(
        df_train, col, models, model_params, splits=SPLITS,
        rand_seed=rand_seed, scoring=SCORING, y='age')

    # BIAS CORRECTION
    final_model, pred_param = steps_of_analysis.bias_correct(df_train, col,
                                                             model_results,
                                                             model_names,
                                                             modality,
                                                             database,
                                                             info=info,
                                                             splits=SPLITS)

    slope_ = pred_param[final_model + "_slope"]
    intercept_ = pred_param[final_model + "_intercept"]
    model_ = model_results[model_names.index(final_model)]

    # TEST
    # How well does the model perform on unseen data?
    df_test = df[~df['train']]
    df_test = df_test.reset_index(drop=True)
    mode = "test"

    if info:
        plots.plot_hist(df_test, mode, modality, df_test['Dataset'], y='age')

    pred, mae, r2 = steps_of_analysis.predict(df_test, col, model_,
                                              final_model,
                                              slope_, intercept_,
                                              modality, mode,
                                              database, info=info)

    return n_outliers, pred, mae, r2, final_model


if __name__ == "__main__":
    analyze = input("Which analysis would you like to run?\
                    \nType\n'1' for bias-correction\
                    \n'2' for brain age prediction in CN\
                    \n'2.1' for validation of brain age prediction in CN\
                    \n'3' for brain age prediction in MCI\
                    \n'4' for association with cognitive\
                    \nperformance/neuropathology")
    if analyze == 1:
        pass
        # TODO
    elif analyze == 2:
        brain_age('PET')
    elif analyze == 2.1:
        mae_range = []
        r2_range = []
        n_outliers_range = []
        models = []
        for i in tqdm(range(50)):
            n_outliers, pred, mae, r2, final_model = brain_age(
                'MRI', imp='validation_random_seeds', rand_seed=i, info=False)
            n_outliers_range.append(n_outliers)
            mae_range.append(mae)
            r2_range.append(r2)
            models.append(final_model)

        print("Range of 50 iterations:\nMAE:",
              np.min(mae_range), "-", np.max(mae_range),
              "(mean: {})".format(np.mean(mae_range)),
              "\nR2:",
              np.min(r2_range), "-", np.max(r2_range),
              "(mean: {})".format(np.mean(r2_range)),
              "\nModels: ", Counter(models))
    elif analyze == 3:
        pass
        # TODO
    elif analyze == 4:
        pass
        # TODO
