#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:27:41 2021

@author: doeringe
"""

from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from skrvm import RVR

from transform_data import split_data
import plots

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


def cross_validate(df_train, col, models, model_params, splits, scoring,
                   rand_seed=0, y='age'):
    """
    Cross-validation.

    Apply cross-validation on training data using models and parameters
    provided by user.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataframe containing input and output variables
    col : list or np.ndarray
        Column(s) to be used as input features
    models : list
        List of algorithms to assess
    model_params : dict
        Dictionary of hyperparameters to assess
    splits : int
        How many folds to split the data into for cross-validation
    scoring : str or list
        Metrics to assess
    rand_seed : int, optional
        Random state to use during all fitting procedures, where applicable
    y : pd.Series, optional
        Column to be used as output variable

    Returns
    -------
    model_results : list
        Best fitted estimator per algorithm
    scores : list
        Average performance during cross-validation of the algorithm

    """
    model_results = []
    scores_results = []
    scaler = 'scaler_robust'

    for i, (model, params) in enumerate(zip(models, model_params)):
        # split data using age-bins
        cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                    df_train['Ageb'])
        cv = list(cv)
        # run julearn function
        scores, final_model = run_cross_validation(X=col, y=y,
                                                   preprocess_X=scaler,
                                                   problem_type='regression',
                                                   data=df_train,
                                                   model=model, cv=cv,
                                                   seed=rand_seed,
                                                   model_params=params,
                                                   return_estimator='all',
                                                   scoring=scoring)
        model_results.append(final_model.best_estimator_)
        scores_results.append(scores)

    return model_results, scores


def pred_uncorr(df_train, col, model_results, splits=5):
    """
    # TODO.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    df_train : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    model_results : TYPE
        DESCRIPTION.
    splits : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    y_pred_rvr = cross_val_prediction(df_train, col, 'age',
                                      model_results[0],
                                      splits)
    y_pred_svr = cross_val_prediction(df_train, col, 'age',
                                      model_results[1],
                                      splits)
    y_pred_uncorr = [y_pred_rvr, y_pred_svr]

    return y_pred_uncorr


# BIAS CORRECTION
# Eliminate linear correlation of brain age delta and chronological age
def bias_correct(df_train, col, model_results, model_names,
                 modality, database, splits, y='age', correct_with_CA=True,
                 return_model='final', info=True, save=True):
    """
    # TODO.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataframe containing input and output variables
    y_pred_uncorr : list, np.ndarray or pd.Series
        Predicted, uncorrected age
    model_names : list
        List of strings naming models to assess
    modality : str
        MRI or PET
    database : str
        CN or MCI
    y : pd.Series, optional
        Column to be considered as output feature. The default is age.
    correct_with_CA : boolean, optional
        whether or not to correct bias with chronological age.
        The default is True.
    info : boolean, optional
        whether or not to create and save plots. The default is True.
    save : boolean, optional
        Whether final model should be saved

    Returns
    -------
    final_model : sklearn model
        # TODO
    pred_param : dict
        dictionary containing bias-corrected age, slope, intercept,
        and pearson r value of bias between BPAD and CA

    """
    # BIAS CORRECTION
    y_true = df_train['age']
    y_pred_uncorr = pred_uncorr(df_train, col, model_results, splits=splits)
    predictions = {}
    predictions['name'] = df_train['name']
    pred_param = {}
    pred_param['withCA'] = [str(correct_with_CA)]

    for y in range(len(y_pred_uncorr)):
        predictions[model_names[y] + '_uncorr'] = y_pred_uncorr[y]
        check_bias = plots.check_bias(y_true,
                                      y_pred_uncorr[y],
                                      model_names[y],
                                      modality,
                                      database,
                                      correct_with_CA,
                                      info=info,
                                      save=save)
        slope_ = check_bias[0]
        intercept_ = check_bias[1]
        check_ = check_bias[2]

        if info:
            print("Significant association between ", model_names[y],
                  "-predicted age delta and CA:",
                  check_)

        if correct_with_CA is None:
            bc = y_pred_uncorr[y]
            predictions[model_names[y] + '_bc'] = bc
        elif correct_with_CA:
            # for age correction WITH chronological age
            bc = y_pred_uncorr[y] - (slope_*y_true + intercept_)
            predictions[model_names[y] + '_bc'] = bc
        else:
            # for age correction WITHOUT chronological age
            bc = (y_pred_uncorr[y] - intercept_)/slope_
            predictions[model_names[y] + '_bc'] = bc

        r2_corr = r2_score(y_true, bc)
        mae_corr = mean_absolute_error(y_true, bc)

        pred_param[model_names[y] + '_slope'] = [slope_]
        pred_param[model_names[y] + '_intercept'] = [intercept_]
        pred_param[model_names[y] + '_check'] = [check_]
        pred_param[model_names[y] + '_r2'] = [r2_corr]
        pred_param[model_names[y] + '_mae'] = [mae_corr]
        r2_uncorr = r2_score(y_true, y_pred_uncorr[y])
        mae_uncorr = mean_absolute_error(y_true, y_pred_uncorr[y])
        pred_param[model_names[y] + '_rsq_uncorr'] = [r2_uncorr]
        pred_param[model_names[y] + '_ma_uncorr'] = [mae_uncorr]

        if save:
            pickle.dump(pred_param, open("../results/" + database +
                                         "/models_and_params_" + modality + "_" +
                                         str(correct_with_CA) +
                                         ".p", "wb"))
            pickle.dump(predictions, open("../results/" + database +
                                          "/cross-val_pred_" + modality + "_" +
                                          str(correct_with_CA) +
                                          ".p", "wb"))
        df = pd.DataFrame(pred_param)
        df.to_csv("../results/" + database + "/models_and_params_"
                  + modality + "_" + str(correct_with_CA) + ".csv")

    if return_model == 'final':
        final_model, final_mae, final_r2 = find_final_model(y_true,
                                                            y_pred_uncorr,
                                                            pred_param,
                                                            model_names,
                                                            modality,
                                                            info)

        return final_model, pred_param
    elif return_model == 'all':
        return model_results, pred_param


def find_final_model(y_true, y_pred_uncorr,
                     pred_param, model_names, modality,
                     correct_with_CA=True, info=True):
    """
    # TODO.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred_uncorr : TYPE
        DESCRIPTION.
    pred_param : TYPE
        DESCRIPTION.
    model_names : TYPE
        DESCRIPTION.
    info : TYPE, optional
        DESCRIPTION. The default is True.
    save : boolean, optional
        Whether final_model should be saved

    Returns
    -------
    final_model : TYPE
        DESCRIPTION.
    final_mae : TYPE
        DESCRIPTION.
    final_r2 : TYPE
        DESCRIPTION.

    """
    final_model_idx = np.argmin([v for k, v in pred_param.items()
                                 if '_mae' in k])
    final_r2 = [v for k, v in pred_param.items()
                if '_r2' in k][final_model_idx]
    final_mae = [v for k, v in pred_param.items()
                 if '_mae' in k][final_model_idx]
    final_model = model_names[final_model_idx]

    if info:
        print("-\033[1m--CROSS-VALIDATION---\n",
              "Final model (smallest MAE): {}\nMAE: {}, R2: {}\033[0m".format(
                  final_model, final_mae[0], final_r2[0]))
    return final_model, final_mae, final_r2


def cross_val_prediction(df_train, col, y, model_, splits):
    """
    # TODO.

    Parameters
    ----------
    df_train : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model_ : TYPE
        DESCRIPTION.
    splits : TYPE
        DESCRIPTION.

    Returns
    -------
    pred : TYPE
        DESCRIPTION.

    """
    cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                df_train['Ageb'])
    cv = list(cv)
    pred = cross_val_predict(model_, df_train[col], df_train[y], cv=cv)

    return pred


def predict(df_test, col, model_, final_model_name,
            slope_, intercept_, modality, database,
            train_test='test', y='age', correct_with_CA=True, info=True):
    """
    Predicts brain age using trained algorithms.

    Parameters
    ----------
    df_test : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    model_ : ExtendedDataFramePipeline
        final model to be used for prediction
    final_model_name : str
        name of final model to be used for saving of plots
    y : str
        DESCRIPTION.
    slope_ : float
        DESCRIPTION.
    intercept_ : float
        DESCRIPTION.
    modality : str
        PET or MRI
    train_test : str
        Whether train or test data is predicted
    database : str
        CN or MCI
    correct_with_CA : boolean, optional
        DESCRIPTION
    info : boolean, optional
        whether or not to create and save plots. The default is True.

    Returns
    -------
    y_pred_bc : np.ndarray
        Bias-corrected brain age of individuals from test set

    """
    y_pred = model_.predict(df_test[col])

    # plot model predictions against GT in test set
    if correct_with_CA is None:
        y_pred_bc = y_pred
    elif correct_with_CA:
        # for age correction WITH chronological age
        y_pred_bc = y_pred - (slope_*df_test[y] + intercept_)
    else:
        # for age correction WITHOUT chronological age
        y_pred_bc = (y_pred - intercept_)/slope_

    mae = mean_absolute_error(df_test[y], y_pred_bc)
    r2 = r2_score(df_test[y], y_pred_bc)

    plots.real_vs_pred_2(df_test[y], y_pred_bc, final_model_name, modality,
                         train_test, database, correct_with_CA=correct_with_CA,
                         info=info, database_list=df_test['Dataset'])

    if info:
        df = pd.DataFrame({'PTID': df_test['PTID'],
                           'Age': df_test[y],
                           'Prediction': y_pred_bc})
        df.to_csv("../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    return y_pred_bc, mae, r2


def brain_age(dir_mri_csv, dir_pet_csv, modality, return_model='final',
              correct_with_CA=True, rand_seed=0, cv=5, imp='main', info=True,
              info_init=False, save=True):
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
    info_init : boolean, optional
        DESCRIPTION
    save : boolean, optional
        Whether final model should be saved

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
    df_mri = pd.read_csv(dir_mri_csv, sep=";")
    df_pet = pd.read_csv(dir_pet_csv, sep=";")
    col = df_mri.columns[3:-1].tolist()
    pickle.dump(col, open("../config/columns.p", "wb"))
    n_outliers = split_data(df_mri, df_pet, col, imp=imp, info=info_init,
                            rand_seed=rand_seed)

    # LOAD DATA
    database = "CN"
    mode = "train"
    df = pd.read_csv('../data/{}/test_train_'.format(imp) + modality +
                     '_' + str(rand_seed) + '.csv')
    df = df[df['AGE_CHECK'] & df['IQR']]
    df_train = df[df['train']]
    df_train = df_train.reset_index(drop=True)

    if info_init:
        plots.plot_hist(df_train, mode, modality, df_train['Dataset'], y='age')

    # CROSS-VALIDATE MODELS
    # define models and model names (some are already included in julearn)
    models = [RVR(), 'svm']
    model_names = ['rvr', 'svm']
    SCORING = ['r2']
    model_params = pickle.load(open("../config/hyperparams_allmodels.p",
                                    "rb"))

    model_results, scores = cross_validate(
        df_train, col, models, model_params, splits=cv,
        rand_seed=rand_seed, scoring=SCORING, y='age')

    final_model_name, pred_param = bias_correct(
        df_train, col, model_results, model_names, modality,
        database, correct_with_CA=correct_with_CA, info=info_init,
        return_model=return_model, splits=cv, save=save)
    final_model = model_results[model_names.index(final_model_name)]
    if save:
        pickle.dump(final_model, open("../results/final_model_{}_{}.p".format(
            modality, str(correct_with_CA)), "wb"))

    slope_ = pred_param[final_model_name + "_slope"]
    intercept_ = pred_param[final_model_name + "_intercept"]

    # TEST
    # How well does the model perform on unseen data?
    df_test = df[~df['train']]
    df_test = df_test.reset_index(drop=True)
    mode = "test"

    if info_init:
        plots.plot_hist(df_test, mode, modality, df_test['Dataset'], y='age')

    pred, mae, r2 = predict(df_test, col, final_model, final_model_name,
                            slope_, intercept_, modality, database,
                            correct_with_CA=correct_with_CA,
                            train_test='test', info=info)

    if info:
        plots.permutation_imp(df_test, col, final_model, final_model_name,
                              modality, rand_seed=rand_seed)

    return n_outliers, pred, mae, r2, final_model
