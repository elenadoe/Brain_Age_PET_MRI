#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:27:41 2021

@author: doeringe
"""

from sklearn.model_selection import train_test_split
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

import plots

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


def outlier_check(df_mri, df_pet, col, threshold=3):
    """
    Checks for outliers, where outliers are defined as
    individuals whose brain signal in one or more regions is
    outside of threshold*interquartile range of this/these regions.

    Parameters
    ----------
    data_mri : pd.dataframe
        parcels derived from MRI data
    data_pet : pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    threshold : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    df_mri : TYPE
        DESCRIPTION.
    df_pet : TYPE
        DESCRIPTION.

    """
    mri_train = df_mri[df_mri['train']]
    pet_train = df_pet[df_pet['train']]

    q1_mri = mri_train.quantile(0.25)
    q1_pet = pet_train.quantile(0.25)
    q3_mri = mri_train.quantile(0.75)
    q3_pet = pet_train.quantile(0.75)
    IQR_mri = q3_mri - q1_mri
    IQR_pet = q3_pet - q1_pet

    low_out_mri = q1_mri - IQR_mri*threshold
    high_out_mri = q3_mri + IQR_mri*threshold
    low_out_pet = q1_pet - IQR_pet*threshold
    high_out_pet = q3_pet + IQR_pet*threshold

    insiderange = (~((df_mri[col] < low_out_mri) |
                     (df_mri[col] > high_out_mri) |
                     (df_pet[col] < low_out_pet) |
                     (df_pet[col] > high_out_pet)).any(axis=1))

    df_mri['IQR'] = insiderange
    df_pet['IQR'] = insiderange

    return df_mri, df_pet


def split_data(df_mri, df_pet, col, imp, test_size=0.3, train_data="ADNI",
               older_65=True, check_outliers=True, info=True,
               rand_seed=42):
    """
    Splits data into train and test sets.

    Parameters
    ----------
    data_mri : pd.dataframe
        parcels derived from MRI data
    data_pet : pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    imp : str
        main analysis or validation_random_seeds
    test_size : float, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        The default is 0.3.
    train_data : str, optional
        Which dataset to use for training
    older_65 : boolean, optional
        Whether to consider only individuals over age 65. The default is True.
    check_outliers : boolean, optional
        Whether to check for and exclude outliers. The default is True.
    rand_seed : int, optional
        Controls the shuffling applied to the data before applying the split.


    Returns
    -------
    None.

    """

    same_ids = all(df_mri['name'] == df_pet['name'])

    # raise error if not all individuals are the same across modalities
    if same_ids is False:
        raise ValueError("IDs between modalities don't match.")

    if info:
        print("First column: {}".format(col[0]) +
              " (should be 'X17Networks_LH_VisCent_ExStr_1')" +
              "\nLast column: {}".format(col[-1]) +
              " (should be 'CAU-lh)")

    # exclude individuals younger than 65 if older_65 == True
    if older_65:
        older_65_mri = df_mri['age'] >= 65
        older_65_pet = df_pet['age'] >= 65

        df_pet['AGE_CHECK'] = older_65_mri & older_65_pet
        df_mri['AGE_CHECK'] = older_65_mri & older_65_pet
        if info:
            print(sum(df_pet['AGE_CHECK'])-len(df_pet),
                  "individuals younger than 65 years discarded.")
    else:
        df_pet['AGE_CHECK'] = True
        df_mri['AGE_CHECK'] = True

    # divide into age bins of "young old", "middle old" and "oldest old"
    # use mri to do so --> same age bins for both modalities
    df_mri['Ageb'] = [0 if x < 74 else 1
                      if x < 84 else 2 for x in df_mri['age']]
    df_pet['Ageb'] = [0 if x < 74 else 1
                      if x < 84 else 2 for x in df_pet['age']]

    # only ADNI data of individuals older than 65 (if older_65 == True)
    # to be considered in train_test split
    # OASIS data to be reserved as additional test set
    split_data = (df_mri['Dataset'] == "ADNI") & df_mri['AGE_CHECK']

    # prepare input (X) and output (y) for train-test split
    X = df_mri[col][split_data].values
    y = df_mri['age'][split_data].values
    y_pseudo = df_mri['Ageb'][split_data]

    # make len(rand_seed) train-test splits
    x_tr, x_te,  y_tr, y_te, id_tr, id_te = train_test_split(
                X, y, df_mri['name'][split_data],
                test_size=test_size, random_state=rand_seed, stratify=y_pseudo)
    df_mri['train'] = [True if x in id_tr.values
                       else False for x in df_mri['name']]
    df_pet['train'] = [True if x in id_tr.values
                       else False for x in df_pet['name']]

    if check_outliers:
        df_mri, df_pet = outlier_check(df_mri, df_pet, col)
        n_outliers = len(df_mri) - sum(df_mri['IQR'])
        if info:
            print(n_outliers, "outliers discarded.")
            print("Outliers in train set: ",
                  sum(df_mri['train']) -
                  sum(df_mri['IQR'][df_mri['train']]))
            print("Outliers in test set: ",
                  sum(~df_mri['train']) -
                  sum(df_mri['IQR'][~df_mri['train']]))
    else:
        n_outliers = 0
        df_mri['IQR'] = True
        df_pet['IQR'] = True
    df_mri.to_csv('../data/{}/'.format(imp) +
                  'test_train_MRI_{}.csv'.format(str(rand_seed)))
    df_pet.to_csv('../data/{}/'.format(imp) +
                  'test_train_PET_{}.csv'.format(str(rand_seed)))

    return n_outliers


def cross_validate(df_train, col, models, model_params, splits, scoring,
                   rand_seed=42, y='age'):
    """
    Applies cross-validation on training data using models and parameters
    provided by user

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


# BIAS CORRECTION
# Eliminate linear correlation of brain age delta and chronological age
def bias_correct(df_train, col, model_results, model_names,
                 modality, database, splits, y='age', correct_with_CA=True,
                 info=True):
    """


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

    Returns
    -------
    pred_param : dict
        dictionary containing bias-corrected age, slope, intercept,
        and pearson r value of bias between BPAD and CA

    """
    # BIAS CORRECTION
    y_true = df_train['age']
    y_pred_rvr = cross_val_prediction(df_train, col, 'age',
                                      model_results[0],
                                      splits)
    y_pred_svr = cross_val_prediction(df_train, col, 'age',
                                      model_results[1],
                                      splits)
    y_pred_uncorr = [y_pred_rvr, y_pred_svr]

    pred_param = {}
    pred_param['withCA'] = correct_with_CA

    for y in range(len(y_pred_uncorr)):
        pred_param[model_names[y] + '_uncorr'] = y_pred_uncorr[y]
        check_bias = plots.check_bias(y_true,
                                      y_pred_uncorr[y],
                                      model_names[y],
                                      modality,
                                      database,
                                      correct_with_CA,
                                      info=info)
        slope_ = check_bias[0]
        intercept_ = check_bias[1]
        check_ = check_bias[2]

        if info:
            print("Significant association between ", model_names[y],
                  "-predicted age delta and CA:",
                  check_bias[2])

        if correct_with_CA:
            # for age correction WITH chronological age
            bc = y_pred_uncorr[y] - (slope_*y_true + intercept_)
            pred_param[model_names[y] + '_bc'] = bc
        else:
            # for age correction WITHOUT chronological age
            bc = (y_pred_uncorr[y] - intercept_)/slope_
            pred_param[model_names[y] + '_bc'] = bc

        r2_corr = r2_score(y_true, bc)
        mae_corr = mean_absolute_error(y_true, bc)
        r2_uncorr = r2_score(y_true, y_pred_uncorr[y])
        mae_uncorr = mean_absolute_error(y_true, y_pred_uncorr[y])
        pred_param[model_names[y] + '_slope'] = slope_
        pred_param[model_names[y] + '_intercept'] = intercept_
        pred_param[model_names[y] + '_check'] = check_
        pred_param[model_names[y] + '_r2'] = r2_corr
        pred_param[model_names[y] + '_mae'] = mae_corr
        pred_param[model_names[y] + '_rsq_uncorr'] = r2_uncorr
        pred_param[model_names[y] + '_ma_uncorr'] = mae_uncorr

        pickle.dump(pred_param, open("../results/" + database +
                                     "/models_and_params_" + modality +
                                     ".p", "wb"))

    final_model_idx = np.argmax([v for k, v in pred_param.items()
                                 if '_r2' in k])

    final_model_r2 = np.max([v for k, v in pred_param.items()
                             if '_r2' in k])
    final_model_mae = [v for k, v in pred_param.items()
                       if '_mae' in k][final_model_idx]
    final_model = model_names[final_model_idx]

    if info:
        print("Final model (highest R2): {}\nMAE: {}, R2: {}".format(
                final_model, final_model_mae, final_model_r2))

    return final_model, pred_param


def cross_val_prediction(df_train, col, y, model_, splits):
    cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                df_train['Ageb'])
    cv = list(cv)
    pred = cross_val_predict(model_, df_train[col], df_train[y], cv=cv)

    return pred


def predict(df_test, col, model_, final_model_name,
            slope_, intercept_, modality, train_test,
            database, y='age', info=True):
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
    y : TYPE
        DESCRIPTION.
    slope_ : TYPE
        DESCRIPTION.
    intercept_ : TYPE
        DESCRIPTION.
    modality : TYPE
        DESCRIPTION.
    train_test : str
        Whether train or test data is predicted
    database : str
        CN or MCI
    info : boolean, optional
        whether or not to create and save plots. The default is True.

    Returns
    -------
    y_pred_bc : np.ndarray
        Bias-corrected brain age of individuals from test set

    """
    # plot model predictions against GT in test set
    y_pred = model_.predict(df_test[col])
    y_pred_bc = y_pred - (slope_*df_test[y] + intercept_)

    mae = mean_absolute_error(df_test[y], y_pred_bc)
    r2 = r2_score(df_test[y], y_pred_bc)

    plots.real_vs_pred_2(df_test[y], y_pred_bc, final_model_name, modality,
                         train_test, database,
                         info=info, database_list=df_test['Dataset'])
    return y_pred_bc, mae, r2
