#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:27:41 2021

@author: doeringe
"""

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold

import plots

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def split_data(df_mri, df_pet, col, test_size=0.3, train_data = "ADNI",
               older_65=True, rand_seed=42):
    """


    Parameters
    ----------
    data_mri : pd.dataframe
        parcels derived from MRI data
    data_pet : pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    test_size : float, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        The default is 0.3.
    train_data : str, optional
        Which dataset to use for training
    older_65 : boolean, optional
        Whether to consider only individuals over age 65. The default is True.
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
        
    print("First column: {}".format(col[0]) +
      " (should be 'X17Networks_LH_VisCent_ExStr_1')" +
      "\nLast column: {}".format(col[-1]) +
      " (should be 'CAU.lh)")
    
    # exclude individuals younger than 65 if older_65 == True
    if older_65:
        older_65_mri = df_mri['age'] >= 65
        older_65_pet = df_pet['age'] >= 65

        df_pet['AGE_CHECK'] = older_65_mri & older_65_pet
        df_mri['AGE_CHECK'] = older_65_mri & older_65_pet
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
    if isinstance(rand_seed, (list, np.ndarray)):
        for r in rand_seed:
            x_tr, x_te,  y_tr, y_te, id_tr, id_te = train_test_split(
                X, y, df_mri['name'][split_data],
                test_size=test_size, random_state=r, stratify=y_pseudo)
            df_mri['train'] = [True if x in id_tr.values
                               else False for x in df_mri['name']]
            df_pet['train'] = [True if x in id_tr.values
                               else False for x in df_pet['name']]

            df_mri.to_csv('../data/merged/validation_random_seeds/' +
                          'test_train_MRI_{}.csv'.format(str(r)))
            df_pet.to_csv('../data/merged/validation_random_seeds/' +
                          'test_train_PET_{}.csv'.format(str(r)))

    if isinstance(rand_seed, int):
        x_tr, x_te,  y_tr, y_te, id_tr, id_te = train_test_split(
            X, y, df_mri['name'][df_mri['Dataset'] == "ADNI"],
            test_size=test_size, random_state=rand_seed, stratify=y_pseudo)
        df_mri['train'] = [True if x in id_tr.values
                           else False for x in df_mri['name']]
        df_pet['train'] = [True if x in id_tr.values
                           else False for x in df_pet['name']]

        df_mri.to_csv('../data/merged/' +
                      'test_train_MRI_{}.csv'.format(str(rand_seed)))
        df_pet.to_csv('../data/merged/' +
                      'test_train_PET_{}.csv'.format(str(rand_seed)))


def cross_validate(df_train, col, models, model_params, splits, scoring,
                   rand_seed=42, y='age'):
    """
    

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

    for i, (model, params) in enumerate(tqdm(zip(models, model_params),
                                             total=len(models))):
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
def bias_correct(df_train, y_pred_uncorr, model_names, modality, database,
                 y='age', correct_with_CA=True):
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

    Returns
    -------
    pred_param : TYPE
        DESCRIPTION.

    """
    # BIAS CORRECTION
    y_true = df_train['age']

    pred_param = {}
    pred_param['withCA'] = correct_with_CA

    for y in range(len(y_pred_uncorr)):
        check_bias = plots.check_bias(y_true,
                                      y_pred_uncorr[y],
                                      model_names[y],
                                      modality,
                                      database,
                                      correct_with_CA)
        slope_ = check_bias[0]
        intercept_ = check_bias[1]
        check_ = check_bias[2]

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

        pred_param[model_names[y] + '_slope'] = slope_
        pred_param[model_names[y] + '_intercept'] = intercept_
        pred_param[model_names[y] + '_check'] = check_

    return pred_param
