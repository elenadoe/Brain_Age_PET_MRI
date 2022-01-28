#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:17:23 2022

@author: doeringe
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def outlier_check(df_mri, df_pet, col, threshold=3):
    """
    Check for outliers.

    Outliers are defined as individuals whose brain signal
    in one or more regions is outside of threshold*interquartile range (IQR)
    of this/these regions.

    Parameters
    ----------
    data_mri : pd.dataframe
        parcels derived from MRI data
    data_pet : pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    threshold : int or float, optional
        threshold*IQR defines the range in which data is not considered
        an outlier. Threshold=1.5 defines normal outliers,
        threshold=3 defines extreme outliers. The default is 3.

    Returns
    -------
    df_mri : pd.dataframe
        parcels derived from MRI data without outliers
    df_pet : pd.dataframe
        parcels derived from PET data without outliers

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
               rand_seed=0):
    """
    Split data into train and test sets.

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
            print(len(df_pet)-sum(df_pet['AGE_CHECK']),
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
            print("Total participants: ", len(df_mri),
                  "Inside IQR of all regions: ", len(df_mri[df_mri['IQR']]),
                  "\n({}".format(n_outliers),
                  "participants discarded as outliers)")
            print("Outliers in train set: ",
                  sum(df_mri['train']) -
                  sum(df_mri['IQR'][df_mri['train']]),
                  "Outliers in test set: ",
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


def neuropsych_merge(df_pred, df_neuropsych,
                     neuropsych_var):
    """
    Merge predictions with neuropsychology.

    Parameters
    ----------
    df_pred : TYPE
        DESCRIPTION.
    df_neuropsych : TYPE
        DESCRIPTION.
    neuropsych_var : TYPE
        DESCRIPTION.

    Returns
    -------
    merged : TYPE
        DESCRIPTION.

    """
    df_dem = pd.read_csv("../data/main/ADNI_Neuropsych_Neuropath.csv",
                         sep=";")
    df_dem['PTGENDER'] = [1 if x == "Female"
                          else 2 if x == "Male"
                          else np.nan for x in df_dem['PTGENDER']]
    df_pred['BPAD'] = df_pred['Prediction']-df_pred['Age']
    dem_var = ['PTID', 'PTGENDER', 'PTEDUCAT']
    merged = df_pred.merge(df_dem[dem_var], how="left", on="PTID")
    merged = merged.merge(df_neuropsych[['RID'] + neuropsych_var],
                          how='left', on='RID')
    return merged


def neuropath_merge(df_pred, df_neuropath1, df_neuropath2,
                    neuropath1_var, neuropath2_var):
    """
    

    Parameters
    ----------
    df_pred : TYPE
        DESCRIPTION.
    df_neuropath1 : TYPE
        DESCRIPTION.
    df_neuropath2 : TYPE
        DESCRIPTION.
    neuropath1_var : TYPE
        DESCRIPTION.
    neuropath2_var : TYPE
        DESCRIPTION.

    Returns
    -------
    merged : TYPE
        DESCRIPTION.

    """
    df_dem = pd.read_csv("../data/main/ADNI_Neuropsych_Neuropath.csv",
                         sep=";")
    df_dem['PTGENDER'] = [1 if x == "Female"
                          else 2 if x == "Male"
                          else np.nan for x in df_dem['PTGENDER']]
    df_pred['BPAD'] = df_pred['Prediction']-df_pred['Age']
    dem_var = ['PTID', 'PTGENDER', 'PTEDUCAT']
    merged = df_pred.merge(df_dem[dem_var],
                           how='left', on='PTID')
    merged = merged.merge(df_neuropath1[['PTID'] + neuropath1_var],
                          how='left', on='PTID')
    merged = merged.merge(df_neuropath2[['RID'] + neuropath2_var],
                          how='left', on='RID')
    merged['ABETA'][merged['ABETA'] == '>1700'] = 1700
    merged['TAU'][merged['TAU'] == '>1300'] = 1300
    merged['PTAU'][merged['PTAU'] == '>120'] = 120
    merged['PTAU'][merged['PTAU'] == '<8'] = 8
    merged['TAU'][merged['TAU'] == '<80'] = 80

    return merged


def dx_merge(df_pred, df_dx):
    """
    Merge predictions with diagnosis after 24 months.

    Parameters
    ----------
    df_pred : TYPE
        DESCRIPTION.
    df_dx : TYPE
        DESCRIPTION.

    Returns
    -------
    merged : TYPE
        DESCRIPTION.

    """
    merged = df_pred.merge(df_dx[['PTID', 'DX']], how='left', on='PTID')
    return merged
