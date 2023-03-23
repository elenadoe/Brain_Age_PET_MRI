#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:17:23 2022

@author: doeringe
"""
import pandas as pd
import numpy as np
import pdb
import random
import pickle

from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import ttest_ind
from os.path import exists

def outlier_check_main(df_mri_ADNI, df_pet_ADNI, col, atlas, threshold=3):
    """
    Check for outliers.

    Outliers are defined as individuals whose brain signal
    in one or more regions is outside of threshold*interquartile range (IQR)
    of this/these regions.

    Parameters
    ----------
    data_mri_ADNI : pd.dataframe
        parcels derived from MRI data
    data_pet_ADNI : pd.dataframe
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
    mri_train = df_mri_ADNI[df_mri_ADNI['train']]
    pet_train = df_pet_ADNI[df_pet_ADNI['train']]

    # get quantiles
    q1_mri = mri_train[col].quantile(0.25)
    q1_pet = pet_train[col].quantile(0.25)
    q3_mri = mri_train[col].quantile(0.75)
    q3_pet = pet_train[col].quantile(0.75)
    IQR_mri = q3_mri - q1_mri
    IQR_pet = q3_pet - q1_pet

    # define outlier ranges
    low_out_mri = q1_mri - IQR_mri*threshold
    high_out_mri = q3_mri + IQR_mri*threshold
    low_out_pet = q1_pet - IQR_pet*threshold
    high_out_pet = q3_pet + IQR_pet*threshold

    # get boolean index of which datapoints are inside (True)
    # or outside (False) of range
    insiderange = (~((df_mri_ADNI[col] < low_out_mri) |
                     (df_mri_ADNI[col] > high_out_mri) |
                     (df_pet_ADNI[col] < low_out_pet) |
                     (df_pet_ADNI[col] > high_out_pet)).any(axis=1))

    df_mri_ADNI['IQR'] = insiderange
    df_pet_ADNI['IQR'] = insiderange

    # save outliers per modality
    save_outliers = open("../results/999_OUTLIERS/outliers_ADNI.txt", "a+")
    save_outliers.write("Outliers PET train:\n")
    [save_outliers.write(x+"\n") for x in df_pet_ADNI['name'][(
        df_pet_ADNI['train']) & (
            (df_pet_ADNI[col] > high_out_pet).any(axis=1) |
            (df_pet_ADNI[col] < low_out_pet).any(axis=1)).tolist()]]
    save_outliers.write("\nOutliers PET test:\n")
    [save_outliers.write(x+"\n") for x in df_pet_ADNI['name'][(
        df_pet_ADNI['train'] == False) & (
            (df_pet_ADNI[col] > high_out_pet).any(axis=1) |
            (df_pet_ADNI[col] < low_out_pet).any(axis=1)).tolist()]]
    save_outliers.write("\nOutliers MRI train:\n")
    [save_outliers.write(x+"\n") for x in df_mri_ADNI['name'][(
        df_mri_ADNI['train'].values) & (
            (df_mri_ADNI[col] > high_out_mri).any(axis=1) |
            (df_mri_ADNI[col] < low_out_mri).any(axis=1)).tolist()]]
    save_outliers.write("\nOutliers MRI test:\n")
    [save_outliers.write(x+"\n") for x in df_mri_ADNI['name'][(
        df_mri_ADNI['train'] == False) & (
            (df_mri_ADNI[col] > high_out_mri).any(axis=1) |
            (df_mri_ADNI[col] < low_out_mri).any(axis=1)).tolist()]]
    save_outliers.write("\n\n")
    save_outliers.close()

    # save outlier ranges for application to DELCODE CN PET data
    pd.DataFrame({'col': col,
                  'high': high_out_pet,
                  'low': low_out_pet}).to_csv(
                      "../results/999_OUTLIERS/{}_outlier_ranges.csv".format(
                          atlas))

    return df_mri_ADNI, df_pet_ADNI

def prepare_data_other(df_other, database, modality, atlas, fold,
                        group, check_outliers, threshold=3):
    """
    Check for outliers in other data (MCI or DELCODE).

    Outliers are defined as individuals whose brain signal
    in one or more regions is outside of threshold*interquartile range (IQR)
    of this/these regions.

    Parameters
    ----------
    df_other: pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    threshold : int or float, optional
        threshold*IQR defines the range in which data is not considered
        an outlier. Threshold=1.5 defines normal outliers,
        threshold=3 defines extreme outliers. The default is 3.

    Returns
    -------
    df_other: pd.dataframe
        parcels derived from PET data without outliers
    """
    ADNI_CN_train = pd.read_csv(
            "../data/ADNI/CN/test_train_{}_{}_{}.csv".format(
                modality, atlas, fold))
    ADNI_CN_train = ADNI_CN_train[ADNI_CN_train.train]

    age_check = df_other['age'] >= 60

    if database == 'ADNI':  # ADNI SCD or MCI
        # for ADNI data additionally check that individuals are > 60 in
        # both modalities

        second_mod = ['PET' if modality == 'MRI' else 'MRI'][0]
        second_df = pd.read_csv(
            '../data/{}/{}/{}_{}_{}_{}_parcels.csv'.format(
                database, group, database, second_mod, group, atlas))

        # Age check
        second_age_check = second_df['age'] >= 60
        age_check = age_check.values & second_age_check.values

        # No outlier check (clinical groups)
        df_other['IQR'] = True
        second_df['IQR'] = True

    elif database == 'OASIS':  # OASIS CN
        second_mod = ['PET' if modality == 'MRI' else 'MRI'][0]
        second_df = pd.read_csv(
            '../data/{}/{}/{}_{}_{}_{}_parcels.csv'.format(
                database, group, database, second_mod, group, atlas))
        assert len(df_other.index) == len(second_df.index),\
            "Dataframes don't have the same length!"

        # Age check
        second_age_check = second_df['age'] >= 60
        age_check = age_check.values & second_age_check.values
        
        # Outlier check
        if check_outliers:
            ranges = pd.read_csv("../results/999_OUTLIERS/" +
                                 "{}_outlier_ranges.csv".format(atlas))
            col = ranges['col']
            low_out_pet = ranges['low']
            high_out_pet = ranges['high']
            insiderange = (~((df_other[col] < low_out_pet) |
                             (df_other[col] > high_out_pet)).any(axis=1))
            df_other['IQR'] = insiderange
            second_df['IQR'] = insiderange

            # Save outliers
            save_outliers = open("../results/999_OUTLIERS/" +
                                 "outliers_{}_{}_{}.txt".format(
                                     database, atlas, group), "a+")
            save_outliers.write("Outliers PET:\n")
            [save_outliers.write(x+"\n") for x in df_other['name'][(
                (df_other[col] > high_out_pet).any(axis=1) |
                (df_other[col] < low_out_pet).any(axis=1)).tolist()]]
            save_outliers.close()
        else:
            df_other['IQR'] = True
            second_df['IQR'] = True

    elif database == "DELCODE":
        # No outlier check (clinical samples), age check from one modality
        df_other['IQR'] = True

    df_other['Ageb'] = [0 if x < 74 else 1
                        if x < 84 else 2 for x in df_other['age']]
    df_other['Dataset'] = database
    df_other['AGE_CHECK'] = age_check
    df_other['Group'] = group
    
    # Exclude individuals younger than 60
    df_other = df_other[df_other['AGE_CHECK']]

    # If young subjects were excluded
    len_init = len(df_other.index)
    print("Detected {} subjects younger than 60 who were removed.".format(
        len_init-len(df_other.index)) + "from analyses")

    df_other.to_csv("../data/{}/{}/{}_parcels_{}_{}_{}_{}.csv".format(
        database, group, modality, atlas, group, database, fold))
    if 'second_mod' in locals():
        second_df['Dataset'] = database
        second_df['Group'] = group
        second_df['AGE_CHECK'] = age_check
        second_df = second_df[second_df['AGE_CHECK']]
        second_df.to_csv("../data/{}/{}/{}_parcels_{}_{}_{}_{}.csv".format(
            database, group, second_mod, atlas, group, database, fold))

    # Compare to age of current training fold
    print("T-test age ADNI CN - {} {}:".format(database, group),
          ttest_ind(ADNI_CN_train[['age']], df_other[['age']])[1][0])
    return df_other


def feature_selection(df, col, y, modality, atlas, i, percentile=50):
    """
    Select features for brain age estimation.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    modality : TYPE
        DESCRIPTION.
    atlas : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    percentile : TYPE, optional
        DESCRIPTION. The default is 75.

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    sel_col : TYPE
        DESCRIPTION.
    unsel_col : TYPE
        DESCRIPTION.

    """
    sel = SelectPercentile(score_func=mutual_info_regression,
                           percentile=percentile)

    # Get and fit FS to  train subset of outer CV loop
    df_train = df[df['train']]
    sel.fit(df_train[col], df_train[y])
    pickle.dump(sel, open(
            f"../results/0_FINAL_MODELS/feature_selector_{modality}_{i}_{atlas}.p", "wb"))

    sel_idx = sel.get_support(indices=True)
    sel_col = np.array(col)[sel_idx].tolist()
    unsel_col = [c for c in col if c not in sel_col]

    # remove unselected columns
    df.drop(unsel_col, axis=1, inplace=True)

    return df, sel_col, unsel_col


def prepare_data(df_mri, df_pet, col, atlas,
                 train_data="ADNI", older_60=True, info=True):
    """
    Check minimum age and ADNI study phase of participants.

    Parameters
    ----------
    data_mri : pd.dataframe
        parcels derived from MRI data
    data_pet : pd.dataframe
        parcels derived from PET data
    col : list or np.array
        columns to consider for brain age prediction
    imp : str
        main analysis, validation_random_seeds or neurocorrelations
    splits : int
        How many folds to split the data into for cross-validation
    test_size : float, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        The default is 0.3.
    train_data : str, optional
        Which dataset to use for training
    older_60: boolean, optional
        Whether to consider only individuals over age 60. The default is True.
    check_outliers : boolean, optional
        Whether to check for and exclude outliers. The default is True.
    rand_seed : int, optional
        Controls the shuffling applied to the data before applying the split.


    Returns
    -------
    None.

    """
    # raise error if not all individuals are the same across modalities
    assert df_mri['name'].tolist() == df_pet['name'].tolist(),\
        "IDs between modalities don't match."

    adniinfo = pd.read_csv(
        "../data/ADNI/CU/FDG_BASELINE_HEALTHY_4_15_2021_unique.csv", sep=";")

    # ADDED IN REVIEW 1: add group
    df_mri = pd.merge(df_mri, adniinfo[['Subject', 'Group', 'StudyPhase']],
                      left_on='name', right_on='Subject', how='left')
    df_pet = pd.merge(df_pet, adniinfo[['Subject', 'Group', 'StudyPhase']],
                      left_on='name', right_on='Subject', how='left')

    if info:
        if atlas == 'Sch_Tian_1mm':
            print("First column: {}".format(col[0]) +
                  " (should be 'X17Networks_LH_VisCent_ExStr_1')" +
                  "\nLast column: {}".format(col[-1]) +
                  " (should be 'CAU-lh)")
        elif atlas == 'AAL1_cropped':
            print("First column: {}".format(col[0]) +
                  " (should be 'Precentral_L')" +
                  "\nLast column: {}".format(col[-1]) +
                  " (should be 'Temporal_Inf_R')")

    # exclude individuals younger than 60 if older_60 == True
    if older_60:
        older_60_mri = df_mri['age'] >= 60
        older_60_pet = df_pet['age'] >= 60

        df_pet['AGE_CHECK'] = older_60_mri & older_60_pet
        df_mri['AGE_CHECK'] = older_60_mri & older_60_pet
        if info:
            print(len(df_pet.index)-sum(df_pet['AGE_CHECK']),
                  "individuals younger than 60 years discarded.")
    else:
        df_pet['AGE_CHECK'] = True
        df_mri['AGE_CHECK'] = True

    # divide into age bins of "young old", "middle old" and "oldest old"
    # use mri to do so --> same age bins for both modalities
    df_mri['Ageb'] = [0 if x < 74 else 1
                      if x < 84 else 2 for x in df_mri['age']]
    df_pet['Ageb'] = [0 if x < 74 else 1
                      if x < 84 else 2 for x in df_pet['age']]

    # only ADNI data of individuals older than 60 (if older_60 == True)
    # to be considered in train_test split
    # OASIS data to be reserved as additional test set
    # ADDED IN REVIEW 1: only CN
    # SCD and CU as additional test set
    df_mri = df_mri[df_mri.StudyPhase != 'ADNI Baseline']
    df_pet = df_pet[df_pet.StudyPhase != 'ADNI Baseline']

    # Drop individuals younger than 60
    df_mri['use_data'] = (df_mri['Dataset'] == train_data) & df_mri['AGE_CHECK']
    df_pet['use_data'] = (df_pet['Dataset'] == train_data) & df_pet['AGE_CHECK']
    df_mri = df_mri[df_mri['use_data']]
    df_pet = df_pet[df_pet['use_data']]
    
    assert len(df_mri.index) == len(df_pet.index)

    return df_mri, df_pet


def split_data(df_mri, df_pet, col, i, id_tr, atlas,
               y='age', check_outliers=True, feat_sel=True, info=True):
    """
    

    Parameters
    ----------
    df_mri : TYPE
        DESCRIPTION.
    df_pet : TYPE
        DESCRIPTION.
    use_data : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    id_tr : TYPE
        DESCRIPTION.
    atlas : TYPE
        DESCRIPTION.
    y : TYPE, optional
        DESCRIPTION. The default is 'age'.
    check_outliers : TYPE, optional
        DESCRIPTION. The default is True.
    select_features : TYPE, optional
        DESCRIPTION. The default is True.
    info : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    df_mri : TYPE
        DESCRIPTION.
    df_pet : TYPE
        DESCRIPTION.
    sel_col_mri : TYPE
        DESCRIPTION.
    sel_col_pet : TYPE
        DESCRIPTION.

    """

    names = df_mri['name'].to_numpy()
    train_names = names[id_tr]

    df_mri['train'] = [True if x in train_names
                       else False for x in df_mri['name']]
    df_pet['train'] = df_mri['train']

    if check_outliers:
        df_mri, df_pet = outlier_check_main(df_mri, df_pet, col,
                                            atlas=atlas)
        n_outliers = len(df_mri) - sum(df_mri['IQR'])
        if info:
            print("Total participants: ", len(df_mri),
                  "Inside IQR of all regions: ",
                  len(df_mri[df_mri['IQR']]),
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
        print("Total participants: ", len(df_mri),
              "Train set: ",
              len(df_mri[df_mri['train']]),
              "Test set: ",
              len(df_mri[~df_mri['train']]))

    if feat_sel:
        df_mri, sel_col_mri, unsel_col_mri = feature_selection(
            df_mri, col, y, 'MRI', atlas, i)
        df_pet, sel_col_pet, unsel_col_pet = feature_selection(
            df_pet, col, y, 'PET', atlas, i)
    else:
        sel_col_mri = col
        sel_col_pet = col

    # Save for reference
    df_mri.to_csv('../data/ADNI/CN/' +
                  'test_train_MRI_{}_{}.csv'.format(atlas, i))
    df_pet.to_csv('../data/ADNI/CN/' +
                  'test_train_PET_{}_{}.csv'.format(atlas, i))

    df_mri = df_mri[df_mri['AGE_CHECK'] & df_mri['IQR']]
    df_pet = df_pet[df_pet['AGE_CHECK'] & df_pet['IQR']]

    # these do not change here, only save last
    """df_mri_dk.to_csv('../data/ADNI/CU/' +
                     'ADNI_MRI_CU_{}_parcels.csv'.format(atlas))
    df_pet_dk.to_csv('../data/ADNI/CU/' +
                     'ADNI_PET_CU_{}_parcels.csv'.format(atlas))"""

    return df_mri, df_pet, sel_col_mri, sel_col_pet


def neuropsych_merge(df_pred, df_neuropsych,
                     neuropsych_var):
    """
    Merge predictions with neuropsychology.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe containing bias-corrected brain age of
        individuals from test set
    df_neuropsych : pd.DataFrame
        Dataframe containing variables of interest for cognitive performance
    neuropsych_var : list
        variables of cognitive performance to be considered.
        ADNI-MEM (memory) and ADNI-EF (executive function)

    Returns
    -------
    merged : pd.DataFrame
        Dataframe containing bias-corrected brain age and variables of
        cognitive performance of individuals from test set.

    """
    df_dem = pd.read_csv("../data/ADNI/PsychPath/ADNI_Neuropsych_Neuropath.csv",
                         sep=";")
    df_dem['PTGENDER'] = [1 if x == "Female"
                          else 2 if x == "Male"
                          else np.nan for x in df_dem['PTGENDER']]
    df_dem['APOE4'] = [1 if x > 0 else 0 for x in df_dem['APOE4']]
    df_pred['BAG'] = df_pred['Prediction']-df_pred['Age']
    dem_var = ['PTID', 'PTGENDER', 'PTEDUCAT', 'APOE4']
    merged = df_pred.merge(df_dem[dem_var], how="left", on="PTID")
    merged = merged.merge(df_neuropsych[['RID'] + neuropsych_var],
                          how='left', on='RID')
    return merged


def neuropath_merge(df_pred, df_neuropath1, df_neuropath2,
                    neuropath1_var, neuropath2_var):
    """
    Merge predictions with neuropathology.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe containing bias-corrected brain age of
        individuals from test set
    df_neuropath1 : pd.DataFrame
        Dataframe containing first part of variables of interest
        for neuropathology
    df_neuropath2 : pd.DataFrame
        Dataframe containing second part of variables of interest
        for neuropathology
    neuropath1_var : list
        List of variables to be considered from df_neuropath1
    neuropath2_var : list
        List of variables to be considered from df_neuropath2

    Returns
    -------
    merged : pd.DataFrame
        Dataframe containing bias-corrected brain age and variables of
        neuropathology of individuals from test set.

    """
    df_dem = pd.read_csv("../data/ADNI/PsychPath/ADNI_Neuropsych_Neuropath.csv",
                         sep=";")
    df_dem['PTGENDER'] = [1 if x == "Female"
                          else 2 if x == "Male"
                          else np.nan for x in df_dem['PTGENDER']]
    df_dem['APOE4'] = [1 if x > 0 else 0 for x in df_dem['APOE4']]
    df_pred['BAG'] = df_pred['Prediction']-df_pred['Age']
    dem_var = ['PTID', 'PTGENDER', 'PTEDUCAT', 'APOE4']
    merged = df_pred.merge(df_dem[dem_var],
                           how='left', on='PTID')
    merged = merged.merge(df_neuropath1[['PTID'] + neuropath1_var],
                          how='left', on='PTID')
    merged = merged.merge(df_neuropath2[['RID'] + neuropath2_var],
                          how='left', on='RID')

    # set extreme values to most extreme still observable value.
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
    df_pred : pd.DataFrame
        Dataframe containing bias-corrected brain age of
        individuals from test set
    df_dx : pd.DataFrame
        Dataframe containing diagnoses at m24 (month 24). Brain age
        of all individuals of the current analysis was calculated at
        baseline, thus the diagnosis at month 24 constitutes the diagnosis
        obtained 24 months after the observed brain age.

    Returns
    -------
    merged : pd.DataFrame
        Dataframe containing bias-corrected brain age and diagnoses after
        24 months of individuals from test set.

    """
    merged = df_pred.merge(df_dx[['PTID', 'DX']], how='left', on='PTID')
    return merged

def neuro_merge(group, psych_or_path, modality, atlas,
                fold="BAGGED"):
    """
    Merge brain age and BAG with neuropsychology and neuropathology

    Parameters
    ----------
    group : str
        CN or MCI
    psych_or_path : str
        cognitive performance (PSYCH) or neuropathology (PATH)
    modality : str
        PET or MRI
    fold : int or "BAGGED"
        which model results are observed. "BAGGED" refers to the mean of
        all five models.

    Returns
    -------
    sign : dict
        r-values of significant correlations

    """
    if group == "CN":
        add_ = ""
    else:
        add_ = "_"+str(fold)
        print("Model " + str(fold))
    df_pred = pd.read_csv(
        "../results/ADNI/{}/{}-predicted_age_{}_{}{}.csv".format(
            group, modality, atlas, group, add_))
    # ADNI RID = last 4 digits of ADNI PTID (required for merging)
    df_pred['RID'] = df_pred['PTID'].str[-4:].astype(int)

    y_true = df_pred['Age']
    y_pred = df_pred['Prediction']
    y_diff = y_pred - y_true

    # merge predictions with cognitive performance
    if (psych_or_path == "PSYCH") or (psych_or_path == "psych"):
        df_neuropsych = pd.read_csv(
            "../data/ADNI/PsychPath/UWNPSYCHSUM_Feb2022.csv", sep=";")
        var_ = ['ADNI_MEM', 'ADNI_EF']
        merged = neuropsych_merge(df_pred, df_neuropsych,
                                  var_)

    # merge predictions with neuropathology
    elif (psych_or_path == "PATH") or (psych_or_path == "path"):
        df_neuropath1 = pd.read_csv(
            "../data/ADNI/PsychPath/ADNI_Neuropsych_Neuropath.csv", sep=";")
        df_neuropath2 = pd.read_csv(
            "../data/ADNI/PsychPath/UPENNBIOMK_MASTER_bl_unique.csv", sep=";")
        neuropath1_var = ['AV45', 'ABETA', 'TAU', 'PTAU']
        # tau meta-roi did not have enough entries so far
        neuropath2_var = []
        var_ = neuropath1_var + neuropath2_var
        merged = neuropath_merge(df_pred, df_neuropath1, df_neuropath2,
                                 neuropath1_var, neuropath2_var)

    merged["BAG"] = y_diff
    merged.to_csv("../results/ADNI/{}/pred_merged_{}_{}_{}_{}.csv".format(
            group, atlas, group, modality, psych_or_path), index=False)