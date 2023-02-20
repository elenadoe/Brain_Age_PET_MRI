#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:15:44 2023

@author: doeringe
"""

from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_aal
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score


def make_parcels(scan_paths):
    """
    Extract AAL1 parcels from given data.

    Parameters
    ----------
    scan_paths : list
        contains all scan paths

    Returns
    -------
    None.

    """
    atlas = '../templates/AAL1_TPMcropped.nii'
    atlas = nib.load(atlas)
    labels = fetch_atlas_aal().labels
    output_csv = '../results/AAL1_parcels.csv'

    image_list = []
    subj = {}
    subj['age'] = []
    subj['name'] = []

    # create list of regional data and subject IDs
    for scan in scan_paths:
        this_image = nib.load(scan)
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj['name'].append(scan)
        subj['age'].append('')
    pause = input(
        """Atlas parcelation successful.
        Please enter chronological (true) age of all subjects in the csv
        file, which was stored at {}.
        Enter 'done' when done.\n""".format(output_csv))

    features = np.array(image_list)
    x, y, z = features.shape
    features = features.reshape(x, z)
    df = pd.DataFrame(features, columns=labels)
    df_sub = pd.DataFrame(subj)
    df_final = pd.concat([df_sub, df], axis=1)

    df_final.to_csv(output_csv, index=False)

    return pause


def predict(df, col, model_, final_model_name,
            slope_, intercept_, modality, r,
            y='age', correct_with_CA=True):
    """
    Predicts brain age using trained algorithms.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing input and output variables
    col : list or np.ndarray
        Column(s) to be used as input features
    model_ : ExtendedDataFramePipeline
        final model to be used for prediction
    final_model_name : str
        name of final model to be used for saving of plots
    y : str, optional
        Column to be considered as output feature. The default is age.
    slope_ : float
        Slope of linear model for bias correction
    intercept_ : float
        Intercept of linear model for bias correction
    modality : str
        PET or MRI
    r : int
        which round of cv is saved
    correct_with_CA : boolean, optional
        Whether or not to correct bias with chronological age.
        The default is True.

    Returns
    -------
    y_pred_bc : np.ndarray
        Bias-corrected brain age of individuals from test set

    """
    print("Model {}:".format(r))
    y_pred = model_.predict(df[col])
    if r == 0:
        print("n = ", len(df), "mean age = ",
              np.round(np.mean(df.age), 2),
              np.round(np.std(df.age), 2))

    if correct_with_CA is None:
        y_pred_bc = y_pred
    elif correct_with_CA:
        # for age correction WITH chronological age
        y_pred_bc = y_pred - (slope_*df[y] + intercept_)
    else:
        # for age correction WITHOUT chronological age
        y_pred_bc = (y_pred - intercept_)/slope_

    y_diff = y_pred_bc - df[y]
    linreg = stats.linregress(df[y], y_diff)
    r_val = linreg[2]
    p_val = linreg[3]
    check = p_val > 0.05
    mae = mean_absolute_error(df[y], y_pred_bc)
    r2 = r2_score(df[y], y_pred_bc)

    # scsv file of predictions of test set
    df_final = pd.DataFrame({'name': df['name'],
                             'age': df[y],
                             'brain age': y_pred_bc,
                             'BAG': y_pred_bc - df[y]})

    print("Bias between chronological age and BPAD eliminated:",
          check, "(r =", r_val, "p =", p_val, ")")
    print("MAE: {}, R2: {}".format(mae, r2))

    return df_final
