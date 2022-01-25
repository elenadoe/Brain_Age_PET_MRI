#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""

from collections import Counter
from tqdm import tqdm
from steps_of_analysis import brain_age, predict
from neuropsychology_correlations import neuro_correlation, \
    conversion_analysis
from plots import plot_hist
import pickle
import numpy as np
import pandas as pd

dir_mri_csv = '../data/main/MRI_parcels_all.csv'
dir_pet_csv = '../data/main/PET_parcels_all.csv'

analyze = 1
modality = 'PET'
rand_seed = 42


def main(analyze, modality, rand_seed=rand_seed):
    """
    Execute analysis.

    Parameters
    ----------
    analyze : int/float
        Which analysis would you like to run?
        1 for bias-correction
        2 for brain age prediction in CN
        2.1 for validation of brain age prediction in CN
        3 for brain age prediction in MCI
        4 for association with cognitive performance/neuropathology
    modality : str
        PET or MRI

    Returns
    -------
    None.

    """
    if analyze == 1:
        correct_with_CA = [None, True, False]
        bias_results = {}
        info_init = [True, False, False]
        for c in correct_with_CA:
            print("\033[1m{} Correction with CA:".format(
                str(correct_with_CA.index(c)+1) + "/3"), str(c), "\033[0m")
            result = brain_age(dir_mri_csv, dir_pet_csv,
                               modality, correct_with_CA=c,
                               info_init=info_init[correct_with_CA.index(c)],
                               save=info_init, rand_seed=rand_seed)
            bias_results[str(c) + '_model'] = result[4]
            bias_results[str(c) + '_MAE'] = result[2]
            bias_results[str(c) + '_R2'] = result[3]
    elif analyze == 2:
        brain_age(dir_mri_csv, dir_pet_csv, modality,
                  correct_with_CA='True', rand_seed=rand_seed,
                  info=True, info_init=True)
    elif analyze == 2.1:
        mae_range = []
        r2_range = []
        n_outliers_range = []
        models = []
        for i in tqdm(range(50)):
            n_outliers, pred, mae, r2,\
                final_model, final_model_name = brain_age(
                    dir_mri_csv, dir_pet_csv,
                    modality, imp='validation_random_seeds',
                    rand_seed=i, info=False, info_init=False, save=False)
            n_outliers_range.append(n_outliers)
            mae_range.append(mae)
            r2_range.append(r2)
            models.append(final_model_name)

        print("Range of 50 iterations:\nMAE:",
              np.min(mae_range), "-", np.max(mae_range),
              "(mean: {})".format(np.mean(mae_range)),
              "\nR2:",
              np.min(r2_range), "-", np.max(r2_range),
              "(mean: {})".format(np.mean(r2_range)),
              "\nModels: ", Counter(models))
    elif analyze == 3:
        file_ = pd.read_csv("../data/MCI/MCI_" + modality + "_parcels.csv",
                            sep=";")
        plot_hist(file_, database='MCI', train_test='MCI',
                  modality=modality, database_list=file_['Dataset'])
        col = pickle.load(open("../config/columns.p", "rb"))
        final_model_name = "svm"
        final_model = pickle.load(open(
            "../results/final_model_{}_True.p".format(
                modality), "rb"))
        params = pd.read_csv(
            "../results/CN/models_and_params_{}_True.csv".format(modality))
        slope_ = params['{}_slope'.format(final_model_name)][0]
        intercept_ = params['{}_intercept'.format(final_model_name)][0]

        pred, mae, r2 = predict(file_, col, final_model, final_model_name,
                                slope_, intercept_, modality, database="MCI",
                                train_test="MCI")
    elif analyze == 4.1:
        group = "CN"
        neuro_correlation(group, "BPAD", "PSYCH", modality)
    elif analyze == 4.2:
        group = "CN"
        neuro_correlation(group, "BPAD", "PATH", modality)
    elif analyze == 4.3:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PSYCH", modality)
    elif analyze == 4.4:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PATH", modality)
    elif analyze == 5.1:
        group = "CN"
        conversion_analysis(group, modality)
    elif analyze == 5.2:
        group = "MCI"
        conversion_analysis(group, modality)