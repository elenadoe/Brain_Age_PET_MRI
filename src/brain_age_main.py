#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""

from collections import Counter
from tqdm import tqdm
from steps_of_analysis import brain_age

import numpy as np

dir_mri_csv = '../data/main/MRI_parcels_all.csv'
dir_pet_csv = '../data/main/PET_parcels_all.csv'

analyze = 1
modality = 'PET'


def main(analyze, modality):
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
                               info_init=info_init[correct_with_CA.index(c)])
            bias_results[str(c) + '_model'] = result[4]
            bias_results[str(c) + '_MAE'] = result[2]
            bias_results[str(c) + '_R2'] = result[3]
    elif analyze == 2:
        brain_age(dir_mri_csv, dir_pet_csv, modality,
                  correct_with_CA='True')
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
