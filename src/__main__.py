#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""

from collections import Counter
# from tqdm import tqdm
from steps_of_analysis import brain_age

import numpy as np

dir_mri_csv = '../data/main/MRI_parcels_all.csv'
dir_pet_csv = '../data/main/PET_parcels_all.csv'
modality = 'PET'

if __name__ == "__main__":
    analyze = input("Which analysis would you like to run?\
                    \nType\n'1' for bias-correction\
                    \n'2' for brain age prediction in CN\
                    \n'2.1' for validation of brain age prediction in CN\
                    \n'3' for brain age prediction in MCI\
                    \n'4' for association with cognitive\
                    \nperformance/neuropathology")
    if analyze == 1:
        # TODO. not finished/tested
        correct_with_CA = [None, True, False]
        bias_results = {}
        for c in correct_with_CA:
            result = brain_age(dir_mri_csv, dir_pet_csv,
                               modality, correct_with_CA=c)
            bias_results[str(c) + '_model'] = result[4]
            bias_results[str(c) + '_MAE'] = result[2]
            bias_results[str(c) + '_R2'] = result[3]
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
