#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""

from steps_of_analysis2 import brain_age, predict_other
from neuropsychology_correlations import neuro_correlation

dir_mri_csv = '../data/main/MRI_parcels_all.csv'
dir_pet_csv = '../data/main/PET_parcels_all.csv'

analyze = 1
modality = 'PET'
all_modalities = ["PET", "MRI"]
rand_seed = 0


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
        4.1 for association of BPAD with cognitive performance in CN
        4.2 for association of BPAD with neuropathology in CN
        4.3 for association of BPAD with cognitive performance in MCI
        4.4 for association of BPAD with neuropathology in MCI
        5.1 for difference of BPAD between disease progressors and stables
            in CN
        5.2 for difference of BPAD between disease progressors and stables
            in MCI
    modality : str
        PET or MRI

    Returns
    -------
    None.

    """
    if analyze == 1:
        correct_with_CA = [None, True, False]
        bias_results = {}
        for c in correct_with_CA:
            print("\033[1m{} Correction with CA:".format(
                str(correct_with_CA.index(c)+1) + "/3"), str(c), "\033[0m")
            n_outliers, mae, r2,\
                final_model, final_model_name = brain_age(
                    dir_mri_csv, dir_pet_csv, modality, correct_with_CA=c,
                    info_init=False, save=False, rand_seed=rand_seed)
            bias_results[str(c) + '_model'] = final_model_name
            bias_results[str(c) + '_MAE'] = mae
            bias_results[str(c) + '_R2'] = r2
        print(bias_results)
    elif analyze == 2:
        group = "CN"
        brain_age(dir_mri_csv, dir_pet_csv, modality,
                  correct_with_CA='True', imp="main",
                  rand_seed=rand_seed,
                  info=True, save=True, info_init=True)
    elif analyze == 3:
        csv_file = [dir_mri_csv if modality == "MRI"
                    else dir_pet_csv][0]
        csv_file_other = [dir_pet_csv if modality == "MRI"
                          else dir_mri_csv][0]
        predict_other(csv_file, csv_file_other,
                      what="OASIS", modality=modality)
    elif analyze == 4:
        all_modalities.remove(modality)
        other_modality = all_modalities[0]
        csv_mci = "../data/MCI/MCI_" + modality + "_parcels.csv"
        csv_mci_othermod = "../data/MCI/MCI_" + other_modality + "_parcels.csv"
        predict_other(csv_mci, csv_mci_othermod, what="MCI", modality=modality)
    elif analyze == 5.1:
        group = "CN"
        neuro_correlation(group, "BPAD", "PSYCH", modality)
    elif analyze == 5.2:
        group = "CN"
        neuro_correlation(group, "BPAD", "PATH", modality)
    elif analyze == 5.3:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PSYCH", modality)
    elif analyze == 5.4:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PATH", modality)
