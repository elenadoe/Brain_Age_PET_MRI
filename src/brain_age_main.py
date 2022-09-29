#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""

from steps_of_analysis2 import brain_age, predict_other
from neuropsychology_correlations import neuro_correlation

dir_mri_csv = '../data/CN/MRI_parcels_all.csv'
dir_pet_csv = '../data/CN/PET_parcels_all.csv'

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
    if analyze == "bias correction":
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
    elif analyze == "predict ADNI CN":
        group = "CN"
        brain_age(dir_mri_csv, dir_pet_csv, modality,
                  correct_with_CA='True',
                  rand_seed=rand_seed,
                  info=True, save=True, info_init=True)

    elif analyze == "predict OASIS CN":
        predict_other(database="OASIS", group="CN", modality=modality)

    elif analyze == "predict DELCODE SCD":
        assert modality == "PET", "DELCODE SCD only available for PET."
        predict_other(database="DELCODE", group="SCD", modality=modality)

    elif analyze == "predict ADNI MCI":
        predict_other(database="ADNI", group="MCI", modality=modality)

    elif analyze == "predict DELCODE MCI":
        assert modality == "MRI", "DELCODE MRI only available for PET."
        predict_other(database="DELCODE", group="MCI", modality=modality)

    elif analyze == 5.1:
        group = "CN"
        neuro_correlation(group, "BPAD", "PSYCH", modality, fold="all")
    elif analyze == 5.2:
        group = "CN"
        neuro_correlation(group, "BPAD", "PATH", modality, fold="all")
    elif analyze == 5.3:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PSYCH", modality)
    elif analyze == 5.4:
        group = "MCI"
        neuro_correlation(group, "BPAD", "PATH", modality)
