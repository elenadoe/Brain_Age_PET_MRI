#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:32:44 2021

@author: doeringe
"""
import numpy as np
from steps_of_analysis import brain_age, predict_other


# TODO: make package (create __init__.py file)

all_modalities = ["PET", "MRI"]
rand_seed = 0


def main(analyze, modality, atlas, rand_seed_np, rand_seed=rand_seed):
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
    while atlas != 'Sch_Tian' and atlas != 'AAL3':
        atlas = input("Invalid argument, which atlas would you" +
                      " like to use?\nPossible arguments: Sch_Tian or AAL3")
    dir_mri_csv = '../data/ADNI/CN/ADNI_MRI_CN_{}_1mm_parcels.csv'.format(
        atlas)
    dir_pet_csv = '../data/ADNI/CN/ADNI_PET_CN_{}_1mm_parcels.csv'.format(
        atlas)

    if analyze == "bias correction":
        correct_with_CA = [None, True, False]
        bias_results = {}
        for c in correct_with_CA:
            print("\033[1m{} Correction with CA:".format(
                str(correct_with_CA.index(c)+1) + "/3"), str(c), "\033[0m")
            n_outliers, mae, r2,\
                final_model, final_model_name = brain_age(
                    dir_mri_csv, dir_pet_csv, modality, atlas=atlas,
                    correct_with_CA=c, info_init=False, save=False,
                    rand_seed=rand_seed)
            bias_results[str(c) + '_model'] = final_model_name
            bias_results[str(c) + '_MAE'] = mae
            bias_results[str(c) + '_R2'] = r2
        print(bias_results)
    elif analyze == "predict ADNI CN":
        group = "CN"
        brain_age(dir_mri_csv, dir_pet_csv, modality, atlas=atlas,
                  correct_with_CA='True',
                  rand_seed=rand_seed,
                  info=True, save=True, info_init=True)

    else:
        database = analyze.split()[1]
        group = analyze.split()[2]
        assert database in ["OASIS", "ADNI", "DELCODE"],\
            "database not recognized"
        assert group in ["CN", "SMC", "MCI"],\
            "group not recognized"

        predict_other(database=database, group=group, modality=modality,
                      atlas=atlas, rand_seed_np=rand_seed_np)
