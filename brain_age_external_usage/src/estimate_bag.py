#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:34:24 2023

@author: doeringe
"""
from glob import glob
import pandas as pd
import sys
import os
import pickle
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter('ignore', UserWarning)

from utils_ import make_parcels, predict

def estimate_bag():
    print("##### BAG estimation #####")
    # get data directories
    print("Required for BAG estimation: the directory where Nifti scans are stored and a csv file indicating the true (chronological) age of each subject.")
    dir_scans = input("""
                      Enter path where scans are stored in Nifti format.\n""")
    if dir_scans[-1] == "/":
        dir_scans = dir_scans[:-1]
    scans_ = sorted(glob(dir_scans+"/**.nii"))

    assert len(scans_)>0, "no Nifti files found in directory and subdirectories"

    # infer modality
    modality = input("""
                     Enter modality of the scan(s). (MRI or FDG-PET)\n""")
    while modality.upper() not in ["MRI", "FDG-PET"]:
        modality = input("""
                         Invalid modality. Enter modality of the scan(s). (MRI or FDG-PET)\n""")
    if modality == "FDG-PET":
        modality = "PET"

    # check if prerequisites are fulfilled
    prep = eval(input("""
                      Was the data pre-processed according to Doering et al.? (True or False)\n"""))
    if not prep:
        print("Note that brain age estimation was only assessed for the preprocessing outlined in Doering et al. BAG estimates may be false.")
    adni = eval(input("""
                      Is this ADNI data? (True or False)\n"""))
    if adni:
        print("Note that the model has been trained on ADNI data, BAG estimates may be skewed.\n")

    # make parcels and require manual entering of chronological age
    make_parcels(scans_)
    pause = input("""
                  Please enter chronological (true) age of all subjects
                  in the csv file, which was stored at ../results/AAL1_parcels.csv. Enter 'done' when done.""")
    while pause != "done" and pause != "'done'":
        pause = input("Please enter chronological (true) age of all subjects in the csv\
            file, which was stored at ../results/AAL1_parcels.csv. Enter 'done' when done.")

    # read df containing AAL parcels and chronological age
    df = pd.read_csv("../results/AAL1_parcels.csv")

    # don't break if Excel/office program changing separator
    if len(df.columns)<2:
        df = pd.read_csv("../results/AAL1_parcels.csv", sep=";")
    len_init = len(df.index)

    # label data and assure all subjects are older than 60
    df['Dataset'] = "new_data"
    df['AGE_CHECK'] = df['age'] >= 60
    df = df[df['AGE_CHECK']]
    if len(df.index)-len_init > 0:
        print("{} individuals were younger than 60. No BAG estimation was possible.".format(len(df.index)-len_init))

    col = pickle.load(open("../templates/columns_AAL1_cropped.p", "rb"))
    assert col[0] == "Precentral_L" and col[-1] == "Temporal_Inf_R", "wrong columns selected"

    # load models from all five folds and predict new data
    predictions = []
    bags = []
    for i in range(5):
        model = pickle.load(open(
                "../templates/0_FINAL_MODELS/final_model_{}_AAL1_cropped_True_{}.p".format(
                    modality, i), "rb"))
        final_model_name = ['svm' if 'svm' in model.named_steps.keys() else 'rvr'][0]
        params = pd.read_csv(
                "../templates/0_FINAL_MODELS/" +
                "models_and_params_{}_AAL1_cropped_True_{}.csv".format(
                    modality, i))
        slope_ = params['{}_slope'.format(final_model_name)][0]
        intercept_ = params['{}_intercept'.format(final_model_name)][0]
        pred_df = predict(df, col, model, final_model_name,
                          slope_, intercept_, modality, r=i)
        predictions.append(pred_df['brain age'].tolist())
        bags.append(pred_df['BAG'].tolist())

    # steal ID and chronological age from last fold, average brain age and BAG over all folds
    pred_df['brain age'] = np.mean(predictions, axis=0)
    pred_df['BAG'] = np.mean(bags, axis=0)
    pred_df.to_csv("../results/{}-predicted_BAG.csv".format(modality), index=False)
    print("Results stored under ../results/{}-predicted_BAG.csv".format(modality))
    
    sns.scatterplot(data=pred_df, x='brain age', y='age', s=100, alpha=0.7)
    
if __name__ == "__main__":
    estimate_bag()