#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:14:53 2021

@author: doeringe
"""

import pickle
import pandas as pd

import neuropsychology_correlations
import plots

# %%
# LOAD DATA
# load and inspect data, set modality
modality = 'PET'
database = "ADNI"
mode = "test"
df = pd.read_csv('../data/ADNI/ADNI_MCI_{}_Sch_Tian_1mm_parcels_NP.csv'.format(
    modality), sep=",")
df = df[df['age'] > 65]
df = df.reset_index(drop=True)
# select columns with '_' which are col's with features
col = df.columns[3:-19].tolist()

final_model = ['gb' if modality == 'PET' else 'gb' if modality == "MRI" else "NAN"][0]

# %%
model_all = pickle.load(open('../results/model_{}_{}.p'.format(
    final_model, modality), "rb"))
intercept_ = model_all['intercept']
slope_ = model_all['slope']
model_ = model_all['model']
y_true = df['age'].values
# %%
# predict and apply bias correction
pred = model_.predict(df[col])
pred_bc = (pred - intercept_)/slope_
# pred_bc = pred - (slope_*y_true + intercept_)

plots.real_vs_pred_2(df['age'], pred_bc, final_model, mode,
                     modality, database, group="MCI")

# %%
# CHECK BIAS
plots.check_bias(y_true, pred, "rvr", "PET_test_MCI",
                 database="merged")

plots.check_bias(y_true, pred_bc, "rvr", "PET_test_MCI",
                 database="merged", corrected=True)
# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df.columns[-18:]

neuropsychology_correlations.neuropsych_correlation(df['age'], pred_bc, "BPAD",
                                                    npt,
                                                    df,
                                                    modality,
                                                    database,
                                                    group='MCI')

