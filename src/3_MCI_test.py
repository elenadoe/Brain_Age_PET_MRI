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
modality = 'MRI'
database = "ADNI"
mode = "test"
df = pd.read_csv('../data/ADNI/ADNI_MCI_{}_Sch_Tian_1mm_parcels_NP.csv'.format(
    modality), sep=";")
df = df[df['Age'] > 65]
df = df.reset_index(drop=True)
# select columns with '_' which are col's with features
col = df.columns[3:-19].tolist()

# %%
model_all = pickle.load(open('../results/model_svr_{}.p'.format(
    modality), "rb"))
intercept_ = model_all['intercept']
slope_ = model_all['slope']
model_ = model_all['model']

# %%
# predict and apply bias correction
pred = model_.predict(df[col])
pred_bc = (pred - intercept_)/slope_

plots.real_vs_pred_2(df['Age'], pred_bc, "svr", mode,
                   modality, database, group="MCI")

# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df.columns[-18:]
neuropsychology_correlations.neuropsych_correlation(df['Age'], pred_bc, "BPA",
                                                    npt,
                                                    df,
                                                    modality,
                                                    database,
                                                    group='MCI')
# Difference between PA-CA+ and PA-CA-
neuropsychology_correlations.plot_bpad_diff(df['Age'], pred_bc,
                                            npt,
                                            df,
                                            modality,
                                            database,
                                            group='MCI')