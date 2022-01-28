#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:14:53 2021

@author: doeringe
"""

import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats

import neuropsychology_correlations
import plots


# %%
# LOAD DATA
# load and inspect data, set modality
modality = 'PET'
database = "2_MCI_ADNI"
mode = "test_MCI"
df = pd.read_csv('../data/ADNI/MCI_{}_parcels_withNP_nooutliers.csv'.format(
    modality), sep=",")
df = df[df['QC'] == True]
df = df.reset_index(drop=True)

# select columns 
col = df.columns[4:-19].tolist()

final_model = ['svr' if modality == 'PET' else 'svr' if modality == "MRI" else "NAN"][0]

# %%
model_all = pickle.load(open('../results/1_CN_ADNI_OASIS/model_{}_{}.p'.format(
    final_model, modality), "rb"))
intercept_ = model_all['intercept']
slope_ = model_all['slope']
model_ = model_all['model']
y_true = df['age'].values

# %%
# predict and apply bias correction
pred = model_.predict(df[col])
#pred_bc = (pred - intercept_)/slope_
pred_bc = pred - (slope_*y_true + intercept_)

plots.real_vs_pred_2(df['age'], pred_bc, final_model, mode,
                     modality, database)

# %%
# CHECK BIAS
plots.check_bias(y_true, pred, final_model, "{}_test_MCI".format(modality),
                 database="merged")

plots.check_bias(y_true, pred_bc, final_model, "{}_test_MCI".format(modality),
                 database="merged", corrected=True)
y_diff = pred_bc - y_true
print("Range of BPAD: ", np.min(y_diff), np.max(y_diff))

# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df.columns[-18:]

sign_npt = neuropsychology_correlations.neuropsych_correlation(y_true,
                                                               pred_bc, "BPAD",
                                                               npt,
                                                               df,
                                                               modality,
                                                               database)

# %%
# INTERACTION EFFECTS
y_diff = pred_bc - y_true
for k in sign_npt:
    exc = np.isnan(df[k])
    pos = df['BPAD Category']=='positive'
    neg = df['BPAD Category']=='negative'
    
    pos_bool = np.array(~exc) & np.array(pos)
    neg_bool = np.array(~exc) & np.array(neg)
    pearson_pos = stats.pearsonr(y_diff[pos_bool],
                                 df[k][pos_bool])
    pearson_neg = stats.pearsonr(y_diff[neg_bool],
                                 df[k][neg_bool])
    print(k, "\033[1msignificant in positive BPAD: ", pearson_pos[1] < 0.05,
          "\033[0m", pearson_pos,
          "\n\033[1msignificant in negative BPAD: ", pearson_neg[1] < 0.05,
          "\033[0m", pearson_neg)
    
