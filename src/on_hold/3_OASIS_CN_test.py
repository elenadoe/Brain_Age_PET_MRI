#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:49:54 2021

@author: doeringe
"""

import pickle
import pandas as pd
import plots

#%%
# LOAD DATA
# load and inspect data, set modality
modality = 'PET'
database = "OASIS"
mode = "test"
df = pd.read_csv('../data/OASIS/OASIS_parcels_FDG.csv')
df = df[df['Age']>65]
df = df.reset_index(drop=True)
# select columns with '_' which are col's with features
col = df.columns[2:].tolist()

#%%
model_all = pickle.load(open('../results/model_rvr_{}.p'.format(modality),"rb"))
intercept_ = model_all['intercept']
slope_ = model_all['slope']
model_ = model_all['model']

#%%
# predict and apply bias correction
pred = model_.predict(df[col])
pred_bc = (pred - intercept_)/slope_

plots.real_vs_pred_2(df['Age'], pred_bc, "rvr", mode, 
                   modality, database)