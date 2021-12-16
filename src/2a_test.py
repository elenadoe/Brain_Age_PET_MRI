#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:02:12 2021
*
@author: doeringe
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

import plots
import neuropsychology_correlations
# %%
# matplotlib config
cm_all = pickle.load(open("../data/config/plotting_config.p", "rb"))

# %%
# LOAD DATA
# load and inspect data, set modality
# TODO: stratify by age group (young old, middle old, oldest-old)
# modality = input("Which modality are you analyzing? ")
modality = 'MRI'
database = "1_CN_ADNI_OASIS"
mode = "test"
df = pd.read_csv('../data/ADNI/test_train_' + modality + '.csv')
df['PTGENDER'] = [0 if x == "Male" else 1 for x in df['PTGENDER'].values]
final_model = ['svr' if modality == 'PET' else 'svr' if modality == "MRI" else "NAN"][0]

# during cross-validation, gb was best for PET
# and rvr was best for MRI
model_all = pickle.load(open('../results/' + database +
                             '/model_{}_{}.p'.format(
    final_model, modality), "rb"))
intercept_ = model_all['intercept']
slope_ = model_all['slope']
model_ = model_all['model']

# select columns with '_' which are col's with features
col = df.columns[4:-23].tolist()
print("First column: {}".format(col[0]) +
      "\n(should be 'X17Networks_LH_VisCent_ExStr_1')" +
      "\nLast column: {}".format(col[-1]) +
      " (should be 'CAU.lh)")

# %%
# TESTING

# How well does the model perform on unseen data?
df_test = df[df['train'] == False]
df_test = df_test.reset_index(drop=True)
db_test = df_test['Dataset'].tolist()
mode = "test"

# investigate age distribution
# plot hist with Ages of train data
sns.displot(df_test, x='age', kde=True, hue='Dataset', palette=cm_all)
plt.title('Age distribution in test set')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.savefig('../results/{}/plots/{}_age_distribution'.format(database,
                                                             modality) +
            '_test.png',
            bbox_inches="tight")
plt.show()

X_test = df_test[col]
y_true = df_test['age'].values

# plot model predictions against GT in test set
y_pred = model_.predict(X_test)
#y_pred_bc = (y_pred - intercept_)/slope_
y_pred_bc = y_pred - (slope_*y_true + intercept_)


# %%
plots.real_vs_pred_2(y_true, y_pred_bc, final_model, modality,
                     mode, database, db_test)

print(mean_absolute_error(y_true,y_pred),
      r2_score(y_true,y_pred),
      np.mean(y_pred-y_true))

# %%
# CHECK BIAS
plots.check_bias(y_true, y_pred, final_model, "{}_test".format(modality),
                 database=database)

plots.check_bias(y_true, y_pred_bc, final_model, "{}_test".format(modality),
                 database=database, corrected=True)

# %%
# SAVE RESULTS
# Create table of (corrected) predicted and chronological age in this modality
# rvr had better performance in both MAE and R2 --> take rvr as final model
y_diff = y_pred_bc - y_true
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_true, columns=["age"]),
                      pd.DataFrame(y_pred, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/pred_age_{}_{}.csv'.format(modality, final_model))


# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df_test.columns[-19:-2].values

neuropsychology_correlations.neuropsych_correlation(df_test['age'], y_pred_bc,
                                                    "BPAD",
                                                    npt,
                                                    df_test,
                                                    modality,
                                                    database)
# %%
# PERMUTATION IMPORTANCE
pi = permutation_importance(model_,
                            X_test, y_true,
                            n_repeats=1000)


# %%
plots.permutation_imp(pi, final_model, modality, database)
