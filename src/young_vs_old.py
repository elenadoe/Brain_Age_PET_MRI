#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import mean_absolute_error
from julearn import run_cross_validation
from julearn.utils import configure_logging
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting
from nilearn import image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
from lib.create_splits import stratified_splits

df = pd.read_csv('data/test_train_MRI.csv', index_col=0)

df_train = df[df['train'] == "T"]
# round to no decimal place
df_train['Age'] = np.around(df_train['Age'].values)
df_train = df_train.reset_index(drop=True)
df_train['old'] = df['Age'] > df['Age'].median()
col = [x for x in df_train.columns if 'H_' in x]
# plt.hist(y, bins=30)
rand_seed = 42
num_bins = 10
model_names = ['gauss', 'svm']
splits = 5

model_results = []
scores_results = []
res = {}
res['model'] = []
res['iter'] = []
res['pred'] = []
res['ind'] = []
# res = pd.Series(index=df_train.index)
for i, model in enumerate(model_names):
    cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                df_train['old'])
    cv = list(cv)
    scores, final_model = run_cross_validation(X=col, y='old',
                                               # preprocess_X='scaler_robust',
                                               # problem_type='regression',
                                               data=df_train,
                                               model=model, cv=cv,
                                               return_estimator='all',
                                               seed=rand_seed,
                                               scoring=['accuracy',
                                                        'balanced_accuracy'])
    model_results.append(final_model)
    scores_results.append(scores)
    for iter in range(splits):
        pred = scores.estimator[iter].predict(df_train.iloc[cv[iter][1]][col])
        res['pred'].append(pred)
        res['iter'].append(iter)
        res['model'].append(str(model))
        res['ind'].append(cv[iter][1])
    print(model)
    print(scores['test_accuracy'].mean())
    print(scores['test_balanced_accuracy'].mean())

df_res = pd.DataFrame(res)
age_pred = {}
age_pred['subj'] = []
age_pred['pred'] = []
age_pred['real'] = []
age_pred['model'] = []
for i, fold in enumerate(df_res['ind']):
    for ind, sample in enumerate(fold):
        age_pred['real'].append(df_train.iloc[sample]['old'])
        age_pred['pred'].append(df_res['pred'].iloc[i][ind])
        age_pred['subj'].append(df_train.iloc[sample]['Subject'])
        age_pred['model'].append(df_res.iloc[i]['model'])

df_ages = pd.DataFrame(age_pred)
