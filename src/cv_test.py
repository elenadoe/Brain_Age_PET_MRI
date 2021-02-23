#!/home/antogeo/anaconda3/envs/julearn/bin/python

import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.svc
from sklearn.model_selection import train_test_split, StratifiedKFold
from lib.create_splits import stratified_splits

%config IPCompleter.use_jedi = False

df = pd.read_csv('/home/antogeo/codes/PET_MRI_age/tests/test_train.csv',
                 index_col=0)

df_train = df[df['train'] == "T"]

col = [x for x in df_train.columns if 'cereb' in x]

X = df_train[col].values
y_pseudo = df_train['Age_bins']
y = df_train['Age'].values

rand_seed = 42
num_bins = 5
# creates dictionary of test indices for different repeats
cv_folds = stratified_splits(bins_on=y_pseudo, num_bins=num_bins,
                             data=df_train, num_splits=5,
                             shuffle=True, random_state=42)
cv = StratifiedKFold(n_splits=5).split(X, y_pseudo)

scores = run_cross_validation(X=X, y=y,
                              preprocess_X='scaler_robust',
                              problem_type='regression',
                              model='svm', cv=cv,
                              return_estimator='final',
                              seed=rand_seed)
