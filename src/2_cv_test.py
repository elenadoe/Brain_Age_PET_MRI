#!/home/antogeo/anaconda3/envs/julearn/bin/python

import pandas as pd
# pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR
from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from lib.create_splits import stratified_splits

configure_logging(level='INFO')
# in case seaborn conflicts with autocomplete
df = pd.read_csv('/home/antogeo/codes/PET_MRI_age/tests/test_train.csv',
                 index_col=0)

df_train = df[df['train'] == "T"]

col = [x for x in df_train.columns if 'cereb' in x]

X = df_train[col].values
y_pseudo = df_train['Age_bins']
y = df_train['Age'].values

rand_seed = 42
num_bins = 5
rvr = RVR()
models = [rvr, 'svm']
model_names = ['rvr', 'svm']
# creates dictionary of test indices for different repeats

cv_folds = stratified_splits(bins_on=y_pseudo, num_bins=num_bins,
                             data=df_train, num_splits=5,
                             shuffle=True, random_state=42)
scores = {}
for i, model in enumerate(models):
    cv = StratifiedKFold(n_splits=5).split(X, y_pseudo)
    scores[model_names[i]] = []
    scores[model_names[i]].append(run_cross_validation(X=X, y=y,
                                  preprocess_X='scaler_robust',
                                  problem_type='regression',
                                  model=model, cv=cv,
                                  return_estimator='final',
                                  seed=rand_seed, scoring=['r2', 'r2']))
print(scores)
