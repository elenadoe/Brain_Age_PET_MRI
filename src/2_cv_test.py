import pandas as pd
import numpy as np
import sys
# pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR
from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.model_selection import train_test_split, StratifiedKFold
sys.path.append("../lib")
from create_splits import stratified_splits

#configure_logging(level='INFO')
# in case seaborn conflicts with autocomplete
df = pd.read_csv('../data/test_train_FDG.csv', index_col=0)

df_train = df[df['train'] == "T"]
col = [x for x in df_train.columns if 'H_' in x]

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
                                  seed=rand_seed, scoring=['r2', 'neg_mean_absolute_error']))

print("RESULTS")
print("\nRIDGE VECTOR REGRESSION:\nR2: ", np.round(np.mean(scores['rvr'][0][0]['test_r2']),2),
      "\nMAE: ", np.abs(np.round(np.mean(scores['rvr'][0][0]['test_neg_mean_absolute_error']),2)))
print("\nSUPPORT VECTOR MACHINE:\nR2: ", np.round(np.mean(scores['svm'][0][0]['test_r2']),2),
      "\nMAE: ", np.abs(np.round(np.mean(scores['svm'][0][0]['test_neg_mean_absolute_error']),2)))
