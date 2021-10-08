# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuropsychology_correlations
import plots
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression


# %%
# LOAD DATA
# load and inspect data, set modality
# TODO: read in bootstrapping samples @antogeo
# modality = input("Which modality are you analyzing? ")
modality = 'PET'
mode = "train"
df = pd.read_csv('../data/ADNI/test_train_' + modality + '_NP_amytau_olderthan65_42.csv')
df_train = df[df['train'] == True]
# select columns with '_' which are col's with features
col = [x for x in df_train.columns if ('_' in x)]
df_train = df_train.reset_index(drop=True)

# plot hist with Ages of train data
plt.hist(df_train['age'], bins=30)
#%%
# PREPARATION
rand_seed = 42
num_bins = 5

# define models and model names (some are already included in julearn)
rvr = RVR()

# models to test & names
models = [rvr, 'svm', 'gradientboost']
model_names = ['rvr', 'svm', 'gradientboost']
splits = 5

model_results = []
scores_results = []
res = {}
res['model'] = []
res['iter'] = []
res['pred'] = []
res['ind'] = []

#%%
# TRAINING
# train models using 5-fold cross-validation
# TODO: read in bootstrapping samples @antogeo
for i, model in enumerate(models):
    # split data using age-bins instead of real age
    cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                df_train['Agebins'])
    cv = list(cv)
    # run julearn function
    scores, final_model = run_cross_validation(X=col, y='age',
                                               preprocess_X='scaler_robust',
                                               problem_type='regression',
                                               data=df_train,
                                               model=model, cv=cv,
                                               return_estimator='all',
                                               seed=rand_seed,
                                               scoring=['r2',
                                                    'neg_mean_absolute_error'])
    model_results.append(final_model)
    scores_results.append(scores)
    # iterate over julearn results to and save results of each iteration
    for iter in range(splits):
        pred = scores.estimator[iter].predict(df_train.iloc[cv[iter][1]][col])
        res['pred'].append(pred)
        res['iter'].append(iter)
        res['model'].append(str(model))
        res['ind'].append(cv[iter][1])

df_res = pd.DataFrame(res)
age_pred = {}
age_pred['subj'] = []
age_pred['pred'] = []
age_pred['real'] = []
age_pred['model'] = []

for i, fold in enumerate(df_res['ind']):
    for ind, sample in enumerate(fold):
        age_pred['real'].append(df_train.iloc[sample]['age'])
        age_pred['pred'].append(df_res['pred'].iloc[i][ind])
        age_pred['subj'].append(df_train.iloc[sample]['name'])
        age_pred['model'].append(df_res.iloc[i]['model'])

df_ages = pd.DataFrame(age_pred)

# assess scaling parameters for test data
scaler = RobustScaler().fit(df_train[col])
# %%
# BIAS CORRECTION
# Eliminate linear correlation of brain age difference and chronological age

# relevance Vectors Regression
y_true = df_ages[df_ages['model'] == 'RVR()']['real']
y_pred = df_ages[df_ages['model'] == 'RVR()']['pred']

# fit a linear model for bias correction for rvr
lm_rvr = LinearRegression()
lm_rvr.fit(np.array(y_pred).reshape(-1, 1), np.array(y_true).reshape(-1, 1))
slope_rvr = lm_rvr.coef_[0][0]
intercept_rvr = lm_rvr.intercept_[0]
y_pred_bc = (y_pred - intercept_rvr)/slope_rvr

# plot real_vs_pred
plots.real_vs_pred(y_true, y_pred_bc, "rvr", mode, modality)

# SVM
y_true_svm = df_ages[df_ages['model'] == 'svm']['real']
y_pred_svm = df_ages[df_ages['model'] == 'svm']['pred']

# fit a linear model for bias correction for svm
lm_svr = LinearRegression()
lm_svr.fit(np.array(y_pred_svm).reshape(-1, 1), np.array(y_true_svm).reshape(-1, 1))
slope_svr = lm_svr.coef_[0][0]
intercept_svr = lm_svr.intercept_[0]
y_pred_svm_bc = (y_pred_svm - intercept_svr)/slope_svr

# plot real_vs_pred
plots.real_vs_pred(y_true_svm, y_pred_svm_bc, "svr", mode, modality)

# Gradient Boost
y_true = df_ages[df_ages['model'] == 'gradientboost']['real']
y_pred = df_ages[df_ages['model'] == 'gradientboost']['pred']


# fit a linear model for bias correction for gaussian
lm_gradboost = LinearRegression()
lm_gradboost.fit(np.array(y_pred).reshape(-1, 1), np.array(y_true).reshape(-1, 1))
slope_gradboost = lm_gradboost.coef_[0][0]
intercept_gradboost = lm_gradboost.intercept_[0]
y_pred_bc = (y_pred - intercept_gradboost)/slope_gradboost

# plot real_vs_pred
plots.real_vs_pred(y_true, y_pred_bc, "gradboost", mode, modality)

# %%
# TESTING
# How well does the model perform on unseen data?
df_test = df[df['train'] == False]
# apply scale parameters from training data
#df_test_scaled = scaler.transform(df_test[col])
mode = "test"

X_test = df_test[col].values
y_true = df_test['age'].values

# plot rvr predictions against GT in test set
y_pred_rvr = model_results[0]['rvr'].predict(X_test)
y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred(y_true, y_pred_rvr_bc, "rvr", mode, modality)

# plot svr predictions against GT in test set
y_pred_svr = model_results[1]['svm'].predict(X_test)
y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred(y_true, y_pred_svr_bc, "svr", mode, modality)


# plot gradboost predictions against GT in test set
y_pred_gradb = model_results[3]['gradientboost'].predict(X_test)
y_pred_gradb_bc = (y_pred_gradb - intercept_gradboost)/slope_gradboost

plots.real_vs_pred(y_true, y_pred_gradb_bc, "gradboost", mode, modality)
#%%
# PERMUTATION IMPORTANCE
pi = permutation_importance(model_results[0]['rvr'],
                            X_test, y_true,
                            n_repeats = 1000)

#%%
plots.permutation_imp(pi, 'rvr', modality)
# %%
# SAVE RESULTS
# Create table of (corrected) predicted and chronological age in this modality
# rvr had better performance in both MAE and R2 --> take rvr as final model
y_diff = y_pred_svr_bc - y_true
df_test = df_test.reset_index(drop=True)
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_true, columns=["age"]),
                      pd.DataFrame(y_pred_svr, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_svr_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/pred_age_{}_svr.csv'.format(modality))

y_diff = y_pred_rvr_bc - y_true
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_true, columns=["age"]),
                      pd.DataFrame(y_pred_rvr, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_rvr_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/pred_age_{}_rvr.csv'.format(modality))

y_diff = y_pred_gradb_bc - y_true
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_true, columns=["age"]),
                      pd.DataFrame(y_pred_gradb, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_gradb_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/pred_age_{}_gradb.csv'.format(modality))

 # %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df_test.columns[-12:].values
neuropsychology_correlations.neuropsych_correlation(y_true, y_pred_rvr_bc, "BPA",
                                                    npt, df_test, modality)
# Correlation with Neuropsychology - brain age difference ( BA- CA)
y_diff = y_pred_rvr_bc - y_true
neuropsychology_correlations.neuropsych_correlation(y_true, y_diff, "BPAD",
                                                    npt, df_test, modality)
