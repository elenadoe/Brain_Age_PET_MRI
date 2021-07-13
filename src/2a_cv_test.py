#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import neuropsychology_correlations
import plots
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

# %%
modality = "MRI"
mode = "train"
df = pd.read_csv('../data/ADNI/test_train_MRI_ADNI_NP.csv', sep = ";")
df_train = df[df['train'] == True]
col = [x for x in df_train.columns if ('_' in x)]

# round to no decimal place
df_train = df_train.reset_index(drop=True)

plt.hist(df_train['Age'], bins=30)

rand_seed = 42
num_bins = 5
rvr = RVR()
models = [rvr, 'svm', 'gauss']
model_names = ['rvr', 'svm', 'gauss']
splits = 5

model_results = []
scores_results = []
res = {}
res['model'] = []
res['iter'] = []
res['pred'] = []
res['ind'] = []

for i, model in enumerate(models):
    cv = StratifiedKFold(n_splits=splits).split(df_train[col],
                                                df_train['Agebins'])
    cv = list(cv)
    scores, final_model = run_cross_validation(X=col, y='Age',
                                         #preprocess_X='scaler_robust',
                                         problem_type='regression',
                                         data=df_train,
                                         model=model, cv=cv,
                                         return_estimator='all',
                                         seed=rand_seed,
                                         scoring=[
                                            'r2', 'neg_mean_absolute_error'])
    model_results.append(final_model)
    scores_results.append(scores)
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
        age_pred['real'].append(df_train.iloc[sample]['Age'])
        age_pred['pred'].append(df_res['pred'].iloc[i][ind])
        age_pred['subj'].append(df_train.iloc[sample]['Subject'])
        age_pred['model'].append(df_res.iloc[i]['model'])

df_ages = pd.DataFrame(age_pred)
# %%
y_true = df_ages[df_ages['model'] == 'svm']['real']
y_pred = df_ages[df_ages['model'] == 'svm']['pred']

# fit a linear model for bias correction
lm_rvr = LinearRegression()
lm_rvr.fit(np.array(y_pred).reshape(-1,1), np.array(y_true).reshape(-1,1))
slope_rvr = lm_rvr.coef_[0][0]
intercept_rvr = lm_rvr.intercept_[0]
y_pred_bc = (y_pred - intercept_rvr)/slope_rvr

# plot real_vs_pred
plots.real_vs_pred(y_true,y_pred_bc, "rvr", mode, modality)

y_true = df_ages[df_ages['model'] == 'RVR()']['real']
y_pred = df_ages[df_ages['model'] == 'RVR()']['pred']

# fit a linear model for bias correction
lm_svr = LinearRegression()
lm_svr.fit(np.array(y_pred).reshape(-1,1), np.array(y_true).reshape(-1,1))
slope_svr = lm_svr.coef_[0][0]
intercept_svr = lm_svr.intercept_[0]
y_pred_bc = (y_pred - intercept_svr)/slope_svr

# plot real_vs_pred
plots.real_vs_pred(y_true,y_pred_bc, "svr", mode, modality)

# %%
# TESTING
df_test = df[df['train'] == False]
mode = "test"
col = [x for x in df_train.columns if '_' in x]

X_test = df_test[col].values
y_true = df_test['Age'].values

# plot rvr
y_pred_rvr = model_results[0]['rvr'].predict(X_test)
y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred(y_true,y_pred_rvr_bc, "rvr", mode, modality)

# plot svr
y_pred_svr = model_results[1]['svm'].predict(X_test)
y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred(y_true,y_pred_svr_bc, "svr", mode, modality)

# %%
# Create table of (corrected) predicted and chronological age in this modality
# svr had better performance in both MAE and R2 --> take svr as final model
y_diff = y_true - y_pred_svr_bc
df_test = df_test.reset_index(drop=True)
pred_csv = pd.concat((df_test["Subject"],
                      pd.DataFrame(y_true, columns = ["Age"]),
                      pd.DataFrame(y_pred_svr, columns = ["RawPredAge"]),
                      pd.DataFrame(y_pred_svr_bc, columns = ["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns = ["BPAD"])), axis = 1)

pred_csv.to_csv('../results/pred_age_{}.csv'.format(modality))

# %%
# Correlation with Neuropsychology - brain age
npt = df.columns[-14:].values
neuropsychology_correlations.neuropsych_correlation(y_true, y_pred_svr_bc,
                                                    npt, df_test)
# Correlation with Neuropsychology - brain age difference (CA - BA)
neuropsychology_correlations.neuropsych_correlation(y_true, y_diff,
                                                    npt, df_test)

# %%
# PERMUTATION IMP
rvr_feature_importance = permutation_importance(model_results[0]['rvr'], X_test, y_true,
                                                scoring="r2", n_repeats = 1000)
svr_feature_importance = permutation_importance(model_results[1]['svm'], X_test, y_true, scoring="r2", n_repeats = 1000)



# %%
