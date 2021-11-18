# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuropsychology_correlations
import plots
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pickle

# %%
# LOAD DATA
# load and inspect data, set modality
# TODO: read in bootstrapping samples @antogeo
# modality = input("Which modality are you analyzing? ")
modality = 'MRI'
database = "ADNI"
mode = "train"
df = pd.read_csv('../data/ADNI/test_train_' + modality + '_NP.csv', sep = ";")
df_train = df[df['train'] == True]
# select columns with '_' which are col's with features
col = [x for x in df_train.columns if ('_' in x)]
# exclude RAVLT memory scores
col = col[:-3]

df_train = df_train.reset_index(drop=True)

# plot hist with Ages of train data
sns.displot(df_train, x = 'age', kde = True)
plt.title('Age distribution in train set')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.savefig('../results/{}/plots/{}_age_distribution.png'.format(database,
                                                                 modality),
            bbox_inches = "tight")
plt.show()
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

# hyperparameters svr & rvr
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
degree = [2,3]
cs = [0.001, 0.01, 0.1, 1, 10, 100]
# hyperparameters gb
loss = ['friedman_mse', 'squared_error', 'absolute_error']
n_estimators = [10, 100, 200]
learning_rate = [0.0001, 0.001, 0.01]
max_depth = [4, 5, 6,7,8,9,10]

model_params = [{'rvr__C': cs, 
                 'rvr__degree': degree,
                 'rvr__kernel': kernels},
                {'svm__C': cs, 
                 'svm__degree': degree,
                 'svm__kernel': kernels},
                {'gradientboost__n_estimators': n_estimators,
                  'gradientboost__learning_rate': learning_rate,
                  'gradientboost__max_depth': max_depth,
                  'gradientboost__random_state': [rand_seed]}]

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

for i, (model, params) in enumerate(zip(models, model_params)):
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
                                               seed=rand_seed,
                                               model_params=params,
                                               return_estimator='all',
                                               scoring=['r2',
                                                'neg_mean_absolute_error'])
    model_results.append(final_model.best_estimator_)
    scores_results.append(scores)
    print(model,scores['test_neg_mean_absolute_error'].mean())

    # iterate over julearn results to and save results of each iteration
    for iter in range(splits):
        pred = final_model.best_estimator_.predict(df_train.iloc[cv[iter][1]][col])
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

# %%
# BIAS CORRECTION
# Eliminate linear correlation of brain age difference and chronological age
def bias_correction(y_pred, y_true):
    lm = LinearRegression()
    lm.fit(np.array(y_pred).reshape(-1,1),
           np.array(y_true).reshape(-1,1))
    slope = lm.coef_[0][0]
    intercept = lm.intercept_[0]
    y_pred_bc = (y_pred - intercept)/slope
    
    return intercept, slope, y_pred_bc

# relevance Vectors Regression
y_true = df_ages[df_ages['model'] == 'RVR()']['real']
y_pred_rvr = df_ages[df_ages['model'] == 'RVR()']['pred']

intercept_rvr, slope_rvr, y_pred_rvr_bc = bias_correction(y_pred_rvr,
                                                          y_true)
plots.real_vs_pred(y_true, y_pred_rvr, "rvr", mode, 
                   modality, database)

# SVM
y_pred_svr = df_ages[df_ages['model'] == 'svm']['pred']

intercept_svr, slope_svr, y_pred_svr_bc = bias_correction(y_pred_svr,
                                                          y_true)
plots.real_vs_pred(y_true, y_pred_svr_bc, "svr", mode, 
                   modality, database)

# Gradient Boost
y_pred_gb = df_ages[df_ages['model'] == 'gradientboost']['pred']


# fit a linear model for bias correction for gaussian
intercept_gb, slope_gb, y_pred_gb_bc = bias_correction(y_pred_gb,
                                                       y_true)
plots.real_vs_pred(y_true, y_pred_gb_bc, "gradboost", mode, 
                   modality, database)
model_rvr = {'intercept':intercept_rvr,
             'slope':slope_rvr,
             'model':model_results[0]}
model_svr = {'intercept':intercept_svr,
             'slope':slope_svr,
             'model':model_results[1]}
model_gb = {'intercept':intercept_gb,
             'slope':slope_gb,
             'model':model_results[2]}
pickle.dump(model_rvr,open("../results/model_rvr.p","wb"))
pickle.dump(model_svr,open("../results/model_svr.p","wb"))
pickle.dump(model_gb,open("../results/model_gb.p","wb"))
            
# %%
# TESTING
# How well does the model perform on unseen data?
df_test = df[df['train'] == False]
# apply scale parameters from training data
#df_test_scaled = scaler.transform(df_test[col])
mode = "test"

X_test = df_test[col]
y_true = df_test['age'].values

# plot rvr predictions against GT in test set
y_pred_rvr = model_results[0].predict(X_test)
y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred(y_true, y_pred_rvr_bc, "rvr", mode, 
                   modality, database)

# plot svr predictions against GT in test set
y_pred_svr = model_results[1].predict(X_test)
y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred(y_true, y_pred_svr_bc, "svr", mode, 
                   modality, database)

# plot gradboost predictions against GT in test set
y_pred_gb = model_results[2].predict(X_test)
y_pred_gradb_bc = (y_pred_gb - intercept_gb)/slope_gb

plots.real_vs_pred(y_true, y_pred_gradb_bc, "gradboost", mode, 
                   modality, database)
#%%
# PERMUTATION IMPORTANCE
pi = permutation_importance(model_results[0],
                            X_test, y_true,
                            n_repeats = 1000)

#%%
plots.permutation_imp(pi, 'rvr', modality, database)
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

"""y_diff = y_pred_gradb_bc - y_true
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_true, columns=["age"]),
                      pd.DataFrame(y_pred_gb, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_gb_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/pred_age_{}_gradb.csv'.format(modality))"""

# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df_test.columns[-15:].values
neuropsychology_correlations.neuropsych_correlation(y_true, y_pred_svr_bc, "BPA",
                                                    npt, 
                                                    df_test, 
                                                    modality,
                                                    database)
# Difference between PA-CA+ and PA-CA-
y_diff = (y_pred_svr_bc - y_true)
neuropsychology_correlations.neuropsych_correlation(y_true, y_diff, "BPAD",
                                                    npt, 
                                                    df_test, 
                                                    modality,
                                                    database)


#%% BINARY --> exclude NA
import seaborn as sns
# no CSF amyloid
#df_test['ABETA_bin'] = [0 if x >= 192 else 1  if x < 192 else -1 for x in df_test['ABETA'].values]
#sns.barplot(x=df_test['ABETA_bin'],y=y_diff)