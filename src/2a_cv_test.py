# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neuropsychology_correlations
import plots
import matplotlib
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import seaborn as sns
import pickle

# %%
# matplotlib config
cm = matplotlib.cm.get_cmap('PuOr')
cm_OASIS = cm(0.8)
cm_ADNI = cm(0.2)
cm_all = np.array([cm_OASIS, cm_ADNI])

# %%
# LOAD DATA
# load and inspect data, set modality
# TODO: stratify by age group (young old, middle old, oldest-old)
# modality = input("Which modality are you analyzing? ")
modality = 'PET'
database = "merged"
mode = "train"
df = pd.read_csv('../data/ADNI/test_train_' + modality + '_NP_' +
                 'withOASIS_olderthan65.csv',
                 sep=";")
df_train = df[df['train'] == True]
# select columns with '_' which are col's with features
col = df.columns[3:-22].tolist()

df_train = df_train.reset_index(drop=True)

# plot hist with Ages of train data
sns.displot(df_train, x='age', kde=True, color=cm_ADNI)
plt.title('Age distribution in train set')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.savefig('../results/{}/plots/{}_age_distribution'.format(database,
                                                             modality) +
            '_train.png', bbox_inches="tight")
plt.show()

# %%
# DATA INVESTIGATION
# how are brain signals distributed in OASIS and ADNI?
# plot global brain signal
df['global'] = np.mean(df[col], axis=1)
sns.displot(df, x='global', kind='kde', hue='Dataset', palette=cm_all)
plt.show()
# %%
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
degree = [2, 3]
cs = [0.001, 0.01, 0.1, 1, 10, 100, 500]
# hyperparameters gb
loss = ['friedman_mse', 'squared_error', 'absolute_error']
n_estimators = [10, 100, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
max_depth = [2, 3, 4, 5, 10, 15]

model_params = [{'rvr__C': cs,
                 'rvr__degree': degree,
                 'rvr__kernel': kernels},
                {'svm__C': cs,
                 'svm__degree': degree,
                 'svm__kernel': kernels},
                {'gradientboost__loss': loss,
                 # 'gradientboost__n_estimators': n_estimators,
                 'gradientboost__learning_rate': learning_rate,
                 'gradientboost__max_depth': max_depth}]

model_results = []
scores_results = []
# %%
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
                                               # confounds='PTGENDER',
                                               model_params=params,
                                               return_estimator='all',
                                               scoring=['r2',
                                               'neg_mean_absolute_error'])
    model_results.append(final_model.best_estimator_)
    scores_results.append(scores)
    print(model, scores['test_neg_mean_absolute_error'].mean())

# %%
y_true = df_train['age']
y_pred_rvr = model_results[0].predict(df_train[col])
y_pred_svr = model_results[1].predict(df_train[col])
slope_rvr, intercept_rvr, rvr_check = plots.check_bias(y_true,
                                                       y_pred_rvr,
                                                       'RVR',
                                                       modality,
                                                       database)
slope_svr, intercept_svr, svr_check = plots.check_bias(y_true,
                                                       y_pred_svr,
                                                       'SVR',
                                                       modality,
                                                       database)

print("Significant association between RVR-predicted age delta and CA:",
      rvr_check)
print("Significant association between SVR-predicted age delta and CA:",
      svr_check)
# %%
# BIAS CORRECTION
# Eliminate linear correlation of brain age delta and chronological age

# relevance Vectors Regression
y_true = df_train['age']
y_pred_rvr = model_results[0].predict(df_train[col])  # + ['PTGENDER']])

y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred_2(y_true, y_pred_rvr_bc, "rvr", modality,
                     mode, database)
plots.check_bias(y_true, y_pred_rvr_bc,
                 "RVR", modality, database,
                 corrected=True)

# SVM
y_pred_svr = model_results[1].predict(df_train[col])  # +['PTGENDER']])

y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred_2(y_true, y_pred_svr_bc, "svr", modality,
                     mode, database)
plots.check_bias(y_true, y_pred_svr_bc,
                 "SVR", modality, database,
                 corrected=True)
# %%
# SAVE MODELS
model_rvr = {'intercept': intercept_rvr,
             'slope': slope_rvr,
             'model': model_results[0]}
model_svr = {'intercept': intercept_svr,
             'slope': slope_svr,
             'model': model_results[1]}

pickle.dump(model_rvr, open("../results/model_rvr_" + modality +
                            ".p", "wb"))
pickle.dump(model_svr, open("../results/model_svr_" + modality +
                            ".p", "wb"))
# %%
# TESTING

# How well does the model perform on unseen data?
df_test = df[df['train'] == False]
df_test = df_test.reset_index(drop=True)
db_test = df_test['Dataset'].tolist()
mode = "test"

# investigate age distribution
# plot hist with Ages of train data
sns.displot(df_test, x='age', kde=True, hue='Dataset')
plt.title('Age distribution in test set')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.savefig('../results/{}/plots/{}_age_distribution'.format(database,
                                                             modality) +
            '_test.png',
            bbox_inches="tight")
plt.show()

X_test = df_test[col]  # +['PTGENDER']]
y_true = df_test['age'].values

# plot rvr predictions against GT in test set
y_pred_rvr = model_results[0].predict(X_test)
y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred_2(y_true, y_pred_rvr_bc, "rvr", modality,
                     mode, database, db_test)

# plot svr predictions against GT in test set
y_pred_svr = model_results[1].predict(X_test)
y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred_2(y_true, y_pred_svr_bc, "svr", modality,
                     mode, database, db_test)

# %%
plt.scatter(df_test['HIP.lh'][df_test['Dataset'] == 'OASIS'],
            df_test['age'][df_test['Dataset'] == 'OASIS'],
            label="OASIS")
plt.scatter(df_test['HIP.lh'][df_test['Dataset'] == 'ADNI'],
            df_test['age'][df_test['Dataset'] == 'ADNI'],
            color='orange', label="ADNI")
plt.xlabel('SUVR in left hippocampus')
plt.ylabel('Chronological age')
plt.legend()
plt.show()

plt.scatter(df_test['X17Networks_LH_VisCent_ExStr_1'][df_test['Dataset'] == 'OASIS'],
            df_test['age'][df_test['Dataset'] == 'OASIS'])
plt.scatter(df_test['X17Networks_LH_VisCent_ExStr_1'][df_test['Dataset'] == 'ADNI'],
            df_test['age'][df_test['Dataset'] == 'ADNI'],
            color='orange')
plt.show()
# %%
# PERMUTATION IMPORTANCE
pi = permutation_importance(model_results[0],
                            X_test, y_true,
                            n_repeats=1000)

# %%
plots.permutation_imp(pi, 'rvr', modality, database)
# %%
# SAVE RESULTS
# Create table of (corrected) predicted and chronological age in this modality
# rvr had better performance in both MAE and R2 --> take rvr as final model
y_diff = y_pred_svr_bc - y_true
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

# %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df_test.columns[-18:].values
neuropsychology_correlations.neuropsych_correlation(y_true, y_pred_rvr_bc,
                                                    "BPA",
                                                    npt,
                                                    df_test,
                                                    modality,
                                                    database)
# Difference between PA-CA+ and PA-CA-
neuropsychology_correlations.plot_bpad_diff(y_true, y_pred_rvr_bc,
                                            npt,
                                            df_test,
                                            modality,
                                            database)
