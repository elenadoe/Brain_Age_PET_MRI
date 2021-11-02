# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuropsychology_correlations
import plots
from skrvm import RVR
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression


# %%
# LOAD DATA
# load and inspect data, set modality
# modality = input("Which modality are you analyzing? ")
modality = 'MRI'
mode = "train"
database = "ADNI"
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
svr = SVR()
gradientboost = GradientBoostingRegressor()

# models to test & names
models = [rvr, svr, gradientboost]
model_names = ['rvr', 'svr', 'gradientboost']
splits = 5

# hyperparameters svr & rvr
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
cs = [0.001, 0.01, 0.1, 1, 10, 100]
# hyperparameters gb
loss = ['friedman_mse', 'squared_error', 'absolute_error']
n_estimators = [10, 100, 1000]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
max_depth = [2, 3, 4, 5, 6]

model_params = [[{'C': cs, 'kernel': kernels}],
                [{'C': cs, 'kernel': kernels}],
                [{'n_estimators': n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'random_state': [rand_seed]}]]

#%%
# TRAINING
# train models using 5-fold cross-validation
scaler = RobustScaler()
X_train = scaler.fit_transform(df_train[col])
cv = StratifiedKFold(n_splits=splits)

final_models = {}
final_scores = {}
final_predictions = {}
for model in models:
    grid_cv = GridSearchCV(model, model_params[models.index(model)],
                           scoring = ['r2','neg_mean_absolute_error'],
                           refit = 'r2',
                           cv = cv)
    final_model = grid_cv.fit(X_train, df_train['age'])
    final_models[str(model)[:-2]] = final_model.best_estimator_
    final_scores[str(model)[:-2]] = final_model.best_score_
    final_predictions[str(model)[:-2]] = final_model.predict(X_train)

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
y_pred_rvr = final_predictions['RVR']
y_true = df_train['age']

intercept_rvr, slope_rvr, y_pred_rvr_bc = bias_correction(y_pred_rvr,
                                                          y_true)
plots.real_vs_pred(y_true, y_pred_rvr, "rvr", mode, 
                   modality, database)

# SVM
y_pred_svr = final_predictions['SVR']

intercept_svr, slope_svr, y_pred_svr_bc = bias_correction(y_pred_svr,
                                                          y_true)
plots.real_vs_pred(y_true, y_pred_svr_bc, "svr", mode, 
                   modality, database)

# Gradient Boost
y_pred_gb = final_predictions['GradientBoostingRegressor']

intercept_gb, slope_gb, y_pred_gb_bc = bias_correction(y_pred_gb,
                                                       y_true)
plots.real_vs_pred(y_true, y_pred_gb_bc, "gradboost", mode, 
                   modality, database)

# %%
# TESTING
# How well does the model perform on unseen data?
df_test = df[df['train'] == False]
# apply scale parameters from training data
X_test = scaler.transform(df_test[col].values)
y_test = df_test['age'].values
mode = "test"

# plot rvr predictions against GT in test set
y_pred_rvr = final_models['RVR'].predict(X_test)
y_pred_rvr_bc = (y_pred_rvr - intercept_rvr)/slope_rvr

plots.real_vs_pred(y_test, y_pred_rvr_bc, "rvr", mode, 
                   modality, database)

# plot svr predictions against GT in test set
y_pred_svr = final_models['SVR'].predict(X_test)
y_pred_svr_bc = (y_pred_svr - intercept_svr)/slope_svr

plots.real_vs_pred(y_test, y_pred_svr_bc, "svr", mode, 
                   modality, database)


# plot gradboost predictions against GT in test set
y_pred_gb = final_models['GradientBoostingRegressor'].predict(X_test)
y_pred_gb_bc = (y_pred_gb - intercept_gb)/slope_gb

plots.real_vs_pred(y_test, y_pred_gb_bc, "gradboost", mode, 
                   modality, database)
#%%
# PERMUTATION IMPORTANCE
pi = permutation_importance(final_models['RVR'],
                            X_test, y_test,
                            n_repeats = 1000)

#%%
plots.permutation_imp(pi, 'rvr', modality, database)
# %%
# SAVE RESULTS
# Create table of (corrected) predicted and chronological age in this modality
# rvr had better performance in both MAE and R2 --> take rvr as final model
y_diff = y_pred_svr_bc - y_test
df_test = df_test.reset_index(drop=True)
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_test, columns=["age"]),
                      pd.DataFrame(y_pred_svr, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_svr_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/'+database+'/pred_age_{}_svr.csv'.format(modality))

y_diff = y_pred_rvr_bc - y_true
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_test, columns=["age"]),
                      pd.DataFrame(y_pred_rvr, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_rvr_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/'+database+'/pred_age_{}_rvr.csv'.format(modality))

y_diff = y_pred_gb_bc - y_test
pred_csv = pd.concat((df_test["name"],
                      pd.DataFrame(y_test, columns=["age"]),
                      pd.DataFrame(y_pred_gb, columns=["RawPredAge"]),
                      pd.DataFrame(y_pred_gb_bc, columns=["CorrPredAge"]),
                      pd.DataFrame(y_diff, columns=["BPAD"])), axis=1)

pred_csv.to_csv('../results/'+database+'/pred_age_{}_gradb.csv'.format(modality))

 # %%
# CORRELATION NEUROPSYCHOLOGY - BRAIN AGE
# Inspect correlation of neuropsychological scores and predicted/corrected
# brain age
npt = df_test.columns[-12:].values
neuropsychology_correlations.neuropsych_correlation(y_test, 
                                                    y_pred_rvr_bc, 
                                                    "BPA",
                                                    npt, df_test, 
                                                    modality,
                                                    database)
# Correlation with Neuropsychology - brain age difference ( BA- CA)
y_diff = y_pred_rvr_bc - y_test
neuropsychology_correlations.neuropsych_correlation(y_test, 
                                                    y_diff, 
                                                    "BPAD",
                                                    npt, df_test, 
                                                    modality,
                                                    database)
