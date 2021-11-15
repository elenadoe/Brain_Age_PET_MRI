import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plots as plots
from sklearn.inspection import permutation_importance
from skrvm import RVR
from sklearn.SVM import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
#cd src
# %%
modality = "multimodal"
mode = "train"
database = "ADNI"
mri = pd.read_csv('../data/ADNI/test_train_PET_NP_amytau_olderthan65_42.csv')
pet = pd.read_csv('../data/ADNI/test_train_MRI_NP_amytau_olderthan65_42.csv')
mri_train = mri[mri['train'] == True]
pet_train = pet[pet['train'] == True]
mri_train = mri_train.reset_index()
pet_train = pet_train.reset_index()
print(mri_train.shape)
# check that all IDs are the same

# changed "Subject" to "name"
if pet_train['name'].equals(mri_train['name']):
    print("Subjects in two modalities match")

col = [x for x in mri_train.columns if ('_' in x)]
# age not Age
plt.hist(mri_train['age'], bins=30)
plt.title('Age distribution (30 bins)')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.show()

# %%
rand_seed = 42
num_bins = 5
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

df_train = pd.DataFrame(pet_train[col].values*mri_train[col].values,
                        columns=col, index=pet_train.index)
df_interact = pd.concat([mri_train.drop(col, axis=1), df_train], axis=1)

model_results = []
scores_results = []
res = {}

# %%
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

# TODO: df appears first here. The same df_train some line bellow!
mri_test = mri[mri['train'] == False]
pet_test = pet[pet['train'] == False]
mri_test = mri_test.reset_index()
pet_test = pet_test.reset_index()
print(mri_test.shape)


df_test = pd.DataFrame(pet_test[col].values*mri_test[col].values,
                       columns=col, index=pet_test.index)
df_interact_test = pd.concat([mri_test.drop(col, axis=1), df_test], axis=1)
# check that all IDs are the same

# changed "Subject" to "name"
if pet_test['name'].equals(mri_test['name']):
    print("Subjects in two modalities match")
mode = "test"

X_test = scaler.transform(df_interact_test[col].values)
y_test = df_interact_test['age'].values

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
# %%
# PERMUTATION IMP
rvr_feature_importance = permutation_importance(final_models['RVR'],
                                                X_test, y_test,
                                                scoring="r2", n_repeats=1000)
svr_feature_importance = permutation_importance(final_models['SVR'],
                                                X_test, y_test,
                                                scoring="r2", n_repeats=1000)

# %%
plots.permutation_imp(rvr_feature_importance, 'rvr', 'pet-mri')
plots.permutation_imp(svr_feature_importance, 'svr', 'pet-mri')
