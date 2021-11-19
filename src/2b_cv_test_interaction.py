import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plots as plots
import seaborn as sns
from sklearn.inspection import permutation_importance
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression

# %%
modality = "multimodal_div"
mode = "train"
database = "ADNI"
mri = pd.read_csv('../data/ADNI/test_train_PET_NP.csv', sep = ";")
pet = pd.read_csv('../data/ADNI/test_train_MRI_NP.csv', sep = ";")

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
# exclude RAVLT memory scores
col = col[:-3]

age_df = pd.DataFrame()
age_df['age'] = (mri_train['age'].tolist() + pet_train['age'].tolist())
age_df['modality'] = (['mri']*len(mri_train) + ['pet']*len(pet_train))
sns.displot(age_df, x = 'age', hue='modality', 
            fill = True, kde = True, alpha = 0.5)
plt.title('Age distribution in Training Data')
plt.xlabel('Age [years]')
plt.ylabel('n Participants')
plt.legend()
plt.show()

# %%
rand_seed = 42
num_bins = 5
rvr = RVR()

models = [rvr, 'svm']
model_names = ['rvr', 'svm']
splits = 5

# hyperparameters svr & rvr
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
degree = [2,3]
cs = [0.01, 0.1, 1, 10, 100]
# hyperparameters gb
loss = ['squared_error', 'absolute_error']
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]

model_params = [{'rvr__C': cs, 
                 'rvr__kernel': kernels,
                 'rvr__degree': degree},
                {'svm__C': cs, 
                 'svm__kernel': kernels,
                 'svm__degree': degree}]
#                {'gradientboost__n_estimators': n_estimators,
#                 'gradientboost__learning_rate': learning_rate}]


df_train = pd.DataFrame(pet_train[col].values/mri_train[col].values,
                        columns=col, index=pet_train.index)
df_interact = pd.concat([mri_train.drop(col, axis=1), df_train], axis=1)

model_results = []
scores_results = []
res = {}
res['model'] = []
res['iter'] = []
res['pred'] = []
res['ind'] = []

# %%
for i, (model, params) in enumerate(zip(models, model_params)):
    cv = StratifiedKFold(n_splits=splits).split(df_interact[col],
                                                df_interact['Agebins'])
    cv = list(cv)
    scores, final_model = run_cross_validation(X=col, y='age',
                                               preprocess_X='scaler_robust',
                                               problem_type='regression',
                                               data=df_interact,
                                               model=model, cv=cv,
                                               return_estimator='all',
                                               model_params=params,
                                               seed=rand_seed,
                                               scoring=['r2',
                                                    'neg_mean_absolute_error'])
    model_results.append(final_model.best_estimator_)
    scores_results.append(scores)
    print(model,scores['test_neg_mean_absolute_error'].mean())

    for iter in range(splits):
        pred = final_model.best_estimator_.predict(
                        df_interact.iloc[cv[iter][1]][col])
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
        age_pred['real'].append(df_interact.iloc[sample]['age'])
        age_pred['pred'].append(df_res['pred'].iloc[i][ind])
        age_pred['subj'].append(df_interact.iloc[sample]['name'])
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

X_test = df_interact_test[col].values
y_true = df_interact_test['age'].values

y_pred = model_results[0]['rvr'].predict(X_test)

y_pred_rvr_bc = (y_pred - intercept_rvr)/slope_rvr

plots.real_vs_pred(y_true, y_pred_rvr_bc, mode, "rvr", modality,
                   database)

y_pred = model_results[1]['svm'].predict(X_test)
y_pred_svr_bc = (y_pred - intercept_svr)/slope_svr

plots.real_vs_pred(y_true, y_pred_svr_bc, mode, "svr", modality,
                   database)
# %%
# PERMUTATION IMP
rvr_feature_importance = permutation_importance(model_results[0]['rvr'],
                                                X_test, y_true,
                                                scoring="r2", n_repeats=1000)
svr_feature_importance = permutation_importance(model_results[1]['svm'],
                                                X_test, y_true,
                                                scoring="r2", n_repeats=1000)

# %%
plots.permutation_imp(rvr_feature_importance, 'rvr', 'pet-mri')
plots.permutation_imp(svr_feature_importance, 'svr', 'pet-mri')
