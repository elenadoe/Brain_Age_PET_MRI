import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.plots as src.plots
from sklearn.inspection import permutation_importance
from skrvm import RVR
from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# %%
modality = "multimodal"
mode = "train"
mri = pd.read_csv('data/ADNI/test_train_PET_NP_amytau_olderthan65_42.csv')
pet = pd.read_csv('data/ADNI/test_train_MRI_NP_amytau_olderthan65_42.csv')
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

mri_scaler = StandardScaler()
pet_scaler = StandardScaler()
#  according to the columns (pet_train or mri_train):
mri_train_data = mri_scaler.fit_transform(mri_train[col])
# KeyError: "['ATHA_rh', 'AMY_lh', 'PUT_lh', 'PTHA_rh'] not in index"
pet_train_data = pet_scaler.fit_transform(pet_train[col])
# KeyError: "['aTHA_rh', 'pTHA_rh', 'AMY_h', 'PUT_h'] not in index"
interact_train = mri_train_data * pet_train_data
interact_train = pd.DataFrame(interact_train, columns=col)
print(interact_train.shape)

interact_train['name'] = mri_train['name']
interact_train['Agebins'] = mri_train['Agebins']
interact_train['age'] = mri_train['age']
print(interact_train.shape)
# %%
for i, model in enumerate(models):
    cv = StratifiedKFold(n_splits=splits).split(interact_train[col],
                                                interact_train['Agebins'])
    cv = list(cv)
    scores, final_model = run_cross_validation(X=col, y='Age',
                                               problem_type='regression',
                                               data=interact_train,
                                               model=model, cv=cv,
                                               return_estimator='all',
                                               seed=rand_seed,
                                               scoring=['r2',
                                                    'neg_mean_absolute_error'])
    model_results.append(final_model)
    scores_results.append(scores)
    for iter in range(splits):
        pred = scores.estimator[iter].predict(
                        interact_train.iloc[cv[iter][1]][col])
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
        age_pred['real'].append(interact_train.iloc[sample]['Age'])
        age_pred['pred'].append(df_res['pred'].iloc[i][ind])
        age_pred['subj'].append(interact_train.iloc[sample]['Subject'])
        age_pred['model'].append(df_res.iloc[i]['model'])

df_ages = pd.DataFrame(age_pred)
# %%
y_true = df_ages[df_ages['model'] == 'svm']['real']
y_pred = df_ages[df_ages['model'] == 'svm']['pred']

# fit a linear model for bias correction
# TODO: bias correction without chronological age
lm_rvr = LinearRegression()
lm_rvr.fit(np.array(y_pred).reshape(-1, 1), np.array(y_true).reshape(-1, 1))
slope_rvr = lm_rvr.coef_
intercept_rvr = lm_rvr.intercept_
y_pred_bc = (y_pred - intercept_rvr[0])/slope_rvr[0][0]

# plot real_vs_pred
plots.real_vs_pred(y_true, y_pred_bc, "rvr", mode, modality)

y_true = df_ages[df_ages['model'] == 'RVR()']['real']
y_pred = df_ages[df_ages['model'] == 'RVR()']['pred']

# fit a linear model for bias correction
lm_svr = LinearRegression()
lm_svr.fit(np.array(y_pred).reshape(-1, 1), np.array(y_true).reshape(-1, 1))
slope_svr = lm_svr.coef_
intercept_svr = lm_svr.intercept_
y_pred_bc = (y_pred - intercept_svr[0])/slope_svr[0][0]


# plot real_vs_pred
plots.real_vs_pred(y_true, y_pred_bc, "svr", mode, modality)

# %%
# TESTING

# TODO: df appears first here. The same df_train some line bellow!
df_test = df[df['train'] == "F"]
mode = "test"
col = [x for x in df_train.columns if '_' in x]

X_test = df_test[col].values
y_true = df_test['Age'].values

y_pred = model_results[0]['rvr'].predict(X_test)

y_pred_bc = y_pred - (y_true*slope_rvr[0]+intercept_rvr[0])

plots.real_vs_pred(y_true, y_pred_bc, mode, "rvr", modality)

y_pred = model_results[1]['svm'].predict(X_test)
y_pred_bc = y_pred - (y_true*slope_svr[0]+intercept_svr[0])

plots.real_vs_pred(y_true, y_pred_bc, mode, "svr", modality)
# %%
# PERMUTATION IMP
rvr_feature_importance = permutation_importance(model_results[0]['rvr'],
                                                X_test, y_true,
                                                scoring="r2", n_repeats=1000)
svr_feature_importance = permutation_importance(model_results[1]['svm'],
                                                X_test, y_true,
                                                scoring="r2", n_repeats=1000)

# %%
