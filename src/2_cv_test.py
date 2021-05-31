# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
#pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from julearn import run_cross_validation
from julearn.utils import configure_logging
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting
from nilearn import image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression


# configure_logging(level='INFO')
# TODO: do with FDG template
# in case seaborn conflicts with autocomplete

df_train = df[df['train'] == "T"]
col = [x for x in df_train.columns if ('_' in x)]

# round to no decimal place
df_train = df_train.reset_index(drop=True)

#col = [x for x in df_train.columns if 'H_' in x]
plt.hist(y, bins=30)

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
# res = pd.Series(index=df_train.index)
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

y_true = df_ages[df_ages['model'] == 'svm']['real']
y_pred = df_ages[df_ages['model'] == 'svm']['pred']

# fit a linear model for bias correction
lm_rvr = LinearRegression()
lm_rvr.fit(np.array(y_true).reshape(-1,1), np.array(y_pred-y_true).reshape(-1,1))
slope_rvr = lm_rvr.coef_
intercept_rvr = lm_rvr.intercept_
y_pred_bc = y_pred - (y_true*slope_rvr[0][0]+intercept_rvr[0])
#y_pred_bc = y_pred

mae = format(mean_absolute_error(y_true, y_pred_bc), '.2f')
corr = format(np.corrcoef(y_pred_bc, y_true)[1, 0], '.2f')

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred_bc)
m, b = np.polyfit(y_true, y_pred_bc, 1)
plt.plot(y_true, m*y_true + b)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
ax.set(xlabel='True values', ylabel='Predicted values')
plt.title('Actual vs Predicted - SVR')
plt.text(xmin + 10, ymax - 0.01 * ymax, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
#plt.savefig("../results/train_performance_svm_FDG_biascorr.jpg")
plt.show()


y_true = df_ages[df_ages['model'] == 'RVR()']['real']
y_pred = df_ages[df_ages['model'] == 'RVR()']['pred']

# fit a linear model for bias correction
lm_svr = LinearRegression()
lm_svr.fit(np.array(y_true).reshape(-1,1), np.array(y_pred-y_true).reshape(-1,1))
slope_svr = lm_svr.coef_
intercept_svr = lm_svr.intercept_
y_pred_bc = y_pred - (y_true*slope_svr[0]+intercept_svr[0])
#y_pred_bc = y_pred

mae = format(mean_absolute_error(y_true, y_pred_bc), '.2f')
corr = format(np.corrcoef(y_pred_bc, y_true)[1, 0], '.2f')

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred_bc)
m, b = np.polyfit(y_true, y_pred_bc, 1)
plt.plot(y_true, m*y_true + b)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
ax.set(xlabel='True values', ylabel='Predicted values')
plt.title('Actual vs Predicted - RVR')
plt.text(xmin + 10, ymax - 0.01 * ymax, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
#plt.savefig("../results/train_performance_rvr_FDG_biascorr.jpg")
plt.show()

y_true = df_ages[df_ages['model'] == 'gauss']['real']
y_pred = df_ages[df_ages['model'] == 'gauss']['pred']

# fit a linear model for bias correction
lm_gauss = LinearRegression()
lm_gauss.fit(np.array(y_true).reshape(-1,1), np.array(y_pred-y_true).reshape(-1,1))
slope_gauss = lm_gauss.coef_
intercept_gauss = lm_gauss.intercept_
y_pred_bc = y_pred - (y_true*slope_gauss[0][0]+intercept_gauss[0])
#y_pred_bc = y_pred

mae = format(mean_absolute_error(y_true, y_pred_bc), '.2f')
corr = format(np.corrcoef(y_pred_bc, y_true)[1, 0], '.2f')

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred_bc)
m, b = np.polyfit(y_true, y_pred_bc, 1)
plt.plot(y_true, m*y_true + b)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
ax.set(xlabel='True values', ylabel='Predicted values')
plt.title('Actual vs Predicted - Gaussian')
plt.text(xmin + 10, ymax - 0.01 * ymax, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
#plt.savefig("../results/train_performance_gauss_FDG_biascorr.jpg")
plt.show()

# PERMUTATION IMP
rvr.fit(X, y)
rvr_feature_importance = permutation_importance(model_results[0]['rvr'], X, y,
                                                scoring="r2")
svm = SVR().fit(X, y)
svr_feature_importance = permutation_importance(svm, X, y, scoring="r2")
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(range(X.shape[1]), rvr_feature_importance.importances_mean)
ax[1].bar(range(X.shape[1]), svr_feature_importance.importances_mean)
plt.show()

schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
atlas = image.load_img(schaefer.maps)
atlas_matrix = image.get_data(atlas)

# create statistical map where each voxel value coresponds to permutation importance
rvr_imp = rvr_feature_importance.importances_mean
atlas_matrix_stat = atlas_matrix.copy()

for x in range(201):
    if x == 0:
        pass
    else:
        atlas_matrix_stat[atlas_matrix_stat == x] = rvr_imp[x-1]
atlas_rvr = image.new_img_like(atlas, atlas_matrix_stat)

svr_imp = svr_feature_importance.importances_mean
atlas_matrix_stat_svr = atlas_matrix.copy()
atlas_svr_dict = {}

for x in range(201):
    if x == 0:
        pass
    else:
        atlas_matrix_stat_svr[atlas_matrix_stat_svr == x] = svr_imp[x-1]
        atlas_svr_dict[x] = svr_imp[x-1]
atlas_svr = image.new_img_like(atlas, atlas_matrix_stat_svr)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plotting.plot_stat_map(atlas_rvr, axes=ax[0])
ax[0].set_title("RVR-relevant regions for aging")
plotting.plot_stat_map(atlas_svr, axes=ax[1])
ax[1].set_title("SVR-relevant regions for aging")
plt.savefig("../results/Permutation_importance_FDG.jpg")
#nib.save(atlas_rvr,"../results/permutation_importance_RVR_FDG.nii")
#nib.save(atlas_svr,"../results/permutation_importance_SVR_FDG.nii")
plt.show()

# TESTING
df_test = df[df['train'] == "F"]
col = [x for x in df_train.columns if '_' in x]

X_test = df_test[col].values
y_true = df_test['Age'].values

# In[26]:


y_pred = model_results[0]['rvr'].predict(X_test)

y_pred_bc = y_pred - (y_true*slope_rvr[0]+intercept_rvr[0])
#y_pred_bc = y_pred

mae = format(mean_absolute_error(y_true, y_pred_bc), '.2f')
corr = format(np.corrcoef(y_pred_bc, y_true)[1, 0], '.2f')

plt.scatter(y_true, y_pred_bc)
sns.set_style("darkgrid")
plt.xlabel("Ground truth")
plt.ylabel("RVR Prediction")
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
plt.title('Actual vs Predicted - Test Set')
plt.text(75, 88, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
plt.xlim(55,90)
plt.ylim(55,90)
plt.savefig("../results/test_performance_rvr_MRI_biascorr.jpg")
plt.show()

y_pred = model_results[1]['svm'].predict(X_test)

y_pred_bc = y_pred - (y_true*slope_svr[0]+intercept_svr[0])
#y_pred_bc = y_pred

mae = format(mean_absolute_error(y_true, y_pred_bc), '.2f')
corr = format(np.corrcoef(y_pred_bc, y_true)[1, 0], '.2f')

plt.scatter(y_true, y_pred_bc)
sns.set_style("darkgrid")
plt.xlabel("Ground truth")
plt.ylabel("SVR Prediction")
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
plt.title('Actual vs Predicted - Test Set')
plt.text(75, 88, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
plt.xlim(55,90)
plt.ylim(55,90)
plt.savefig("../results/test_performance_svm_MRI_biascorr.jpg")
plt.show()