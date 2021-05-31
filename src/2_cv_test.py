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
sys.path.append("../scratch")
from create_splits import stratified_splits

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


# In[4]:
