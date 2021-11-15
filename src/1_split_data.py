import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

modality = 'MRI'
df = pd.read_csv('../data/ADNI/ADNI_'+modality+'_Sch_Tian_1mm_parcels.csv',
                 sep = ",")

df['Agebins'] = df['age'].values // 7
df['Agebins'] = df['Agebins'].astype(int)

col = [x for x in df.columns if '_' in x]

X = df[col].values
y = df['age'].values
#%%
y_pseudo = df['Agebins']

X = X[np.where(y>65)[0]]
id_ = df['name'][np.where(y>65)[0]]
y_pseudo = y_pseudo[np.where(y>65)[0]]
y = y[np.where(y>65)[0]]

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, id_, test_size=.2, random_state=42,
    stratify=y_pseudo)

df['train'] = [True if x in id_train.values else False for x in df[
               'name']]
# TODO: make bootstrapping test_train_splits @antogeo
df.to_csv('../data/ADNI/test_train_'+modality+'.csv')
