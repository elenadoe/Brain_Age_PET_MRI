import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

modality = input("Which modality are you analyzing? ")
df = pd.read_csv('../data/ADNI/ADNI_'+modality+'_Sch_Tian_1mm_parcels.csv', sep = ";")

df['Agebins'] = df['age'].values // 7
df['Agebins'] = df['Agebins'].astype(int)

col = [x for x in df.columns if '_' in x]

X = df[col].values

y_pseudo = df['Agebins']
y = df['age']

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['name'], test_size=.3, random_state=42,
    stratify=y_pseudo)

# drop outliers (absolute z score bigger than 2)
z = np.abs(stats.zscore(df['age']))
df = df.drop(np.where(z>2)[0])

df['train'] = [True if x in id_train.values else False for x in df[
               'name']]
# TODO: make bootstrapping test_train_splits @antogeo
df.to_csv('../data/ADNI/test_train_'+modality+'.csv')
