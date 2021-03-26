#!/home/antogeo/anaconda3/envs/julearn/bin/python

import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

%matplotlib
%config IPCompleter.use_jedi = False

df = pd.read_csv('/home/antogeo/codes/PET_MRI_age/tests/test_csv.csv',
                 index_col=0)
df['Age_bins'] = df['Age'].values // 5
df['Age_bins'] = df['Age_bins'].astype(int)

col = [x for x in df.columns if 'cereb' in x]

X = df[col].values

y_pseudo = df['Age_bins']
y = df['Age']

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['participant_id'], test_size=.2, random_state=42,
    stratify=y_pseudo)

df['train'] = ["T" if x in id_train.values else "F" for x in df[
               'participant_id']]


plt.hist(y_pseudo.values)
plt.hist(y_train // 5)
plt.hist(y_test // 5)
plt.show()

df.to_csv('/home/antogeo/codes/PET_MRI_age/tests/test_train.csv')
