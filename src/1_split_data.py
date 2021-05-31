from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/ADNI/parcels_MRI_ADNI_final.csv', sep = ";")

df['Agebins'] = df['Age'].values // 7
df['Agebins'] = df['Agebins'].astype(int)

col = [x for x in df.columns if '_' in x]

X = df[col].values

y_pseudo = df['Agebins']
y = df['Age']

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['Subject'], test_size=.2, random_state=42,
    stratify=y_pseudo)

df['train'] = ["T" if x in id_train.values else "F" for x in df[
               'Subject']]

#df.to_csv('../data/test_train_FDG.csv')