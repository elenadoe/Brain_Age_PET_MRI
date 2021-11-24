import pandas as pd
from sklearn.model_selection import train_test_split

df_mri = pd.read_csv('../data/ADNI/ADNI_MRI_Sch_Tian_1mm_parcels.csv',
                 sep = ",")
df_pet = pd.read_csv('../data/ADNI/ADNI_PET_Sch_Tian_1mm_parcels.csv',
                 sep = ";")
print(all(df_mri['name'] == df_pet['name']))
#%%
df_mri = df_mri[df_mri['age']>=65]
df_pet = df_mri[df_mri['age']>=65]
df_mri = df_mri.reset_index(drop=True)
df_pet = df_pet.reset_index(drop=True)

df_mri = df_mri[df_pet['age']>=65]
df_mri = df_mri.reset_index(drop=True)
df_pet = df_pet.reset_index(drop=True)
print("Individuals younger than 65 contained in data:",any(df_mri['age']<65))
print("Same IDs contained in both dataframes:", all(df_mri['name'] == df_pet['name']))
#%%
# divide into age bins of "young old", "middle old" and "oldest old"
df_mri['Agebins'] = [0 if x < 74 else 1 if x < 84 else 2 for x in df_mri['age']]
df_pet['Agebins'] = [0 if x < 74 else 1 if x < 84 else 2 for x in df_pet['age']]

col = df_mri.columns[3:-1].tolist()

X = df_mri[col].values
y = df_mri['age'].values
#%%
y_pseudo = df_mri['Agebins']

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, df_mri['name'], test_size=.2, random_state=42,
    stratify=y_pseudo)

df_mri['train'] = [True if x in id_train.values else False for x in df_mri[
               'name']]
df_pet['train'] = [True if x in id_train.values else False for x in df_pet[
               'name']]
print(all((df_mri['train'] == df_pet['train']) and ))
df_mri.to_csv('../data/ADNI/test_train_MRI.csv')
df_pet.to_csv('../data/ADNI/test_train_PET.csv')
