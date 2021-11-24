import pandas as pd
from sklearn.model_selection import train_test_split

modality = 'MRI'
df = pd.read_csv('../data/ADNI/ADNI_'+modality+'_Sch_Tian_1mm_parcels.csv',
                 sep = ",")
df = df[df['age']>65]
df = df.reset_index(drop=True)
print("Individuals younger than 65 contained in data:",any(df['age']<65))

#%%
# divide into age bins of "young old", "middle old" and "oldest old"
df['Agebins'] = [0 if x < 74 else 1 if x < 84 else 2 for x in df['age']]
#df['Agebins'] = df['Agebins'].astype(int)

col = df.columns[3:-1].tolist()

X = df[col].values
y = df['age'].values
#%%
y_pseudo = df['Agebins']

x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['name'], test_size=.2, random_state=42,
    stratify=y_pseudo)

df['train'] = [True if x in id_train.values else False for x in df[
               'name']]

df.to_csv('../data/ADNI/test_train_'+modality+'.csv')
