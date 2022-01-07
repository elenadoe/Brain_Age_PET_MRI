import pandas as pd
import random
import pickle
from sklearn.model_selection import train_test_split

df_mri = pd.read_csv('../data/merged/MRI_parcels_all.csv')
df_pet = pd.read_csv('../data/merged/PET_parcels_all.csv')
print("IDs are the same in PET and MRI: ",
      all(df_mri['name'] == df_pet['name']), "(should be True)")

# %%
older_65_mri = (df_mri['age'] >= 65).tolist()
df_pet = df_pet[older_65_mri]
df_mri = df_mri[older_65_mri]
df_mri = df_mri.reset_index(drop=True)
df_pet = df_pet.reset_index(drop=True)

older_65_pet = (df_pet['age'] >= 65).tolist()
df_mri = df_mri[older_65_pet]
df_pet = df_pet[older_65_pet]
df_mri = df_mri.reset_index(drop=True)
df_pet = df_pet.reset_index(drop=True)

# should be False:
print("Individuals younger than 65 contained in data:",
      any(df_mri['age'] < 65), "(should be False)")
# Should be True:
print("Same IDs contained in both dataframes:",
      all(df_mri['name'] == df_pet['name']), "(should be True)")
# %%
# divide into age bins of "young old", "middle old" and "oldest old"
# use mri to do so --> same age bins for both modalities
# maybe acknowledge as weakness in discussion?
df_mri['Ageb'] = [0 if x < 74 else 1 if x < 84 else 2 for x in df_mri['age']]
df_pet['Ageb'] = [0 if x < 74 else 1 if x < 84 else 2 for x in df_pet['age']]

col = df_mri.columns[3:-2].tolist()

X = df_mri[col][df_mri['Dataset'] == "ADNI"].values
y = df_mri['age'][df_mri['Dataset'] == "ADNI"].values

# %%
y_pseudo = df_mri['Ageb'][df_mri['Dataset'] == "ADNI"]

# ONLY UNCOMMENT TO GENERATE NEW RANDOM SEEDS
# rand_seeds = random.sample(range(0,101),100)
# pickle.dump(rand_seeds, open("../data/config/rand_seeds.p" , "wb"))

# %%
rand_seeds = pickle.load(open("../data/config/rand_seeds.p", "rb"))
for r in rand_seeds:
    x_train, x_test,  y_train, y_test, id_train, id_test = train_test_split(
        X, y, df_mri['name'][df_mri['Dataset'] == "ADNI"],
        test_size=.25, random_state=r,
        stratify=y_pseudo)
    
    df_mri['train'] = [True if x in id_train.values else False for x in df_mri[
                   'name']]
    df_pet['train'] = [True if x in id_train.values else False for x in df_pet[
                   'name']]

    df_mri.to_csv('../data/merged/validation_random_seeds/' +
                  'test_train_MRI_{}.csv'.format(str(r)))
    df_pet.to_csv('../data/merged/validation_random_seeds/' +
                  'test_train_PET_{}.csv'.format(str(r)))
