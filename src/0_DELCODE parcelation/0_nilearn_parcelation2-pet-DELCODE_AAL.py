from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_aal

group = 'SMC'
subjs = pd.read_csv(
    '../../data/DELCODE/{}/{}.csv'.format(group, group), sep=";")
subjs_list = subjs['PTID'].tolist()
data_path = '/media/ukwissarchive/doeringe/BrainAge/DELCODE/FDG_P0/2_SUVR/'

atlas = '../../data/0_ATLAS/AAL1_TPMcropped.nii'
atlas = nib.load(atlas)
labels = fetch_atlas_aal().labels

output_csv = '../../data/DELCODE/{}/DELCODE_PET_{}_AAL1_cropped_parcels2.csv'.format(group, group)

# %%
dates = [date.split('.')[::-1] for date in subjs['visdat']]
sess_list = [date[0]+date[1]+'0' + date[2] if len(date[2]) == 1
             else ''.join(date) for date in dates]
subjs['sess'] = sess_list

# %%
image_list = []
subj_succ = {}
subj_succ['age'] = []
subj_succ['name'] = []
subj_succ['sess'] = []

# create list of regional data and subject IDs
for sub in subjs_list:
    foi = glob(data_path + "SUV*" + sub + "*.nii")
    date = subjs[subjs['PTID'] == sub]['sess'].values[0]
    age = subjs[subjs['PTID'] == sub]['Age'].values[0]
    y = []

    base_ind_ = 0

    if foi and (sub not in subj_succ):
        this_image = nib.load(foi[base_ind_])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ['sess'].append(date)
        subj_succ['age'].append(age)
        subj_succ['name'].append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)
df_final['Dataset'] = 'DELCODE'
df_final['Group'] = group

df_final.to_csv(output_csv, index=False)
