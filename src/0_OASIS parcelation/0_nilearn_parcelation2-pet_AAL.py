from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_aal

group = "CN"
subject_list = '../../data/OASIS/OASIS_CN_IDs_Age.txt'
data_path = '/media/ukwissarchive/doeringe/BrainAge/OASIS/2_SUVR/'
output_csv = '../../data/OASIS/{}/OASIS_PET_{}_AAL1_cropped_parcels.csv'.format(group, group)

atlas = '../../data/0_ATLAS/AAL1_TPMcropped.nii'
atlas = nib.load(atlas)
labels = fetch_atlas_aal().labels

# read IDs and age
subjs = pd.read_csv(subject_list, delimiter="\t")
subj_list = [sub for sub in subjs['SCAN_ID']]
age = subjs['age']

image_list = []
subj_succ = []
subj_miss = []
sessions = []

# create list of regional data and subject IDs
for sub in subj_list:
    sub_name = sub.split("_", 1)[0]
    session = sub.split("d", 1)[1]

    foi = glob(data_path + "SUV*" + sub_name + "*_*" + session + "*.nii")
    if foi:
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ.append(sub)
        sessions.append(session)

# exclude data that could not be pre-processed including frame issues
age = [int(np.round(age[x])) for x in range(len(subj_list)) if subj_list[x] in subj_succ]
subj_succ = [sub.split("_", 1)[0] for sub in subj_succ]

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)

# combine information on subjects, age and regional data
subs = {'name': subj_succ,
        'age': age,
        'sess': sessions}
subs_pd = pd.DataFrame(subs)
df_new = pd.concat([subs_pd, df], axis=1)
df_new['Database'] = 'OASIS'
df_new['Group'] = group
df_new.to_csv(output_csv, index=False)
