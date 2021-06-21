import os
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018

# Edit paths before running
subject_list = 'data/OASIS/OASIS_CN_IDs_Age.txt'
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
# this should include subjects' folders
data_file = '/data/project/cat_12.5/OASIS3'
output_csv = '/data/project/age_prediction/codes/PET_MRI_age/data/OASIS_mri_parcels_mixedAtlas.csv'

text_file = open('/data/project/age_prediction/extras/Tian_Subcortex_S1_3T_label.txt')
labels = text_file.read().split('\n')
labels = np.append(atlas['labels'], np.array(labels[:-1]))

# NOTE: 'sub-OAS30775_ses-d2893', 'sub-OAS31018_ses-d0469' need to be
# excluded as not all frames were measured in PET
# therefore pre-processing was not possible!
excl_ids = ['sub-OAS30775_ses-d2893', 'sub-OAS31018_ses-d0469']

# read IDs and age
subjs = pd.read_csv(subject_list, delimiter="\t")
subj_list = ['sub-' + sub for sub in subjs['SCAN_ID']]
age = subjs['age']

count = 0
image_list = []
subj_succ = {}
subj_succ['name'] = []
subj_succ['sess'] = []
# subj_succ['age'] = []
# Strips the newline character
for sub in subj_list:
    sub_name = sub.split("_", 1)[0]
    session = sub.split("_", 1)[1]
    # /data/project/cat_12.5/HCP/993675/mri/m0wp1993675.nii.gz
    foi = glob(op.join(data_file, sub_name, '*/mri', '*.nii*'))
    if foi:
        this_image = nib.load(foi[0])
        path = os.path.normpath(foi[0])
        sess = [dir for dir in path.split(os.sep) if dir.startswith('ses')]

        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img='data/schaefer200-17_Tian.nii',
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ['sess'].append(sess[0])
        # subj_succ['age'].append()
        subj_succ['name'].append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)

for excl_id in excl_ids:
    df_final.drop(df_final[df_final['name'] == excl_id].index, inplace=True)
# exclude data where PET could not be pre-processed (not all frames measured)

df_final.to_csv(output_csv, index=False)
