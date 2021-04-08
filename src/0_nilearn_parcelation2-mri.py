from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018

# Edit paths before running
subject_list = '../data/OASIS_CN_IDs_Age.txt'
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
# this should include subjects' folders
data_file = '/data/project/cat_12.5/OASIS3'
output_csv = '/data/project/age_prediction/codes/PET_MRI_age/data/parcels.csv'


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
subj_succ = []
# Strips the newline character
for sub in subj_list:
    sub_name = sub.split("_", 1)[0]
    session = sub.split("_", 1)[1]
    # /data/project/cat_12.5/HCP/993675/mri/m0wp1993675.nii.gz
    foi = glob(op.join(data_file, sub_name, '*/mri', '*.nii*'))
    if foi:
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas.maps,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ.append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=atlas.labels)

# exclude data where PET could not be pre-processed (not all frames measured)
age = [age[x] for x in range(len(subj_succ)) if subj_list[x] not in excl_ids]
# TODO: The following line is deleting everything.'not in' maybe
# TMO is better to delete it
# from the initial list. Another way would be to add a QC column or common.
subj_succ = [x for x in subj_succ if x in excl_ids]

subs = {'Subject': subj_succ,
        'Age': age}
subs_pd = pd.DataFrame(subs)
df_new = pd.concat([subs_pd, df], axis=1)

df_new.to_csv(output_csv, index=False)
