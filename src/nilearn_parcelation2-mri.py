from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob

# Edit paths before running
subject_list = '/data/project/age_prediction/codes/PET_MRI_age/data/OASIS_PET_IDs.txt'
atlas_path = '/data/project/age_prediction/codes/PET_MRI_age/data/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.nii.gz'
# this should include subjects' folders
data_file = '/data/project/cat_12.5/OASIS3'
output_csv = '/data/project/age_prediction/codes/PET_MRI_age/data/parcels.csv'

# Using readlines()
file1 = open(subject_list, 'r')
subjs = file1.read()
subjs = subjs.splitlines()
subj_list = ['sub-' + sub for sub in subjs]
print(subj_list)

count = 0
image_list = []
subj_succ = []
# Strips the newline character
for sub in subj_list:
    sub_name = sub.split("_", 1)[0]
    session = sub.split("_", 1)[1]
    # /data/project/cat_12.5/HCP/993675/mri/m0wp1993675.nii.gz
    foi = glob(op.join(data_file, sub_name, session, 'mri', '*.nii*'))
    if foi:
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas_path,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ.append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features)

subs_pd = pd.DataFrame(subj_succ, columns=['subject'])
df_new = pd.concat([subs_pd, df], axis=1)
df_new.to_csv(output_csv, index=False)
