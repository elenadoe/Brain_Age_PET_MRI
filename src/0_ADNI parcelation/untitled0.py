#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:23:26 2022

@author: doeringe
"""

from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_aal

subjs = pd.read_csv('../data/ADNI_merge_nooutliers_rev1.csv')
subjs_list = subjs['PTID'].tolist()
data_path = '../../SUVR/'

atlas = fetch_atlas_aal()
labels = atlas.labels

output_csv = '../../data/ADNI_parcels_rev1.csv'

# %%
image_list = []
subj_succ = {}
subj_succ['age'] = []

# create list of regional data and subject IDs
for sub in subjs_list:
    foi = glob(data_path + "SUV*" + sub + "*.nii")

    base_ind_ = 0

    if foi:
        this_image = nib.load(foi[base_ind_])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas.maps,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ['name'].append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)

df_final.to_csv(output_csv, index=False)
