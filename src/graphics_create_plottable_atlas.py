#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:58:02 2023

@author: doeringe
"""
# Translate values in AAL to range 1-90 for graphics
# in AAL, values span range from 2000 to 9000 therefore plotted regions
# look very big, which is misleading
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_aal

# get atlas
atlas = fetch_atlas_aal()
atlas_nii = image.load_img(atlas.maps)
trans_dict = {
    x: i+1 for x, i in zip(atlas.indices, range(len(atlas.indices[:90])))}

mat = atlas_nii.get_fdata()
for key, value in trans_dict.items():
    mat[mat == int(key)] = int(value)
# remove cerebellum
mat[mat >= 9000] = 0

atlas_for_plotting = image.new_img_like(atlas_nii, mat)

nib.save(atlas_for_plotting, "../data/0_ATLAS/plottable_atlas.nii")