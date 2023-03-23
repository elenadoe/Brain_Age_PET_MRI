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
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.datasets import fetch_atlas_aal, fetch_surf_fsaverage
from nilearn import plotting, surface

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

surf = fetch_surf_fsaverage()
mesh = surface.vol_to_surf(atlas_for_plotting, surf['pial_left'])
plotting.plot_surf_roi(surf['pial_left'], mesh, view='lateral',
                       hemi='left', bg_map = surf['sulc_left'],
                       bg_on_data=True, darkness=0.2)
plt.savefig("../data/0_ATLAS/graphics_AAL_parcellation.png", dpi=300)