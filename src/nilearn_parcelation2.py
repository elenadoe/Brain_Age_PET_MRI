from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import os
import numpy as np
import pandas as pd
from glob import glob

# Edit paths before running
subject_list = 'path/to/text/file/with/subjects_id.txt'
atlas_path = '/path/to/parcelation_atlas.nii'
data_file = '/path/to/data/file' # this should include subjects' folders
output_csv = '/path/to/ouput/file/with/format.csv'

# Using readlines()
file1 = open(subject_list, 'r')
subjs = file1.read()
subjs = subjs.splitlines()
print(subjs)

count = 0
image_list = []
# Strips the newline character
for sub in subjs:
        foi = glob(op.join(data_file, sub, 'mri', '*.nii*')) #/data/project/cat_12.5/HCP/993675/mri/m0wp1993675.nii.gz
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas_path,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)

features = np.array(image_list)
features =  features.reshape(398, 264)
df =  pd.DataFrame(features)

subs_pd = pd.DataFrame(subjs,columns = ['subject'])
df_new = pd.concat([subs_pd, df], axis=1)
df_new.to_csv(output_csv, index=False)


