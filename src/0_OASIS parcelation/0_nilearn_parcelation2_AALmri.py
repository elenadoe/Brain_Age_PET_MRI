import os
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob

# Edit paths before running
subject_list = 'data/OASIS/OASIS_CN_IDs_Age.txt'
# fetch AAL atlas and labels
atlas_file = 'data/0_ATLAS/AAL3v1_1mm.nii'
atlas = nib.load(atlas_file)
label_file = 'data/0_ATLAS/AAL3v1_1mm.nii.txt'
label_list = open(label_file)
label_elems = label_list.read().split('\n')
labels = [x.split(' ')[1] for x in label_elems if len(x.split(' '))>1]
# remove labels that were redefined in AAL3 and left empty for comparability
exclude = ['Cingulate_Ant_L', 'Cingulate_Ant_R', 
            'Thalamus_L', 'Thalamus_R']
labels = [i for i in labels if i not in exclude]

# this should include subjects' folders
data_file = '/data/project/cat_12.5/OASIS3'
output_csv = '/data/project/age_prediction/codes/PET_MRI_age/data/OASIS_AAL_1mm_parcels.csv'

# NOTE: 'sub-OAS30775_ses-d2893', 'sub-OAS31018_ses-d0469' need to be
# excluded as not all frames were measured in PET
# therefore pre-processing was not possible!
excl_ids = ['sub-OAS30775_ses-d2893', 'sub-OAS31018_ses-d0469']

# read IDs and age
subjs = pd.read_csv(subject_list, delimiter="\t")
subj_list = ['sub-' + sub for sub in subjs['SCAN_ID']]
age = subjs['age']

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
        masker = NiftiLabelsMasker(labels_img=atlas,
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
