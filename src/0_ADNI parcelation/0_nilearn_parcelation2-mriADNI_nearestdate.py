import os
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018
from datetime import datetime
import re


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# Edit paths before running
subject_list = 'data/ADNI/FDG_BASELINE_HEALTHY_4_15_2021.csv'

# fetch schaefer atlas. We need the labels from it. Then load Tian labels
# and concat
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
text_file = open('/data/project/age_prediction/extras/Tian_Subcortex_S1_3T_label.txt')
labels = text_file.read().split('\n')
labels = np.append(atlas['labels'], np.array(labels[:-1]))

# this should include subjects' folders
data_file = '/data/project/cat_12.5/ADNI_complete'
output_csv = '/data/project/age_prediction/codes/PET_MRI_age/data/ADNI_Sch_Tian_1mm_parcels_nearest.csv'


# ids to be excluded
excl_ids = []

# read IDs and age
subjs = pd.read_csv(subject_list)
subj_list = ['sub-' + sub.replace('_', '') for sub in subjs['Subject']]
dates = [date.split('/')[::-1] for date in subjs['AcqDate']]
sess_list = [date[0]+date[1]+'0' + date[2] if len(date[2]) == 1
             else ''.join(date) for date in dates]
subjs['sess'] = sess_list
age = subjs['Age']

image_list = []
subj_succ = {}
subj_succ['name'] = []
subj_succ['sess'] = []
subj_succ['Age'] = []
# subj_succ['age'] = []

# create list of regional data and subject IDs
for sub in subjs['Subject']:
    sub_name = 'sub-' + sub.replace('_', '')
    year = subjs[subjs['Subject'] == sub]['AcqDate'].values[0].split('/')[2]
    day = subjs[subjs['Subject'] == sub]['AcqDate'].values[0].split('/')[1]
    month = subjs[subjs['Subject'] == sub]['AcqDate'].values[0].split('/')[0]
    pet_date = datetime(int(year), int(month), int(day), 0, 0)
    fois = glob(op.join(data_file, sub_name, 'ses-' + '*', 'mri', '*.nii*'))
    if fois:
        datetimeobject = []
        for foi in fois:
            result = re.search('ses-(.*)/mri', foi)
            datetimeobject.append(datetime.strptime(result.group(1)[0:8],
                                  '%Y%m%d'))
    nearest_scan = nearest(datetimeobject, pet_date)
    nearest_foi = glob(op.join(data_file, sub_name, 'ses-' + str(
                                 nearest_scan.year) + '*' + str(
                                 nearest_scan.month) + '*' + str(
                                 nearest_scan.day) + '??????', 'mri', '*.nii*'))
    for near_foi in nearest_foi:
        this_image = nib.load(near_foi)
        path = os.path.normpath(near_foi)
        sess = [dir for dir in path.split(os.sep) if dir.startswith('ses')]

        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img='data/schaefer200_17_Tian1mm.nii',
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ['sess'].append(sess[0])
        # subj_succ['age'].append()
        subj_succ['name'].append(sub_name)
        new_age = subjs[subjs['Subject'] == sub][
                                'Age'].values[0] + (nearest_scan.year - int(year))
        subj_succ['Age'].append(new_age)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)

for excl_id in excl_ids:
    df_final.drop(df_final[df_final['name'] == excl_id].index, inplace=True)

df_final.to_csv(output_csv, index=False)
