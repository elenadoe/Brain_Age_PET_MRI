import os
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_aal
from datetime import datetime
import re


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# fetch AAL atlas and labels
atlas_file = 'data/0_ATLAS/AAL1_TPMcropped.nii'
atlas = nib.load(atlas_file)
atlas = nib.load(atlas_file)
label_list = fetch_atlas_aal()
labels = pd.DataFrame(label_list)[['labels', 'indices']]

# this should include subjects' folders
data_file = '/data/project/cat_12.5/ADNI_complete'

# Edit paths before running
# groups = ['CN', 'MCI', 'SMC']
groups = ['MCI']
for group in groups:  
    f_path = 'data/ADNI/' + group + '/ADNI_PET_' + group + '_AAL1_cropped_parcels.csv'
    subject_list = pd.read_csv(f_path)
    subject_list = subject_list[subject_list.Group == group]
    subject_list.name

    output_csv = 'data/ADNI/{}/ADNI_MRI_{}_AAL1_cropped_parcels.csv'.format(group, group)

    # ids to be excluded
    excl_ids = []

    # read IDs and age
    subjs = subject_list[['age', 'name', 'sess']]

    # initiate lists
    image_list = []
    subj_succ = {}
    subj_succ['name'] = []
    subj_succ['sess'] = []
    subj_succ['Age'] = []

    # create list of regional data and subject IDs
    for sub in subjs['name']:
        sub_name = 'sub-' + sub.replace('_', '')
        year = subjs[subjs['name'] == sub]['sess'].values[0].astype(str)[0:4]
        day = subjs[subjs['name'] == sub]['sess'].values[0].astype(str)[4:6]
        month = subjs[subjs['name'] == sub]['sess'].values[0].astype(str)[6:8]
        pet_date = datetime(int(year), int(month), int(day), 0, 0)
        fois = glob(op.join(data_file, sub_name, 'ses-2' + '*', 'mri', '*.nii*'))
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
            masker = NiftiLabelsMasker(labels_img=atlas,
                                    standardize=False,
                                    memory='nilearn_cache',
                                    resampling_target='data')
            parcelled = masker.fit_transform(niimg)
            image_list.append(parcelled)
            subj_succ['sess'].append(sess[0])
            subj_succ['name'].append(sub_name)
            new_age = subjs[subjs['name'] == sub][
                                    'age'].values[0] + (nearest_scan.year - int(year))
            subj_succ['Age'].append(new_age)

    features = np.array(image_list)
    x, y, z = features.shape
    features = features.reshape(x, z)
    df = pd.DataFrame(features, columns=labels['labels'])
    df_sub = pd.DataFrame(subj_succ)
    df_final = pd.concat([df_sub, df], axis=1)

    for excl_id in excl_ids:
        df_final.drop(df_final[df_final['name'] == excl_id].index, inplace=True)

    df_final.to_csv(output_csv, index=False)
