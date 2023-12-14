from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob

group = "MCI"
if group == "MCI":
    subjs = pd.read_csv('/path/to/ADNI_overview_CSV.csv')
    data_path = '/path/to/SUVR_data/'
else:
    subjs = pd.read_csv('/path/to/ADNI_overview_CSV.csv')
    data_path = '/path/to/SUVR_data/'
subjs = subjs[[group in x for x in subjs.Group]]
subjs_list = subjs['Subject'].tolist()

atlas = '../../templates/AAL1_TPMcropped.nii'
atlas = nib.load(atlas)
labels = fetch_atlas_aal().labels

output_csv = '/path/to/output_CSV.csv'

# format dates in ADNI overview file
dates = [date.split('/')[::-1] for date in subjs['Acq Date']]
sess_list = [date[0]+date[1]+'0' + date[2] if len(date[2]) == 1
             else ''.join(date) for date in dates]
subjs['sess'] = sess_list

# initiate lists
image_list = []
subj_succ = {}
subj_succ['age'] = []
subj_succ['name'] = []
subj_succ['sess'] = []

# create list of regional data and subject IDs
for sub in subjs_list:
    foi = glob(data_path + "SUV*" + sub + "*.nii")
    date = subjs[subjs['Subject'] == sub]['sess'].values[0]
    age = subjs[subjs['Subject'] == sub]['Age'].values[0]

    # if there are several scans, only extract regional values for the first
    if len(foi) > 1:
        y = []
        for n in range(len(foi)):
            y.append(int(foi[n][122:126]))
            assert date[:4] == foi[n][122:126],\
                "{}: dates are not the same".format(sub)
        base_ind_ = y.index(np.min(y))
    else:
        base_ind_ = 0

    if foi and (sub not in subj_succ):
        this_image = nib.load(foi[base_ind_])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas,
                                   labels=labels,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(this_image)
        image_list.append(parcelled)
        subj_succ['sess'].append(date)
        subj_succ['age'].append(age)
        subj_succ['name'].append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)
df_final['Group'] = group
df_final['Dataset'] = "ADNI"

df_final.to_csv(output_csv, index=False)
