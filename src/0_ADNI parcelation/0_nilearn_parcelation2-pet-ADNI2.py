from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018

subjs = pd.read_csv('../../data/ADNI/FDG_BASELINE_HEALTHY_4_15_2021.csv')
subjs_list = subjs['Subject'].tolist()
data_path = '/home/doeringe/Dokumente/brain age/4_SUVR/'

atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
text_file = open('../../data/Tian_Subcortex_S1_3T_label.txt')
labels = text_file.read().split('\n')
labels = np.append(atlas['labels'], np.array(labels[:-1]))

output_csv = '/../../data/ADNI_PET_Sch_Tian_1mm_parcels.csv'

#%%
dates = [date.split('/')[::-1] for date in subjs['AcqDate']]
sess_list = [date[0]+date[1]+'0' + date[2] if len(date[2]) == 1
             else ''.join(date) for date in dates]
subjs['sess'] = sess_list
age = subjs['Age']
#%%
image_list = []
subj_succ = {}
subj_succ['name'] = []
subj_succ['sess'] = []

# create list of regional data and subject IDs
for sub in subjs_list:
    foi = glob(data_path + "SUV*" + sub + "*.nii")
    date = subjs[subjs['Subject'] == sub]['sess']
    age = subjs[subjs['Subject'] == sub]['Age']
    y = []
    
    # if there are several scans, only extract regional values for the first
    for n in range(len(foi)):
        y.append(int(foi[n][95:99]))
    base_ind_ = y.index(np.min(y))
    
    if foi and (sub not in subj_succ):
        this_image = nib.load(foi[base_ind_])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas.maps,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
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

df_new.to_csv(output_csv, index=False)
