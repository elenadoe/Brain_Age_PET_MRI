from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018

subjects = pd.read_csv('../FDG_BASELINE_HEALTHY_4_15_2021.csv')
subject_list = subjects['Subject'].tolist()
data_path = '/home/doeringe/Dokumente/brain age/4_SUVR/'
output_csv = '../data/parcels_FDG_tpm_ADNI.csv'
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks = 17)

image_list = []
subj_succ = []
subj_miss = []
subj_year = []
subj_month = []
subj_age = []

# create list of regional data and subject IDs
for sub in subject_list:
    # CHANGE TO MRI FILE BEGINNING
    foi = glob(data_path + "SUV*" + sub + "*.nii")
    y = []
    
    # if there are several scans, only extract regional values for the first
    for n in range(len(foi)):
        # CHANGE WHERE IN STRING YEAR OCCURS HERE
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
        subj_succ.append(sub)
        # CHANGE WHERE IN STRING YEAR OCCURS HERE
        subj_year.append(foi[base_ind_][95:99])
        # CHANGE WHERE IN STRING MONTH OCCURS HERE
        subj_month.append(foi[base_ind_][99:101])
        subj_age.append(np.min(subjects['Age'][subjects['Subject']==sub]))
        
    
features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=atlas.labels)

# combine information on subjects, age and regional data
subs = {'Subject' : subj_succ,
       'Age' : subj_age,
       'Year' : subj_year,
       'Month' : subj_month}
subs_pd = pd.DataFrame(subs)
df_new = pd.concat([subs_pd, df], axis=1)
df_new.to_csv(output_csv, index=False)
