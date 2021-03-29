from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_schaefer_2018

# Edit paths before running
subject_list = '../data/OASIS_CN_IDs_Age.txt'
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks = 17)

# this should include subjects' folders
data_path = '/DATA/doeringe/Dokumente/BrainAge/2_Segmented/'
output_csv = '../data/parcels_FDG_fdgtemplate.csv'

# read IDs and age
subjs = pd.read_csv(subject_list, delimiter="\t")

subjs = pd.read_csv(subject_list, delimiter="\t")
subj_list = ['sub-' + sub for sub in subjs['SCAN_ID']]
age = subjs['age']

count = 0
image_list = []
subj_succ = []
subj_miss = []

for sub in subj_list:
    sub_name = sub.split("_", 1)[0]
    session = sub.split("_", 1)[1]
    
    foi = glob(data_path + "SUV*" + sub_name + "_" +  session + "*.nii")
    if foi:
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlas.maps,
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
        subj_succ.append(sub)
        
# exclude data that could not be pre-processed including frame issues
age = [age[x] for x in range(len(subj_list)) if subj_list[x] in subj_succ]

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=atlas.labels)

# include age info in dataframe
subs = {'Subject' : subj_succ,
       'Age' : age}
subs_pd = pd.DataFrame(subs)
df_new = pd.concat([subs_pd, df], axis=1)
df_new.to_csv(output_csv, index=False)
