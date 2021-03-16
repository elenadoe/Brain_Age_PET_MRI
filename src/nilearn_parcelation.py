from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import os.path as op
import os
import mask
import logging
from glob import glob
import pandas as pd
import numpy as np
from qc import qc_z


dataset_path = '/home/antogeo/data/cluster/test_folder/'


extras = '/home/antogeo/data/extras'
schaefer_path = op.join(extras, 'schaefer', 'MNI',
                        'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz')
sub_cortex_path = op.join(extras,
                          'Tian_Subcortex_S1_3T.nii')
cereb_path = op.join(os.environ.get('FSLDIR'),
    'data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-2mm.nii.gz')

pipeline = 'ants'
sample_path = op.join(dataset_path, 'derivatives', pipeline)
vol_patern = op.join(sample_path, 'sub*', '*mod', '*.nii*')
vol_sample = glob(vol_patern)
atlases = [schaefer_path, sub_cortex_path, cereb_path]
atlases_resized = []
for atlas in atlases:
    atlases_resized.append(mask.atlas_prep(atlas, vol_sample[0]))

atlas_features = [
    {'name': 'schaefer_atlas',
     'atlas_fname': atlases_resized[0]}
    # ,
    # {'name': 'sub_cortex',
    #  'atlas_fname': atlases_resized[1]},
    # {'name': 'cereb',
    #  'atlas_fname': atlases_resized[2]}
     ]

meta_fname = glob(dataset_path + '/participants*v')
what = ''

subj_path = op.join(dataset_path, 'derivatives', pipeline)

# List all the subjects in subj_path
subjects = sorted(
    [x.split('/')[-1] for x in glob(op.join(subj_path, 'sub-[!.]*'))
        if op.isdir(x)])

# Read the file
filename, file_extension = op.splitext(meta_fname[0])
if file_extension == '.csv':
    df = pd.read_csv(meta_fname[0])
elif file_extension == '.tsv':
    df = pd.read_csv(meta_fname[0], delimiter='\t')
else:
    raise ValueError(
        'I do not know how to read {}'.format(file_extension))
# Read the nifti/img file
images = []
read_subjects = []
# 
for subject in subjects:
    s_path = op.join(subj_path, subject)
    # read all subjects with selected keyword 
    if (what == '_s2'):
        foi = op.join(s_path, '*mod', what + '*.nii*')
        files = glob(foi)
        read_subjects.append(subject)

logging.info(
    "Reading subjects' volume. Total number of subjects: {}"
    .format(len(read_subjects)))


# Create one list with those subjects that exist in metadata file and one
# with those missing
exist_subj = []
missing = []
image_list = []
for i, subj in enumerate(read_subjects):
    if df['participant_id'].str.contains(subj[4:]).any():
        exist_subj.append(subj[4:])
        foi = glob(op.join(subj_path, '*' + subj, '*mod', '*.nii*'))
        this_image = nib.load(foi[0])
        niimg = check_niimg(this_image, atleast_4d=True)
        masker = NiftiLabelsMasker(labels_img=atlases_resized[0],
                                   standardize=False,
                                   memory='nilearn_cache',
                                   resampling_target='data')
        parcelled = masker.fit_transform(niimg)
        image_list.append(parcelled)
    else:
        missing.append(subj)
logging.info(
    "Missing metadata for subjects: {}".format(missing[:]))
# Get only the data for the subjects in the list

cols = list(range(image_list[0].shape[1]))
df = pd.DataFrame(np.vstack(image_list), columns=cols)
t_df = df[df['participant_id'].isin(exist_subj)]
df.to_csv(f_name)
# Sort
