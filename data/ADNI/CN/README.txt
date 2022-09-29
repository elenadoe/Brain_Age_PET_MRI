File explanations

FDG_BASELINE_HEALTY --> FDG_BASELINE_HEALTHY_unique
1. excluded second baseline visit of participants
1.5 also excluded corresponding scans from 2_SUVR

ADNI_{mod}_Sch_Tian created from parcelation python file

ADNI_{mod}_Sch_Tian --> {mod}_parcels_all:
1. check each ID only occurs once
1. exclude scans only available in one modality (MRI not in PET: ~5, PET not in MRI: 8)
2. exclude scans where PET and MRI were done more than one year apart