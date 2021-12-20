STEPS OF ANALYSIS
1. make parcels for PET and MRI
2. make sure PET and MRI come from same individuals and same time point (+/- one year) 
   and all participants are at least 65
n = 370
3. add OASIS test set >= 65 years to csv (n = 60)
4. exclude individuals younger than 65 from OASIS test set
5. perform train/test split on ADNI data (100 randomly generated, pickle saved seeds)
   file also excludes individuals younger than 65 in either modality automatically
   file automatically sets OASIS as test set
6. perform outlier exclusion on train set (outliers = outside 2.5xIQR)
7. merge train-test csv with neuropsychology/neuropathology
8. apply outlier exclusion to both (ADNI + OASIS) test sets
