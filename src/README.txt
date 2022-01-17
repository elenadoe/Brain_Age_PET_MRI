STEPS OF ANALYSIS
1. make parcels for PET and MRI in ADNI
2. make sure PET and MRI come from same individuals and same time point (+/- one year) 
   and all participants are at least 65
3. add OASIS test set >= 65 years to csv
4. perform train/test split on ADNI data (50 randomly generated, pickle saved seeds)
   file excludes individuals younger than 65 in either modality automatically
   file automatically sets OASIS as test set
5. perform outlier exclusion on train set (outliers = outside 3xIQR)
6. apply outlier exclusion boundaries to test set to exclude outliers in test set
7. apply outlier exclusion to both (ADNI + OASIS) test sets
8. run analyses
