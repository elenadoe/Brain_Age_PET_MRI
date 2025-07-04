Steps of pre-processing for PET (If scans were downloaded in "Co-registered, Averaged" format from ADNI, step 2 is not necessary):
1. If data is downloaded in DICOM format, it needs to be converted to Nifti using DicomImport.m
2. Average across frames for static acquisition
3. Set origin of PET scans
4. Spatially normalize PET data
5. Create SUVRs using the pons as reference
