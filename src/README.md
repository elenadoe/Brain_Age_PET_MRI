## **STEPS OF ANALYSIS**
**Assumes parcelation of neuroimaging data is already completed. A demonstration of Code can be found in demo.ipynb**
Code dependency is visualized below.
1. assure all participants in ADNI and OASIS are older than 65
2. create train-test splits from ADNI data with given random seed (default = 0), assign OASIS as second test set
3.1 determine outlier ranges from training data
3.2 exclude data outside ranges from training and test data
4. train classifiers using k-fold stratified cross-validation for (hyper)parameter optimization
5. compare results with and without bias correction, with and without chronological age (*demo.ipynb main(1, modality)*)
6. predict bias-corrected brain age of ADNI and OASIS CN test set (*demo.ipynb main(2, modality)*)
7. iterate over different random stratified train-test splits to validate MAE and R2 by repeating steps 1:6 50 times (*demo.ipynb main(2.1, modality)*)
8. predict bias-corrected brain age of ADNI MCI test set (*demo.ipynb main(3, modality)*)
9. Calculate partial spearman correlations between cognitive performance/neuropathology and BPAD in CN and MCI correcting for sex (*demo.ipyb main(4.1:4.4, modality)*)
10. Calculate Mann-Whitney-U test between frequencies of diagnoses (CN, MCI, AD) after two years (*demo.ipynb main(5.1:5.2, modality)*)

![Code dependency](analysis_code_hierarchy.jpg)
