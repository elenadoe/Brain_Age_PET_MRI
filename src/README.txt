STEPS OF ANALYSIS (Assumes parcelation of neuroimaging data is already completed. A demonstration of Code can be found in demo.ipynb)
1. assure all participants in ADNI and OASIS are older than 65
2. create train-test splits from ADNI data with given random seed (default = 0), assign OASIS as second test set
3.1 determine outlier ranges from training data
3.2 exclude data outside ranges from training and test data
4. train classifiers using k-fold stratified cross-validation for (hyper)parameter optimization
'''
# demo.ipynb
main(2, modality)
'''
5. 
