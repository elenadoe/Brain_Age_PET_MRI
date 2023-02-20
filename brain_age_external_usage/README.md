This folder contains all files necessary to estimate MRI or FDG BAG from new data.

Before usage, please assure that 
1) the modality of the data is either MRI or FDG-PET
2) the data has been pre-processed according to Doering et al. (pre-processing scripts available in src folder). Importantly, data must be in MNI152 space.
3) TODO

Instructions for usage:
1) clone this folder by entering the following into your terminal
'''
git clone https://github.com/elenadoe/Brain_Age_PET_MRI/brain_age_external_usage
''' 
2) use pip install to install the following dependencies:
# TODO
3) execute the brain age estimation program from the src folder, either using jupyter notebook, or Python3. For Python3, enter the following into your terminal and follow instructions:
'''
python3 predict_BAG.py
'''

