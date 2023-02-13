#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:28:22 2023

@author: doeringe
"""

### Data investigation ADNI, OASIS, DELCODE ###
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.image import load_img
from scipy.stats import ttest_ind

# %%%
# LOAD DATA
adni_cn = pd.read_csv(
    "../data/ADNI/CN/ADNI_PET_CN_AAL3_1mm_parcels.csv")
adni_scd = pd.read_csv(
    "../data/ADNI/SMC/ADNI_PET_SMC_AAL3_1mm_parcels.csv")
oasis_cn = pd.read_csv(
    "../data/OASIS/CN/OASIS_PET_CN_AAL3_1mm_parcels.csv")
delcode_scd = pd.read_csv(
    "../data/DELCODE/SMC/DELCODE_PET_SMC_AAL3_1mm_parcels.csv")

# %%
# INSPECT SUVR in ROIS
cols = adni_cn.columns[3:-2].values
adni_cn_means = adni_cn[cols].mean(axis=0)
adni_scd_means = adni_scd[cols].mean(axis=0)
oasis_cn_means = oasis_cn[cols].mean(axis=0)
delcode_scd_means = delcode_scd[cols].mean(axis=0)
# %%
adni_cn_age = adni_cn[['age']].mean().values[0]
oasis_cn_age = oasis_cn[['age']].mean().values[0]
plt.plot(adni_cn_means, label="ADNI CN")
plt.plot(oasis_cn_means, label="OASIS CN")
plt.title("Cognitively normal individuals")
plt.legend()
plt.show()
print("ADNI age: {}, OASIS age: {}".format(adni_cn_age, oasis_cn_age))
print(ttest_ind(adni_cn[['age']], oasis_cn[['age']]))
# %%
adni_scd_age = adni_scd[['age']].mean().values[0]
delcode_scd_age = delcode_scd[['age']].mean().values[0]
plt.plot(adni_scd_means, label="ADNI SCD")
plt.plot(adni_cn_means, label="ADNI CN")
plt.plot(delcode_scd_means, label="DELCODE SCD")
plt.legend()
plt.show()
print("ADNI CN age: {}, ADNI SCD age: {}, DELCODE age: {}".format(
    adni_cn_age, adni_scd_age, delcode_scd_age))
print(ttest_ind(adni_cn[['age']], delcode_scd[['age']]))


# %%
adni_cn = pd.read_csv(
    "../data/ADNI/CN/ADNI_PET_CN_AAL1_cropped_parcels.csv")
for i in range(2,98):
    plt.scatter(range(len(adni_cn.index)), adni_cn[adni_cn.columns.tolist()[-i]])
    plt.title(adni_cn.columns.tolist()[-i])
    plt.show()