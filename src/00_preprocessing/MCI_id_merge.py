#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:57:16 2021

@author: doeringe
"""

# make list of files in SUVR folder only IDs
import os
import pandas as pd
path_ = "/media/projects/gatekeeping_amyloid_positivity/SUVR/"
adni_info = pd.read_csv("/media/projects/brain_age/PET_MRI_age/data/ADNI/FDG_BASELINE_MCI_11_17_2021.csv")
text_ = open("../data/ADNI/ADNI_MCI_ids.txt","w+")

text_.write("SCAN_ID\tage")
for file in os.listdir(path_):
    age = []
    pat = file[12:22]
    age = adni_info['Age'][adni_info['Subject'] == pat].values
    if len(age)>0:
        text_.write("\n"+pat+"\t"+str(age[0]))
        
text_.close()