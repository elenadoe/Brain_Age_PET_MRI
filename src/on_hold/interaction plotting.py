#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:28:37 2021

@author: doeringe
"""
import matplotlib.pyplot as plt

y_db_cat = [0 if x == "ADNI" else 1 for x in df_test['Dataset']]
plt.scatter(mri, pet, c=cm[y_db_cat])
plt.xlim(60,90)
plt.ylim(60,90)
plt.xlabel('MRI age')
plt.ylabel('PET age')
plt.show()
print(r2_score(mri,pet))

mri_diff = mri - df_test['age']
pet_diff = pet - df_test['age']
pr = stats.pearsonr(mri_diff, pet_diff)
print(pr)
plt.scatter(mri_diff, pet_diff, c=cm[y_db_cat])
plt.xlim(-9,9)
plt.ylim(-9,9)
plt.xlabel('MRI age')
plt.ylabel('PET age')
slope, intercept = np.polyfit(mri_diff, pet_diff, 1)
plt.plot(y_diff, slope*y_diff+intercept, linestyle="--",
         color = "gray", zorder=0, alpha=0.3)
plt.savefig('../results/3_interaction/Correlation_MRI-PET_BPAD.jpg',
            bbox_inches='tight')
plt.show()


