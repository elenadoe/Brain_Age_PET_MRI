#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:17:08 2021

@author: doeringe
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
def neuropsych_correlation(y_true, y_pred, age_or_diff, neuropsych_var, df_test, modality):
    """

    Parameters
    ----------
    y_pred : predicted age (int or float)
    y_true: true age (int or float)
    neuropsych : list of str containing all neuropsychological test names
    to be assessed
    df_test : dataframe containing the neuropsychological test scores for all
    subjects

    Returns
    -------
    None.

    """
    print("Significant correlations between {} and Neuropsychology".format(age_or_diff))
    
    sign = {}
    for n in neuropsych_var:
        exc = np.isnan(df_test[n])
        pearson = stats.pearsonr(y_pred[~exc],
                                 df_test[n][~exc])
        if pearson[1] < 0.05:
            if age_or_diff == "BPA":
                sign[n] = pearson[0]
                fig, ax = plt.subplots(1, figsize = [12,8])
                sns.regplot(y_true, df_test[n], ax = ax, scatter_kws = {'alpha' : 0.3}, label = "Age")
                sns.regplot(y_pred, df_test[n], ax = ax, scatter_kws = {'alpha' : 0.3}, color = "red", label = age_or_diff)
                plt.xlabel("Age [years]")
                plt.legend()
                plt.title(n)
            else:
                sign[n] = pearson[0]
                fig, ax = plt.subplots(1, figsize = [12,8])
                sns.regplot(y_pred, df_test[n], ax = ax, scatter_kws = {'alpha' : 0.3}, color = "red", label = age_or_diff)
                plt.xlabel("PA - CA [years]")
                plt.legend()
                plt.title(n)
            plt.savefig(fname = "../results/plots/"+modality+"_"+
                        age_or_diff+"_"+n+".png")

    for key in sign:
        print(key, ":", np.round(sign[key],3))
        
