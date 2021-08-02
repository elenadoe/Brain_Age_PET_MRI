#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:17:08 2021

@author: doeringe
"""
import numpy as np
import scipy.stats as stats
def neuropsych_correlation(y_pred, neuropsych_var, df_test):
    """

    Parameters
    ----------
    y_pred : predicted age (int or float)
    neuropsych : list of str containing all neuropsychological test names
    to be assessed
    df_test : dataframe containing the neuropsychological test scores for all
    subjects

    Returns
    -------
    None.

    """
    
    print("Significant correlations between predicted brain age and Neuropsychology")

    sign = {}
    for n in neuropsych_var:
        exc = np.isnan(df_test[n])
        pearson = stats.pearsonr(df_test[n][~exc], 
                             y_pred[~exc])
        if pearson[1] < 0.05:
            sign[n] = pearson[0]
    for key in sign:
        print(key, ":", np.round(sign[key],3))

