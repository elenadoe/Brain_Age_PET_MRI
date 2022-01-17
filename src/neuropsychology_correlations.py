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
import pandas as pd
import warnings
import pickle
from transform_data import neuropsych_merge

warnings.filterwarnings("ignore")
# matplotlib config
cm = pickle.load(open("../data/config/plotting_config.p", "rb"))


def neuropsych_correlation(database, age_or_diff, modality,
                           neuropsych_ind=[4:]):
    """
    # TODO.

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
    df_neuropsych = pd.read_csv(
        "../data/main/ADNI_Neuropsych_Neuropath.csv", sep=";")
    neuropsych_var = df_neuropsych.columns[neuropsych_ind]
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    y_true = df_pred['Age']
    y_pred = df_pred['Prediction']
    merged = neuropsych_merge(df_pred, df_neuropsych, database,
                              neuropsych_var)
    print("Significant correlations between {} ".format(age_or_diff) +
          "and Neuropsychology")

    sign = {}
    for n in neuropsych_var:
        merged[n] = pd.to_numeric(merged[n])
        exc = np.isnan(merged[n])
        if age_or_diff == "BPA":
            pearson = stats.pearsonr(y_pred[~exc],
                                     merged[n][~exc])
            if pearson[1] < 0.05:
                sign[n] = pearson[0]
                fig, ax = plt.subplots(1, figsize=[12, 8])
                text = 'r = ' + \
                    str(np.round(pearson[0], 3)) + \
                    ' p = ' + str(np.round(pearson[1], 3))
                plt.title('Difference BPAD - {}'.format(n))
                sns.regplot(y_true, merged[n], ax=ax,
                            scatter_kws={'alpha': 0.3}, label="Age")
                sns.regplot(y_pred, merged[n], ax=ax,
                            scatter_kws={'alpha': 0.3},
                            color="red", label=age_or_diff)
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                plt.legend()
                plt.text(xmin + 0.01 * xmin, ymax - 0.1 * ymax, text,
                         fontsize=12, verticalalignment='bottom',
                         horizontalalignment='left')
                plt.title(n)

        elif age_or_diff == "BPAD":
            y_pred_rnd = np.round(y_pred, 0)
            y_diff = y_pred_rnd - y_true
            y_diff_cat = ["negative" if x < 0 else "BPAD = 0" if x ==
                          0 else "positive" for x in y_diff]
            df_pred["y_pred_rnd"] = y_pred_rnd
            df_pred["y_diff"] = y_diff
            df_pred["BPAD Category"] = y_diff_cat
            pearson = stats.pearsonr(y_diff[~exc],
                                     merged[n][~exc])
            if pearson[1] < 0.05:
                sign[n] = pearson
                slope, intercept = np.polyfit(y_diff[~exc], merged[n][~exc],
                                              1)
                sns.lmplot("y_diff", n, data=merged,
                           scatter_kws={'alpha': 0.3},
                           palette="YlOrBr", hue="BPAD Category")
                plt.plot(y_diff, slope*y_diff+intercept, linestyle="--",
                         label="all", color="gray", zorder=0, alpha=0.3)
                plt.xlabel("BPAD [years]")
                plt.title(n)
                plt.savefig(fname="../results/" + database + "/plots/" +
                            modality + "_" + age_or_diff +
                            "_" + n + ".png", bbox_inches="tight", dpi=300)
                plt.show()

    for key in sign:
        print(key, ":", np.round(sign[key][0], 3), sign[key][1])

    return sign


def plot_bpad_diff(y_true, y_pred, neuropsych_var,
                   df_test, modality, database,
                   group="CN"):
    """
    Create boxplots of BPAD differences significant as per t-test.

    Parameters
    ----------
    y_true: list of floating point values or integers, representing ground
        truth values
    y_pred: list of floating point/integers values, representing predictions
    alg: algorithm used for current task (used for saving)
    modality: image modality used (MRI/PET; used for saving)
    train_test: str indicating whether train or test data is plotted
        (used for saving)
    database: str indicating which database was used

    Returns
    -------
    None. (plots and saves plots)

    """
    print('---BPAD---')
    y_pred = np.round(y_pred, 0)
    y_diff = y_pred - y_true
    y_diff_cat = [-1 if x < 0 else 0 if x == 0 else 1 for x in y_diff]

    sign = {}
    for n in neuropsych_var:
        ttest = stats.ttest_ind(df_test[n][np.where(np.array(y_diff_cat) > 0)[0]],
                                df_test[n][np.where(
                                    np.array(y_diff_cat) < 0)[0]],
                                nan_policy='omit')
        print(n, "p-value: ", ttest[1])
        if ttest[1] < 0.05:
            sign[n] = ttest[0]
            fig, ax = plt.subplots(1, figsize=[12, 8])
            sns.boxplot(x=y_diff_cat, y=df_test[n],
                        palette='PuOr')
            plt.xlabel('BPAD [years]')
            plt.xticks((0, 1, 2), ('-', 'o', '+'))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            text = 't = ' + \
                str(np.round(ttest[0], 3)) + ' p = ' + \
                str(np.round(ttest[1], 3))
            plt.title('Difference BPAD - {}'.format(n))
            plt.text(xmax - 0.01 * xmax, ymax - 0.01 * ymax,
                     text, verticalalignment='top',
                     horizontalalignment='right', fontsize=12)

            plt.savefig(fname="../results/" + database + "/plots/" +
                        group + "/" + modality +
                        "_boxplot_BPAD" + "_" + n + ".png", dpi=300)
            plt.show()
    print("t-values of significant tests:")
    for key in sign:
        print(key, ":", np.round(sign[key], 3))
