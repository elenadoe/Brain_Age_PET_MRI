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
from transform_data import neuropsych_merge, dx_merge

warnings.filterwarnings("ignore")


def neuropsych_correlation(database, age_or_diff, neuropsych_ind, modality):
    """
    Correlations between BPAD and neuropsychology/-pathology.

    Parameters
    ----------
    database : str
        CN or MCI
    age_or_diff : str
        Brain-predicted age (BPA) or brain-predicted age difference (BPAD)
    neuropsych_ind : list
        indexes of columns to consider
    modality : str
        PET or MRI

    Returns
    -------
    sign : dict
        r-values of significant correlations

    """
    df_neuropsych = pd.read_csv(
        "../data/main/ADNI_Neuropsych_Neuropath.csv", sep=";")
    neuropsych_var = df_neuropsych.columns[neuropsych_ind].tolist()
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    y_true = df_pred['Age']
    y_pred = df_pred['Prediction']
    merged = neuropsych_merge(df_pred, df_neuropsych, database,
                              neuropsych_var)

    print("\033[1m---SIGNIFICANT CORRELATIONS BETWEEN {} ".format(
        age_or_diff) + "& NEUROPSYCHOLOGY/NEUROPATHOLOGY---\033[0m")

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
            merged["y_pred_rnd"] = y_pred_rnd
            merged["y_diff"] = y_diff
            merged["BPAD Category"] = y_diff_cat
            if len(y_diff[~exc]) < 2:
                print("Not enough observations of", n)
                continue
            pearson = stats.pearsonr(y_diff[~exc],
                                     merged[n][~exc])
            if pearson[1] < 0.05:
                sign[n] = pearson
                slope, intercept = np.polyfit(y_diff[~exc], merged[n][~exc],
                                              1)
                if database == "CN":
                    cm_np = pickle.load(open(
                        "../config/plotting_config_np_{}.p".format(
                            modality), "rb"))
                    sns.set_palette(cm_np)
                elif database == "MCI":
                    cm_np_mci = pickle.load(open(
                        "../config/plotting_config_np_mci_{}.p".format(
                            modality), "rb"))
                    sns.set_palette(cm_np_mci)
                sns.lmplot("y_diff", n, data=merged,
                           scatter_kws={'alpha': 0.4},
                           hue="BPAD Category")
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

    neuropsychology_BPAD_interaction(merged, sign, y_diff)

    return sign


def neuropsychology_BPAD_interaction(merged, sign, y_diff):
    """
    Assess interaction effects of BPAD and correlation.

    Parameters
    ----------
    merged : TYPE
        DESCRIPTION.
    sign : TYPE
        DESCRIPTION.
    y_diff : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print("\n---INTERACTION EFFECTS---")
    for k in sign:
        exc = np.isnan(merged[k])
        pos = merged['BPAD Category'] == 'positive'
        neg = merged['BPAD Category'] == 'negative'

        pos_bool = np.array(~exc) & np.array(pos)
        neg_bool = np.array(~exc) & np.array(neg)
        pearson_pos = stats.pearsonr(y_diff[pos_bool],
                                     merged[k][pos_bool])
        pearson_neg = stats.pearsonr(y_diff[neg_bool],
                                     merged[k][neg_bool])
        print(k, "significant in positive BPAD: ",
              pearson_pos[1] < 0.05,
              pearson_pos,
              "\nsignificant in negative BPAD: ", pearson_neg[1] < 0.05,
              pearson_neg)


def conversion_analysis(database, modality):
    """
    Analyze conversion rates.

    Analyze how many participants convert to AD after 24 months.

    Parameters
    ----------
    database : TYPE
        DESCRIPTION.
    modality : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # cm_np = pickle.load(open(
    #                    "../config/plotting_config_np_{}.p".format(
    #                        modality), "rb"))
    df_dx = pd.read_csv(
        "../data/MCI/MCI_DX_after24months.csv", sep=";")
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    df_pred['BPAD'] = np.round(df_pred['Prediction'] - df_pred['Age'], 0)
    merge = dx_merge(df_pred, df_dx)
    sns.violinplot(x='BPAD', y='DX', data=merge,
                   order=["CN", "MCI", "Dementia"],
                   hue_order=["CN", "MCI", "Dementia"])
    plt.ylabel("Diagnosis after 24 months")
    # plt.ylabel("Percent of whole group")
    plt.savefig(fname="../results/" + database + "/plots/" +
                modality + "_.png", box_inches="tight", dpi=300)
    plt.show()
