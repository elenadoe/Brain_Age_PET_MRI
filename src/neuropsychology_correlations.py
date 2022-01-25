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
from pingouin import partial_corr
from transform_data import neuropsych_merge, neuropath_merge, dx_merge

warnings.filterwarnings("ignore")


def neuro_correlation(database, age_or_diff, psych_or_path, modality):
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
    # TODO: delete neuropsych_ind (not used anymore)

    Returns
    -------
    sign : dict
        r-values of significant correlations

    """
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    df_pred['RID'] = df_pred['PTID'].str[-4:].astype(int)
    y_true = df_pred['Age']
    y_pred = df_pred['Prediction']
    y_diff = y_pred - y_true
    std_diff = np.nanstd(y_diff)
    y_diff_cat = ["negative" if x < -std_diff
                  else "neutral" if (-std_diff < x < std_diff)
                  else "positive" for x in y_diff]

    if (psych_or_path == "PSYCH") or (psych_or_path == "psych"):
        df_neuropsych = pd.read_csv(
            "../data/main/UWNPSYCHSUM_12_13_21.csv", sep=";")
        var_ = ['ADNI_MEM', 'ADNI_EF']
        merged = neuropsych_merge(df_pred, df_neuropsych,
                                  var_)

    elif (psych_or_path == "PATH") or (psych_or_path == "path"):
        df_neuropath1 = pd.read_csv(
            "../data/main/ADNI_Neuropsych_Neuropath.csv", sep=";")
        df_neuropath2 = pd.read_csv(
            "../data/main/ADNI_TauPET.csv", sep=";")
        neuropath1_var = ['ABETA', 'AV45', 'TAU', 'PTAU']
        neuropath2_var = ['TAU_METAROI']
        var_ = neuropath1_var + neuropath2_var
        merged = neuropath_merge(df_pred, df_neuropath1, df_neuropath2,
                                 neuropath1_var, neuropath2_var)

    merged["y_diff"] = y_diff
    merged["BPAD Category"] = y_diff_cat
    merged.to_csv("../results/pred_merged_{}_{}_{}.csv".format(
            database, modality, psych_or_path), index=False)
    print("\033[1m---SIGNIFICANT CORRELATIONS BETWEEN {} ".format(
        age_or_diff) + "& NEURO{}---\033[0m".format(psych_or_path))

    sign = {}
    for n in var_:
        # TODO: change to partial correlation pingouin, print sample size
        merged[n] = pd.to_numeric(merged[n])
        exc = np.isnan(merged[n])
        if len(y_diff[~exc]) < 3:
            print("Not enough observations of", n)
            continue
        if age_or_diff == "BPAD":
            norm = stats.shapiro(merged[n][~exc])
            if norm[1] > 0.05:
                stat = stats.pearsonr(y_diff[~exc],
                                      merged[n][~exc])
                method_str = "Pearson"
            else:
                stat = stats.spearmanr(y_diff[~exc],
                                       merged[n][~exc])
                method_str = "Spearman"
            if stat[1] < 0.05:
                sign[n] = stat
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
                           scatter_kws={'alpha': 0.4})
                plt.plot(y_diff, slope*y_diff+intercept, linestyle="--",
                         label="all", color="gray", zorder=0, alpha=0.3)
                plt.xlabel("BPAD [years]")
                plt.title(method_str + "-Correlation " + n + " X BPAD")
                plt.savefig(fname="../results/" + database + "/plots/" +
                            modality + "_" + age_or_diff +
                            "_" + n + ".png", bbox_inches="tight", dpi=300)
                plt.show()

    for key in sign:
        print(key, ":", np.round(sign[key][0], 3), sign[key][1])

    neuropsychology_BPAD_interaction(merged, sign, y_diff, modality, database)

    return sign


def neuropsychology_BPAD_interaction(merged, sign, y_diff, modality, database):
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
    if database == "CN":
        cm_np = pickle.load(open(
            "../config/plotting_config_np_{}.p".format(
                modality), "rb"))
    elif database == "MCI":
        cm_np = pickle.load(open(
            "../config/plotting_config_np_mci_{}.p".format(
                modality), "rb"))
    sns.set_palette(cm_np)
    print("\n---INTERACTION EFFECTS---")
    for k in sign:
        p = 0.05/len(sign)
        exc = np.isnan(merged[k])
        pos = merged['BPAD Category'] == 'positive'
        neg = merged['BPAD Category'] == 'negative'
        zer = merged['BPAD Category'] == 'neutral'

        pos_bool = np.array(~exc) & np.array(pos)
        neg_bool = np.array(~exc) & np.array(neg)
        zer_bool = np.array(~exc) & np.array(zer)
        norm_pos = stats.shapiro(merged[k][pos_bool])
        norm_neg = stats.shapiro(merged[k][neg_bool])
        norm_zer = stats.shapiro(merged[k][zer_bool])
        if (norm_pos[1] > 0.05) and (norm_neg[1] > 0.05) and (norm_zer[1] > 0.05):
            stat_pos = stats.pearsonr(y_diff[pos_bool],
                                      merged[k][pos_bool])
            stat_neg = stats.pearsonr(y_diff[neg_bool],
                                      merged[k][neg_bool])
            stat_zer = stats.pearsonr(y_diff[zer_bool],
                                      merged[k][zer_bool])
            method_str = "Pearson Correlation"
        else:
            stat_pos = stats.spearmanr(y_diff[pos_bool],
                                       merged[k][pos_bool])
            stat_neg = stats.spearmanr(y_diff[neg_bool],
                                       merged[k][neg_bool])
            stat_zer = stats.spearmanr(y_diff[zer_bool],
                                       merged[k][zer_bool])
            method_str = "Spearman Correlation"
        print(k, "significant in positive BPAD: ",
              stat_pos[1] < p, "\n",
              stat_pos,
              "\nsignificant in neutral BPAD: ",
              stat_zer[1] < p, "\n",
              stat_zer,
              "\nsignificant in negative BPAD: ",
              stat_neg[1] < p, "\n",
              stat_neg)
        if (stat_pos[1] < p) or (stat_neg[1] < p) or (stat_zer[1] < p):
            sns.lmplot("y_diff", k, data=merged,
                       scatter_kws={'alpha': 0.4},
                       hue="BPAD Category", palette=cm_np)
            # plt.plot(y_diff, slope*y_diff+intercept, linestyle="--",
            #         label="all", color="gray", zorder=0, alpha=0.3)
            plt.xlabel("BPAD [years]")
            plt.title(method_str + "-Correlation " + k +
                      " X BPAD by BPAD Category")
            plt.savefig(fname="../results/" + database + "/plots/" +
                        modality + "_" +
                        "_" + k + "_GROUPEFFECT.png", bbox_inches="tight",
                        dpi=300)
            plt.show()

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
    df_dx = pd.read_csv(
        "../data/MCI/MCI_DX_after24months.csv", sep=";")
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}.csv".format(
            database, modality, database))
    df_pred['BPAD'] = np.round(df_pred['Prediction'] - df_pred['Age'], 0)
    merge = dx_merge(df_pred, df_dx)
    sns.boxplot(y='BPAD', x='DX', data=merge, palette='plasma',
                   order=["CN", "MCI", "Dementia"],
                   hue_order=["CN", "MCI", "Dementia"])
    plt.xlabel("Diagnosis after 24 months")
    # plt.ylabel("Percent of whole group")
    plt.savefig(fname="../results/" + database + "/plots/Conversion_" +
                modality + ".png", bbox_inches="tight", dpi=300)
    plt.show()
