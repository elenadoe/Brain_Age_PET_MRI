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
    std_diff = np.nanstd(y_diff)/2
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

    merged["BPAD"] = y_diff
    merged["BPAD Category"] = y_diff_cat
    merged.to_csv("../results/pred_merged_{}_{}_{}.csv".format(
            database, modality, psych_or_path), index=False)
    print("\033[1m---SIGNIFICANT CORRELATIONS BETWEEN {} ".format(
        age_or_diff) + "& NEURO{}---\033[0m".format(psych_or_path))

    sign = {}
    all_ = open(
        "../results/{}/associations/{}_associations_{}-BPAD.txt".format(
            database, psych_or_path, modality), "w+")
    all_.write("Variable\tr\tp\tn\n")
    p = 0.05/len(var_)
    for n in var_:
        # TODO: change to partial correlation pingouin, print sample size
        merged[n] = pd.to_numeric(merged[n])
        exc = np.where(np.isnan(merged[n]))[0]
        merged_foranalysis = merged.drop(exc, axis=0)
        y_diff = merged_foranalysis['BPAD']
        if len(y_diff) < 3:
            print("Not enough observations of", n)
            all_.write(n + "\tNA\tNA\t<3")
            all_.write("\n")
            continue
        if age_or_diff == "BPAD":
            norm = stats.shapiro(merged_foranalysis[n])
            if norm[1] > 0.05:
                stat = partial_corr(data=merged_foranalysis, x="BPAD",
                                    y=n, covar="PTGENDER",
                                    method="pearson")
                method_str = "Pearson"
            else:
                stat = partial_corr(data=merged_foranalysis, x="BPAD",
                                    y=n, covar="PTGENDER",
                                    method="spearman")
                method_str = "Spearman"
            if stat['p-val'][0] < p:
                sign[n] = stat
                slope, intercept = np.polyfit(y_diff, merged_foranalysis[n],
                                              1)
                cm_np = pickle.load(open(
                        "../config/plotting_config_gender_{}.p".format(
                            database), "rb"))
                sns.set_palette(cm_np)
                merged_foranalysis['PTGENDER'] = \
                    ["Female" if x == 1 else "Male" if x == 2
                     else np.nan for x in merged_foranalysis['PTGENDER']]
                sns.lmplot("BPAD", n, data=merged_foranalysis,
                           scatter_kws={'alpha': 0.4}, hue="PTGENDER")
                ymin, ymax = plt.gca().get_ylim()
                xmin, xmax = plt.gca().get_xlim()
                plt.text(0.7*xmax, 0.9*ymax, "n = {}".format(stat["n"][0]))
                plt.xlabel("BPAD [years]")
                plt.title(method_str + "-Correlation " + n + " X BPAD")
                plt.savefig(fname="../results/" + database + "/plots/" +
                            modality + "_" + age_or_diff +
                            "_" + n + ".png", bbox_inches="tight", dpi=300)
                plt.show()
        all_.write(n + "\t" + str(stat['r'][0]) + "\t" +
                   str(stat['p-val'][0]) + "\t" + str(stat['n'][0]))
        all_.write("\n")

    for key in sign:
        print(key, ":", np.round(sign[key]["r"][0], 3), sign[key]["p-val"][0])

    neuropsychology_BPAD_interaction(merged, sign, y_diff, modality, database)
    all_.close()

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
        exc_ = np.where(exc)[0]
        merged_foranalysis = merged.drop(exc_, axis=0)

        pos = merged_foranalysis['BPAD Category'] == 'positive'
        neg = merged_foranalysis['BPAD Category'] == 'negative'
        zer = merged_foranalysis['BPAD Category'] == 'neutral'

        """pos_bool = np.array(~exc) & np.array(pos)
        neg_bool = np.array(~exc) & np.array(neg)
        zer_bool = np.array(~exc) & np.array(zer)"""
        norm_pos = stats.shapiro(merged_foranalysis[k][pos])
        norm_neg = stats.shapiro(merged_foranalysis[k][neg])
        norm_zer = stats.shapiro(merged_foranalysis[k][zer])

        merged_pos = merged_foranalysis[pos]
        merged_neg = merged_foranalysis[neg]
        merged_zer = merged_foranalysis[zer]
        if (norm_pos[1] > 0.05) and (norm_neg[1]
                                     > 0.05) and (norm_zer[1] > 0.05):
            stat_pos = partial_corr(data=merged_pos, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="pearson")
            stat_neg = partial_corr(data=merged_neg, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="pearson")
            stat_zer = partial_corr(data=merged_zer, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="pearson")
            method_str = "Pearson Correlation"
        else:
            stat_pos = partial_corr(data=merged_pos, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="spearman")
            stat_neg = partial_corr(data=merged_neg, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="spearman")
            stat_zer = partial_corr(data=merged_zer, x="BPAD",
                                    y=k, covar="PTGENDER",
                                    method="spearman")
            method_str = "Spearman Correlation"
        print(k, "significant in positive BPAD: ",
              stat_pos['p-val'][0] < p, "\n",
              stat_pos,
              "\nsignificant in neutral BPAD: ",
              stat_zer['p-val'][0] < p, "\n",
              stat_zer,
              "\nsignificant in negative BPAD: ",
              stat_neg['p-val'][0] < p, "\n",
              stat_neg)
        if (stat_pos['p-val'][0] < p) or (stat_neg['p-val'][0]
                                       < p) or (stat_zer['p-val'][0] < p):
            sns.lmplot("BPAD", k, data=merged,
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
    sns.boxplot(y='BPAD', x='DX', data=merge[
        merge['DX'].isin([database, "MCI", "Dementia"])], color="gray",
        order=set([database, "MCI", "Dementia"]))
    plt.xlabel("Diagnosis after 24 months")
    # plt.ylabel("Percent of whole group")
    plt.savefig(fname="../results/" + database + "/plots/Conversion_" +
                modality + ".png", bbox_inches="tight", dpi=300)
    plt.show()
