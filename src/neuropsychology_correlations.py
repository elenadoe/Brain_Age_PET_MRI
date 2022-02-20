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
from pingouin import corr, partial_corr, mwu
from transform_data import neuropsych_merge, neuropath_merge, dx_merge

warnings.filterwarnings("ignore")


def neuro_correlation(group, age_or_diff, psych_or_path, modality, fold=0):
    """
    Correlations between BPAD and cognitive performance/neuropathology.

    Parameters
    ----------
    group : str
        CN or MCI
    age_or_diff : str
        Brain-predicted age (BPA) or brain-predicted age difference (BPAD)
    psych_or_path : str
        cognitive performance (PSYCH) or neuropathology (PATH)
    modality : str
        PET or MRI

    Returns
    -------
    sign : dict
        r-values of significant correlations

    """
    # for publication, report MCI results for first model
    # put correlations of the four other models' predictions in
    # supplementaries
    if group == "CN":
        add_ = ""
    elif group == "MCI":
        add_ = "_"+str(fold)
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}{}.csv".format(
            group, modality, group, add_))
    # ADNI RID = last 4 digits of ADNI PTID (required for merging)
    df_pred['RID'] = df_pred['PTID'].str[-4:].astype(int)
    y_true = df_pred['Age']
    y_pred = df_pred['Prediction']
    y_diff = y_pred - y_true
    # categorize BPAD into residuals with standard deviation (SD):
    std_diff = np.nanstd(y_diff)/2
    y_diff_cat = ["negative" if x < -std_diff
                  else "neutral" if (-std_diff < x < std_diff)
                  else "positive" for x in y_diff]

    # merge predictions with cognitive performance
    if (psych_or_path == "PSYCH") or (psych_or_path == "psych"):
        df_neuropsych = pd.read_csv(
            "../data/main/UWNPSYCHSUM_12_13_21.csv", sep=";")
        var_ = ['ADNI_MEM', 'ADNI_EF']
        merged = neuropsych_merge(df_pred, df_neuropsych,
                                  var_)

    # merge predictions with neuropathology
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
            group, modality, psych_or_path), index=False)
    print("\033[1m---SIGNIFICANT CORRELATIONS BETWEEN {} ".format(
        age_or_diff) + "& NEURO{}---\033[0m".format(psych_or_path))

    # test significant correlations by correcting for sex
    # type of correlation depending on fulfillment of normality
    # criterion
    sign = {}
    all_ = open(
        "../results/neurocorrelations/" +
        group + "_{}_associations_{}-BPAD.txt".format(psych_or_path,
                                                      modality), "w+")
    all_.write("Variable\tr\tp\tn\tp\tp_Bonferroni\tmethod\n")
    # p Bonferroni-corrected
    p = 0.05
    for n in var_:
        merged[n] = pd.to_numeric(merged[n])
        exc = np.where(np.isnan(merged[n]))[0]
        merged_foranalysis = merged.drop(exc, axis=0)
        y_diff = merged_foranalysis['BPAD']
        # at least three individuals must have information
        # on n
        if len(y_diff) < 3:
            print("Not enough observations of", n)
            all_.write(n + "\tNA\tNA\t<3")
            all_.write("\n")
            continue
        # if data distributin is normal, use pearson correlation
        # if not, use spearman rank correlation
        norm = stats.shapiro(merged_foranalysis[n])
        if norm[1] > 0.05:
            stat = corr(merged_foranalysis['BPAD'], merged_foranalysis[n],
                        method='pearson')
            stat_par = partial_corr(data=merged_foranalysis, x="BPAD",
                                    y=n, covar=["Age", "PTGENDER"],
                                    method="pearson")
            method_str = "Pearson"
        else:
            stat = corr(merged_foranalysis['BPAD'], merged_foranalysis[n],
                        method='spearman')
            stat_par = partial_corr(data=merged_foranalysis, x="BPAD",
                                    y=n, covar=["Age", "PTGENDER"],
                                    method="spearman")
            method_str = "Spearman"

        # if n is significantly associated with BPAD in zero-order
        # correlation or partial correlation,
        # save correlation parameters and plots
        if (stat_par['p-val'][0] < p) or (stat['p-val'][0] < p):
            print("Significant correlation between BPAD and {}: ".format(n),
                  stat['p-val'][0] < p,
                  "\nSignificant correlation between BPAD and", n,
                  "after controlling for the effect of sex and age:",
                  stat_par['p-val'][0] < p)
            sign[n] = [stat, stat_par]
            cm_np = pickle.load(open(
                    "../config/plotting_config_gender_{}.p".format(
                        group), "rb"))
            # sns.set_palette("cm_np")
            merged_foranalysis['PTGENDER'] = \
                ["Female" if x == 1 else "Male" if x == 2
                 else np.nan for x in merged_foranalysis['PTGENDER']]
            sns.lmplot("BPAD", n, data=merged_foranalysis,
                       scatter_kws={'alpha': 0.4, 'color': 'black'},
                       line_kws={'color': 'black'})  # , hue="PTGENDER")
            ymin, ymax = plt.gca().get_ylim()
            xmin, xmax = plt.gca().get_xlim()
            plt.text(0.7*xmax, 0.8*ymax, "n = {}\n".format(stat["n"][0]) +
                     r"$r_{zero-order}$ = " + str(np.round(stat['r'][0], 3)) +
                     r", $p_{zero-order}$ = " +
                     str(np.round(stat['p-val'][0], 3)) + "\n" +
                     r"$r_{partial}$ = " + str(np.round(stat_par['r'][0], 3)) +
                     r"$, p_{partial}$ = " +
                     str(np.round(stat_par['p-val'][0], 3)))
            plt.xlabel("BPAD [years]")
            plt.title(method_str + "-Correlation " + n + " X BPAD")
            plt.savefig(fname="../results/neurocorrelations/plots/" +
                        group + "_" + modality + "_" + age_or_diff +
                        "_" + n + ".png", bbox_inches="tight", dpi=300)
            plt.show()
        all_.write(n + "\t" + str(stat['r'][0]) + "\t" +
                   str(stat['p-val'][0]) + "\t" + str(stat['n'][0]) +
                   "\t" + method_str + "\t" + str(p) +
                   "\t" + str(p/len(var_)) + "\tzero-order")
        all_.write("\n")
        all_.write(n + "\t" + str(stat_par['r'][0]) + "\t" +
                   str(stat_par['p-val'][0]) + "\t" + str(stat_par['n'][0]) +
                   "\t" + method_str + "\t" + str(p) +
                   "\t" + str(p/len(var_)) + "\tpartial")
        all_.write("\n")

    neuropsychology_BPAD_group(merged, sign, y_diff, modality, group)
    all_.close()

    return sign


def neuropsychology_BPAD_group(merged, sign, y_diff, modality, group):
    """
    Assess effects of residual group on BPAD X cognitive performance/pathology.

    Parameters
    ----------
    merged : pd.DataFrame
        merged dataframe containing BPAD, sex and variables of interest
    sign : dict
        dictionary containing variable names of significant (zero-order
        or partial) correlations
    y_diff : pd.Series
        BPAD
    modality : str
        PET or MRI
    group : str
        CN or MCI

    Returns
    -------
    None.

    """
    if group == "CN":
        cm_np = pickle.load(open(
            "../config/plotting_config_np_{}.p".format(
                modality), "rb"))
    elif group == "MCI":
        cm_np = pickle.load(open(
            "../config/plotting_config_np_mci_{}.p".format(
                modality), "rb"))
    sns.set_palette(cm_np)
    print("\n---RESIDUAL CATEGORY GROUP EFFECTS---")
    for k in sign:
        p = 0.05/len(sign)
        exc = np.isnan(merged[k])
        exc_ = np.where(exc)[0]
        merged_foranalysis = merged.drop(exc_, axis=0)

        # get array of indexes for each category
        pos = merged_foranalysis['BPAD Category'] == 'positive'
        neg = merged_foranalysis['BPAD Category'] == 'negative'
        zer = merged_foranalysis['BPAD Category'] == 'neutral'

        # test normality of data for each category
        norm_pos = stats.shapiro(merged_foranalysis[k][pos])
        norm_neg = stats.shapiro(merged_foranalysis[k][neg])
        norm_zer = stats.shapiro(merged_foranalysis[k][zer])

        merged_pos = merged_foranalysis[pos]
        merged_neg = merged_foranalysis[neg]
        merged_zer = merged_foranalysis[zer]

        # choose method of correlation depending on normality
        if (norm_pos[1] > 0.05) and (norm_neg[1]
                                     > 0.05) and (norm_zer[1] > 0.05):
            stat_pos = partial_corr(data=merged_pos, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="pearson")
            stat_neg = partial_corr(data=merged_neg, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="pearson")
            stat_zer = partial_corr(data=merged_zer, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="pearson")
            method_str = "Pearson Correlation"
        else:
            stat_pos = partial_corr(data=merged_pos, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="spearman")
            stat_neg = partial_corr(data=merged_neg, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="spearman")
            stat_zer = partial_corr(data=merged_zer, x="BPAD",
                                    y=k, covar=["Age", "PTGENDER"],
                                    method="spearman")
            method_str = "Spearman Correlation"

        # if there is a significant correlation in one or more
        # residual categories, plot and save plot
        if (stat_pos['p-val'][0] < p) or (stat_neg['p-val'][0]
                                          < p) or (stat_zer['p-val'][0] < p):
            print(k, "significant in positive BPAD: ",
                  stat_pos['p-val'][0] < p, "\n",
                  stat_pos,
                  "\nsignificant in neutral BPAD: ",
                  stat_zer['p-val'][0] < p, "\n",
                  stat_zer,
                  "\nsignificant in negative BPAD: ",
                  stat_neg['p-val'][0] < p, "\n",
                  stat_neg)
            sns.lmplot("BPAD", k, data=merged,
                       scatter_kws={'alpha': 0.4},
                       hue="BPAD Category", palette=cm_np)
            plt.xlabel("BPAD [years]")
            plt.title(method_str + "-Correlation " + k +
                      " X BPAD by BPAD Category")
            plt.savefig(fname="../results/neurocorrelations/plots/" +
                        group + "_" + modality + "_" +
                        "_" + k + "_GROUPEFFECT.png", bbox_inches="tight",
                        dpi=300)
            plt.show()


def conversion_analysis(group, modality):
    """
    Analyze conversion rates.

    Analyze how many participants convert to next stage of AD spectrum
    after 24 months.

    Parameters
    ----------
    group : str
        CN or MCI
    modality : str
        PET or MRI

    Returns
    -------
    None.

    """
    df_dx = pd.read_csv(
        "../data/MCI/MCI_DX_after24months.csv", sep=";")
    df_pred = pd.read_csv(
        "../results/{}/{}-predicted_age_{}_0.csv".format(
            group, modality, group))
    df_pred['BPAD'] = np.round(df_pred['Prediction'] - df_pred['Age'], 0)
    merge = dx_merge(df_pred, df_dx)

    # Mann Whitney U Test
    if group == 'CN':
        sns.boxplot(y='BPAD', x='DX', data=merge[
            merge['DX'].isin(["CN", "MCI"])], color="gray",
            order=["CN", "MCI"])
        mwu_ = mwu(merge['BPAD'][merge['DX'] == 'CN'],
                   merge['BPAD'][merge['DX'] == 'MCI'])
    else:
        sns.boxplot(y='BPAD', x='DX', data=merge[
            merge['DX'].isin(["MCI", "Dementia"])], color="gray",
            order=["MCI", "Dementia"])
        mwu_ = mwu(merge['BPAD'][merge['DX'] == 'MCI'],
                   merge['BPAD'][merge['DX'] == 'Dementia'])
    plt.xlabel("Diagnosis after 24 months")
    plt.savefig(fname="../results/neurocorrelations/plots/Conversion_" +
                modality + ".png", bbox_inches="tight", dpi=300)
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    plt.text(xmax+0.05*xmax, ymax-0.1*ymax, "U = {}, p = {}".format(
        np.round(mwu_['U-val'][0], 3), np.round(mwu_['p-val'][0], 3)))
    plt.show()
    print(mwu_)
