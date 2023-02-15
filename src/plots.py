"""
Created on Tue Dec 21 16:27:41 2021.

@author: doeringe
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
import scipy.stats as stats
import pandas as pd
import pickle
import warnings
# import pdb
from nilearn import plotting, image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from nilearn.datasets import fetch_atlas_aal
warnings.filterwarnings("ignore")
np.random.seed(0)
# %%
# matplotlib config
cm_main = pickle.load(open("../config/plotting_config_main.p", "rb"))


def plot_hist(df, group, train_test, modality, database_list, atlas, r,
              y='age'):
    """
    Plot histogram of y.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing age information for all individuals
    train_test : str
        Whether train or test data is predicted
    group : str
        CN or MCI
    modality : str
        PET or MRI
    database_list : list
        Database (ADNI or OASIS) of each individual
    y : str, optional
        Variable of interest. Must be a column name in df.
        The default is 'age'.

    Returns
    -------
    None.

    """
    sns.set_palette(cm_main)
    if (train_test == 'train') or (train_test == "MCI"):
        # plot hist with Ages of train data
        sns.displot(df, x='age', kde=True, color=cm_main[0])
    else:
        sns.displot(df, x='age', kde=True, hue=database_list)
        plt.ylim(0, 60)
    plt.title('Age distribution in {} set'.format(train_test))
    plt.xlabel('Age [years]')
    plt.ylabel('n Participants')
    plt.savefig('../results/{}/{}/plots/{}_{}_{}_age_distribution_{}'.format(
        database_list[0], group, train_test, modality, atlas, r) + '.png',
        bbox_inches="tight")
    plt.show()


def real_vs_pred_2(y_true, y_pred, alg, modality, train_test, group, r,
                   database, atlas, correct_with_CA=True,
                   database_list=None):
    """
    Plot predicted age against chronological age.

    Parameters
    ----------
    y_true : int
        Chronological age
    y_pred : int
        Predicted age
    alg : str
        Algorithm used
    modality : str
        PET or MRI
    train_test : str
        train or test set
    group: str
        CN or MCI

    Returns
    -------
    None.

    """
    sns.set_palette(cm_main)
    y_diff = np.array(y_pred) - np.array(y_true)
    # r2 and mae of whole dataset
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2_adni = np.nan
    mae_adni = np.nan
    r2_oasis = np.nan
    mae_oasis = np.nan
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # if CN test set, color datapoints in scatterplot according to
    # database (ADNI or OASIS) and calcuate r2 and mae for each database
    if train_test == 'test':
        r2_adni = r2_score(np.array(y_true)[
            np.array(database_list) == database],
                           np.array(y_pred)[
                               np.array(database_list) == database])
        mae_adni = mean_absolute_error(
            np.array(y_true)[np.array(database_list) == database],
            np.array(y_pred)[np.array(database_list) == database])
        """r2_oasis = r2_score(
            np.array(y_true)[np.array(database_list) == 'OASIS'],
            np.array(y_pred)[np.array(database_list) == 'OASIS'])
        mae_oasis = mean_absolute_error(
            y_true[np.array(database_list) == 'OASIS'],
            np.array(y_pred)[np.array(database_list) == 'OASIS'])
        y_db_cat = [0 if x == "ADNI" else 1 for x in database_list]"""

        plt.scatter(y_true, y_pred, color=cm_main[0], alpha=0.9)
        plt.fill([60, 60, 95], [60, 95, 95],
                 zorder=0, color='gray', alpha=0.3)
        plt.fill([60, 95, 95], [60, 60, 95],
                 zorder=0, color='gray', alpha=0.1)

        # print test metrics
        """print("Round: {}\n".format(r), "On average, predicted age of", group,
              "differed by ", np.mean(y_diff),
              " years from their chronological age.")
        print("MAE = {}, R2 = {}".format(mae, r2))
        print("ADNI:\nMAE = {}, R2 = {}".format(mae_adni, r2_adni))
        print("OASIS:\nMAE = {}, R2 = {}\n\n".format(mae_oasis, r2_oasis))"""

    # if MCI test set, plot in ADNI color and print test metrics
    else:
        if group == "MCI":
            plt.xlim(50, 100)
            plt.ylim(50, 100)
            plt.scatter(y_true, y_pred, color=cm_main[0], zorder=1)
            plt.fill([50, 50, 100], [50, 100, 100],
                     zorder=0, color=cm_main[0], alpha=0.4)
            plt.fill([50, 100, 100], [50, 50, 100],
                     zorder=0, color=cm_main[0], alpha=0.2)
            """print("Round: {}\n".format(r), "On average, predicted age of",
                  group,
                  "differed by ", np.mean(y_diff),
                  " years from their chronological age.")
            print("MAE = {}, R2 = {}".format(mae, r2))"""

        # if CN train set or OASIS CN set (latter may be shown in
        # paper, therefore use OASIS color)
        else:
            plt.xlim(60, 95)
            plt.ylim(60, 95)
            plt.scatter(y_true, y_pred, color=cm_main[1], zorder=1)
            plt.fill([60, 60, 95], [60, 95, 95],
                     zorder=0, color=cm_main[0], alpha=0.4)
            plt.fill([60, 95, 95], [60, 60, 95],
                     zorder=0, color=cm_main[0], alpha=0.2)
            """print("Round: {}\n".format(r), "On average, predicted age of",
                  group,
                  "differed by ", np.mean(y_diff),
                  " years from their chronological age.")
            print("MAE = {}, R2 = {}".format(mae, r2))"""

    # plot line where chronological age = brain-predicted age
    plt.plot([np.min(y_pred), np.max(y_pred)],
             [np.min(y_pred), np.max(y_pred)],
             linestyle="--", color="black", label="CA = BPA")
    # equalize axes so plot is square
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel('{}-Predicted Age ({})'.format(alg, modality))
    plt.xlabel('Chronological Age [Years]')
    plt.legend()
    plt.savefig("../results/{}/{}/plots/real_vs_pred".format(database, group) +
                "_{}_{}_{}_{}_{}_{}.jpg".format(modality, atlas,
                                             train_test,
                                             alg,
                                             str(correct_with_CA),
                                             r),
                bbox_inches='tight', dpi=300)
    plt.close()

    # save prediction metrics
    results = open("../results/{}/{}/evaluation/eval_{}_{}_{}_{}_{}_{}.txt".format(
        database, group, modality, atlas, train_test, alg,
        str(correct_with_CA), r), 'w+')
    results.write("MAE\tR2\tME\tMAE_ADNI\tR_2ADNI\tMAE_OASIS\tR2_OASIS" +
                  "\n" + str(mae) + "\t" + str(r2) + "\t" +
                  str(np.mean(y_diff)) +
                  "\t" + str(mae_adni) + "\t" + str(r2_adni) +
                  "\t" + str(mae_oasis) + "\t" + str(r2_oasis))
    results.close()


def check_bias(y_true, y_pred, alg, modality, group, r, database, atlas,
               corr_with_CA=True, corrected=False, info=True, save=True):
    """
    Bias check & provision of correction parameters.

    Check whether there is a significant association (= bias)
    between chronological age (CA) and brain-age delta.

    Parameters
    ----------
    y_true : list
        Chronological age
    y_pred : list
        Predicted age
    alg : str
        algorithm used for current task (used for saving)
    modality : str
        image modality used (MRI/PET; used for saving)
    group : str
        CN or MCI
    corr_with_CA : boolean, optional
        whether to use (True) chronological age for brain age
        correction. Default is False.
    corrected : boolean, optional
        whether y_pred is corrected (True) or not
        Default is False.
    info : boolean, optional
        whether or not to create and save plots
        Default is True.
    save : boolean, optional
        DESCRIPTION

    Returns
    -------
    slope: slope of correlation between true and predicted age
    intercept: intercept of correlation between true and predicted age
    check: a boolean represenation of whether there is a significant
        association between CA and brain-age delta with p < 0.05

    """
    sns.set_palette(cm_main)
    linreg = LinearRegression()

    if corr_with_CA is None:
        slope_ = np.nan
        intercept_ = np.nan
        y_pred_bc = y_pred

    elif corr_with_CA:
        # linear regression between CA and age delta
        # source: Beheshti et al.
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6861562/
        y_diff = np.array(y_pred) - np.array(y_true)
        linreg.fit(np.array(y_true).reshape(-1, 1), y_diff)
        slope_ = linreg.coef_[0]
        intercept_ = linreg.intercept_
        y_pred_bc = y_pred - (slope_*y_true + intercept_)

    else:
        # linear regression between CA and predicted age
        # source: https://github.com/james-cole/UKBiobank-Brain-Age/
        # blob/master/ukb_multimodal_brain_age_lasso.Rmd
        linreg.fit(np.array(y_true).reshape(-1, 1), y_pred)
        slope_ = linreg.coef_[0]
        intercept_ = linreg.intercept_
        y_pred_bc = (y_pred - intercept_)/slope_

    # calculate statistical linear regression to obtain r and p-value
    # if uncorrected, y_pred_bc is equal to y_pred
    y_diff = np.array(y_pred) - np.array(y_true)
    linreg_plotting = stats.linregress(y_true, y_diff)
    r_plotting = linreg_plotting[2]
    p_plotting = linreg_plotting[3]
    check = p_plotting < 0.05
    y_diff_bc = np.array(y_pred_bc) - np.array(y_true)
    linreg_plotting_bc = stats.linregress(y_true, y_diff_bc)
    r_plotting_bc = linreg_plotting_bc[2]
    p_plotting_bc = linreg_plotting_bc[3]

    # plot bias between chronological age and brain age delta
    if info:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10))
        sns.regplot(y_diff, y_true,
                    line_kws={'label': "r = {}, p = {}".format(np.round(
                        r_plotting, 2),
                        np.round(
                        p_plotting, 5))}, ax=ax[0])
        sns.regplot(y_diff_bc, y_true,
                    line_kws={'label': "r = {}, p = {}".format(np.round(
                        r_plotting_bc, 2),
                        np.round(
                        p_plotting_bc, 5))}, ax=ax[1])
        plt.ylabel('Chronological Age [years]')
        ax[0].set_xlabel('uncorrected BPAD [years]')
        ax[1].set_xlabel('corrected BPAD [years]')
        plt.legend()
        plt.title('Association between brain-age ' +
                  'delta and chronological age {}'.format(alg))

        if save:
            plt.savefig(
                '../results/{}/{}/bias-corrected_{}_{}_{}_{}_{}.jpg'.format(
                database, group, corr_with_CA, modality, atlas, alg, r),
                dpi=300)

        plt.close()

    return slope_, intercept_, check


def feature_imp(df_test, col, final_model, final_model_name,
                modality, atlas, r, y='age', rand_seed=0):
    """
    Plot feature importance.

    Feature importance = weights assigned to features
    (coef_ argument of regression algorithms of shape (n_features, n_classes),
     here: (216,1)
     https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

    Parameters  # TODO
    ----------
    feature_imp: dictionary-like object
        from calling sklearn.inspection.permutation_importance
    final_model_name : str
        DESCRIPTION
    modality: str
        modality with which brain age was assessed (MRI/PET; used for saving)

    Returns
    -------
    none (plots and saves plots)
    """
    # get feature importance for linear kernels
    # (coefficients are not available for other kernels,
    # best kernel is svm most of the times)
    if final_model_name != 'svm':
        print("No coefficients available for RVR.")
    else:
        if final_model.named_steps[final_model_name].kernel == "linear":
    
            imp = final_model.named_steps[final_model_name].coef_[0]
    
            # get labels
            if atlas.startswith("Sch_Tian"):
                # get atlas
                atlas_file = '../data/0_ATLAS/schaefer200-17_Tian.nii'
                atlas_matrix = image.get_data(atlas_file)
                labels = open('../data/0_ATLAS/composite_atlas_labels.txt')
                labels = labels.read().split('\n')[:-1]
            elif atlas.startswith("AAL3"):
                atlas_file = '../data/0_ATLAS/AAL3v1_1mm.nii'
                
                labels = open('../data/0_ATLAS/AAL3v1_1mm.nii.txt')
                labels = labels.read().split('\n')[:-1]
                # remove labels that were redefined in AAL3 and left empty for comparability
                labels = [i for i in labels if i not in ['35 Cingulate_Ant_L ',
                                                         '36 Cingulate_Ant_R ',
                                                         '81 Thalamus_L ',
                                                         '82 Thalamus_R ']]
                # remove numbers from names, only keep region definition
                labels = [x.split()[1] for x in labels]
            elif atlas.startswith("AAL1"):
                atlas_file = '../data/0_ATLAS/AAL1_TPMcropped.nii'
                labels = fetch_atlas_aal().labels

            atlas_file = image.load_img(atlas_file)
            atlas_matrix = image.get_data(atlas_file)

            # put feature importance into dataframe and save
            df_imp = pd.DataFrame({'region': labels,
                                   'feat_importance': imp})
            df_imp.to_csv('../results/ADNI/CN/evaluation/' +
                          'weighted_importance_{}_{}_{}_{}.csv'.format(
                              modality, atlas, final_model_name, r))
    
            # create statistical map where each voxel valueo
            # coresponds to weight coefficients
            atlas_matrix_stat = np.around(atlas_matrix).copy().astype(int)
            label_nums = sorted(list(set(atlas_matrix_stat.flatten())))
    
            # 0 is background in atlas matrix
            # but first element (index = 0) of imp is
            # weight of first actual feature
            for x in range(1, len(label_nums)):
                tmp = label_nums[x]
                atlas_matrix_stat[atlas_matrix_stat == tmp] = imp[x-1]
            # create niimg from atlas_matrix
            atlas_final = image.new_img_like(atlas_file, atlas_matrix_stat)
    
            # plot feature importance, save output plot and niimg
            plotting.plot_stat_map(atlas_final)
            plt.title("{}-relevant regions for aging".format(final_model_name))
            plt.savefig("../results/ADNI/" +
                        "CN/evaluation/weighted_importance_{}_{}_{}_{}.jpg".format(
                            modality, atlas, final_model_name, r))
            nib.save(atlas_final, "../results/ADNI/CN/evaluation" +
                     "/feature_importance_{}_{}_{}_{}.nii".format(
                         modality, atlas, final_model_name, r))
            plt.show()
            plt.close()
        else:
            print("Kernel not linear, no weight coefficients available.")
