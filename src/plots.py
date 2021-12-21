import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
import scipy.stats as stats
import pandas as pd
import pickle
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting, image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# %%
# matplotlib config
cm = pickle.load(open("../data/config/plotting_config.p", "rb"))


def plot_hist(df_train, train_test, modality, database_list, y='age'):
    if train_test == 'train':
        # plot hist with Ages of train data
        sns.displot(df_train, x='age', kde=True, color=cm[0])
    else:
        sns.displot(df_train, x='age', kde=True, hue=database_list,
                    color=cm)
        plt.ylim(0, 70)
        plt.title('Age distribution in {} set'.format(train_test))
        plt.xlabel('Age [years]')
        plt.ylabel('n Participants')
        plt.savefig('../results/{}/plots/{}_age_distribution'.format(
            train_test, modality) + '.png', bbox_inches="tight")
        plt.show()


def real_vs_pred_2(y_true, y_pred, alg, modality, train_test, database_name,
                   database_list=None):
    """
    plots predicted age against chronological age

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
    database_name : str
        the database used

    Returns
    -------
    None.

    """

    y_diff = y_pred - y_true
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mae_adni = np.nan
    r2_adni = np.nan
    mae_oasis = np.nan
    r2_oasis = np.nan

    print("---", alg, "---")
    print("On average, predicted age of {} differed ".format(database_name) +
          "by {} years from their chronological age.".format(np.mean(y_diff)))
    print("MAE = {}, R2 = {}".format(mae, r2))
    # uncomment if coloring in scatterplot is supposed to be
    # depending on CA-PA
    """
    y_diff = np.round(y_diff,0)
    y_diff_cat = [0 if x < 0 else 1 if x == 0 else 2 for x in y_diff]
    # y_diff_label = ['PA-CA negative', 'PA = CA', 'PA-CA positive']
    cm_neg = cm(0.2)
    cm_0 = 'black'
    cm_pos = cm(0.8)
    cm_final = np.array([cm_neg, cm_0, cm_pos])
    plt.scatter(y_pred, y_true, c=cm_final[y_diff_cat])"""
    if train_test == 'test':
        y_db_cat = [0 if x == "ADNI" else 1 for x in database_list]
        plt.scatter(y_true, y_pred, c=cm[y_db_cat], alpha=0.8)
        plt.fill([60, 60, 90], [60, 90, 90],
                 zorder=0, color='gray', alpha=0.3)
        plt.fill([60, 90, 90], [60, 60, 90],
                 zorder=0, color='gray', alpha=0.1)
        print("Orange color representing ADNI, " +
              "purple color representing OASIS")

        r2_oasis = r2_score(
            np.array(y_true)[np.array(database_list) == 'OASIS'],
            np.array(y_pred)[np.array(database_list) == 'OASIS'])
        mae_oasis = mean_absolute_error(
            y_true[np.array(database_list) == 'OASIS'],
            np.array(y_pred)[np.array(database_list) == 'OASIS'])
        print("OASIS:\nMAE = {}, R2 = {}".format(mae_oasis, r2_oasis))

        # return evaluation scores
        r2_adni = r2_score(np.array(y_true)[np.array(database_list) == 'ADNI'],
                           np.array(y_pred)[np.array(database_list) == 'ADNI'])
        mae_adni = mean_absolute_error(
            np.array(y_true)[np.array(database_list) == 'ADNI'],
            np.array(y_pred)[np.array(database_list) == 'ADNI'])
        print("ADNI:\nMAE = {}, R2 = {}".format(mae_adni, r2_adni))
    else:
        if database_name == "2_MCI_ADNI":
            plt.xlim(55, 95)
            plt.ylim(55, 95)
            plt.scatter(y_true, y_pred, color=cm[0], zorder=1)
            plt.fill([55, 55, 95], [55, 95, 95],
                     zorder=0, color=cm[0], alpha=0.4)
            plt.fill([55, 95, 95], [55, 55, 95],
                     zorder=0, color=cm[0], alpha=0.2)
        else:
            plt.xlim(60, 90)
            plt.ylim(60, 90)
            plt.scatter(y_true, y_pred, color=cm[0], zorder=1)
            plt.fill([60, 60, 90], [60, 90, 90],
                     zorder=0, color=cm[0], alpha=0.4)
            plt.fill([60, 90, 90], [60, 60, 90],
                     zorder=0, color=cm[0], alpha=0.2)

        database_list = ['ADNI']*np.array(y_true).shape[0]

    plt.plot([np.min(y_pred), np.max(y_pred)],
             [np.min(y_pred), np.max(y_pred)],
             linestyle="--", color="black", label="CA = PA")

    plt.ylabel('{}-Predicted Age ({})'.format(alg, modality))
    plt.xlabel('Chronological Age [Years]')
    plt.legend()
    plt.savefig("../results/{}/plots/real_vs_pred".format(database_name) +
                "_{}_{}_{}.jpg".format(modality,
                                       train_test,
                                       alg),
                bbox_inches='tight', dpi=300)
    plt.show()

    results = open("../results/{}/eval_{}_{}_{}.txt".format(database_name,
                                                            modality,
                                                            train_test,
                                                            alg), 'w+')
    results.write("MAE\tR2\tME\tMAE_ADNI\tR_2ADNI\tMAE_OASIS\tR2_OASIS" + "\n"
                  + str(mae) + "\t" + str(r2) + "\t" + str(np.mean(y_diff))
                  + "\t" + str(mae_adni) + "\t" + str(r2_adni)
                  + "\t" + str(mae_oasis) + "\t" + str(r2_oasis))


def check_bias(y_true, y_pred, alg, modality, database,
               corr_with_CA=False, corrected=False):
    """
    checks whether there is a significant association (= bias)
    between chronological age (CA) and brain-age delta

    Parameters
    ----------
    y_true: list of floating point values or integers, representing ground
        truth values
    y_pred: list of floating point/integers values, representing predictions
    alg: algorithm used for current task (used for saving)
    modality: image modality used (MRI/PET; used for saving)
    database: str indicating which database was used
    corr_with_CA: boolean indicating whether to use (True) chronological
        age for brain age correction. Default is False.
    corrected: boolean indicating whether y_pred is corrected (True) or not.
        Default is False.

    Returns
    -------
    slope: slope of correlation between true and predicted age
    intercept: intercept of correlation between true and predicted age
    check: a boolean represenation of whether there is a significant
        association between CA and brain-age delta with p < 0.05

    """
    y_diff = y_pred-y_true
    linreg = LinearRegression()
    if corr_with_CA:
        # linear regression between CA and age delta
        # slope and intercept are needed for bias correction
        # source: Population-based neuroimaging reveals traces of childbirth
        linreg.fit(np.array(y_true).reshape(-1, 1), y_diff)
        slope = linreg.coef_[0]
        intercept = linreg.intercept_

    else:
        # linear regression between CA and predicted age
        linreg.fit(np.array(y_true).reshape(-1, 1), y_pred)
        slope = linreg.coef_[0]
        intercept = linreg.intercept_

    linreg_plotting = stats.linregress(y_true, y_diff)
    r_plotting = linreg_plotting[2]
    p_plotting = linreg_plotting[3]
    check = p_plotting < 0.05
    sns.regplot(y_diff, y_true,
                line_kws={'label': "r = {}, p = {}".format(np.round(
                                                    r_plotting, 2),
                                                           np.round(
                                                    p_plotting, 5))})
    plt.ylabel('Chronological Age [years]')
    plt.xlabel('BPAD [years]')
    plt.legend()
    plt.title('Association between brain-age ' +
              'delta and chronological age {}'.format(alg))

    # save figures
    if corrected:
        plt.savefig('../results/{}/bias-corrected_{}_{}.jpg'.format(database,
                                                                    modality,
                                                                    alg),
                    dpi=300)
    else:
        plt.savefig('../results/{}/bias-uncorrected_{}_{}.jpg'.format(database,
                                                                      modality,
                                                                      alg),
                    dpi=300)
    plt.show()
    return slope, intercept, check


# plot permutation importance
def permutation_imp(feature_imp, alg, modality, database):
    """Plots permutation importance as evaluated in test set
    inputs:
    feature_imp: dictionary-like object from calling
        sklearn.inspection.permutation_importance
    alg : string of algorithm used for current task (used for saving)
    modality: str representing the modality with which brain age was
        assessed (MRI/PET; used for saving)
    database: str indicating which database was used

    outputs: none (plots and saves plots)
    """
    schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
    text_file = open('../data/Tian_Subcortex_S1_3T_label.txt')
    labels = text_file.read().split('\n')
    labels = np.append(schaefer['labels'], np.array(labels[:-1]))
    df_imp = pd.DataFrame({'region': labels,
                          'perm_importance': feature_imp.importances_mean})
    df_imp.to_csv('../results/{}/'.format(database) +
                  'permutation_importance_{}_{}.csv'.format(modality, alg))

    atlas = '../data/schaefer200-17_Tian.nii'
    atlas = image.load_img(atlas)
    atlas_matrix = image.get_data(atlas)

    # create statistical map where each voxel value coresponds to permutation
    # importance
    imp = feature_imp.importances_mean
    atlas_matrix_stat = atlas_matrix.copy()

    for x in range(217):
        if x == 0:
            pass
        else:
            atlas_matrix_stat[atlas_matrix_stat == x] = imp[x-1]
    atlas_final = image.new_img_like(atlas, atlas_matrix_stat)

    plotting.plot_stat_map(atlas_final)
    plotting.view_img_on_surf(atlas_final, threshold="90%")
    plt.title("{}-relevant regions for aging".format(alg))
    plt.savefig("../results/" + database +
                "/Permutation_importance_{}_{}.jpg".format(modality, alg))
    nib.save(atlas_final, "../results/" + database +
             "/permutation_importance_{}_{}.nii".format(modality, alg))
    plt.show()
