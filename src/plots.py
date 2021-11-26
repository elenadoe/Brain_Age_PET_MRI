import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
import scipy.stats as stats
import pandas as pd
import matplotlib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting, image
from sklearn.metrics import mean_absolute_error, r2_score


def real_vs_pred_2(y_true, y_pred, alg, modality, train_test, database_name,
                   database_list=None, group="CN"):
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
    group : str, optional
        whether cognitively normal (CN) or mild cognitive impairment (MCI)
        was investigated. The default is "CN".

    Returns
    -------
    None.

    """

    y_diff = y_pred - y_true
    cm = matplotlib.cm.get_cmap('PuOr')
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
        cm = matplotlib.cm.get_cmap('PuOr')
        cm_0 = cm(0.2)
        cm_1 = cm(0.8)
        cm_final = np.array([cm_0, cm_1])
        plt.scatter(y_pred, y_true, c=cm_final[y_db_cat])
        print("Purple color representing ADNI, " +
              "orange color representing OASIS")
    else:
        plt.scatter(y_pred, y_true)

    plt.plot([np.min(y_pred), np.max(y_pred)],
             [np.min(y_pred), np.max(y_pred)],
             linestyle="--", color="black", label="CA = PA")
    plt.xlim(np.min(y_pred)-2, np.max(y_pred)+2)
    plt.ylim(np.min(y_pred)-2, np.max(y_pred)+2)
    plt.xlabel('{}-Predicted Age ({})'.format(alg, modality))
    plt.ylabel('Chronological Age')
    plt.legend()
    plt.savefig("../results/{}/plots/real_vs_pred".format(database_name) +
                "_{}_{}_{}_{}.jpg".format(group,
                                          modality,
                                          train_test,
                                          alg),
                bbox_inches='tight')
    plt.show()

    # return evaluation scores
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2_adni = r2_score(np.array(y_true[database_list=='ADNI'], n
                                p.array(y_pred)[database_list=='ADNI'])
    mae_adni = mean_absolute_error(np.array(y_true[database_list=='ADNI'], 
                                            np.array(y_pred)[database_list=='ADNI'])
    r2_oasis = r2_score(np.array(y_true[database_list=='OASIS'], 
                                 np.array(y_pred)[database_list=='OASIS'])
    mae_oasis = mean_absolute_error(y_true[database_list=='OASIS'], 
                                 np.array(y_pred)[database_list=='OASIS'])
    results = open("../results/{}/eval_{}_{}_{}_{}.txt".format(database_name,
                                                               group,
                                                               modality,
                                                               train_test,
                                                               alg), 'w+')
    results.write("MAE\tR2\tME\n" + str(mae) + "\t" +
                  str(r2) + "\t" + str(np.mean(y_diff)))
    print("---", alg, "---")
    print("On average, predicted age of CN differed " +
          "by {} years from their chronological age.".format(np.mean(y_diff)))
    print("MAE = {}, R2 = {}".format(mae, r2))
    print("ADNI:\nMAE = {}, R2 = {}".format(mae_adni, r2_adni))
    print("OASIS:\nMAE = {}, R2 = {}".format(mae_oasis, r2_oasis))


def check_bias(y_true, y_pred, alg, modality, database, corrected=False):
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
    train_test: str indicating whether train or test data is plotted
        (used for saving)
    database: str indicating which database was used

    Returns
    -------
    slope: slope of correlation between true and predicted age
    intercept: intercept of correlation between true and predicted age
    check: a boolean represenation of whether there is a significant
        association between CA and brain-age delta with p < 0.05

    """
    # linear regression between CA and predicted age
    # slope and intercept are needed for bias correction
    linreg_pa_ca = stats.linregress(y_true, y_pred)
    slope = linreg_pa_ca[0]
    intercept = linreg_pa_ca[1]

    # linear regression between brain-age delta and CA
    # to check whether there is a significant correlation
    linreg = stats.linregress(y_pred-y_true, y_true)
    r = linreg[2]
    p = linreg[3]
    check = p < 0.05

    sns.regplot(y_pred-y_true, y_true,
                line_kws={'label': "r = {}, p = {}".format(np.round(r, 2),
                                                           np.round(p, 5))})
    plt.xlabel('True Age [years]')
    plt.ylabel('brain-age delta')
    plt.legend()
    plt.title('Association between brain-age ' +
              'delta and chronological age {}'.format(alg))

    # save figures
    if corrected:
        plt.savefig('../results/{}/bias-corrected_{}_{}.jpg'.format(database,
                                                                    modality,
                                                                    alg))
    else:
        plt.savefig('../results/{}/bias-uncorrected_{}_{}.jpg'.format(database,
                                                                      modality,
                                                                      alg))
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
