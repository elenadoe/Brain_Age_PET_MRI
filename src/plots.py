import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting
from nilearn import image
from sklearn.metrics import mean_absolute_error, r2_score


# plot ground truth against predictions
def real_vs_pred(y_true, y_pred, alg, modality, train_test, database):
    """Plots True labels against the predicted ones.
    inputs:
    y_true: list of floating point values or integers, representing ground
        truth values
    y_pred: list of floating point/integers values, representing predictions
    alg: algorithm used for current task (used for saving)
    modality: image modality used (MRI/PET; used for saving)
    train_test: str indicating whether train or test data is plotted
        (used for saving)
    database: str indicating which database was used
    outputs: none (plots and saves plots)
    """
    mae = format(mean_absolute_error(y_true, y_pred), '.2f')
    corr = format(r2_score(y_true, y_pred), '.2f')

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    sns.regplot(y_true, y_pred, ax = ax)
    ax.set_xlim(np.min(y_true)-1, 
                np.max(y_true)+1)
    ax.set_ylim(np.min(y_true)-1, 
                np.max(y_true)+1)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
    ax.set(xlabel='True values', ylabel='Predicted values')
    plt.title('Actual vs Predicted {}'.format(alg))
    plt.text(xmin + 10, ymax - 0.01 * ymax, text, verticalalignment='top',
             horizontalalignment='right', fontsize=12)
    plt.savefig("../results/"+database+
                "/plots/real_vs_pred_{}_{}_{}.jpg".format(
        train_test, modality, alg))
    plt.show()


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
    print("Most important regions: {}".format(
        np.array(labels)[np.where(feature_imp.importances_mean>1e-02)]))
    
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

    plotting.plot_stat_map(atlas_final, threshold = 0)
    plt.title("{}-relevant regions for aging".format(alg))
    plt.savefig("../results/"+database+
                "/Permutation_importance_{}_{}.jpg".format(
        modality, alg))
    nib.save(atlas_final, "../results/"+database+
             "/permutation_importance_{}_{}.nii".format(
        modality, alg))
    plt.show()
