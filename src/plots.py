import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting
from nilearn import image
from nilearn import plotting
from nilearn import image
from sklearn.metrics import mean_absolute_error

# plot ground truth against predictions
def real_vs_pred(y_true, y_pred, alg, modality, train_test):
    """Plots True labels against the predicted ones.
    inputs: 
    y_true = list of floating point values or integers, representing ground truth values
    y_pred = list of floating point values or integers, representing predicted values
    alg = algorithm used for current task (used for saving)
    modality = modality with which brain age was assessed (MRI/PET; used for saving)
    train_test = str indicating whether train or test data is plotted (used for saving)
    
    outputs: 
    none (plots and saves plots)
    """
    mae = format(mean_absolute_error(y_true, y_pred), '.2f')
    corr = format(np.corrcoef(y_pred, y_true)[1, 0], '.2f')

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.scatter(y_true, y_pred)
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
    ax.set(xlabel='True values', ylabel='Predicted values')
    plt.title('Actual vs Predicted')
    plt.text(xmin + 10, ymax - 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='right', fontsize=12)
    plt.savefig("../results/real_vs_pred_{}_{}_{}.jpg".format(train_test, modality, alg))
    plt.show()

# plot permutation importance
def permutation_imp(feature_imp, alg, modality):
    """Plots permutation importance as evaluated in test set
    inputs: 
    feature_imp = dictionary-like object from calling sklearn.inspection.permutation_importance
    alg = algorithm used for current task (used for saving)
    modality = modality with which brain age was assessed (MRI/PET; used for saving)
        
    outputs: 
    none (plots and saves plots)
    """
    schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
    atlas = image.load_img(schaefer.maps)
    atlas_matrix = image.get_data(atlas)

    # create statistical map where each voxel value coresponds to permutation importance
    imp = feature_imp.importances_mean
    atlas_matrix_stat = atlas_matrix.copy()

    for x in range(201):
        if x == 0:
            pass
        else:
            atlas_matrix_stat[atlas_matrix_stat == x] = imp[x-1]
    atlas_final = image.new_img_like(atlas, atlas_matrix_stat)

    plotting.plot_stat_map(atlas_final)
    plt.title("{}-relevant regions for aging".format(alg))
    plt.savefig("../results/Permutation_importance_{}_{}.jpg".format(modality, alg))
    nib.save(atlas_final,"../results/permutation_importance_{}_{}.nii".format(modality, alg))
    plt.show()