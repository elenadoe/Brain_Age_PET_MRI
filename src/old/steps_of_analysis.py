"""
Created on Tue Dec 21 16:27:41 2021.

@author: doeringe
"""

from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from skrvm import RVR
from os.path import exists
import pdb

from transform_data import split_data, outlier_check_other
import plots

import numpy as np
import pandas as pd
import pickle
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")
np.random.seed(0)


def cross_validate(df_train, col, models, model_params, splits, scoring,
                   rand_seed, y='age', info=False):
    """
    Cross-validation.

    Apply cross-validation on training data using models and parameters
    provided by user.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataframe containing input and output variables
    col : list or np.ndarray
        Column(s) to be used as input features
    models : list
        List of algorithms to assess
    model_params : dict
        Dictionary of hyperparameters to assess
    splits : int
        How many folds to split the data into for cross-validation
    scoring : str or list
        Metrics to assess
    rand_seed : int, optional
        Random state to use during all fitting procedures, where applicable
    y : pd.Series, optional
        Column to be used as output variable
    info : boolean, optional
        Whether to print status updates

    Returns
    -------
    model_results : list
        Best fitted estimator per algorithm
    scores : list
        Average performance during cross-validation of the algorithm
    results : dict
        Dictionary containing true and cross-validated predicted age

    """
    if info:
        print("Cross-validating...")
    model_results = []
    scores_results = []
    results = {}
    results['RVR()'] = {}
    results['svm'] = {}

    # scaler = 'scaler_robust'
    scaler = 'scaler_minmax'

    for i, (model, params) in enumerate(zip(models, model_params)):
        # split data using age-bins

        cv = StratifiedKFold(n_splits=splits,
                             random_state=rand_seed,
                             shuffle=True).split(df_train[col],
                                                 df_train['Ageb'])
        cv = list(cv)

        # run julearn function
        scores, final_model = run_cross_validation(X=col, y=y,
                                                   preprocess_X=scaler,
                                                   problem_type='regression',
                                                   data=df_train,
                                                   model=model, cv=cv,
                                                   seed=rand_seed,
                                                   model_params=params,
                                                   return_estimator='all',
                                                   scoring=scoring)
        model_results.append(final_model.best_estimator_)
        scores_results.append(scores)

        # save cross-validated predictions
        N = len(df_train)
        results[str(model)]['pred'] = N * [0]
        results[str(model)]['true'] = N * [0]
        for iter in range(splits):
            valid_ind = cv[iter][1]
            pred = scores.estimator[iter].predict(
                   df_train.iloc[valid_ind][col])

            for i, iv in enumerate(valid_ind):
                results[str(model)]['pred'][iv] = pred[i]
                results[str(model)]['true'][iv] = df_train.iloc[iv]['age']
    if info:
        print("done.")

    return model_results, scores, results


# BIAS CORRECTION
# Eliminate linear correlation of brain age delta and chronological age
def bias_correct(results, df_train, col, model_results, model_names,
                 modality, atlas, group, r, database='ADNI', y='age',
                 correct_with_CA=True, info_init=False, save=True):
    """
    Correct for bias between CA and BPA.

    Parameters
    ----------
    results : dict
        Dictionary containing true and cross-validated predicted age
        from cross_validate
    df_train : pd.DataFrame
        Dataframe containing input and output variables
    y_pred_uncorr : list, np.ndarray or pd.Series
        Predicted, uncorrected age
    model_names : list
        List of strings naming models to assess
    modality : str
        MRI or PET
    group : str
        CN or MCI
    y : str, optional
        Column to be considered as output feature. The default is age.
    correct_with_CA : boolean, optional
        Whether or not to correct bias with chronological age.
        The default is True.
    info_init : boolean, optional
        whether or not to create and save plots. The default is True.
    save : boolean, optional
        Whether final model should be saved

    Returns
    -------
    final_model : str
        Name of final model
    pred_param : dict
        dictionary containing bias-corrected age, slope, intercept,
        and pearson r value of bias between BPAD and CA

    """
    # get true and predicted age from df_ages
    # y_true is the same for rvr and svr (validated)
    y_true = np.array(results['RVR()']['true'])
    y_pred_rvr = results['RVR()']['pred']
    y_pred_svr = results['svm']['pred']
    y_pred = [y_pred_rvr, y_pred_svr]

    # save predictions with and without bias correction in dictionary
    predictions = {}
    predictions['name'] = df_train['name'].values
    pred_param = {}
    pred_param['withCA'] = [str(correct_with_CA)]

    # get bias-correction parameters
    for y in range(len(y_pred)):
        predictions[model_names[y] + '_uncorr'] = y_pred[y]
        check_bias = plots.check_bias(y_true,
                                      y_pred[y],
                                      model_names[y],
                                      modality,
                                      group,
                                      database=database,
                                      atlas=atlas,
                                      r=r,
                                      corr_with_CA=correct_with_CA,
                                      info=info_init,
                                      save=save)
        slope_ = check_bias[0]
        intercept_ = check_bias[1]
        check_ = check_bias[2]

        if info_init:
            print("Significant association between ", model_names[y],
                  "-predicted age delta and CA:",
                  check_)
        # apply desired bias-correction
        if correct_with_CA is None:
            # no bias correction
            bc = y_pred[y]
            predictions[model_names[y] + '_bc'] = bc
        elif correct_with_CA:
            # bias correction WITH chronological age
            bc = y_pred[y] - (slope_*y_true + intercept_)
            predictions[model_names[y] + '_bc'] = bc
        else:
            # bias correction WITHOUT chronological age
            bc = (y_pred[y] - intercept_)/slope_
            predictions[model_names[y] + '_bc'] = bc

        r2_corr = r2_score(y_true, bc)
        mae_corr = mean_absolute_error(y_true, bc)

        # save bias-correction parameters and metrics
        pred_param[model_names[y] + '_slope'] = [slope_]
        pred_param[model_names[y] + '_intercept'] = [intercept_]
        pred_param[model_names[y] + '_check'] = [check_]
        pred_param[model_names[y] + '_r2'] = [r2_corr]
        pred_param[model_names[y] + '_mae'] = [mae_corr]
        r2_uncorr = r2_score(y_true, y_pred[y])
        mae_uncorr = mean_absolute_error(y_true, y_pred[y])
        pred_param[model_names[y] + '_rsq_uncorr'] = [r2_uncorr]
        pred_param[model_names[y] + '_ma_uncorr'] = [mae_uncorr]

        if save:
            """pickle.dump(pred_param, open(
            "../results/0_FINAL_MODELS/models_and_params_{}_{}_{}_{}.p".format(
                modality, atlas, correct_with_CA, r), "wb"))
            pickle.dump(predictions, open(
                "../results/{}/{}/cross-val_pred_{}_{}_{}_{}.p".format(
                    database, group, modality,
                    atlas, correct_with_CA, r), "wb"))"""

            df = pd.DataFrame(pred_param)
            df.to_csv(
                "../results/0_FINAL_MODELS/" +
                "models_and_params_{}_{}_{}_{}.csv".format(
                    modality, atlas, correct_with_CA, r))

    # compare predictions to find final model
    final_model, final_mae, final_r2 = find_final_model(pred_param,
                                                        model_names,
                                                        modality,
                                                        info_init=info_init)

    # scatterplot of bias-corrected results from cross-validation
    if info_init:
        plots.real_vs_pred_2(y_true, bc, final_model, modality,
                             database='ADNI', atlas=atlas, train_test='train',
                             group='CN', r=r)

    return final_model, pred_param


def find_final_model(pred_param, model_names, modality,
                     correct_with_CA=True, info_init=False):
    """
    Compare transition models to find final model.

    The transition model with the smallest mean absolute error (MAE)
    on uncorrected brain age prediction against chronological age
    is the final model.

    Parameters
    ----------
    pred_param : dict
        dictionary containing bias-corrected age, slope, intercept,
        and pearson r value of bias between BPAD and CA for best rvr and svr
    model_names : list
        List of strings naming models to assess
    info : boolean, optional
        whether or not to create and save plots. The default is True.
    save : boolean, optional
        Whether final model should be saved

    Returns
    -------
    final_model : str
        Name of final model
    final_mae : float
        Mean absolute error of final model
    final_r2 : float
        R squared of final model

    """
    # find index (0 or 1 for rvr or svr) of smallest mae after bias
    # correction --> if smallest mae before bias correction is desired,
    # change to _ma_uncorr
    final_model_idx = np.argmin([v for k, v in pred_param.items()
                                 if '_mae' in k])
    final_r2 = [v for k, v in pred_param.items()
                if '_r2' in k][final_model_idx]
    final_mae = [v for k, v in pred_param.items()
                 if '_mae' in k][final_model_idx]
    final_model = model_names[final_model_idx]

    if info_init:
        print("-\033[1m--CROSS-VALIDATION---\n",
              "Final model (smallest MAE): {}\nMAE: {}, R2: {}\033[0m".format(
                  final_model, final_mae[0], final_r2[0]))
    return final_model, final_mae, final_r2


def predict(df_test, col, model_, final_model_name,
            slope_, intercept_, modality, atlas, group, r, database="ADNI",
            train_test='test', y='age', correct_with_CA=True, info_init=True):
    """
    Predicts brain age using trained algorithms.

    Parameters
    ----------
    df_test : pd.DataFrame
        Dataframe containing input and output variables
    col : list or np.ndarray
        Column(s) to be used as input features
    model_ : ExtendedDataFramePipeline
        final model to be used for prediction
    final_model_name : str
        name of final model to be used for saving of plots
    y : str, optional
        Column to be considered as output feature. The default is age.
    slope_ : float
        Slope of linear model for bias correction
    intercept_ : float
        Intercept of linear model for bias correction
    modality : str
        PET or MRI
    train_test : str
        Whether train or test data is predicted
    group : str
        CN or MCI
    r : int
        which round of cv is saved
    correct_with_CA : boolean, optional
        Whether or not to correct bias with chronological age.
        The default is True.
    info : boolean, optional
        Whether or not to create and save plots. The default is True.

    Returns
    -------
    y_pred_bc : np.ndarray
        Bias-corrected brain age of individuals from test set

    """
    # reduce to only the sample for which metrics are expected
    """df_test.drop(df_test[(df_test['Dataset'] != database) |
                         (df_test['Group'] != group)].index,
                 inplace=True)"""

    y_pred = model_.predict(df_test[col])
    if r == 0:
        print("n = ", len(df_test), "mean age = ",
              np.round(np.mean(df_test.age), 2),
              np.round(np.std(df_test.age), 2))
    # plot model predictions against GT in test set
    if correct_with_CA is None:
        y_pred_bc = y_pred
    elif correct_with_CA:
        # for age correction WITH chronological age
        y_pred_bc = y_pred - (slope_*df_test[y] + intercept_)
    else:
        # for age correction WITHOUT chronological age
        y_pred_bc = (y_pred - intercept_)/slope_

    y_diff = y_pred_bc - df_test[y]
    linreg = stats.linregress(df_test[y], y_diff)
    r_val = linreg[2]
    p_val = linreg[3]
    check = p_val > 0.05
    mae = mean_absolute_error(df_test[y], y_pred_bc)
    r2 = r2_score(df_test[y], y_pred_bc)
    mean_diff = np.mean(y_pred_bc - df_test[y])

    # scatterplot and csv file of predictions of test set
    if r == 0 and info_init:
        plots.real_vs_pred_2(df_test[y], y_pred_bc, final_model_name, modality,
                             train_test, group, database=database, atlas=atlas,
                             correct_with_CA=correct_with_CA, r=r,
                             database_list=df_test['Dataset'])
    df = pd.DataFrame({'PTID': df_test['name'],
                       'Age': df_test[y],
                       'Prediction': y_pred_bc})
    df.to_csv("../results/{}/{}/{}-predicted_age_{}_{}_{}.csv".format(
              database, group, modality, atlas, group, r))
    print("Bias between chronological age and BPAD eliminated:",
          check, "(r =", r_val, "p =", p_val, ")")

    return y_pred_bc, mae, r2, mean_diff


def brain_age(dir_mri_csv, dir_pet_csv, modality, rand_seed, atlas,
              database="ADNI", correct_with_CA=True, cv_outer=5, cv_inner=5,
              info=True, info_init=False, save=True):
    """
    Execute brain age prediction pipeline.

    Main function uses functions defined in steps_of_analysis
    to run all steps: (1) train-test split, (2) hyperparameter
    tuning using cross-validation, (3) bias correction using
    cross-validated predictions, (4) prediction of test set.

    Parameters
    ----------
    modality : str
        MRI or PET
    rand_seed : int, optional
        Random seed to use throughout pipeline. The default is 0.
    imp : str, optional
        Main analysis with one random seed, validation
        with several random seeds or neurocorrelation. The default is 'ADNI'.
    info : boolean, optional
        Whether to print intermediate info. Recommended to set
        to False for validation_random_seeds. The default is True.
    info_init : boolean, optional
        Whether to print initial info.
    save : boolean, optional
        Whether final model should be saved

    Returns
    -------
    n_outliers : int
        Number of outliers excluded prior to splitting.
    pred : list
        Predicted & bias-corrected age.
    mae : float
        Mean absolute error of pred.
    r2 : float
        R squared of chronological age and pred.

    """
    # LOAD RAW DATA
    df_mri = pd.read_csv(dir_mri_csv)
    df_pet = pd.read_csv(dir_pet_csv)

    col = df_mri.columns[3:-2].tolist()
    pickle.dump(col, open("../config/columns.p", "wb"))

    # SPLIT AND LOAD TRAIN-TEST DATA
    # save csv files and save number of outliers in n_outliers
    # only load data of current modality
    group = "CN"
    mode = "train"
    mae_all = []
    r2_all = []
    model_all = []
    mean_diff_all = []

    for r in range(cv_outer):
        # to avoid new train-test splitting at each iteration
        if not(exists('../data/ADNI/CN/test_train_{}_{}_{}.csv'.format(
                modality, atlas, r))):
            split_data(df_mri, df_pet, col, splits=cv_outer, atlas=atlas,
                       info=info_init, rand_seed=rand_seed)

        df = pd.read_csv('../data/ADNI/CN/test_train_{}_{}_{}.csv'.format(
                modality, atlas, r))
        df = df[df['AGE_CHECK'] & df['IQR']]
        df_train = df[df['train']]
        df_train = df_train.reset_index(drop=True)

        if info_init:
            plots.plot_hist(df_train, group, mode, modality,
                            df_train['Dataset'], atlas=atlas, r=r, y='age')

        # CROSS-VALIDATE MODELS
        # define models and model names (some are already included in julearn)
        models = [RVR(), 'svm']
        model_names = ['rvr', 'svm']
        SCORING = ['r2']
        model_params = pickle.load(open("../config/hyperparams_allmodels.p",
                                        "rb"))

        model_results, scores, results = cross_validate(
            df_train, col, models, model_params, splits=cv_inner,
            rand_seed=rand_seed, scoring=SCORING, y='age', info=info_init)

        # APPLY BIAS CORRECTION AND FIND FINAL MODEL
        final_model_name, pred_param = bias_correct(
            results, df_train, col, model_results, model_names, modality,
            atlas, group, database=database, r=r,
            correct_with_CA=correct_with_CA,
            info_init=info_init, save=save)
        slope_ = pred_param[final_model_name + "_slope"]
        intercept_ = pred_param[final_model_name + "_intercept"]
        final_model = model_results[model_names.index(final_model_name)]

        # YIELD TEST PREDICTIONS FOR CN
        # How well does the model perform on unseen data (ADNI & OASIS)?
        df_test = df[~df['train']]
        df_test = df_test.reset_index(drop=True)
        mode = "test"

        if info_init:
            plots.plot_hist(df_test, group, mode, modality,
                            df_test['Dataset'], y='age', atlas=atlas, r=r)
            plots.feature_imp(df_test, col, final_model, final_model_name,
                              modality, atlas=atlas, r=r, rand_seed=rand_seed)

        pred, mae, r2, mean_diff = predict(
            df_test, col, final_model, final_model_name,
            slope_, intercept_,
            modality=modality, atlas=atlas,
            group=group,
            correct_with_CA=correct_with_CA,
            r=r, train_test='test',
            info_init=info)
        mae_all.append(mae)
        r2_all.append(r2)
        mean_diff_all.append(mean_diff)
        model_all.append(final_model_name)
        if save:
            pickle.dump(
                final_model,
                open("../results/0_FINAL_MODELS/" +
                     "final_model_{}_{}_{}_{}.p".format(
                         modality, atlas, str(correct_with_CA), str(r)), "wb"))
    results = {"Round": list(range(cv_outer)),
               "MAE": mae_all,
               "R2": r2_all,
               "Mean diff": mean_diff_all,
               "Model": model_all}
    results = pd.DataFrame(results)

    # SHOW METRICS
    print("\033[1m---TEST---\033[0m")
    print(results)
    print(results.describe())

    # combine all predictions
    df_final = pd.read_csv("../results/{}/{}/".format(database, group) +
                           "{}-predicted_age_{}_{}_0.csv".format(
                               modality, atlas, group))

    for f in range(1, cv_outer):
        df_add = pd.read_csv("../results/{}/{}/".format(database, group) +
                             "{}-predicted_age_{}_{}_{}.csv".format(
                               modality, atlas, group, str(f)))
        df_final = pd.concat([df_final, df_add])
    df_final.to_csv("../results/{}/{}/".format(database, group) + modality +
                    "-predicted_age_{}_{}.csv".format(atlas, group))

    return results


def predict_other(database, group, modality, atlas, rand_seed_np,
                  info_init=False):
    """
    Predict age of MCI patients.

    Parameters
    ----------
    csv_other : pd.dataframe
        Containing data for input and output
    group : str
        which group is being investigated, "CN", "SCD" or "MCI"
    modality : str
        PET or MRI
    atlas : str
        which atlas to use, "Sch_Tian" or "AAL"
    info_init : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    print("\033[1m---{}-{}---\033[0m".format(database.upper(),
                                             modality.upper()))

    col = pickle.load(open("../config/columns.p", "rb"))

    # make main predictions
    final_model = pickle.load(open(
        "../results/0_FINAL_MODELS/final_model_{}_{}_True_0.p".format(
            modality, atlas), "rb"))
    final_model_name = ['svm' if 'svm' in final_model.named_steps.keys()
                        else 'rvr'][0]
    params = pd.read_csv(
        "../results/0_FINAL_MODELS/models_and_params_{}_{}_True_0.csv".format(
            modality, atlas))
    slope_ = params['{}_slope'.format(final_model_name)][0]
    intercept_ = params['{}_intercept'.format(final_model_name)][0]

    if database == "OASIS":
        train_test = "validation"

    elif database == "DELCODE":
        train_test = "test"

    elif (database == "ADNI") and (group == "MCI"):
        train_test = "test"

    nooutliers_all = []
    pred_all = []
    mae_all = []
    r2_all = []
    mean_diff_all = []

    for i in range(0, 5):
        # only re-load file for OASIS, as outliers are not excluded for MCI
        """if database == "OASIS" or database == "DELCODE":
            # cognitively normal individuals from OASIS
            # outlier check has already been conducted with ADNI data
            file_ = pd.read_csv(
                "../data/{}/{}/{}_{}_{}_{}_1mm_parcels.csv".format(
                    database, group, database, modality, group, atlas))

        elif (database == "ADNI") and (group == "SMC"):
            pass  # TODO
        else:  # if ADNI MCI data is investigated
            file_ = pd.read_csv(
                "../data/ADNI/MCI/MCI_{}_parcels_init.csv".format(
                    modality), sep=";")
            file_['Group'] = 'MCI'"""
        file_ = pd.read_csv(
                    "../data/{}/{}/{}_{}_{}_{}_parcels.csv".format(
                        database, group, database, modality, group, atlas))
        print("../data/{}/{}/{}_{}_{}_{}_parcels.csv".format(
                        database, group, database, modality, group, atlas))
        file_['age'] = np.round(file_['age'], 0)
        file_ = outlier_check_other(file_, group=group, database=database,
                                    atlas=atlas, modality=modality, fold=i,
                                    rand_seed_np=rand_seed_np)

        nooutliers_all.append(
            file_.name[file_['AGE_CHECK'] & file_['IQR']].values.tolist())

    # for bagging, only consider data that is not an outlier in any of the
    # models
    nooutliers_all = [id_ for sublist in nooutliers_all for id_ in sublist]
    nooutliers_union = {x: nooutliers_all.count(x) == 5
                        for x in nooutliers_all}
    # make list of individuals occuring in all five models
    nooutliers_ids = [key for key, value in nooutliers_union.items() if value]
    nooutliers_bool = [x in nooutliers_ids for x in file_.name.values.tolist()]
    file_ = file_[nooutliers_bool]
    for i in range(0, 5):
        final_model = pickle.load(open(
            "../results/0_FINAL_MODELS/final_model_{}_{}_True_{}.p".format(
                modality, atlas, str(i)), "rb"))
        params = pd.read_csv(
            "../results/0_FINAL_MODELS/" +
            "models_and_params_{}_{}_True_{}.csv".format(
                modality, atlas, str(i)))
        slope_ = params['{}_slope'.format(final_model_name)][0]
        intercept_ = params['{}_intercept'.format(final_model_name)][0]
        pred, mae, r2, mean_diff = predict(file_, col, final_model,
                                           final_model_name,
                                           slope_, intercept_, modality,
                                           atlas=atlas,
                                           group=group,
                                           database=database, r=i,
                                           train_test="test")
        pred_all.append(pred)
        mae_all.append(mae)
        r2_all.append(r2)
        mean_diff_all.append(mean_diff)

    # bagging (mean) of results for neurocorrelations
    pred_bagged = np.mean(pred_all, axis=0)
    mae_bagged = mean_absolute_error(file_['age'], pred_bagged)
    r2_bagged = r2_score(file_['age'], pred_bagged)
    mean_diff_bagged = np.mean(pred_bagged - file_['age'])
    df = pd.DataFrame({'PTID': file_['name'],
                       'Age': file_['age'],
                       'Prediction': pred_bagged})
    df.to_csv("../results/{}/{}/{}-predicted_age_{}_{}_BAGGED.csv".format(
                  database, group, modality, atlas, group))

    results = pd.DataFrame({"MAE": mae_all,
                            "R2": r2_all,
                            "mean_diff": mean_diff_all})
    print("\n---BAGGED RESULTS---")
    print("MAE: ", mae_bagged, "\nR2 Score: ", r2_bagged, "\nMean error: ",
          mean_diff_bagged)
    print("\n---RESULTS ACROSS MODELS---")
    print(results)
    print(results.describe())
