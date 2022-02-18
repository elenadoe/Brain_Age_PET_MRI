"""
Created on Tue Dec 21 16:27:41 2021.

@author: doeringe
"""

from julearn import run_cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from skrvm import RVR
# import pdb

from transform_data import split_data, split_data_np
import plots

import numpy as np
import pandas as pd
import pickle
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")


def cross_validate(df_train, col, models, model_params, splits, scoring,
                   rand_seed=0, y='age'):
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

    Returns
    -------
    model_results : list
        Best fitted estimator per algorithm
    scores : list
        Average performance during cross-validation of the algorithm
    results : dict
        Dictionary containing true and cross-validated predicted age

    """
    model_results = []
    scores_results = []
    results = {}
    results['RVR()'] = {}
    results['svm'] = {}

    scaler = 'scaler_robust'

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
    return model_results, scores, results


# BIAS CORRECTION
# Eliminate linear correlation of brain age delta and chronological age
def bias_correct(results, df_train, col, model_results, model_names,
                 modality, group, y='age', correct_with_CA=True,
                 info_init=False, save=True):
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
                                      correct_with_CA,
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
            pickle.dump(pred_param, open("../results/" + group +
                                         "/bias-correction/models_and_params_"
                                         + modality + "_" +
                                         str(correct_with_CA) + ".p", "wb"))
            pickle.dump(predictions, open("../results/" + group +
                                          "/cross-val_pred_" + modality + "_" +
                                          str(correct_with_CA) +
                                          ".p", "wb"))

        df = pd.DataFrame(pred_param)
        df.to_csv("../results/" + group + "/models_and_params_"
                  + modality + "_" + str(correct_with_CA) + ".csv")

    # compare predictions to find final model
    final_model, final_mae, final_r2 = find_final_model(pred_param,
                                                        model_names,
                                                        modality,
                                                        info_init=info_init)

    # scatterplot of bias-corrected results from cross-validation
    if info_init:
        plots.real_vs_pred_2(y_true, bc, final_model,
                             modality, 'train', 'CN')

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
            slope_, intercept_, modality, group, imp, r=0,
            train_test='test', y='age', correct_with_CA=True, info=True):
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
    imp : str
        main analysis, validation_random_seeds or neurocorrelations
    r : int
        which round of cv is saved (only important for neurocorrelations)
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
    y_pred = model_.predict(df_test[col])

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
    p = linreg[3]
    check = p > 0.05
    mae = mean_absolute_error(df_test[y], y_pred_bc)
    r2 = r2_score(df_test[y], y_pred_bc)

    # scatterplot and csv file of predictions of test set
    if info:
        plots.real_vs_pred_2(df_test[y], y_pred_bc, final_model_name, modality,
                             train_test, group,
                             correct_with_CA=correct_with_CA,
                             database_list=df_test['Dataset'])
        df = pd.DataFrame({'PTID': df_test['name'],
                           'Age': df_test[y],
                           'Prediction': y_pred_bc})
        if imp == "neurocorrelations":
            df.to_csv("../results/{}/{}-predicted_age_{}_fold{}.csv".format(
                imp, modality, group, r))
        else:
            df.to_csv("../results/{}/{}-predicted_age_{}_{}.csv".format(
                group, modality, group, r))
        print("Bias between chronological age and BPAD eliminated:",
              check, "(r =", r, "p =", p, ")")

    return y_pred_bc, mae, r2


def brain_age(dir_mri_csv, dir_pet_csv, modality,
              correct_with_CA=True, rand_seed=0, cv=5, imp='main', info=True,
              info_init=False, save=True):
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
        Random seed to use throughout pipeline. The default is 42.
    imp : str, optional
        Main analysis with one random seed, validation
        with several random seeds or neurocorrelation. The default is 'main'.
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
    df_mri = pd.read_csv(dir_mri_csv, sep=";")
    df_pet = pd.read_csv(dir_pet_csv, sep=";")
    col = df_mri.columns[3:-1].tolist()
    pickle.dump(col, open("../config/columns.p", "wb"))

    # SPLIT AND LOAD TRAIN-TEST DATA
    # save csv files and save number of outliers in n_outliers
    # only load data of current modality
    group = "CN"
    mode = "train"
    mae_all = {}
    r2_all = {}
    model_all = {}
    if imp == "neurocorrelations":
        rounds = cv
    else:
        rounds = 1
    for r in range(cv):
        if imp == "neurocorrelations":
            n_outliers = split_data_np(df_mri, df_pet, col, imp=imp,
                                       info=info_init, rand_seed=rand_seed)
            df = pd.read_csv('../data/neurocorrelations/test_train_' +
                             modality + '_' + str(r) + '.csv')
        else:
            n_outliers = split_data(df_mri, df_pet, col, imp=imp,
                                    info=info_init, rand_seed=rand_seed)
            df = pd.read_csv('../data/{}/test_train_'.format(imp) + modality +
                             '_' + str(rand_seed) + '.csv')
        df = df[df['AGE_CHECK'] & df['IQR']]
        df_train = df[df['train']]
        df_train = df_train.reset_index(drop=True)

        if info_init:
            plots.plot_hist(df_train, group, mode,
                            modality, df_train['Dataset'], y='age')

        # CROSS-VALIDATE MODELS
        # define models and model names (some are already included in julearn)
        models = [RVR(), 'svm']
        model_names = ['rvr', 'svm']
        SCORING = ['r2']
        model_params = pickle.load(open("../config/hyperparams_allmodels.p",
                                        "rb"))
        model_results, scores, results = cross_validate(
            df_train, col, models, model_params, splits=cv,
            rand_seed=rand_seed, scoring=SCORING, y='age')

        # APPLY BIAS CORRECTION AND FIND FINAL MODEL
        final_model_name, pred_param = bias_correct(
            results, df_train, col, model_results, model_names, modality,
            group, correct_with_CA=correct_with_CA, info_init=info_init,
            save=save)
        slope_ = pred_param[final_model_name + "_slope"]
        intercept_ = pred_param[final_model_name + "_intercept"]
        final_model = model_results[model_names.index(final_model_name)]
        if save:
            pickle.dump(final_model,
                        open("../results/final_model_{}_{}.p".format(
                            modality, str(correct_with_CA)), "wb"))

        # YIELD TEST PREDICTIONS FOR CN
        # How well does the model perform on unseen data (ADNI & OASIS)?
        df_test = df[~df['train']]
        df_test = df_test.reset_index(drop=True)
        mode = "test"

        if info_init:
            plots.plot_hist(df_test, group, mode,
                            modality, df_test['Dataset'], y='age')
            plots.feature_imp(df_test, col, final_model, final_model_name,
                              modality, rand_seed=rand_seed)
        pred, mae, r2 = predict(df_test, col, final_model, final_model_name,
                                slope_, intercept_, modality, group,
                                correct_with_CA=correct_with_CA, imp=imp, r=r,
                                train_test='test', info=info)
        mae_all[0] = mae
        r2_all[0] = r2
        model_all[0] = final_model_name
        if save:
            pickle.dump(final_model,
                        open("../results/final_model_{}_{}_{}.p".format(
                            modality, str(correct_with_CA), str(r)), "wb"))
    results = {"Round": list(range(rounds)),
               "MAE": mae_all,
               "R2": r2_all,
               "Model": model_all}
    results = pd.DataFrame(results)
    print(results)
    print("\033[1mResults over {} rounds\033[0m".format(rounds))
    print(results[["MAE", "R2", "Model"]].describe())

    # combine all predictions
    df_neuro = pd.read_csv("../results/neurocorrelations/" +
                           modality + "-predicted_age_" +
                           group + "_fold0.csv")

    for f in range(1, rounds):
        df_add = pd.read_csv("../results/neurocorrelations/" + modality +
                             "-predicted_age_" + group + "_fold" +
                             str(f) + ".csv")
        df_neuro = pd.concat([df_neuro, df_add])
        df_neuro.to_csv("../results/CN/" + modality +
                        "-predicted_age_CN.csv")

    return n_outliers, results
