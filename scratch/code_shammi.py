#!/home/smore/.venvs/py3smore/bin/python3
import pickle
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from julearn import run_cross_validation
# from julearn.utils import configure_logging
from julearn.transformers import register_transformer

from skrvm import RVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
import sklearn.gaussian_process as gp
from sklearn.neural_network import MLPRegressor
from create_splits import *
from glmnet import ElasticNet
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Data path")
    parser.add_argument("output_filenm", type=str, help="Output file name")

    args = parser.parse_args()
    data = args.data_path
    output_filenm = args.output_filenm
    output_path = '/data/project/brainage/brainage_julearn/results/' + output_filenm
    print('output_path', output_path)

    # data = '/data/project/brainage/data_new/ixi/ixi_bsf_423'
    # output_path = '/data/project/brainage/brainage_julearn/results/ixi/ixi_test'
    # configure_logging(level='INFO')

    data_df = pickle.load(open(data, 'rb'))
    print(data_df.columns)
    print(data_df.index)
    data_df.rename(columns=lambda X: str(X), inplace=True)  # convert numbers to strings as column names
    X = data_df.columns[4:].tolist() # don't use 'site', 'subject', 'age', 'gender' from the dataframe as features
    y = 'age'

    age = data_df['age'].round().astype(int) # round off age and convert to integer
    data_df['age'] = age # update age
    data_df = data_df[data_df['age'].between(18, 90)].reset_index(drop=True)
    data_df.sort_values(by='age', inplace=True, ignore_index=True)

    # check for duplicates (multiple sessions for one subject)
    duplicated_subs_1 = data_df[data_df.duplicated(['subject'], keep='first')]
    duplicated_subs_2 = data_df[data_df.duplicated(['subject'], keep='last')]
    data_df = data_df.drop(duplicated_subs_1.index).reset_index(drop=True)

    rand_seed = 200
    num_splits = 5  # how many train and test splits
    num_repeats = 10
    num_bins = math.floor(len(data_df)/num_splits) # num of bins to be created = num of labels created
    # test_indices = repeated_stratified_splits(bins_on=data_df.index, , data=data_df, num_splits=num_splits,
    #                                           num_repeats=num_repeats, random_state=rand_seed)
    test_indices = stratified_splits(bins_on=data_df.index, num_bins=num_bins, data=data_df, num_splits=num_splits, shuffle=False, random_state=None)  # creates dictionary of test indices for different repeats

    all_idx = np.array(range(0, len(data_df)))

    scores_cv = {k: {} for k in test_indices.keys()}
    models = {k: {} for k in test_indices.keys()}
    results = {k: {} for k in test_indices.keys()}

    for repeat_key in test_indices.keys():

        test_idx = test_indices[repeat_key]  # get test indices
        train_idx = np.delete(all_idx, test_idx)  # get train indices
        train_df, test_df = data_df.loc[train_idx,:], data_df.loc[test_idx,:]  # get test and train dataframes
        print('train size:', train_df.shape, 'test size:', test_df.shape)

        qc = pd.cut(train_df['age'].tolist(), bins=5, precision=1)  # create bins for only train set
        print('age_bins', qc.categories, 'age_codes', qc.codes)

        # Define all models
        rvr = RVR()
        pls = PLSRegression() #plsregression
        kernel_ridge = KernelRidge() #kernelridge
        mlp = MLPRegressor()
        lasso = ElasticNet(alpha=1, n_lambda=10, n_splits=5)
        elasticnet = ElasticNet(alpha=0.5, n_lambda=10, n_splits=5)

        model_list = ['ridge', 'rf', rvr, kernel_ridge, 'gauss', lasso, elasticnet]
        model_names = ['ridge', 'rf', 'rvr', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet']

        model_para_list = [{'ridge__alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'cv': 5},
                           {'rf__n_estimators': 500, 'rf__criterion': 'mse', 'cv': 5},
                           {'rvr__kernel': 'linear', 'cv': 5},
                           {'kernelridge__alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0], 'kernelridge__kernel': 'polynomial',
                            'kernelridge__degree': [1, 2], 'cv': 5},
                           {'gauss__kernel': gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0,
                                                                                                          (1e-3, 1e3)),
                            'gauss__n_restarts_optimizer': 10, 'gauss__alpha': 0.1, 'gauss__normalize_y': True,
                            'cv': 5}
                           ]

        for i in range(0, len(model_names)):
            print(i, model_names[i])

            if model_names[i] == 'mlp' and model_names[i] == 'lasso_new':
                preprocess_X = 'zscore'
            else:
                preprocess_X = None

            if model_names[i] != 'elasticnet' and model_names[i] != 'lasso':
                # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=rand_seed).split(train_df, qc.codes)
                cv = StratifiedKFold(n_splits=5).split(train_df, qc.codes)
                scores, model = run_cross_validation(X=X, y=y, data=train_df, preprocess_X=preprocess_X,
                                                     problem_type='regression', model=model_list[i], cv=cv,
                                             return_estimator='final', model_params=model_para_list[i], seed=rand_seed,
                                                     scoring=
                                             ['neg_mean_absolute_error', 'neg_mean_squared_error','r2'])
            else:
                # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=rand_seed).split(train_df, qc.codes)
                cv = StratifiedKFold(n_splits=5).split(train_df, qc.codes)
                scores, model = run_cross_validation(X=X, y=y, data=train_df, preprocess_X=preprocess_X,
                                                     problem_type='regression', model=model_list[i], cv=cv,
                                                     return_estimator='final', seed=rand_seed, scoring=
                                                     ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])

            scores_cv[repeat_key][model_names[i]] = scores

            if model_names[i] == 'rf' or model_names[i] == 'rvr' or model_names[i] == 'gauss' or \
                    model_names[i] == 'mlp' or model_names[i] == 'elasticnet' or model_names[i] == 'lasso':
                models[repeat_key][model_names[i]] = model
            else:
                models[repeat_key][model_names[i]] = [model.best_estimator_ , model.best_params_]
                print('best para', model.best_params_)

            y_true = test_df[y]
            y_pred = model.predict(test_df[X]).ravel()
            y_delta = y_true - y_pred
            print(y_true.shape, y_pred.shape)
            mae = round(mean_absolute_error(y_true, y_pred), 2)
            mse = round(mean_squared_error(y_true, y_pred), 2)
            corr = round(np.corrcoef(y_pred, y_true)[1, 0], 2)

            print('----------', mae, mse, corr)
            results[repeat_key][model_names[i]] = {'predictions': y_pred, 'true': y_true, 'test_idx': test_idx,
                                                   'delta': y_delta, 'mae': mae, 'mse': mse, 'corr': corr}

    pickle.dump(results, open(output_path + '.results', "wb"))
    pickle.dump(scores_cv, open(output_path + '.scores', "wb"))
    pickle.dump(models, open(output_path + '.models', "wb"))
