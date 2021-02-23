#!/home/smore/.venvs/py3smore/bin/python3
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold



def stratified_splits(bins_on, num_bins, data, num_splits, shuffle, random_state):
    """
    :param bins_on: variable used to create bins
    :param num_bins: num of bins/classes to create
    :param data: data to create cv splits on
    :param num_splits: number of cv splits to create
    :param shuffle: shuffle the data or not
    :param random_state: random seed to use if shuffle=True
    :return: a dictionary with index
    """
    qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins
    cv = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_state)
    test_idx = {}
    rpt_num = 0
    for train_index, test_index in cv.split(data, qc.codes):
        key = 'repeat_' + str(rpt_num)
        test_idx[key] = test_index
        rpt_num = rpt_num + 1
    return test_idx



def repeated_stratified_splits(bins_on, num_bins, data, num_splits, num_repeats, random_state):
    qc = pd.cut(bins_on.tolist(), num_bins)
    cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=random_state)
    test_idx = {}
    rpt_num = 0
    for train_index, test_index in cv.split(data, qc.codes):
        key = 'repeat_' + str(rpt_num)
        test_idx[key] = test_index
        rpt_num = rpt_num + 1
    return test_idx







