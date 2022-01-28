#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:10:14 2022

@author: doeringe
"""


def plot_bpad_diff(y_true, y_pred, neuropsych_var,
                   df_test, modality, database,
                   group="CN"):
    """
    Create boxplots of BPAD differences significant as per t-test.

    Parameters
    ----------
    y_true : list
        Chronological age
    y_pred: list
        Predicted age
    alg: algorithm used for current task (used for saving)
    modality: image modality used (MRI/PET; used for saving)
    train_test: str indicating whether train or test data is plotted
        (used for saving)
    database: str indicating which database was used

    Returns
    -------
    None. (plots and saves plots)
    """
    
    print('---BPAD---')
    y_pred = np.round(y_pred, 0)
    y_diff = y_pred - y_true
    y_diff_cat = [-1 if x < 0 else 0 if x == 0 else 1 for x in y_diff]

    sign = {}
    for n in neuropsych_var:
        ttest = stats.ttest_ind(df_test[n][np.where(np.array(y_diff_cat) > 0)[0]],
                                df_test[n][np.where(
                                    np.array(y_diff_cat) < 0)[0]],
                                nan_policy='omit')
        print(n, "p-value: ", ttest[1])
        if ttest[1] < 0.05:
            sign[n] = ttest[0]
            fig, ax = plt.subplots(1, figsize=[12, 8])
            sns.boxplot(x=y_diff_cat, y=df_test[n],
                        palette='PuOr')
            plt.xlabel('BPAD [years]')
            plt.xticks((0, 1, 2), ('-', 'o', '+'))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            text = 't = ' + \
                str(np.round(ttest[0], 3)) + ' p = ' + \
                str(np.round(ttest[1], 3))
            plt.title('Difference BPAD - {}'.format(n))
            plt.text(xmax - 0.01 * xmax, ymax - 0.01 * ymax,
                     text, verticalalignment='top',
                     horizontalalignment='right', fontsize=12)

            plt.savefig(fname="../results/" + database + "/plots/" +
                        group + "/" + modality +
                        "_boxplot_BPAD" + "_" + n + ".png", dpi=300)
            plt.show()
    print("t-values of significant tests:")
    for key in sign:
        print(key, ":", np.round(sign[key], 3))"""