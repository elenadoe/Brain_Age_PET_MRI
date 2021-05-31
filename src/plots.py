import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn import plotting
from nilearn import image
from sklearn.metrics import mean_absolute_error

def real_vs_pred(y_true, y_pred)
    """Plots True labels against the predicted ones.
    inputs: 
    y_true = list of floating point values or integers, representing ground truth values
    y_pred = list of floating point values or integers, representing predicted values
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
    plt.show()

# plot permutation importance