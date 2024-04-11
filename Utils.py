# Utility functions.

import math
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score, log_loss
from scipy.stats import pearsonr, spearmanr, kendalltau


cpu_num=8

# Function to set number of cpus to use.
def SetCPUNum(num):
    global cpu_num
    cpu_num = num


# Function to plot pandas DataFrame as matplotlib table.
def DFAsTable(df, dtype='float'):
    
    # Create figure.
    fig, ax = plt.subplots()
    
    if dtype == 'str':
        info_table = ax.table(df.to_numpy(), rowLabels=df.index, colLabels=df.columns, loc='center')
    else:
        info_table = ax.table(np.around(df.to_numpy(), 2), rowLabels=df.index, colLabels=df.columns, loc='center')
    info_table.auto_set_font_size(False)
    info_table.set_fontsize(16)
    info_table.scale(1, 2)
    ax.axis('off')

    # Write column and index headers in bold.
    for (row, col), cell in info_table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(size=16, weight='bold'))
        
    return fig, ax


# Function to calculate regression model performance metrics from input test data and model predictions.
def GetRegMetrics(y_test, y_pred):
    
    # Ensure test set and prediction arrays are the same size.
    if y_pred.shape != y_test.shape:
        raise ValueError('Prediction and test matrices must be of equal shape.')
    
    # Names of the metrics we will use.
    cols_names = ['MAE', 'RMSE', 'MAPE', '\N{GREEK SMALL LETTER RHO}\N{LATIN SUBSCRIPT SMALL LETTER P}', 'R\N{SUPERSCRIPT TWO}',
                  '\N{GREEK SMALL LETTER RHO}\N{LATIN SUBSCRIPT SMALL LETTER S}', '\N{GREEK SMALL LETTER TAU}']
    cols = ['mae', 'rmse', 'mape', 'pearson', 'r2', 'spearman', 'kendall']
    
    # Calculate metrics populations.
    metrics = np.zeros(7)
    metrics[0] = mean_absolute_error(y_test, y_pred)
    metrics[1] = mean_squared_error(y_test, y_pred, squared=True)
    metrics[2] = mean_absolute_percentage_error(y_test, y_pred)
    metrics[3] = pearsonr(y_test, y_pred)[0]
    metrics[4] = r2_score(y_test, y_pred)
    metrics[5] = spearmanr(y_test, y_pred)[0]
    metrics[6] = kendalltau(y_test, y_pred)[0]
    
    return xr.DataArray(metrics, dims=('Metrics'), coords={'Metrics': cols}, attrs={'Metric Names': cols_names})


# Function to calculate classification model performance metrics from input test data and model predictions.
def GetClfMetrics(y_test, y_pred, y_prob, beta):
    
    # Ensure test set and prediction arrays are the same size.
    if y_pred.shape != y_test.shape:
        raise ValueError('Prediction and test matrices must be of equal shape.')
    
    # Names of the metrics we will use.
    cols_names = ['Accuracy', 'Precision', 'Recall', f'F-{beta} Score', 'Log Loss', 'AUC-ROC']
    cols = ['acc', 'prec', 'rec', 'f', 'log_loss', 'auc_roc']
    
    # Calculate metrics populations.
    metrics = np.zeros(6)
    metrics[0] = accuracy_score(y_test, y_pred)
    metrics[1] = precision_score(y_test, y_pred)
    metrics[2] = recall_score(y_test, y_pred)
    metrics[3] = fbeta_score(y_test, y_pred, beta=beta)
    metrics[4] = log_loss(y_test, y_prob)
    metrics[5] = roc_auc_score(y_test, y_prob)
    
    print(metrics.shape, len(cols), len(cols_names))
    
    return xr.DataArray(metrics, dims=('Metrics'), coords={'Metrics': cols}, attrs={'Metric Names': cols_names})


# Function to calculate classification model performance metrics from input test data and model predictions for specific groups.
# Since some groups may only have one class, we must slightly change the types of metrics we calculate for group based
# performance analysis.
def GetGroupClfMetrics(y_test, y_pred, y_prob, beta):
    
    # Ensure test set and prediction arrays are the same size.
    if y_pred.shape != y_test.shape:
        raise ValueError('Prediction and test matrices must be of equal shape.')
    
    # Names of the metrics we will use.
    cols_names = ['Accuracy', 'Missclassification', 'Precision', 'Recall', f'F-{beta} Score']
    cols = ['acc', 'miss', 'prec', 'rec', 'f']
    
    # Calculate metrics populations.
    metrics = np.zeros(6)
    metrics[0] = accuracy_score(y_test, y_pred)
    metrics[1] = 1 - metrics[0]
    metrics[2] = precision_score(y_test, y_pred)
    metrics[3] = recall_score(y_test, y_pred)
    metrics[4] = fbeta_score(y_test, y_pred, beta=beta)
    
    return xr.DataArray(metrics, dims=('Metrics'), coords={'Metrics': cols}, attrs={'Metric Names': cols_names})


# Function to represent DataFrame as barplot.
def DFAsBar(df, value, threshold=None, **kwargs):
    
    fig, ax = plt.subplots()
    
    ax.set_xticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.index, rotation=90)
    ax.set_ylabel(value, weight='bold', labelpad=15)
    ax.bar(np.arange(len(df.index)), df[value].to_numpy(), facecolor='cyan', edgecolor='black', **kwargs)
    if threshold is not None:
        left, right = ax.get_xlim()
        ax.plot(np.linspace(left, right, len(df.index)), np.repeat(threshold, len(df.index)),
                color='red', linestyle='dashed')
        ax.set_xlim(left, right)
    
    return fig, ax


# Time to return an amount of seconds as a string formatted as xh ym zs.
def FormattedTime(t):
  h, remainder = divmod(t, 3600)
  m, s = divmod(remainder, 60)
  return f'{int(h)}h {int(m)}m {int(round(s))}s'
