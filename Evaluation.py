# Functions to evaluate the performance of a model.

import os
import glob
import subprocess
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.tree import export_graphviz
from sklearn.dummy import DummyRegressor, DummyClassifier
from mlxtend.evaluate import paired_ttest_5x2cv, combined_ftest_5x2cv
from mlxtend.evaluate import mcnemar, mcnemar_table
import Utils
from Utils import GetRegMetrics


cpu_num = Utils.cpu_num

def EvalRegModel(model, X_test, y_test):
    
    # Fit input model and make predictions.
    y_pred = model.predict(X_test)
    
    # Evaluate model.
    metrics = GetRegMetrics(y_test, y_pred)
    metrics_df = pd.DataFrame(np.around(metrics.to_numpy().reshape(1, -1), 4),
                              index=['Model'], columns=metrics.attrs['Metric Names'])
    
    return metrics_df


# Function to compare predictions and experimental results, as well as a dummy predictor.
def DrawPredVsReal(model, X_train, y_train, X_test, y_test, **kwargs):
    
    # Fit and make predictions for input model.
    y_pred = model.predict(X_test)
        
    # Fit and make predictions for dummy control model.
    dummy_model = DummyRegressor(strategy='mean')
    dummy_model.fit(X_train, y_train)
    y_dummy = dummy_model.predict(X_test)
    
    # Evaluate models.
    model_metrics = GetRegMetrics(y_test, y_pred)
    rmse_model = round(float(model_metrics.loc['rmse']), 2)
    dummy_metrics = GetRegMetrics(y_test, y_dummy)
    rmse_dummy = round(float(dummy_metrics.loc['rmse']), 2)
    
    # Plot data.
    fig, ax = plt.subplots()
    
    ax.scatter(y_test, y_pred, facecolor='none', edgecolor='grey', **kwargs)
    ax.plot(y_test, y_test, color='black', linewidth=3,
            label=f'Model RMSE: {rmse_model}')
    ax.plot(y_test, y_dummy, color='red', linestyle='dashed', linewidth=2,
            label=f'Dummy RMSE: {rmse_dummy}')
    ax.legend(loc='best')
    
    return fig, ax


# Function to perform a t-test or f-test to check if two models have similar performance or
# if one outperforms the other.
def PairedTTest(model1, X, y, model2=DummyRegressor(strategy='mean'), strategy='t-test',
                  scoring=None, significance=0.05):
    
    # Check that both models are regressors.
    if not is_regressor(model1) or not is_regressor(model2):
        raise ValueError(f'Both models must be regressors in order to carry out a paired {strategy}')
    
    # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
    if scoring is None and is_regressor(model1):
        scoring = 'neg_mean_squared_error'
    elif scoring is None and is_classifier(model1):
        scoring = 'roc_auc'

    # Perform 5x2cv paired t-test or 5x2cv combined F test, based on input
    if strategy == 't-test':
        # Perform 5x2cv paired t-test.   
        test, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring=scoring)
    elif strategy == 'f-test':
        #Perform 5x2cv combined F test.
        test, p = combined_ftest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring=scoring)
    
    # p value in scientific notation.
    p_scientific = '{:.3e}'.format(p)
    
    # Print result.
    print('Null hypothesis: model A and model B have equal performance.')
    print(f'Significance set to \N{GREEK SMALL LETTER ALPHA} = {significance}.')
    print('')
    if strategy == 't-test':
        print(f't statistic: {round(test, 3)}')
    elif strategy == 'f-test':
        print(f'F statistic: {round(test, 3)}')
    print(f'p value: {p_scientific}')
    if p < significance:
        print('Since p < \N{GREEK SMALL LETTER ALPHA}, the null hypothesis is rejected.')
    else:
        print('Since p > \N{GREEK SMALL LETTER ALPHA}, the null hypothesis cannot be rejected.')


# Function to draw a number of random trees from the estimators used in a random forest model.
# The pictures will be saved as png files in the working directory.
def DrawGraphs(model, features, num_graphs, depth):
    
    # Check if input model is ensemble model.
    if not hasattr(model, 'estimators_'):
        raise ValueError('Input model is not a ensemble model, son trees cannot be drawn.')
    
    # Store names of graphs in this list.
    graph_names = []

    for i, n in enumerate(random.sample(range(len(model.estimators_)), num_graphs)):
        
        graph_names.append(f'graph{i}.png')
        
        # Check if model is a classifier or regressor, then draw trees using graphviz.
        if is_classifier(model):
            export_graphviz(
                decision_tree=model.estimators_[n],
                out_file=f'graph{i}.dot',
                feature_names=features,
                max_depth=depth,
                class_names=['Inactive', 'Active'],
                rounded=True,
                filled=True
            )
        elif is_regressor(model):
            export_graphviz(
                decision_tree=model.estimators_[n],
                out_file=f'graph{i}.dot',
                feature_names=features,
                max_depth=depth,
                rounded=True,
                filled=True
            )

        # Create png images from dot files.
        subprocess.run(['dot', '-Tpng', f'graph{i}.dot', '-o', f'graph{i}.png'], subprocess.DEVNULL)

    # Delete dot files.
    for file in glob.glob('*.dot'):
        os.remove(file)

    return graph_names


# Function to perform McNemar's test to compare the performance of two classifiers.
# It assumes model1 was fitted to some cross-validation object.
def McNemarsTest(cv_model, model2, significance=0.05, display='total_count'):
    
    # Check that both models are classifiers.
    if not is_classifier(cv_model.model) or not is_classifier(model2):
        raise ValueError('Both models must be classifiers in order to carry out McNemars test')
    
    # Check that input arrays are equal dimension.
    if not hasattr(cv_model, 'outer_cv_split'):
        raise ValueError('"model1" must be fitted to a cross-validation object (not neccessary for "model2").')
    
    # Chech that input display is valid, to fitting model2 in vain.
    if display not in ['percentage', 'total_count']:
        raise ValueError('"display" must be one of "percentage" or "total_count".')
    
    # Get arrays with all the predictions made during cross-validation and their true values.
    y_true = np.hstack([cv_model.cv_array.data_vars[var][0, :].to_numpy() for var in cv_model.cv_array.data_vars])
    y_model1 = np.hstack([cv_model.cv_array.data_vars[var][1, :].to_numpy() for var in cv_model.cv_array.data_vars])
    
    # Remove NaN values.
    y_true = y_true[~np.isnan(y_true)]
    y_model1 = y_model1[~np.isnan(y_model1)]
    
    # Train model2 on same train sets used to train model1, then make predictions on same test sets used for model1.
    clone_model_2 = clone(model2)
    y_model2_list = []
    for key in cv_model.outer_cv_split.keys():
        X_train_tmp = cv_model.outer_cv_split[key]['X_train']
        y_train_tmp = cv_model.outer_cv_split[key]['y_train']
        clone_model_2.fit(X_train_tmp, y_train_tmp)
        y_model2_list.append(clone_model_2.predict(cv_model.outer_cv_split[key]['X_test']))
    y_model2 = np.concatenate(y_model2_list)
    
    # Compute McNemar table
    table = mcnemar_table(y_target=y_true,  y_model1=y_model1,  y_model2=y_model2)
    
    # Format according to input display.
    if display == 'percentage':
        disp_table = np.around((table/table.sum()), 2)
    elif display == 'total_count':
        disp_table = table
    else:
        raise ValueError('display must be "percentage" or "total_count"')

    # Plot McNemar table.
    fig, ax = plt.subplots()
    
    if display == 'percentage':
        sns.heatmap(disp_table, vmin=0, vmax=1, cmap='Blues', annot=True, fmt='.2%', ax=ax)
    elif display == 'total_count':    
        sns.heatmap(disp_table, vmin=0, vmax=table.ravel().sum(), cmap='Blues', annot=True, fmt='.0f', ax=ax)
    ax.set_xticklabels(['Correct', 'Incorrect'], fontweight='bold', ha='center')
    ax.set_yticklabels(['Correct', 'Incorrect'], fontweight='bold', va='center')
    ax.set_xlabel('Model2', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Model1', fontsize=14, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', which='major', pad=10, labelsize=12)

    # Setup colorbar.
    cbar = ax.collections[0].colorbar
    if display == 'percentage':
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # McNemar's test.
    chi2, p = mcnemar(ary=table, corrected=True)
    
    # p value in scientific notation.
    p_scientific = '{:.3e}'.format(p)
    
    # Print result.
    print('Null hypothesis: model A and model B have equal performance.')
    print(f'Significance set to \N{GREEK SMALL LETTER ALPHA} = {significance}.')
    print('')
    print(f'\N{GREEK SMALL LETTER CHI}\N{SUPERSCRIPT TWO} statistic: {round(chi2, 3)}')
    print(f'p value: {p_scientific}')
    if p < significance:
        print('Since p < \N{GREEK SMALL LETTER ALPHA}, the null hypothesis is rejected.')
    else:
        print('Since p > \N{GREEK SMALL LETTER ALPHA}, the null hypothesis cannot be rejected.')

    return fig, ax


# Counterfactual
