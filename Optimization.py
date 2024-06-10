# Functions to help in the optimization of a model.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from mlxtend.evaluate import bias_variance_decomp
import Utils


cpu_num=Utils.cpu_num

# Function to automatically search for the best hyperparameters for a given model
# using k-fold cross-validation. If rand is set to a number i, i iterations of random search
# will be performed instead (for cases where full grid search is too slow).
def FindBestParams(model, params, X, y, strategy='grid_search', cv=10, scoring=None,
                   iters=100, factor=3, n_candidates='exhaust'):
    
    # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
    if scoring is None and is_regressor(model):
        scoring = 'neg_mean_squared_error'
    elif scoring is None and is_classifier(model):
        scoring = 'roc_auc'
    
    # Save results for different score in this dictionary.
    scores_dict = {}
    
    # For every scoring metric, search for the the parameters using input strategy and save them.
    for score in scoring:
        if strategy == 'grid_search':
            grid_search = GridSearchCV(model, params, cv=cv, scoring=score, n_jobs=cpu_num)
            grid_search.fit(X, y)
            scores_dict[score] = grid_search.best_params_
        elif strategy == 'random_search':
            random_search = RandomizedSearchCV(model, params, n_iter=iters, cv=cv, scoring=score, n_jobs=cpu_num)
            random_search.fit(X, y)
            scores_dict[score] = random_search.best_params_
        elif strategy == 'halving_grid_search':
            halving_search = HalvingGridSearchCV(model, params, factor=factor, cv=cv, scoring=score, n_jobs=cpu_num)
            halving_search.fit(X, y)
            scores_dict[score] = halving_search.best_params_
        elif strategy == 'random_halving_search':
            halving_random = HalvingRandomSearchCV(model, params, n_candidates=n_candidates,
                                                      factor=factor, cv=cv, scoring=score, n_jobs=cpu_num)
            halving_random.fit(X, y)
            scores_dict[score] = halving_random.best_params_
        else:
            raise ValueError('Invalid input strategy. Must be one of grid_search, random_search\
            halving_grid_search or random_halving_search.')
    
    # Format the input scoring metrics so they look better.
    formatted_metrics = []
    for score in scoring:
        if score.startswith('neg_'):
            score.replace('neg_', '')
        formatted_metrics.append(' '.join([word.capitalize() for word in score.split('_')]))
     
    # Columns for DataFrame.
    cols = list(params.keys())
    
    # Parameter matrix for DataFrame
    param_arr = np.array([scores_dict[score][param] for score in scoring\
                          for param in cols]).reshape(len(scoring), len(cols))
    
    param_df = pd.DataFrame(param_arr, index=formatted_metrics, columns=cols)
    
    return param_df, scores_dict


# Function to plot the validation curve of a hyperparameter for a given model.
def PlotValidationCurve(model, X, y, hyper_param, value_range, scoring=None,
                        strategy= 'k-fold', cv=5, iters=1, shuffle=False, **kwargs):
    
    # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
    if scoring is None and is_regressor(model):
        scoring = 'neg_mean_squared_error'
    elif scoring is None and is_classifier(model):
        scoring = 'roc_auc'
    
    # Create cross-validator following the strategy given in the input.
    if strategy == 'k-fold':
        cv_model = KFold(n_splits=cv, shuffle=shuffle)
    elif strategy == 'repeated_k-fold':
        if iters <= 1:
            raise ValueError('Iters must be bigger than 1 for repeated k-fold')
        cv_model = RepeatedKFold(n_splits=cv, n_repeats=iters)
    else:
        raise ValueError('Strategy must be one of: k-fold or repeated_k-fold.')
    
    # If input score_metric is a string instead of a list, turn it into a list so that
    # it can be iterated.
    if type(scoring) == str:
        score_metrics = [scoring]
    
    # Create figure and axes objects, whose size depends on the number of score metrics
    # we want to draw.
    if len(score_metrics) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes = np.array(axes)
    else:   
        fig, axes = plt.subplots(nrows=int(math.ceil(len(score_metrics)/2)), ncols=2)
        if (len(score_metrics) % 2) != 0:
            axes[axes.shape[0]-1, axes.shape[1]-1].axis('off')
    
    for ax, score_metric in zip(axes.flatten(), score_metrics):
        
        # Calculate training scores and validation score for different values of
        # the input hyper parameter using cross-validation at every value.
        train_scores, val_scores = validation_curve(model, X, y,
                                                    param_name=hyper_param,
                                                    param_range=value_range,
                                                    scoring=score_metric,
                                                    cv=cv_model,
                                                    n_jobs=cpu_num)
        
        # Calculate the means and standard deviations of the cross-validation experiments
        # for every value of the input hyper parameter for the training and validation sets.
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Some metrics in scikit-learn use negative values, change them to positive.
        if score_metric.startswith('neg_'):
            train_scores_mean, val_scores_mean = -train_scores_mean, -val_scores_mean

        # Format score metric so it looks better.
        formatted_metric = score_metric
        if formatted_metric.startswith('neg_'):
            formatted_metric = formatted_metric.replace('neg_', '')
        formatted_metric = ' '.join([word.capitalize() for word in formatted_metric.split('_')])

        # Plot results.
        ax.set_xlabel(hyper_param, fontsize=12, fontweight='bold', labelpad=14)
        ax.set_ylabel(formatted_metric, fontsize=12, fontweight='bold', labelpad=12)
        ax.plot(value_range, train_scores_mean, label="Training score",
                color="darkorange", lw=2, **kwargs)
        ax.fill_between(value_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2,
                        color="darkorange", lw=2)
        ax.plot(value_range, val_scores_mean, label="Validation score",
                color="navy", lw=2, **kwargs)
        ax.fill_between(value_range, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.12,
                        color="navy", lw=2)
        ax.legend(loc='best')

    return fig, ax


# Function to plot the learning curve of a model.
def PlotLearningCurve(model, X, y, start, steps, scoring=None, strategy='k-fold',
                      cv=5, iters=1, shuffle=False, **kwargs):
    
    # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
    if scoring is None and is_regressor(model):
        scoring = 'neg_mean_squared_error'
    elif scoring is None and is_classifier(model):
        scoring = 'roc_auc'
    
    # Create cross-validator following the strategy given in the input.
    if strategy == 'k-fold':
        cv_model = KFold(n_splits=cv, shuffle=shuffle)
    elif strategy == 'repeated_k-fold':
        if iters <= 1:
            raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
        cv_model = RepeatedKFold(n_splits=cv, n_repeats=iters)
    else:
        raise ValueError('Strategy must be one of: k-fold or repeated_k-fold.')
    
    # Calculate train and validation scores for different sizes of the total dataset
    # using cross-validation.
    start = start/np.size(X, 0)
    total_steps, train_scores, val_scores = learning_curve(model, X, y,
                                                           train_sizes=np.linspace(start, 1.0, steps),
                                                           cv=cv_model,
                                                           scoring=scoring,
                                                           n_jobs=cpu_num)

    # Calculate the means and standard deviations of the cross-validation experiments
    # for every size of the dataset for the training and validation sets.
    mean_train_scores = np.mean(train_scores, 1)
    mean_val_scores = np.mean(val_scores, 1)
    std_train_scores = np.std(train_scores, 1)
    std_val_scores = np.std(val_scores, 1)

    # Some metrics in scikit-learn use negative values, change them to positive.
    if scoring.startswith('neg_'):
        mean_train_scores, mean_val_scores = -mean_train_scores, -mean_val_scores

    # Format score metric so it looks better.
    formatted_metric = scoring
    if formatted_metric.startswith('neg_'):
        formatted_metric = formatted_metric.replace('neg_', '')
    formatted_metric = ' '.join([word.capitalize() for word in formatted_metric.split('_')])

    # Arrays to draw error bars.
    train_above = mean_train_scores - std_train_scores
    train_below = mean_train_scores + std_train_scores
    val_above = mean_val_scores - std_val_scores
    val_below = mean_val_scores + std_val_scores

    # Plot data.
    fig, ax = plt.subplots()
    
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold', labelpad=14)
    ax.set_ylabel(formatted_metric, fontsize=12, fontweight='bold', labelpad=12)
    ax.plot(total_steps, mean_train_scores, label='Training score', color='red', lw=2, **kwargs)
    ax.plot(total_steps, mean_val_scores, label='Validation score', color='green', lw=2, **kwargs)
    ax.fill_between(total_steps, train_below, train_above, alpha=0.12, color='red', lw=2)
    ax.fill_between(total_steps, val_below, val_above, alpha=0.12, color='green', lw=2)
    ax.legend(loc='best')
    
    return fig, ax


# Function to plot the bias-variance decomposition of a given loss function based on a set
# of values of an input hyperparameter.
def LossDecomposition(model, X, y, hyper_param, value_range, loss=None,
                      strategy= 'k-fold', cv=5, iters=1, shuffle=False, **kwargs):
    
    # If no input loss function is given, choose appropriate one based on model type.
    if loss is None and is_regressor(model):
        loss = 'mse'
    elif loss is None and is_classifier(model):
        loss = '0-1_loss'
    
    # Create cross-validator following the strategy given in the input.
    if strategy == 'k-fold':
        cv_model = KFold(n_splits=cv, shuffle=shuffle)
    elif strategy == 'repeated_k-fold':
        if iters <= 1:
            raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
        cv_model = RepeatedKFold(n_splits=cv, n_repeats=iters)
    else:
        raise ValueError('Strategy must be one of: k-fold or repeated_k-fold.')
    
    # Arrays to store info for all hyperparameter values of the loss, bias and variance.
    loss_arr = np.zeros((len(value_range), 2))
    bias_arr = np.zeros((len(value_range), 2))
    var_arr = np.zeros((len(value_range), 2))
    
    # Iterate over every value of the input hyperparameters.
    for i, value in enumerate(value_range):
        
        # Create new model with the parameter analyzed in this iteration.
        clone_model = clone(model)
        param = {hyper_param: value}
        clone_model.set_params(**param)
    
        # Cross-validation.
        cv_data = np.zeros((cv*iters, 3))
        for n, (train_index, test_index) in enumerate(cv_model.split(X, y)):

            # Calculate train and test sets for this cv iteration.
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_test = X[test_index, :]
            y_test = y[test_index]

            # Bias-Variance decomposition in this CV iteration.
            avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clone_model, X_train, y_train, X_test,
                                                                        y_test, loss=loss)

            # Store data in array.
            cv_data[n, :] = np.array([avg_expected_loss, avg_bias, avg_var])
        
        # Average all CV iterations.
        cv_mean = np.mean(cv_data, axis=0)
        cv_std = np.std(cv_data, axis=0)
        
        # Store data for this value of the input hyperparameter.
        loss_arr[i, :] = np.array([cv_mean[0], cv_std[0]])
        bias_arr[i, :] = np.array([cv_mean[1], cv_std[1]])
        var_arr[i, :] = np.array([cv_mean[2], cv_std[2]])
    
    # Plot data.
    fig, ax = plt.subplots()
    
    ax.set_xlabel(hyper_param, fontsize=12, fontweight='bold', labelpad=14)
    if loss == 'mse':
        ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold', labelpad=12)
    elif loss == '0-1_loss':
        ax.set_ylabel('0-1 Loss', fontsize=12, fontweight='bold', labelpad=12)
    ax.plot(value_range, loss_arr[:, 0], color='red', lw=3, label='Loss')
    ax.plot(value_range, bias_arr[:, 0], color='chartreuse', lw=2, label='Bias')
    ax.plot(value_range, var_arr[:, 0], color='aqua', lw=2, label='Variance', **kwargs)
    ax.legend(loc='best')
        
    return fig, ax
