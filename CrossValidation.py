# This module contains tools to perform cross-validation and analyze the results.

import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr
from scipy.stats import bootstrap, mode
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import confusion_matrix, roc_curve, auc, det_curve
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from genetic_selection import GeneticSelectionCV
import Utils
from Utils import GetRegMetrics, GetClfMetrics, GetGroupClfMetrics

cpu_num = Utils.cpu_num


# Class to perform cross-validation.
class CrossValidation:
    
    def __init__(self, model, strategy='k-fold', cv_folds=5, iters=1, shuffle=False, loss=None):
        
        # Set input data as attributes.
        self.model = model
        self.strategy = strategy
        self.cv_folds = cv_folds
        if (strategy == 'k-fold' or strategy == 'leave_group_out') and iters > 1:
            self.iters = 1
        else:
            self.iters = iters
        self.shuffle = shuffle
        self.loss = loss

    # Function to perform cross-validation.
    def fit(self, X, y, group=None):
        
        self.X = X
        self.y = y
        self.g = group
        if self.g is not None:
            self.g_num = np.max(self.g) + 1
            self.g_max = np.max(np.bincount(self.g))
        
        # Define cross-validation splitting object and name of the indices based on
        # input cross-validation strategy.
        if self.strategy == 'k-fold':
            self.cv_model = KFold(n_splits= self.cv_folds, shuffle=self.shuffle)
            self.indices = [f'Fold {i+1}' for i in range(self.cv_folds)]
        elif self.strategy == 'repeated_k-fold':
            if self.iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.cv_model = RepeatedKFold(n_splits= self.cv_folds, n_repeats=self.iters)
            self.indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.cv_folds) for i in range(self.iters)]
        elif self.strategy == 'leave_group_out':
            self.cv_model = LeaveOneGroupOut()
            self.indices = [f'Cluster {i+1}' for i in range(len(np.unique(self.g)))]
        elif self.strategy == 'stratified_k-fold':
            self.cv_model = StratifiedKFold(n_splits=self.cv_folds, shuffle=self.shuffle)
            self.indices = [f'Fold {i+1}' for i in range(self.cv_folds)]
        elif self.strategy == 'repeated_stratified_k-fold':
            if self.iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.cv_model = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=self.iters)
            self.indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.cv_folds) for i in range(self.iters)]
        else:
            raise ValueError('Strategy for cross-validation must be one of: k-fold, repeated_k-fold,\
                stratified_k-fold or repeated_stratified_k-fold.')
        
        # Function to split input data according to input cross-validation strategy.
        def cv_split(self):

            # Calculate maximum size of a fold.
            if self.strategy == 'leave_group_out':
                fold_len = self.g_max
            else:
                fold_len = int(len(y)/ self.cv_folds) + len(y)% self.cv_folds

            # Initialize Dataset object to store CV data and array to store bias-variance data.
            cv_array = xr.Dataset()
            if  self.strategy == 'leave_group_out':
                bias_var_arr = np.zeros((fold_len, 3))
            else:
                bias_var_arr = np.zeros(( self.cv_folds* self.iters, 3))
            
            times = []

            # Cross-validation
            for n, (train_index, test_index) in enumerate(self.cv_model.split(X, y,  self.g)):
                
                # Start measuring time.
                start = timeit.default_timer()
            
                # Estimate in how much time the process will end.
                if times:
                    mean_t = np.array(times).mean()
                    t_estimation = Utils.FormattedTime(mean_t*(self.cv_folds*self.iters - n))
                
                # Print progress.
                if times:
                    print(f'\rCross-validation iteration: {n+1}/{self.cv_folds*self.iters}; Estimated time remaining: {t_estimation}',
                        end='     ', flush=True)
                else:
                    print(f'\rCross-validation iteration: {n+1}/{self.cv_folds*self.iters}', end='     ', flush=True)

                # Calculate train and test sets for this cv iteration.
                clone_model = clone(self.model)
                X_train = X[train_index, :]
                y_train = y[train_index]
                X_test = X[test_index, :]
                y_test = y[test_index]
                if self.g is not None:
                    group_train =  self.g[train_index]
                    group_test =  self.g[test_index]

                # Fit train set and make test set predictions.
                clone_model.fit(X_train, y_train)
                y_pred = clone_model.predict(X_test).reshape(-1,)
                if is_classifier(clone_model):
                    y_prob = clone_model.predict_proba(X_test)[:, 1].reshape(-1,)

                # Stack test data, predictions and groups, then fill missing values so that
                # dimensions remain the same in all iterations.
                if self.g is not None and is_regressor(self.model):
                    arr = np.stack((y_test, y_pred, group_test))
                    if arr.shape[1] != fold_len:
                        empty_arr = np.empty((3, fold_len - arr.shape[1]))
                        empty_arr[:, :] = np.nan
                        arr = np.hstack((arr, empty_arr))
                    
                    # Write xarray for this iteration.
                    xarr = xr.DataArray(arr,
                                        dims=('Type', 'Sample'),
                                        coords={'Type': ['Test Set Samples', 'Predictions', 'Groups'],
                                                'Sample': np.arange(1, fold_len+1)}, name=self.indices[n])
                
                elif self.g is not None and is_classifier(self.model):
                    arr = np.stack((y_test, y_pred, y_prob, group_test))
                    if arr.shape[1] != fold_len:
                        empty_arr = np.empty((4, fold_len - arr.shape[1]))
                        empty_arr[:, :] = np.nan
                        arr = np.hstack((arr, empty_arr))
            
                    # Write xarray for this iteration.
                    xarr = xr.DataArray(arr,
                                        dims=('Type', 'Sample'),
                                        coords={'Type': ['Test Set Samples', 'Predictions', 'Probabilities', 'Groups'],
                                                'Sample': np.arange(1, fold_len+1)}, name=self.indices[n])
                
                elif self.g is None and is_regressor(self.model):
                    arr = np.stack((y_test, y_pred))
                    if arr.shape[1] != fold_len:
                        empty_arr = np.empty((2, fold_len - arr.shape[1]))
                        empty_arr[:, :] = np.nan
                        arr = np.hstack((arr, empty_arr))
                    
                    # Write xarray for this iteration.
                    xarr = xr.DataArray(arr,
                                        dims=('Type', 'Sample'),
                                        coords={'Type': ['Test Set Samples', 'Predictions'],
                                                'Sample': np.arange(1, fold_len+1)}, name=self.indices[n])
                
                
                elif self.g is None and is_classifier(self.model):
                    arr = np.stack((y_test, y_pred, y_prob))
                    if arr.shape[1] != fold_len:
                        empty_arr = np.empty((3, fold_len - arr.shape[1]))
                        empty_arr[:, :] = np.nan
                        arr = np.hstack((arr, empty_arr))
                    
                    # Write xarray for this iteration.
                    xarr = xr.DataArray(arr,
                                        dims=('Type', 'Sample'),
                                        coords={'Type': ['Test Set Samples', 'Predictions', 'Probabilities'],
                                                'Sample': np.arange(1, fold_len+1)}, name=self.indices[n])

                # Add xarray from this iteration to Dataset that stores all iterations.
                cv_array[xarr.name] = xarr
                
                # Calculate bias-variance decomposition of the loss for this CV iteration.
                if  self.loss is not None:
                    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clone_model, X_train, y_train, X_test, y_test, loss= self.loss)
                    bias_var_arr[n, :] = np.array([avg_expected_loss, avg_bias, avg_var])
                
                # End measuring time.
                end = timeit.default_timer()
                times.append(end-start)
            
            if  self.loss is not None:
                if  self.strategy == 'leave_group_out':
                    bias_var_arr = xr.DataArray(bias_var_arr,
                                                dims=('Sample', 'Decomposition'),
                                                coords={'Sample': np.arange(1, fold_len+1),
                                                        'Decomposition': ['Average Loss', 'Average Bias', 'Average Variance']})
                else:
                    bias_var_arr = xr.DataArray(bias_var_arr,
                                                dims=('Sample', 'Decomposition'),
                                                coords={'Sample': np.arange(1,  self.cv* self.iters+1),
                                                        'Decomposition': ['Average Loss', 'Average Bias', 'Average Variance']})
            
            return cv_array, bias_var_arr, times
        
        # Add model prediction and bias-variance data as class attributes.
        self.cv_array, self.bias_var_arr, times = cv_split(self)
        
        print(f'\rCross-validation evaluation successfully completed. Total time: {Utils.FormattedTime(np.array(times).sum())}',
              end='     ')

    # Function to calculate permutation score of a model.
    def permutation_score(self, scoring=None, n_permutations=100, **kwargs):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
        if scoring is None and is_regressor(self.model):
            scoring = 'neg_mean_squared_error'
        elif scoring is None and is_classifier(self.model):
            scoring = 'roc_auc'
        
        # Calculate the score of the input model, the scores of the permutations
        # and the empirical p-value.
        score, perm_scores, pvalue = permutation_test_score(self.model, self.X, self.y, scoring=scoring,
                                                            cv=self.cv_model, n_permutations=n_permutations)
        
        # Format input scoring metric so that it look better in the final figure.
        if scoring.startswith('neg_'):
            scoring = scoring.replace('neg_', '')
            score = -score
            perm_scores = -perm_scores
        scoring = ' '.join([word.capitalize() for word in scoring.replace('_', ' ').split()])
        
        # Draw distribution of permutation scores as histogram and
        # the score of the input model as a vertical dashed line.
        fig, ax = plt.subplots()
        
        ax.hist(perm_scores, bins=25, density=True, **kwargs)
        ax.axvline(score, linestyle='--', color='r', label=f'p-value: {round(pvalue, 5)}')
        ax.set_xlabel(scoring, fontsize=11, fontweight='bold', labelpad=12.5)
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold', labelpad=12.5)
        ax.legend(loc='best')
        
        return fig, ax

    # Function to obtain model performance metrics calculated from the
    # cross-validation experiments.
    def get_cv_metrics(self, beta=1):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Write each xarray defining a CV iteration as a pandas Series,
        # then put all Series in a list.
        cv_list = [self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars]
        if is_regressor(self.model):
            metric_list = [GetRegMetrics(metric_arr[0, :].to_numpy(), metric_arr[1, :].to_numpy()) for metric_arr in cv_list]
        elif is_classifier(self.model):
            metric_list = [GetClfMetrics(metric_arr[0, :].to_numpy(), metric_arr[1, :].to_numpy(), metric_arr[2, :].to_numpy(), beta=beta) for metric_arr in cv_list]
        series_list = [pd.Series(x.to_numpy(), index=x.attrs['Metric Names']) for x in metric_list]

        # Write pandas DataFrame by merging all Series in Series list.
        iters_df = pd.concat(series_list, axis=1, keys=self.indices).transpose()
    
        # Compute 95% confidence intervals and standard erros and write them in pandas Series.
        mean_cis = [bootstrap((iters_df[[metric]].to_numpy(),), np.mean).confidence_interval for metric in iters_df.columns]
        median_cis = [bootstrap((iters_df[[metric]].to_numpy(),), np.median).confidence_interval for metric in iters_df.columns]
        mean_stes = [bootstrap((iters_df[[metric]].to_numpy(),), np.mean).standard_error for metric in iters_df.columns]
        median_stes = [bootstrap((iters_df[[metric]].to_numpy(),), np.median).standard_error for metric in iters_df.columns]
        
        mean_ci_df = pd.Series([f'{round(ci.low[0], 2)}-{round(ci.high[0], 2)}' for ci in mean_cis], index=iters_df.columns)
        median_ci_df = pd.Series([f'{round(ci.low[0], 2)}-{round(ci.high[0], 2)}' for ci in median_cis], index=iters_df.columns)
        mean_ste_df = pd.Series([ste[0] for ste in mean_stes], index=iters_df.columns)
        median_ste_df = pd.Series([ste[0] for ste in median_stes], index=iters_df.columns)

        # Write another pandas DataFrame containing statistics abount CV experiments.
        means_df = pd.concat([iters_df.mean(), mean_ci_df, mean_ste_df, iters_df.median(), median_ci_df,
                              median_ste_df, iters_df.min(), iters_df.max()],
                              axis=1, keys=['Mean', '95% CI Mean', 'Standard Error Mean', 'Median', '95% CI Median',
                                            'Standard Error Median', 'Min', 'Max']).transpose()

        return means_df, iters_df
   
    # Function to obtain model performance metrics broken down by groups
    # calculated from the cross-validation experiments.
    def get_group_metrics(self, beta=1):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check if this cv model has been fitted using groups.
        if self.g is None:
            raise ValueError('This cross-validation object does not contain information about groups.')
       
        # Write each xarray defining a CV iteration as a pandas Series,
        # then put all Series in a list.
        data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
        if is_regressor(self.model):
            metric_list = [GetRegMetrics(data[:, data[2, :]==g][0, :], data[:, data[2, :]==g][1, :]) for g in range(self.g_num)]
        elif is_classifier(self.model):
            metric_list = [GetClfMetrics(data[:, data[3, :]==g][0, :], data[:, data[3, :]==g][1, :], data[:, data[3, :]==g][2, :],
                                         beta=beta) for g in range(self.g_num)]
        series_list = [pd.Series(x.to_numpy(), index=x.attrs['Metric Names']) for x in metric_list]

        # Write pandas DataFrame containing statistics abount CV experiments broken down by group.
        groups_df = pd.concat(series_list, axis=1, keys=[f'Cluster {n+1}' for n in range(self.g_num)]).transpose()

        return groups_df
    
    # Function to get the bias-variance decomposition of the loss.
    def bias_variance_decomp(self):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check if loss decomposition was computed when performing cross-validation.
        if not hasattr(self, 'bias_var_arr'):
            raise ValueError('must fit CrossValidation class with loss="mse" or loss="0-1_loss" before performing\
                a bias-variance decomposition.')

        decomp_mean = np.mean(self.bias_var_arr.to_numpy(), axis=0)
        decomp_std = np.std(self.bias_var_arr.to_numpy(), axis=0)
        decomp_median = np.median(self.bias_var_arr.to_numpy(), axis=0)
        decomp_max = np.max(self.bias_var_arr.to_numpy(), axis=0)
        decomp_min = np.min(self.bias_var_arr.to_numpy(), axis=0)
        
        bias_variance_df = pd.DataFrame(np.stack([decomp_mean, decomp_std, decomp_median, decomp_max, decomp_min], axis=0),
                                        columns=self.bias_var_arr.coords['Decomposition'].to_numpy(),
                                        index=['Mean', 'Std', 'Median', 'Max', 'Min'])
        
        return bias_variance_df

    # Function to draw confusion matrix of a classifier.
    def draw_confusion_matrix(self, display='percentage'):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw confusion matrix for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_pred = np.hstack([self.cv_array.data_vars[var][1, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_pred = y_pred[~np.isnan(y_pred)]
        
        # Compute confusion matrix.
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Format according to input display.
        if display == 'percentage':
            conf_matrix = np.around((conf_matrix/conf_matrix.sum()), 2)
        elif display == 'total_count':
            pass
        else:
            raise ValueError('display must be "percentage" or "total_count"')
        
        # Draw confusion matrix.
        fig, ax = plt.subplots()
        
        if display == 'percentage':
            sns.heatmap(conf_matrix, vmin=0, vmax=1, cmap='Blues', annot=True, fmt='.2%', ax=ax)
        elif display == 'total_count':
            sns.heatmap(conf_matrix, vmin=0, vmax=conf_matrix.ravel().sum(), cmap='Blues', annot=True, fmt='.0f', ax=ax)
        ax.set_xticklabels(['Positive', 'Negative'], fontweight='bold', ha='center')
        ax.set_yticklabels(['Positive', 'Negative'], fontweight='bold', va='center')
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('True', fontsize=14, fontweight='bold', labelpad=15)
        ax.tick_params(axis='both', which='major', pad=10, labelsize=12)
        
        # Setup colorbar.
        cbar = ax.collections[0].colorbar
        if display == 'percentage':
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        return fig, ax
    
    # Function to plot the area under the ROC curve.
    def draw_auc_roc(self):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw ROC curve for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_prob = np.hstack([self.cv_array.data_vars[var][2, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_prob = y_prob[~np.isnan(y_prob)]
        
        # Compute ROC curve.
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Compute area under ROC curve.
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve.
        fig, ax = plt.subplots()
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot(fpr, tpr, label=f'ROC Curve (area={round(roc_auc, 2)})', color='darkorange')
        ax.plot(fpr, fpr, linestyle='dashed', color='blue')
        ax.legend(loc='best')
        
        return fig, ax

    # Function to plot the area under the ROC curve.
    def draw_det_curve(self):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw ROC curve for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_prob = np.hstack([self.cv_array.data_vars[var][2, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_prob = y_prob[~np.isnan(y_prob)]
        
        # Compute ROC curve.
        fpr, tpr, _ = det_curve(y_true, y_prob)
        
        # Plot ROC curve.
        fig, ax = plt.subplots()
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f'{20*x}%' for x in np.arange(6)])
        ax.set_yticklabels([f'{20*y}%' for y in np.arange(6)])
        ax.plot(fpr, tpr, color='darkorange')
        
        return fig, ax
    
    # Function to plot predictions from all cross-validation experiments.
    def draw_cv(self, group=None, sel_size=1.0, threshold=None, **kwargs):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that model is a regressor.
        if not is_regressor(self.model):
            raise ValueError ('Model must be a regressor in order to plot prediction values.')
        
        # If a group is given as input, then only that group is plotted.
        # Here we remove data not belonging to that group if group is given
        # and then calculate performance metrics.
        if group is None:
            data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
            rmse, r2 = np.around(GetRegMetrics(data[0, :], data[1, :]).loc[['rmse', 'r2']].to_numpy(), 2)
        else:
            data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
            data = data[:, data[2, :] == group]
            rmse, r2 = np.around(GetRegMetrics(data[0, :], data[1, :]).loc[['rmse', 'r2']].to_numpy(), 2)

        # If selection size is smaller than 1, then only plot a randomly chosen selection
        # of points. In case there is too much data and plot looks cluttered.
        if sel_size < 1.0:
            sample_num = int(data.shape[1]*sel_size)
            mask = np.random.choice(np.arange(data.shape[1]), sample_num, replace=False)
            data = data[:, mask]
        
        # Plot the data.
        # If threshold is given, find the points that are within the threshold and those that are not,
        # then color them differently. Also draw dashed lines representing these thresholds.
        if threshold is not None:
            # Find points over and under threshold and the ratio of points that is under it.
            diff = np.absolute(data[0, :] - data[1, :])
            over_test = data[0, diff > threshold]
            over_pred = data[1, diff > threshold]
            under_test = data[0, diff <= threshold]
            under_pred = data[1, diff <= threshold]
            ratio = round(len(under_pred)/len(diff), 2)
            
            # Plot data.
            fig, ax = plt.subplots()
            
            ax.scatter(over_test, over_pred, facecolor='red', edgecolor='grey', **kwargs)
            ax.scatter(under_test, under_pred, facecolor='green', edgecolor='grey', **kwargs)
            
            space = np.linspace(np.min(data[0, :]), np.max(data[0, :]), 20)
            over_space = np.linspace(np.min(data[0, :]) + threshold, np.max(data[0, :]) + threshold, 20)
            under_space = np.linspace(np.min(data[0, :]) - threshold, np.max(data[0, :]) - threshold, 20)
            
            ax.plot(space, space, color='black', linewidth=3,
                    label=f'RMSE: {rmse}; R\N{SUPERSCRIPT TWO}: {r2}; Ratio: {ratio}')
            ax.plot(space, over_space, space, under_space, color='black', linestyle='dashed')
            
            ax.legend(loc='best')
        
        # If threshold is not given, simply plot data with colorless points.
        else:
            fig, ax = plt.subplots()

            ax.scatter(data[0, :], data[1, :], facecolor='none', edgecolor='grey', **kwargs)
            
            space = np.linspace(np.min(data[0, :]), np.max(data[0, :]), 20)            
            
            ax.plot(space, space, color='black', linewidth=3, label=f'RMSE: {rmse}; R\N{SUPERSCRIPT TWO}: {r2}')
            
            ax.legend(loc='best')

        return fig, ax        
      

# Class to perform nested cross-validation.
class NestedCrossValidation:
    
    def __init__(self, model, params, feature_selection_strategy='forward_sfs', top_features=None,
                 genetic_kwargs={}, hyperparameter_selection_strategy='grid_search', random_iters=1000,
                 inner_strategy='k-fold', outer_strategy='k-fold', inner_cv_folds=5, outer_cv_folds=10,
                 opt_score='neg_mean_squared_error', inner_iters=1, outer_iters=1, shuffle=False, loss=None):

        # Set input data as attributes.
        self.model = model
        self.params = params
        self.opt_score = opt_score
        self.feature_selection_strategy = feature_selection_strategy
        self.top_features = top_features
        self.genetic_kwargs = genetic_kwargs
        self.hyperparameter_selection_strategy = hyperparameter_selection_strategy
        self.random_iters = random_iters
        self.inner_strategy = inner_strategy
        self.outer_strategy = outer_strategy
        self.inner_cv_folds = inner_cv_folds
        self.outer_cv_folds = outer_cv_folds
        self.opt_score = opt_score
        if inner_strategy == 'k-fold' and inner_iters > 1:
            self.inner_iters = 1
        else:
            self.inner_iters = inner_iters
        if outer_strategy == 'k-fold' and outer_iters > 1:
            self.outer_iters = 1
        else:
            self.outer_iters = outer_iters
        self.shuffle = shuffle
        self.loss = loss
    
    # Function to perform nested cross-validation.
    def fit(self, X, y, group=None, feature_names=None):
        
        self.X = X
        self.y = y
        self.g = group
        if self.g is not None:
            self.g_num = np.max(self.g) + 1
        # By default (if number of top features is not specified) use 80% of available features.
        if self.top_features is None:
            self.top_features = int(0.8*self.X.shape[1])
        clone_model = clone(self.model)
        
        # Define inner cross-validation splitting object and name of the indices based on
        # input cross-validation strategy.
        if self.inner_strategy == 'k-fold':
            self.inner_cv = KFold(n_splits=self.inner_cv_folds, shuffle=self.shuffle)
            self.inner_indices = [f'Fold {i+1}' for i in range(self.inner_cv_folds)]
        elif self.inner_strategy == 'repeated_k-fold':
            if self.inner_iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.inner_cv = RepeatedKFold(n_splits=self.inner_cv_folds, n_repeats=self.inner_iters)
            self.inner_indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.inner_cv_folds)
                                  for i in range(self.inner_iters)]
        elif self.inner_strategy == 'stratified_k-fold':
            self.inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=self.shuffle)
            self.inner_indices = [f'Fold {i+1}' for i in range(self.inner_cv_folds)]
        elif self.inner_strategy == 'repeated_stratified_k-fold':
            if self.inner_iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.inner_cv = RepeatedStratifiedKFold(n_splits=self.inner_cv_folds, n_repeats=self.inner_iters)
            self.inner_indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.inner_cv_folds)
                                  for i in range(self.inner_iters)]
        else:
            raise ValueError('Strategy for inner cross-validation must be one of: k-fold, repeated_k-fold,\
                stratified_k-fold or repeated_stratified_k-fold.')
        
        # Do the same for outer cross-validation.
        if self.outer_strategy == 'k-fold':
            self.outer_cv = KFold(n_splits=self.outer_cv_folds, shuffle=self.shuffle)
            self.outer_indices = [f'Fold {i+1}' for i in range(self.outer_cv_folds)]
        elif self.outer_strategy == 'repeated_k-fold':
            if self.outer_iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.outer_cv = RepeatedKFold(n_splits=self.outer_cv_folds, n_repeats=self.outer_iters)
            self.outer_indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.outer_cv_folds)
                                  for i in range(self.outer_iters)]
        elif self.outer_strategy == 'stratified_k-fold':
            self.outer_cv = StratifiedKFold(n_splits=self.outer_cv_folds, shuffle=self.shuffle)
            self.outer_indices = [f'Fold {i+1}' for i in range(self.outer_cv_folds)]
        elif self.outer_strategy == 'repeated_stratified_k-fold':
            if self.outer_iters <= 1:
                raise ValueError('Iters must be bigger than 1 for repeated k-fold.')
            self.outer_cv = RepeatedStratifiedKFold(n_splits=self.outer_cv_folds, n_repeats=self.outer_iters)
            self.outer_indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.outer_cv_folds)
                                  for i in range(self.outer_iters)]
        else:
            raise ValueError('Strategy for outer cross-validation must be one of: k-fold, repeated_k-fold,\
                stratified_k-fold or repeated_stratified_k-fold.')
        
        # Create feature search algorithm.
        if self.feature_selection_strategy is None:
            pass
        elif self.feature_selection_strategy == 'forward_sfs':
            feature_search = SFS(clone_model, k_features=self.top_features, forward=True, floating=False,
                                 scoring=self.opt_score, cv=self.inner_cv, n_jobs=cpu_num)
        elif self.feature_selection_strategy == 'backward_sfs':
            feature_search = SFS(clone_model, k_features=self.top_features, forward=False, floating=False,
                                 scoring=self.opt_score, cv=self.inner_cv, n_jobs=cpu_num)
        elif self.feature_selection_strategy == 'forward_sffs':
            feature_search = SFS(clone_model, k_features=self.top_features, forward=True, floating=True,
                                 scoring=self.opt_score, cv=self.inner_cv, n_jobs=cpu_num)
        elif self.feature_selection_strategy == 'backward_sffs':
            feature_search = SFS(clone_model, k_features=self.top_features, forward=False, floating=True,
                                 scoring=self.opt_score, cv=self.inner_cv, n_jobs=cpu_num)
        elif self.feature_selection_strategy == 'genetic':
            feature_search = GeneticSelectionCV(clone_model, scoring=self.opt_score, cv=self.inner_cv,
                                                n_jobs=cpu_num, **self.genetic_kwargs)
        else:
            raise ValueError ('Input feature selection strategy is invalid. Valid options are forward_sfs and\
            backward_sfs for Forward/Backward Sequential Feature Selection, forward_sffs and backward_sffs\
            for Forward/Backward Sequential Floating Feature Selection and genetic for the Genetic Algorithm.')

        # Create hyperparameter search algorithm.
        if self.hyperparameter_selection_strategy == 'grid_search':
            parameter_search = GridSearchCV(clone_model, self.params, cv=self.inner_cv, scoring=self.opt_score,
                                       n_jobs=cpu_num)
        elif self.hyperparameter_selection_strategy == 'random_search':
            parameter_search = RandomizedSearchCV(clone_model, self.params, n_iter=self.random_iters,
                                               cv=self.inner_cv, scoring=self.opt_score, n_jobs=cpu_num)
        elif self.hyperparameter_selection_strategy == 'halving_grid_search':
            parameter_search = HalvingGridSearchCV(clone_model, self.params, factor=3, cv=self.inner_cv,
                                                   scoring=self.opt_score, n_jobs=cpu_num)
        elif self.hyperparameter_selection_strategy == 'random_halving_search':
            parameter_search = HalvingRandomSearchCV(clone_model, self.params, n_candidates=self.random_iters,
                                                   cv=self.inner_cv, scoring=self.opt_score, n_jobs=cpu_num)
        else:
            raise ValueError('Invalid input hyperparameter_selection_strategy. Must be one of grid_search,\
                  random_search, halving_grid_search or random_halving_search.')
        
        # Function that splits dataset following given cross-validation strategy.
        def cv_split(self, split_model, X, y, g):
            
            # Save info about all train and test sets for all splits in this dictionary.
            cv_splitter = {}
            
            # Cross-validation with groups.
            if g is not None:
                for n, (train_index, test_index) in enumerate(split_model.split(X, y, g)):

                    # Save info about train and test set for this split in this dictionary.
                    split_dict = {}

                    # Calculate train and test sets for this cv iteration.
                    split_dict['X_train'] = X[train_index, :]
                    split_dict['y_train'] = y[train_index]
                    split_dict['group_train'] = g[train_index]
                    split_dict['X_test'] = X[test_index, :]
                    split_dict['y_test'] = y[test_index]
                    split_dict['group_test'] = g[test_index]

                    cv_splitter[f'Iter {n+1}'] = split_dict

            # Cross-validation without groups.
            else:
                for n, (train_index, test_index) in enumerate(split_model.split(X, y)):

                    # Save info about train and test set for this split in this dictionary.
                    split_dict = {}

                    # Calculate train and test sets for this cv iteration.
                    split_dict['X_train'] = X[train_index, :]
                    split_dict['y_train'] = y[train_index]
                    split_dict['X_test'] = X[test_index, :]
                    split_dict['y_test'] = y[test_index]

                    cv_splitter[f'Iter {n+1}'] = split_dict

            return cv_splitter

        # Create initial train and test set from the total dataset.
        X_cv = self.X
        y_cv = self.y.reshape(-1,)
        if self.g is not None:
            g_cv = self.g.reshape(-1,)
        fold_len = int(len(y_cv)/self.outer_cv_folds) + len(y_cv)%self.outer_cv_folds
        
        # Create outer cross-validation split.
        if self.g is not None:
            self.outer_cv_split = cv_split(self, self.outer_cv, X_cv, y_cv, g_cv)
        else:
            self.outer_cv_split = cv_split(self, self.outer_cv, X_cv, y_cv, None)
        
        # Variables to store info about each CV iteration.
        feature_dfs_list = []
        params_dfs_list = []
        cv_array = xr.Dataset()
        bias_var_arr = np.zeros((self.outer_cv_folds*self.outer_iters, 3))
        times = []
        
        # Nested cross-validation loop.
        for n, outsplit in enumerate(self.outer_cv_split.values()):
            
            # Start measuring time.
            start = timeit.default_timer()
            
            # Estimate in how much time the process will end.
            if times:
                mean_t = np.array(times).mean()
                t_estimation = Utils.FormattedTime(mean_t*(len(self.outer_cv_split.values()) - n))
            
            # Print progress.
            if times:
                print(f'\rNested cross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}; Estimated time remaining: {t_estimation}',
                      end='     ', flush=True)
            else:
                print(f'\rNested cross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}', end='     ', flush=True)
            
            # Find best features and create train and test sets.
            if self.feature_selection_strategy is not None:
                self.best_features = feature_search.fit(outsplit['X_train'], outsplit['y_train'])
                if self.feature_selection_strategy == 'genetic':
                    best_feature_idx = np.arange(len(self.best_features.support_))[self.best_features.support_]
                else:
                    best_feature_idx = np.array(self.best_features.k_feature_idx_)
            else:
                best_feature_idx = np.arange(outsplit['X_train'].shape[1])

            X_train_tmp = outsplit['X_train'][:, best_feature_idx]
            X_test_tmp = outsplit['X_test'][:, best_feature_idx]
            y_train_tmp = outsplit['y_train'].reshape(-1,)
            y_test_tmp = outsplit['y_test'].reshape(-1,)
            
            # Find best hyperparameters.
            self.hyperparameters = parameter_search.fit(X_train_tmp, y_train_tmp)
            
            # Fit model using optimal features and hyperparameters.
            tmp_model = clone(self.model).set_params(**self.hyperparameters.best_params_)
            tmp_model.fit(X_train_tmp, y_train_tmp)
            y_pred = tmp_model.predict(X_test_tmp).reshape(-1,)
            if is_classifier(self.model):
                y_prob = tmp_model.predict_proba(X_test_tmp)[:, 1].reshape(-1,)
            
            # Format score metric name.
            if self.opt_score.startswith('neg'):
                score_name = self.opt_score.replace('neg_', '')
            else:
                score_name = self.opt_score
            score_name = ' '.join([word.capitalize() for word in score_name.split('_')])
            
            # Write feature data.
            # Determine which features were choesen and which weren't.
            max_features = outsplit['X_train'].shape[1]
            chosen_features = np.array(['yes' if i in best_feature_idx else 'no' for i in range(max_features)]).reshape(-1, 1)
            
            # Add score of best subset of features.
            if self.feature_selection_strategy is not None:
                if self.feature_selection_strategy == 'genetic':
                    if self.opt_score.startswith('neg_'):
                        feature_data = np.vstack((chosen_features,
                                                -np.around(np.max(self.best_features.generation_scores_), 2)))
                    else:
                        feature_data = np.vstack((chosen_features,
                                                np.around(np.max(self.best_features.generation_scores_), 2)))
                else:
                    if self.opt_score.startswith('neg_'):
                        feature_data = np.vstack((chosen_features,
                                                -np.around(np.array([self.best_features.subsets_[self.top_features]['avg_score']]), 2)))
                    else:
                        feature_data = np.vstack((chosen_features,
                                                np.around(np.array([self.best_features.subsets_[self.top_features]['avg_score']]), 2)))
            
            # Create DataFrame for feature data and add feature names.
            if self.feature_selection_strategy is not None:
                if feature_names is None:
                    feature_dfs_list.append(pd.DataFrame(feature_data,
                                                        index=[f'Feature {i+1}' for i in range(max_features)] + [score_name],
                                                        columns=[self.outer_indices[n]]))
                else:
                    feature_dfs_list.append(pd.DataFrame(feature_data,
                                                        index=feature_names + [score_name],
                                                        columns=[self.outer_indices[n]]))
            
            # Write hyperparameter data.
            if self.opt_score.startswith('neg_'):
                hp_data = np.hstack((np.around(np.array(list(self.hyperparameters.best_params_.values())), 3),
                                np.around(-np.array(self.hyperparameters.best_score_), 2)))
            else:
                hp_data = np.hstack((np.around(np.array(list(self.hyperparameters.best_params_.values())), 2),
                                np.around(np.array(self.hyperparameters.best_score_), 2)))
            params_dfs_list.append(pd.DataFrame(hp_data,
                                                index=list(self.hyperparameters.best_params_.keys()) + [score_name],
                                                columns=[self.outer_indices[n]]))
            
            # Stack test data, predictions and groups, then fill missing values so that
            # dimensions remain the same in all iterations.
            if self.g is not None and is_regressor(self.model):
                arr = np.stack((y_test_tmp, y_pred, outsplit['group_test'].reshape(-1,)))
                if arr.shape[1] != fold_len:
                    empty_arr = np.empty((3, fold_len - arr.shape[1]))
                    empty_arr[:, :] = np.nan
                    arr = np.hstack((arr, empty_arr))
                
                # Write xarray for this iteration.
                xarr = xr.DataArray(arr,
                                    dims=('Type', 'Sample'),
                                    coords={'Type': ['Test Set Samples', 'Predictions', 'Groups'],
                                            'Sample': np.arange(1, fold_len+1)}, name=self.outer_indices[n])
            
            elif self.g is not None and is_classifier(self.model):
                arr = np.stack((y_test_tmp, y_pred, y_prob, outsplit['group_test'].reshape(-1,)))
                if arr.shape[1] != fold_len:
                    empty_arr = np.empty((4, fold_len - arr.shape[1]))
                    empty_arr[:, :] = np.nan
                    arr = np.hstack((arr, empty_arr))
        
                # Write xarray for this iteration.
                xarr = xr.DataArray(arr,
                                    dims=('Type', 'Sample'),
                                    coords={'Type': ['Test Set Samples', 'Predictions', 'Probabilities', 'Groups'],
                                            'Sample': np.arange(1, fold_len+1)}, name=self.outer_indices[n])
            
            elif self.g is None and is_regressor(self.model):
                arr = np.stack((y_test_tmp, y_pred))
                if arr.shape[1] != fold_len:
                    empty_arr = np.empty((2, fold_len - arr.shape[1]))
                    empty_arr[:, :] = np.nan
                    arr = np.hstack((arr, empty_arr))
                
                # Write xarray for this iteration.
                xarr = xr.DataArray(arr,
                                    dims=('Type', 'Sample'),
                                    coords={'Type': ['Test Set Samples', 'Predictions'],
                                            'Sample': np.arange(1, fold_len+1)}, name=self.outer_indices[n])
               
            
            elif self.g is None and is_classifier(self.model):
                arr = np.stack((y_test_tmp, y_pred, y_prob))
                if arr.shape[1] != fold_len:
                    empty_arr = np.empty((3, fold_len - arr.shape[1]))
                    empty_arr[:, :] = np.nan
                    arr = np.hstack((arr, empty_arr))
                
                # Write xarray for this iteration.
                xarr = xr.DataArray(arr,
                                    dims=('Type', 'Sample'),
                                    coords={'Type': ['Test Set Samples', 'Predictions', 'Probabilities'],
                                            'Sample': np.arange(1, fold_len+1)}, name=self.outer_indices[n])
            
            # Add xarray from this iteration to Dataset that stores all iterations.
            cv_array[xarr.name] = xarr
            
            # Calculate bias-variance decomposition of the loss for this CV iteration.
            if self.loss is not None:
                avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clone_model, X_train_tmp, y_train_tmp,
                                                                            X_test_tmp, y_test_tmp, loss=self.loss)
                bias_var_arr[n, :] = np.array([avg_expected_loss, avg_bias, avg_var])
                
            # End measuring time.
            end = timeit.default_timer()
            times.append(end-start)
        
        # Create xarray with all bias variance data combined.
        bias_var_arr = xr.DataArray(bias_var_arr,
                                    dims=('Sample', 'Decomposition'),
                                    coords={'Sample': np.arange(1, self.outer_cv_folds*self.outer_iters + 1),
                                            'Decomposition': ['Average Loss', 'Average Bias', 'Average Variance']})
        
        # Add feature, hyperparameter, model prediction and bias-variance data as class attributes.
        if self.feature_selection_strategy is not None:
            self.features_df = pd.concat(feature_dfs_list, axis=1)
            self.features_df['Total Times Used'] = [np.count_nonzero(self.features_df.loc[idx, :].to_numpy() == 'yes')
                                                for idx in self.features_df.index[:-1]] + ['-']
        self.params_df = pd.concat(params_dfs_list, axis=1)
        
        self.params_df[['Mean Values', 'Median Values', 'Modes', 'Standard Deviations']] = np.around(
            np.vstack((np.mean(self.params_df.to_numpy(), axis=1),
                       np.median(self.params_df.to_numpy(), axis=1),
                       mode(self.params_df.to_numpy(), axis=1)[0].reshape(-1,),
                       np.std(self.params_df.to_numpy(), axis=1))).transpose(), 2)

        self.cv_array = cv_array
        self.bias_var_arr = bias_var_arr
        
        # Notify that Nested cross-validation procedure was completed successfully and the total time used for computation.
        print(f'\rNested cross-validation evaluation successfully completed. Total time: {Utils.FormattedTime(np.array(times).sum())}',
              end='     ')

    # Function to calculate permutation score of a model.
    def permutation_score(self, scoring=None, n_permutations=100, **kwargs):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
        if scoring is None and is_regressor(self.model):
            scoring = 'neg_mean_squared_error'
        elif scoring is None and is_classifier(self.model):
            scoring = 'roc_auc'
        
        # Calculate the score of the input model, the scores of the permutations
        # and the empirical p-value.
        score, perm_scores, pvalue = permutation_test_score(self.model, self.X, self.y, scoring=scoring,
                                                            cv=self.outer_cv, n_permutations=n_permutations)
        
        # Format input scoring metric so that it look better in the final figure.
        if scoring.startswith('neg_'):
            scoring = scoring.replace('neg_', '')
            score = -score
            perm_scores = -perm_scores
        scoring = ' '.join([word.capitalize() for word in scoring.replace('_', ' ').split()])
        
        # Draw distribution of permutation scores as histogram and
        # the score of the input model as a vertical dashed line.
        fig, ax = plt.subplots()
        
        ax.hist(perm_scores, bins=25, density=True, **kwargs)
        ax.axvline(score, linestyle='--', color='r', label=f'p-value: {round(pvalue, 5)}')
        ax.set_xlabel(scoring, fontsize=11, fontweight='bold', labelpad=12.5)
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold', labelpad=12.5)
        ax.legend(loc='best')
        
        return fig, ax

    # Function to obtain model performance metrics calculated from the
    # cross-validation experiments.
    def get_cv_metrics(self, beta=1):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Write each xarray defining a CV iteration as a pandas Series,
        # then put all Series in a list.
        cv_list = [self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars]
        if is_regressor(self.model):
            metric_list = [GetRegMetrics(metric_arr[0, :].to_numpy(), metric_arr[1, :].to_numpy()) for metric_arr in cv_list]
        elif is_classifier(self.model):
            metric_list = [GetClfMetrics(metric_arr[0, :].to_numpy(), metric_arr[1, :].to_numpy(), metric_arr[2, :].to_numpy(), beta=beta) for metric_arr in cv_list]
        series_list = [pd.Series(x.to_numpy(), index=x.attrs['Metric Names']) for x in metric_list]

        # Write pandas DataFrame by mergin all Series in Series list.
        iters_df = pd.concat(series_list, axis=1, keys=self.outer_indices).transpose()
        
        # Compute 95% confidence intervals and standard erros and write them in pandas Series.
        mean_cis = [bootstrap((iters_df[[metric]].to_numpy(),), np.mean).confidence_interval for metric in iters_df.columns]
        #median_cis = [bootstrap((iters_df[[metric]].to_numpy(),), np.median).confidence_interval for metric in iters_df.columns]
        mean_stes = [bootstrap((iters_df[[metric]].to_numpy(),), np.mean).standard_error for metric in iters_df.columns]
        #median_stes = [bootstrap((iters_df[[metric]].to_numpy(),), np.median).standard_error for metric in iters_df.columns]
        
        mean_ci_df = pd.Series([f'{round(ci.low[0], 2)}-{round(ci.high[0], 2)}' for ci in mean_cis], index=iters_df.columns)
        #median_ci_df = pd.Series([f'{round(ci.low[0], 2)}-{round(ci.high[0], 2)}' for ci in median_cis], index=iters_df.columns)
        mean_ste_df = pd.Series([ste[0] for ste in mean_stes], index=iters_df.columns)
        #median_ste_df = pd.Series([ste[0] for ste in median_stes], index=iters_df.columns)

        # Write another pandas DataFrame containing statistics abount CV experiments.
        means_df = pd.concat([iters_df.mean(), mean_ci_df, mean_ste_df, iters_df.median(), iters_df.min(), iters_df.max()],
                              axis=1, keys=['Mean', '95% CI Mean', 'Standard Error Mean', 'Median', 'Min', 'Max']).transpose()
        #means_df = pd.concat([iters_df.mean(), mean_ci_df, mean_ste_df, iters_df.median(), median_ci_df,
        #                      median_ste_df, iters_df.min(), iters_df.max()],
        #                      axis=1, keys=['Mean', '95% CI Mean', 'Standard Error Mean', 'Median', '95% CI Median',
        #                                    'Standard Error Median', 'Min', 'Max']).transpose()

        return means_df, iters_df

    # Function to obtain model performance metrics broken down by groups
    # calculated from the cross-validation experiments.
    def get_group_metrics(self, beta=1):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check if this cv model has been fitted using groups.
        if self.g is None:
            raise ValueError('This cross-validation object does not contain information about groups.')
       
        # Write each xarray defining a CV iteration as a pandas Series,
        # then put all Series in a list.
        data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
        if is_regressor(self.model):
            metric_list = [GetRegMetrics(data[:, data[2, :]==g][0, :], data[:, data[2, :]==g][1, :]) for g in range(self.g_num)]
        elif is_classifier(self.model):
            metric_list = [GetGroupClfMetrics(data[:, data[3, :]==g][0, :], data[:, data[3, :]==g][1, :], data[:, data[3, :]==g][2, :],
                                         beta=beta) for g in range(self.g_num)]
        series_list = [pd.Series(x.to_numpy(), index=x.attrs['Metric Names']) for x in metric_list]

        # Write pandas DataFrame containing statistics abount CV experiments broken down by group.
        groups_df = pd.concat(series_list, axis=1, keys=[f'Cluster {n+1}' for n in range(self.g_num)]).transpose()

        return groups_df

    # Function to get the bias-variance decomposition of the loss.
    def bias_variance_decomp(self):

        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check if loss decomposition was computed when performing cross-validation.
        if not hasattr(self, 'bias_var_arr'):
            raise ValueError('Must fit NestedCrossValidation class with loss="mse" or loss="0-1_loss" before performing\
                   a bias-variance decomposition.')
        
        decomp_mean = np.mean(self.bias_var_arr.to_numpy(), axis=0)
        decomp_std = np.std(self.bias_var_arr.to_numpy(), axis=0)
        decomp_median = np.median(self.bias_var_arr.to_numpy(), axis=0)
        decomp_max = np.max(self.bias_var_arr.to_numpy(), axis=0)
        decomp_min = np.min(self.bias_var_arr.to_numpy(), axis=0)
        
        bias_variance_df = pd.DataFrame(np.stack([decomp_mean, decomp_std, decomp_median, decomp_max, decomp_min], axis=0),
                                        columns=self.bias_var_arr.coords['Decomposition'].to_numpy(),
                                        index=['Mean', 'Std', 'Median', 'Max', 'Min'])
        
        return bias_variance_df

    # Function to draw confusion matrix of a classifier.
    def draw_confusion_matrix(self, display='percentage'):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw confusion matrix for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_pred = np.hstack([self.cv_array.data_vars[var][1, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_pred = y_pred[~np.isnan(y_pred)]
        
        # Compute confusion matrix.
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Format according to input display.
        if display == 'percentage':
            conf_matrix = np.around((conf_matrix/conf_matrix.sum()), 2)
        elif display == 'total_count':
            pass
        else:
            raise ValueError('display must be "percentage" or "total_count"')
        
        # Draw confusion matrix.
        fig, ax = plt.subplots()
        
        if display == 'percentage':
            sns.heatmap(conf_matrix, vmin=0, vmax=1, cmap='Blues', annot=True, fmt='.2%', ax=ax)
        elif display == 'total_count':
            sns.heatmap(conf_matrix, vmin=0, vmax=conf_matrix.ravel().sum(), cmap='Blues', annot=True, fmt='.0f', ax=ax)
        ax.set_xticklabels(['Positive', 'Negative'], fontweight='bold', ha='center')
        ax.set_yticklabels(['Positive', 'Negative'], fontweight='bold', va='center')
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('True', fontsize=14, fontweight='bold', labelpad=15)
        ax.tick_params(axis='both', which='major', pad=10, labelsize=12)
        
        # Setup colorbar.
        cbar = ax.collections[0].colorbar
        if display == 'percentage':
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        return fig, ax
    
    # Function to plot the area under the ROC curve.
    def draw_auc_roc(self):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw ROC curve for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_prob = np.hstack([self.cv_array.data_vars[var][2, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_prob = y_prob[~np.isnan(y_prob)]
        
        # Compute ROC curve.
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Compute area under ROC curve.
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve.
        fig, ax = plt.subplots()
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot(fpr, tpr, label=f'ROC Curve (area={round(roc_auc, 2)})', color='darkorange')
        ax.plot(fpr, fpr, linestyle='dashed', color='blue')
        ax.legend(loc='best')
        
        return fig, ax

    # Function to plot the area under the ROC curve.
    def draw_det_curve(self):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that the model is a classifier.
        if not is_classifier(self.model):
            raise ValueError('Can only draw ROC curve for a classifier estimator.')
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([self.cv_array.data_vars[var][0, :].to_numpy() for var in self.cv_array.data_vars])
        y_prob = np.hstack([self.cv_array.data_vars[var][2, :].to_numpy() for var in self.cv_array.data_vars])
        
        # Remove NaN values.
        y_true = y_true[~np.isnan(y_true)]
        y_prob = y_prob[~np.isnan(y_prob)]
        
        # Compute ROC curve.
        fpr, tpr, _ = det_curve(y_true, y_prob)
        
        # Plot ROC curve.
        fig, ax = plt.subplots()
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f'{20*x}%' for x in np.arange(6)])
        ax.set_yticklabels([f'{20*y}%' for y in np.arange(6)])
        ax.plot(fpr, tpr, color='darkorange')
        
        return fig, ax

    # Function to plot predictions from all cross-validation experiments.
    def draw_cv(self, group=None, sel_size=1.0, threshold=None, **kwargs):
        
        # Check if cv model is fitted.
        if not hasattr(self, 'cv_array'):
            raise ValueError('Cross-validation model is not fitted.')
        
        # Check that model is a regressor.
        if not is_regressor(self.model):
            raise ValueError ('Model must be a regressor in order to plot prediction values.')
        
        # If a group is given as input, then only that group is plotted.
        # Here we remove data not belonging to that group if group is given
        # and then calculate performance metrics.
        if group is None:
            data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
            rmse, r2 = np.around(GetRegMetrics(data[0, :], data[1, :]).loc[['rmse', 'r2']].to_numpy(), 2)
        else:
            data = np.hstack([self.cv_array[iteration].dropna('Sample') for iteration in self.cv_array.data_vars])
            data = data[:, data[2, :] == group]
            rmse, r2 = np.around(GetRegMetrics(data[0, :], data[1, :]).loc[['rmse', 'r2']].to_numpy(), 2)

        # If selection size is smaller than 1, then only plot a randomly chosen selection
        # of points. In case there is too much data and plot looks cluttered.
        if sel_size < 1.0:
            sample_num = int(data.shape[1]*sel_size)
            mask = np.random.choice(np.arange(data.shape[1]), sample_num, replace=False)
            data = data[:, mask]
        
        # Plot the data.
        # If threshold is given, find the points that are within the threshold and those that are not,
        # then color them differently. Also draw dashed lines representing these thresholds.
        if threshold is not None:
            # Find points over and under threshold and the ratio of points that is under it.
            diff = np.absolute(data[0, :] - data[1, :])
            over_test = data[0, diff > threshold]
            over_pred = data[1, diff > threshold]
            under_test = data[0, diff <= threshold]
            under_pred = data[1, diff <= threshold]
            ratio = round(len(under_pred)/len(diff), 2)
            
            # Plot data.
            fig, ax = plt.subplots()
            
            ax.scatter(over_test, over_pred, facecolor='red', edgecolor='grey', **kwargs)
            ax.scatter(under_test, under_pred, facecolor='green', edgecolor='grey', **kwargs)
            
            space = np.linspace(np.min(data[0, :]), np.max(data[0, :]), 20)
            over_space = np.linspace(np.min(data[0, :]) + threshold, np.max(data[0, :]) + threshold, 20)
            under_space = np.linspace(np.min(data[0, :]) - threshold, np.max(data[0, :]) - threshold, 20)
            
            ax.plot(space, space, color='black', linewidth=3,
                    label=f'RMSE: {rmse}; R\N{SUPERSCRIPT TWO}: {r2}; Ratio: {ratio}')
            ax.plot(space, over_space, space, under_space, color='black', linestyle='dashed')
            
            ax.legend(loc='best')

        # If threshold is not given, simply plot data with colorless points.
        else:
            fig, ax = plt.subplots()

            ax.scatter(data[0, :], data[1, :], facecolor='none', edgecolor='grey', **kwargs)
            
            space = np.linspace(np.min(data[0, :]), np.max(data[0, :]), 20)            
            
            ax.plot(space, space, color='black', linewidth=3, label=f'RMSE: {rmse}; R\N{SUPERSCRIPT TWO}: {r2}')
            
            ax.legend(loc='best')

        return fig, ax
