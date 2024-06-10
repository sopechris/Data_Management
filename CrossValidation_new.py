import timeit
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import bootstrap, mode
from sklearn.base import is_classifier, is_regressor, clone
from sklearn.metrics import confusion_matrix, roc_curve, auc, det_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import permutation_test_score
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector
from genetic_selection import GeneticSelectionCV
from Enumeratos import *
from Utils_new import *
from CustomExceptions import *
from FeatureAnalysis_new import Observation
import Utils

cpu_num = Utils.cpu_num


# Best subset selection algorithm classes.
class HPSelector:
    """Class to create best hyperparameter subset selection algorithm."""
    
    def __init__(self, strategy, params, scoring, cv_model, iters=None, factor=None):
        
        self.strategy = HPStrategy(strategy)
        self.params = params
        self.scoring = scoring
        self.cv_model = cv_model
        self.iters = iters
        self.factor = factor
    
    def _validate_input(self):
        """Method to check if input is consistent.

        Raises:
            InvalidHyperparameterSelectorError: Raised if neccessary input for the selected strategy wasn't given.
        """
        
        if self.strategy == HPStrategy.RANDOMIZED and self.iters is None:
            raise InvalidHyperparameterSelectorError(
                '"iters" must be specified if the chosen algorithm is "randomized_search".'
            )
        elif self.strategy == HPStrategy.HALVING and self.factor is None:
            raise InvalidHyperparameterSelectorError(
                '"factor" must be specified if the chosen algorithm is "halving_search".'
            )
    
    def fit(self, model, X_train, y_train):
        """Fit input model and data to hyperparameter selection algorithm.

        Args:
            model (object): Sklearn regressor or classifier.
            X_train (np.array): Feature matrix.
            y_train (np.array): Respose variable matrix.
        """
        
        # Check that all neccessary input was given.
        self._validate_input()
        
        # Create hyperparameter selection strategy based on input.
        if self.strategy == HPStrategy.GRID:
            hp_searcher = GridSearchCV(model, self.params, scoring=self.scoring, cv=self.cv_model, n_jobs=cpu_num)
        elif self.strategy == HPStrategy.RANDOMIZED:
            hp_searcher = RandomizedSearchCV(model, self.params, scoring=self.scoring, n_iter=self.iters, cv=self.cv_model, n_jobs=cpu_num)
        elif self.strategy == HPStrategy.HALVING:
            hp_searcher = HalvingGridSearchCV(model, self.params, scoring=self.scoring, cv=self.cv_model, factor=self.factor, n_jobs=cpu_num)

        # Fit hyperparameter selection strategy to input data.
        hp_searcher.fit(X_train, y_train)
        
        # Save results info as class attributes.
        self.results = pd.DataFrame(hp_searcher.cv_results_)
        self.best_params = hp_searcher.best_params_
        selected_params = np.fromiter(hp_searcher.best_params_.values(), dtype=float)
        param_names = [*hp_searcher.best_params_] + ['Score']
        self.selected_params = pd.DataFrame(np.hstack((selected_params, np.array(hp_searcher.best_score_))),
                                        index=param_names, columns=['Results'])
    
class FSelector:
    """Class to create best hyperparameter subset selection algorithm."""
    
    def __init__(self, strategy, scoring, cv_model, max_features=None, min_features=None, top_features=None,
                 genetic_kwargs={}):
        
        self.strategy = FSStrategy(strategy)
        self.scoring = scoring
        self.cv_model = cv_model
        self.max_features = max_features
        self.min_features = min_features
        self.top_features = top_features
        self.genetic_kwargs = genetic_kwargs
    
    def _validate_input(self):
        """Method to check if input is consistent.

        Raises:
            InvalidFeatureSelectorError: Raised if neccessary input for the selected strategy wasn't given.
        """
        
        if self.strategy in (FSStrategy.SFS, FSStrategy.SBS, FSStrategy.SFFS, FSStrategy.SBFS) and self.top_features is None:
            raise InvalidFeatureSelectorError(
                f'"top_features" must be specified if the chosen algorithm is "{self.strategy.value}".'
            )
        elif self.strategy == FSStrategy.EXHAUSTIVE and (self.min_features is None or self.max_features is None):
            raise InvalidFeatureSelectorError(
                '"min_features" and "max_features" must be specified if the chosen algorithm is "exhaustive".'
            )
    
    def fit(self, model, X_train, y_train, feature_names=None):
        """Fit input model and data to feature selection algorithm.

        Args:
            model (object): Sklearn regressor or classifier.
            X_train (np.array): Feature matrix.
            y_train (np.array): Response variable matrix.
            feature_names (list, optional): List of names for input features. Defaults to None.
        """
        
        # Check that all neccessary input was given.
        self._validate_input()
        
        # If feature names are not given, crete them.
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]

        # Create feature selection strategy based on input.
        if self.strategy == FSStrategy.SFS:
            f_searcher = SequentialFeatureSelector(model, k_features=self.top_features, forward=True,
                                                   floating=False, scoring=self.scoring, cv=self.cv_model,
                                                   n_jobs=cpu_num)
        elif self.strategy == FSStrategy.SBS:
            f_searcher = SequentialFeatureSelector(model, k_features=self.top_features, forward=False,
                                                   floating=False, scoring=self.scoring, cv=self.cv_model,
                                                   n_jobs=cpu_num)
        elif self.strategy == FSStrategy.SFFS:
            f_searcher = SequentialFeatureSelector(model, k_features=self.top_features, forward=True,
                                                   floating=True, scoring=self.scoring, cv=self.cv_model,
                                                   n_jobs=cpu_num)
        elif self.strategy == FSStrategy.SBFS:
            f_searcher = SequentialFeatureSelector(model, k_features=self.top_features, forward=False,
                                                   floating=True, scoring=self.scoring, cv=self.cv_model,
                                                   n_jobs=cpu_num)
        elif self.strategy == FSStrategy.GENETIC:
            f_searcher = GeneticSelectionCV(model, scoring=self.scoring, cv=self.cv_model, n_jobs=cpu_num,
                                            **self.genetic_kwargs)
        elif self.strategy == FSStrategy.EXHAUSTIVE:
            f_searcher = ExhaustiveFeatureSelector(model, min_features=self.min_features, max_features=self.max_features,
                                                   scoring=self.scoring, cv=self.cv_model, n_jobs=cpu_num,
                                                   print_progress=False)
        
        # Fit feature selection strategy to input data.
        f_searcher.fit(X_train, y_train)
        
        # Save results info as class attributes.
        feature_names = feature_names + ['Total Features', 'Best Score']
        
        if self.strategy == FSStrategy.GENETIC:
            self.results = f_searcher.generation_scores_
            self.best_idx = f_searcher.support_
            feature_choice = ['yes' if i in f_searcher.support_ else 'no' for i in range(X_train.shape[1])] +\
                             [len(f_searcher.support_), f_searcher.generation_scores_.max()]
        elif self.strategy == FSStrategy.EXHAUSTIVE:
            self.results = f_searcher.subsets_
            self.best_idx = f_searcher.best_idx_
            feature_choice = ['yes' if i in f_searcher.best_idx_ else 'no' for i in range(X_train.shape[1])] +\
                             [len(f_searcher.best_idx_), f_searcher.best_score_]
        else:
            self.results = f_searcher.subsets_
            self.best_idx = f_searcher.k_feature_idx_
            feature_choice = ['yes' if i in f_searcher.k_feature_idx_ else 'no' for i in range(X_train.shape[1])] +\
                             [len(f_searcher.k_feature_idx_), f_searcher.k_score_]
        
        self.selected_features = pd.DataFrame(feature_choice, index=feature_names, columns=['Results'])


# Cross-validation classes.
class CrossValidation(ABC):
    """Abstract class for cross-validation classes."""
    
    @staticmethod
    def _validate_input(features, X, y, group, feature_names):
        """Method to validate that the input for a cross-validator fit method is valid.

        Raises:
            InvalidCrossValidationInputError: Raised if input for fit method was invalid.
        """

        if features is not None and any(i is not None for i in [X, y, group, feature_names]):
            raise InvalidCrossValidationInputError(
                'If a FeatureList is given as input, X, y and group matrices cannot be given as input.'
                )
        elif features is None and (X is None or y is None):
            raise InvalidCrossValidationInputError(
                'If a no FeatureList is given as input, X and y matrices must be passed as input instead.'
                )
    
    def _check_fit(self):
        """Checks that CrossValidation instance is fitted.

        Raises:
            UnfittedCVError: Raised when CrossValidation instance is not fitted.
        """
        
        if not hasattr(self, 'cv_results'):
            raise UnfittedCVError(
                'This CrossValidation instance is not fitted. Use fit() method before scoring the results.'
                )
    
    @abstractmethod
    def fit(self):
        pass
    
    def score(self):
        """Method to score the cross-validation results.

        Returns:
            ClfScores or RegScores: Dataclasses containing cross-validation result scores and method to visualize them.
        """
        
        # Check that CrossValidation instance is fitted.
        self._check_fit()
        
        # Create appropriate indices for given CV strategy.
        if self.outer_strategy == CVStrategy.KFOLD:
            indices = [f'Fold {i+1}' for i in range(self.outer_cv_folds)]
        elif self.outer_strategy == CVStrategy.REPEATED_KFOLD:
            indices = [f'Iter {i+1}, Fold {j+1}' for j in range(self.outer_cv_folds) for i in range(self.outer_iters)]
        elif self.outer_strategy == CVStrategy.LEAVE_GROUP_OUT:
            indices = [f'Group {i+1}' for i in range(self.group.max()+1)]
        
        # Get preformance scores for every cross-validation iteration.        
        if is_classifier(self.model):
            
            scores_arr = np.zeros((6, len(self.cv_results)))
            
            for n, result in enumerate(self.cv_results):
                scores_arr[:, n] = GetClfMetrics(result[0, :], result[1, :], result[2, :])

        elif is_regressor(self.model):
            
            scores_arr = np.zeros((7, len(self.cv_results)))
            
            for n, result in enumerate(self.cv_results):
                scores_arr[:, n] = GetRegMetrics(result[0, :], result[1, :])
        
        # Get performance score decomposition by group.
        if self.group is not None and self.outer_strategy != CVStrategy.LEAVE_GROUP_OUT:
            
            # Create dictionary to store results for every group.
            groups_dict = {}
            
            # Iterate over every group and then score the points that belong to that group for every cv iteration.
            for g in range(self.group.max()+1):
                
                if is_classifier(self.model):
                
                    group_arr = np.zeros((6, len(self.cv_results)))
                    
                    for n, result in enumerate(self.cv_results):
                        group_arr[:, n] = GetClfMetrics(result[0, :][result[3, :] == g], result[1, :][result[3, :] == g],
                                                        result[2, :][result[3, :] == g])
                    
                    groups_dict[f'group {g}'] = group_arr
                
                elif is_regressor(self.model):
                    
                    group_arr = np.zeros((7, len(self.cv_results)))
                    
                    for n, result in enumerate(self.cv_results):
                        group_arr[:, n] = GetRegMetrics(result[0, :][result[2, :] == g], result[1, :][result[2, :] == g])
                    
                    groups_dict[f'group {g}'] = group_arr

            # Return CVScores instance.
            if is_classifier(self.model):
                return ClfScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 self.cv_results, groups_dict, indices)
            elif is_regressor(self.model):
                return RegScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 scores_arr[6, :], self.cv_results, groups_dict, indices)
        
        elif self.group is not None and self.outer_strategy == CVStrategy.LEAVE_GROUP_OUT:
            # Decompose results by group, which in this case is the same as decomposing by iteration.
            groups_dict = {}
            for g in range(scores_arr.shape[1]):
                groups_dict[f'group {g}'] = scores_arr[:, g].reshape(-1, 1)
        
            # Return CVScores instance.
            if is_classifier(self.model):
                return ClfScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 self.cv_results, groups_dict, indices)
            elif is_regressor(self.model):
                return RegScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 scores_arr[6, :], self.cv_results, groups_dict, indices)

        else:
            # Return CVScores instance.
            if is_classifier(self.model):
                return ClfScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 self.cv_results, None, indices)
            if is_regressor(self.model):
                return RegScores(scores_arr[0, :], scores_arr[1, :], scores_arr[2, :], scores_arr[3, :], scores_arr[4, :], scores_arr[5, :],
                                 scores_arr[6, :], self.cv_results, None, indices)

    def permutation_score(self, n_permutations=100, scoring=None, cv=10, style='hist'):
        """Method to estimate the significance of a cross-validated score. Pertutates the response
           feature and computes the empirical p-value against the null hypothesis that features and
           targets are independent.

        Args:
            n_permutations (int, optional): Number of times the reponse feature is permutated. Defaults to 100.
            scoring (str, optional): Score metric used to evaluate the permutation score. Defaults to None.
            cv (int, optional): Number of folds to use for cross-validation. Defaults to 10.
            style (str, optional): Style of plot to use to represent the premutation scores. Cand be either "hist"
                                   or "kde". Defaults to 'hist'.

        Raises:
            UnfittedCVError: Raised when the CrossValidation object has not yet been fitted. Must fit before using this method.

        Returns:
            figure, axis: Figure and axis of the plot.
        """
        
        # Check that CrossValidation instance is fitted.
        self._check_fit()
        
        # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
        if scoring is None and is_regressor(self.model):
            scoring = 'neg_mean_squared_error'
        elif scoring is None and is_classifier(self.model):
            scoring = 'roc_auc'
        
        # Create cross-validation splitter object.
        cv_model = GetCVSplitter(self.model, 'k-fold', cv_folds=cv)
        
        # Calculate the score of the input model, the scores of the permutations
        # and the empirical p-value.
        score, perm_scores, pvalue = permutation_test_score(self.model, self.X, self.y, scoring=scoring,
                                                            cv=cv_model, n_permutations=n_permutations)
        
        # Format input scoring metric so that it look better in the final figure.
        if scoring.startswith('neg_'):
            scoring = scoring.replace('neg_', '')
            score = -score
            perm_scores = -perm_scores
        scoring = ' '.join([word.capitalize() for word in scoring.split('_')])
        
        # Draw distribution of permutation scores as histogram and
        # the score of the input model as a vertical dashed line.
        fig, ax = plt.subplots()
        
        if Styles(style) == Styles.HIST:
            ax.hist(perm_scores, bins=25, density=True)
        elif Styles(style) == Styles.KDE:
            sns.kdeplot(perm_scores, ax=ax)
        ax.axvline(score, linestyle='dashed', color='red', label=f'p-value: {round(pvalue, 5)}')
        ax.set_xlabel(scoring, fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold', labelpad=15)
        ax.legend(loc='best')
        
        return fig, ax
    
class KFoldCV(CrossValidation):

    def __init__(self, model, strategy='k-fold', cv_folds=10, iters=1, shuffle=True):
        
        super().__init__()
        self.model = clone(model)
        self.outer_strategy = CVStrategy(strategy)
        self.outer_cv_folds = cv_folds
        self.outer_iters = iters
        self.shuffle = shuffle

    def fit(self, dataset=None, X=None, y=None, group=None):
        """Method to fit cross-validator to input data.

        Args:
            features (FeatureList, optional): FeatureList object to use as data for cross-validator. Defaults to None.
            X (np.array, optional): Feature array used to fit cross-validator. Alternative to features, don't use both. Defaults to None.
            y (np.array, optional): Response variable array used to fit cross-validator. Alternative to features, don't use both. Defaults to None.
            group (np.array, optional): Array that specifies which group each observation belongs to. Alternative to features, don't use both. Defaults to None.
            feature_names (list, optional): List with names of input features. Alternative to features, don't use both. Defaults to None.
        """

        # Check that input is valid.
        self._validate_input(dataset, X, y, group, None)
        
        # Shuffle dataset before splitting.
        if self.shuffle:
            dataset.shuffle()
        
        # If input is a FeatureList, define neccessary matrices.
        if dataset is not None:
            X = dataset.feature_matrix()
            y = dataset.response_matrix()
            ids = np.array([observation.id for observation in dataset.observations])
        else:
            self.X = X
            self.y = y
            self.group = group

        # Create CV splitter object.
        cv_model = GetCVSplitter(self.model, self.outer_strategy, cv_folds=self.outer_cv_folds, iters=self.outer_iters)
        
        # Perform cross-validation. Also, measure time for each iteration and estimate time for completion.
        #self.cv_results = []
        times = []

        for n, (train_index, test_index) in enumerate(cv_model.split(X, y)):
            
            # Start measuring time.
            start = timeit.default_timer()
        
            # If more than one iteration has passed, estimate in how much time the process will end.
            if times:
                mean_t = np.array(times).mean()
                t_estimation = Utils.FormattedTime(mean_t*(self.outer_cv_folds*self.outer_iters - n))       
        
            # Print progress.
            if times:
                print(f'\rCross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}; Estimated time remaining: {t_estimation}',
                    end='     ', flush=True)
            else:
                print(f'\rCross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}', end='     ', flush=True)
            
            # Calculate train and test sets for this cv iteration.
            X_train = X[train_index, :]
            y_train = y[train_index]
            #X_test = self.X[test_index, :]
            #y_test = self.y[test_index]
            ids_test = ids[test_index]
            
            # Fit train set and make test set predictions.
            self.model.fit(X_train, y_train)
            for id in ids_test:
                observation = dataset.get_id(id)
                y_pred = self.model.predict(observation.descriptors.to_numpy().reshape(1, -1))[0]
                observation.make_prediction(y_pred)
                if is_classifier(self.model):
                    y_prob = self.model.predict_proba(observation.descriptors.to_numpy().reshape(1, -1))[0][1]
                    observation.make_prob_prediction(y_prob)
            
            # Add dataset with predictions for every observation.
            self.dataset = dataset
            
            # Stack test data, predictions and groups. Then append to list of results of all cv iterations.
            #if is_classifier(self.model) and group is not None:
            #    self.cv_results.append(np.vstack((y_test, y_pred, y_prob, group_test)))
            #elif is_classifier(self.model) and group is None:
            #    self.cv_results.append(np.vstack((y_test, y_pred, y_prob)))
            #elif is_regressor(self.model) and group is not None:
            #    self.cv_results.append(np.vstack((y_test, y_pred, group_test)))
            #elif is_regressor(self.model) and group is None:
            #    self.cv_results.append(np.vstack((y_test, y_pred)))
            
            # End of current iteration. Stop measuring time.
            end = timeit.default_timer()
            times.append(end-start)
    
        print(f'\rCross-validation evaluation successfully completed. Total time: {Utils.FormattedTime(np.array(times).sum())}',
              end='     ')
        
class NestedCV(CrossValidation):
    
    def __init__(self, model, params, opt_scoring, inner_strategy='k-fold', outer_strategy='k-fold', inner_cv_folds=5,
                 outer_cv_folds=10, inner_iters=1, outer_iters=1, feature_selection_strategy='sfs',
                 top_features=None, min_features=None, max_features=None, genetic_kwargs={},
                 hyperparameter_selection_strategy='grid_search', random_iters= 1000, factor=3, shuffle=True):
        super().__init__()
        self.model = clone(model)
        self.params = params
        self.scoring = opt_scoring
        self.inner_strategy = CVStrategy(inner_strategy)
        self.outer_strategy = CVStrategy(outer_strategy)
        self.inner_cv_folds = inner_cv_folds
        self.outer_cv_folds = outer_cv_folds
        self.inner_iters = inner_iters
        self.outer_iters = outer_iters
        self.feature_selection_strategy = FSStrategy(feature_selection_strategy)
        self.top_features = top_features
        self.min_features = min_features
        self.max_features = max_features
        self.genetic_kwargs = genetic_kwargs
        self.hyperparameter_selection_strategy = HPStrategy(hyperparameter_selection_strategy)
        self.random_iters = random_iters
        self.factor = factor
        self.shuffle = shuffle
    
    def fit(self, features=None, X=None, y=None, group=None, feature_names=None):
        """Method to fit cross-validator to input data.

        Args:
            features (FeatureList, optional): FeatureList object to use as data for cross-validator. Defaults to None.
            X (np.array, optional): Feature array used to fit cross-validator. Alternative to features, don't use both. Defaults to None.
            y (np.array, optional): Response variable array used to fit cross-validator. Alternative to features, don't use both. Defaults to None.
            group (np.array, optional): Array that specifies which group each observation belongs to. Alternative to features, don't use both. Defaults to None.
            feature_names (list, optional): List with names of input features. Alternative to features, don't use both. Defaults to None.
        """
        
        # Check that input is valid.
        self._validate_input(features, X, y, group, feature_names)
        
        # If input is a FeatureList, define neccessary matrices.
        if features is not None:
            self.X = features.feature_df.to_numpy()
            self.y = features.response.to_numpy()
            self.group = features.group.to_numpy()
            self.feature_names = features.feature_names
        else:
            self.X = X
            self.y = y
            self.group = group
            self.feature_names = feature_names
        
        # Create CV splitter objects.
        inner_cv_model = GetCVSplitter(self.model, self.inner_strategy, cv_folds=self.inner_cv_folds,
                                        iters=self.inner_iters)
        outer_cv_model = GetCVSplitter(self.model, self.outer_strategy, cv_folds=self.outer_cv_folds,
                                        iters=self.outer_iters)

        # Perform cross-validation. Also, measure time for each iteration and estimate time for completion.
        self.cv_results = []
        best_hp = []
        best_features = []
        times = []
        
        # Outer CV split.
        for n, (train_index, test_index) in enumerate(outer_cv_model.split(self.X, self.y, self.group)):
            
            # Start measuring time.
            start = timeit.default_timer()
        
            # If more than one iteration has passed, estimate in how much time the process will end.
            if times:
                mean_t = np.array(times).mean()
                t_estimation = Utils.FormattedTime(mean_t*(self.outer_cv_folds*self.outer_iters - n))       
        
            # Print progress.
            if times:
                print(f'\rCross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}; Estimated time remaining: {t_estimation}',
                    end='     ', flush=True)
            else:
                print(f'\rCross-validation iteration: {n+1}/{self.outer_cv_folds*self.outer_iters}', end='     ', flush=True)
            
            # Calculate train and test sets for this cv iteration.
            X_train = self.X[train_index, :]
            y_train = self.y[train_index]
            X_test = self.X[test_index, :]
            y_test = self.y[test_index]
            if self.group is not None:
                group_test = self.group[test_index]
            
            if self.feature_selection_strategy is not None:
                # Search for best feature subset for this cv iteration.
                feat_selector = FSelector(self.feature_selection_strategy, self.scoring, inner_cv_model, max_features=self.max_features,
                                        min_features=self.min_features, top_features=self.top_features, genetic_kwargs=self.genetic_kwargs)
                feat_selector.fit(self.model, X_train, y_train, self.feature_names)
                
                # Save subset of selected features, then remove discarded features from train and test matrices.
                best_features.append(feat_selector.selected_features)
                X_train = X_train[:, feat_selector.best_idx]
                X_test = X_test[:, feat_selector.best_idx]
            
            # Search for best hyperparameter subset for this cv iteration.
            hp_selector = HPSelector(self.hyperparameter_selection_strategy, self.params, self.scoring, inner_cv_model,
                                     iters=self.random_iters, factor=self.factor)
            hp_selector.fit(self.model, X_train, y_train)
            
            # Save subset of selected hyperparameters, then train a model with those hyperparameters.
            best_hp.append(hp_selector.selected_params)
            self.model.set_params(**hp_selector.best_params)
            
            # Fit model and make predictions.
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test).reshape(-1,)
            if is_classifier(self.model):
                y_prob = self.model.predict_proba(X_test)[:, 1].reshape(-1,)
            
            # Stack test data, predictions and groups. Then append to list of results of all cv iterations.
            if is_classifier(self.model) and group is not None:
                self.cv_results.append(np.vstack((y_test, y_pred, y_prob, group_test)))
            elif is_classifier(self.model) and group is None:
                self.cv_results.append(np.vstack((y_test, y_pred, y_prob)))
            elif is_regressor(self.model) and group is not None:
                self.cv_results.append(np.vstack((y_test, y_pred, group_test)))
            elif is_regressor(self.model) and group is None:
                self.cv_results.append(np.vstack((y_test, y_pred)))
                
            # End of current iteration. Stop measuring time.
            end = timeit.default_timer()
            times.append(end-start)
        
        # Merge fetures selected through every iteration into single DataFrames.
        if self.feature_selection_strategy is not None:
            self.features_df = pd.DataFrame(np.hstack(best_features), columns=[f'Iteration {i+1}' for i in range(len(best_features))])
            if feature_names is not None:
                self.features_df.index = feature_names + ['Selected Features', 'Score']
            self.features_df['Total'] = [(row.to_numpy()=='yes').sum() for _, row in self.features_df.iloc[:-2, :].iterrows()]\
                                        + self.features_df.iloc[-2:, :].mean(axis=1).to_list()
        
        # Merge hyperparameters selected through every iteration into single DataFrames.
        self.hyperparameter_df = pd.DataFrame(np.hstack(best_hp), index=list(self.params.keys()) + ['Score'],
                                              columns=[f'Iteration {i+1}' for i in range(len(best_hp))])
        
        print(f'\rCross-validation evaluation successfully completed. Total time: {Utils.FormattedTime(np.array(times).sum())}',
              end='     ')
            
    def hp_summary(self):
        """Method to calculate statistics about the hyperparameters used during nested cross-validation procedure.

        Returns:
            DataFrame: DataFrame with statistics about the used hyperparameters.
        """
        
        # Check that CrossValidation instance is fitted.
        self._check_fit()
        
        # Initialize array to store stats.
        stats_arr = np.zeros((self.hyperparameter_df.shape[0]-1, 6))
        
        # Calculate stats for every hyperparameter.
        stats_arr[:, 0] = self.hyperparameter_df.iloc[:-1, :].mean(axis=1)
        stats_arr[:, 1] = self.hyperparameter_df.iloc[:-1, :].median(axis=1)
        stats_arr[:, 2] = mode(self.hyperparameter_df.iloc[:-1, :], axis=1)[0].reshape(-1,)
        stats_arr[:, 3] = self.hyperparameter_df.iloc[:-1, :].std(axis=1)
        stats_arr[:, 4] = self.hyperparameter_df.iloc[:-1, :].min(axis=1)
        stats_arr[:, 5] = self.hyperparameter_df.iloc[:-1, :].max(axis=1)
        
        return pd.DataFrame(stats_arr, index=self.hyperparameter_df.index[:-1], columns=['Mean', 'Median', 'Mode', 'Std', 'Min', 'Max'])
    
    def hp_distribution(self, hp, style='kde', bins=15):
        """Method to plot the distribution of the values of a given hyperparameter that were used during cross-validation.

        Args:
            hp (str): Name of the hyperparameter to plot.
            style (str, optional): Style of the plot. Can be either 'hist' or 'kde'. Defaults to 'kde'.
            bins (int, optional): Number of bins in the histogram. Only relevant if chosen style is 'hist'. Defaults to 15.

        Raises:
            ValueError: Raised when the input hyperparameter wasn't used during cross-validation.

        Returns:
            figure, ax: Figure of the distribution.
        """
        
        # Check that CrossValidation instance is fitted.
        self._check_fit()
        
        # Check that input hyperparameter is a valid hyperparameter that was analyized during cross-validation.
        if hp not in self.hyperparameter_df.index:
            raise ValueError(
                f'{hp} is not a valid hyperparameter.'
            )
        
        # Create figure.
        fig, ax = plt.subplots()
        
        # Create plot and add labels.
        if Styles(style) == Styles.HIST:
            sns.histplot(self.hyperparameter_df.transpose()[hp], bins=bins, ax=ax)
            ax.set_ylabel('Counts', labelpad=15, fontsize=12, fontweight='bold')
        elif Styles(style) == Styles.KDE:
            sns.kdeplot(self.hyperparameter_df.transpose()[hp], ax=ax)
            ax.set_ylabel('Distribution', labelpad=15, fontsize=12, fontweight='bold')
        ax.set_xlabel(hp, labelpad=15, fontsize=12, fontweight='bold')

        return fig, ax
    
    def feature_summary(self):
        """Method to plot the number of times each feature was chosen, and statistics about the number of total choices.

        Returns:
            figure, axis: Figure and axis of the plot.
        """
        
        # Check that CrossValidation instance is fitted.
        self._check_fit()
        
        # Create series of number of times each features was chosen.
        features_series = self.features_df.iloc[:-2, :].sort_values(by='Total', ascending=False)['Total']
        
        # Calculate statistics about number of features chosen at each iteration.
        num_features_df = self.features_df.iloc[-2, :]
        num_features_list = [[num_features_df.mean(), num_features_df.median(), mode(num_features_df)[0][0],num_features_df.std()]]
        
        
        # Create figure.
        fig, ax = plt.subplots()
        
        # Add bar plot.
        ax.bar(np.arange(features_series.shape[0]), features_series.to_numpy(), color='darkblue', edgecolor='black')
        ax.set_xticks(np.arange(features_series.shape[0]))
        ax.set_xticklabels(features_series.index, rotation=90)
        ax.set_ylabel('Nº Times Used', labelpad=15, fontsize=12, fontweight='bold')
        
        # Add table.
        ax.table(cellText=num_features_list, rowLabels=['Nº of Features'], colLabels=['Mean', 'Median', 'Mode', 'Std'], loc='top')
        
        fig.tight_layout()
        
        return fig, ax
    
    def opt_model(self, statistic='median'):
        
        hp_df = self.hp_summary()
        feat_df = self.features_df
        print('Unfinished feature.')
        
        #if Statistics(statistic) == Statistics.MEAN:
        # UNFINISHED!!!!
            
        

# Dataclasses to represent cross-validation results.
@dataclass
class CVScores:

    def results(self):
        """Method that returns all scores obtained across every CV iteration.

        Returns:
            DataFrame: All scores of every CV iteration.
        """

        tmp_dict = asdict(self)
        cols = [format_score(score) for score in list(tmp_dict.keys())[:-3]]
        
        return pd.DataFrame(list(tmp_dict.values())[:-3],
                            index=cols,
                            columns=tmp_dict['indices'])

    def summary(self):
        """Method that returns summary statistics about scores obtained during CV.

        Returns:
            DataFrame: Summary statistics of cross-validation scores.
        """

        tmp_dict = asdict(self)
        cols = [format_score(score) for score in list(tmp_dict.keys())[:-3]]

        # Create array and list to store info. List for confidence intervals is separated, since they are strings
        # instead of floats.
        summary_arr = np.zeros((5, len(tmp_dict.keys())-3))
        ci_list = []

        # Iterate over every score, then compute mean, median, min and max values. Then compute confidence intervals
        # and standard errors using the bootstrap.
        for n, score in enumerate(list(tmp_dict)[:-3]):
            boot = bootstrap((tmp_dict[score],), np.mean)
            ci_list.append('-'.join([str(round(x, 3)) for x in boot.confidence_interval]))
            boot_ste = boot.standard_error
            summary_arr[:, n] = np.array([tmp_dict[score].mean(), boot_ste, np.median(tmp_dict[score]),
                                          tmp_dict[score].min(), tmp_dict[score].max()])

        summary_df =  pd.DataFrame(np.around(summary_arr, 3),
                                   index=['Mean', 'Standard Error Mean', 'Median', 'Min', 'Max'],
                                   columns=cols
                                  ).transpose()
        summary_df.insert(1, '95% CI Mean', ci_list)

        return summary_df.transpose()

    def draw_distribution(self, score, style='kde', bins=25):
        """Method to draw the distribution of values of a given score that were obtained through cross-validation.

        Args:
            score (str): Name of the score whose values will be ploted.
            style (str, optional): Name of the style of plot used to draw distribution, must be "hist" or "kde". Defaults to 'hist'.
            bins (int, optional): Number of bins used to draw histogram. Only used when style is "hist", otherwise ignored. Defaults to 25.

        Raises:
            ValueError: Raised when the input score is invalid.
            ValueError: Raised when the input style is invalid.

        Returns:
            figure, axis: Figure and axis objects of the plot.
        """
        
        tmp_dict = asdict(self)
        
        # Check that input score is valid.
        if score not in list(tmp_dict.keys())[:-3]:
            raise ValueError(
                f'{score} is not a valid score.' 
                )
        
        # Create figure.
        fig, ax = plt.subplots()
        
        # Draw histogram or kde plot of distribution.
        if Styles(style) == Styles.HIST:
            sns.histplot(tmp_dict[score].reshape(-1, 1), bins=bins, ax=ax)
        elif Styles(style) == Styles.KDE:
            sns.kdeplot(tmp_dict[score], ax=ax)

        # Format figure.
        ax.set_xlabel(format_score(score), labelpad=15, fontsize=12, fontweight='bold')
        if style == 'hist':
            ax.set_ylabel('Counts', labelpad=15, fontsize=12, fontweight='bold')
        elif style == 'kde':
            ax.set_ylabel('Density', labelpad=15, fontsize=12, fontweight='bold')

        return fig, ax
    
    def group_summary(self):
        """Method to return mean scores decomposed by group.

        Returns:
            DataFrame: Means of all score metrics for every group.
        """
        
        tmp_dict = asdict(self)
        cols = [format_score(score) for score in list(tmp_dict.keys())[:-3]]
        
        # Create arrays to store mean and std group scores.
        scores_arr = np.zeros((len(self.group_dict.keys()), len(tmp_dict.keys())-3))
        stds_arr = np.zeros((len(self.group_dict.keys()), len(tmp_dict.keys())-3))
        
        # For every group, calculate the mean of every score.
        for n, group in enumerate(self.group_dict.keys()):
            scores_arr[n, :] = np.mean(self.group_dict[group], axis=1)
            stds_arr[n, :] = np.std(self.group_dict[group], axis=1)
        
        means_df = pd.DataFrame(np.around(scores_arr, 2),
                                index=[f'Group {i+1}' for i in range(scores_arr.shape[0])],
                                columns=cols)
        
        stds_df = pd.DataFrame(np.around(stds_arr, 2),
                               index=[f'Group {i+1}' for i in range(scores_arr.shape[0])],
                               columns=cols)
        
        return means_df, stds_df

    def group_score(self, score, threshold=None, group_names=None):
        """Method to plot the group decomposition of input score. 

        Args:
            score (str): Name of score metric to plot.
            threshold (float, optional): If given, horizontal line will be draw on this point, as threshold for score. Defaults to None.
            group_names (list, optional): List of names for group names. Defaults to None.

        Raises:
            ValueError: Raised when input score is invalid.

        Returns:
            figure, axis: Figure and axis objects for the output plot.
        """
        
        # Create means and std dataframes, then create appropriate group names.
        means_df, stds_df = self.group_summary()
        if group_names is None:
            group_names = means_df.index
        
        # Check if input score is formatted, and if it's not, format it.
        if score not in means_df.columns:
            try:
                score = format_score(score)
            except:
                raise ValueError(
                    f'{score} is not a valid score.'
                    )

        # Create figure.
        fig, ax = plt.subplots()

        # Plot figures.
        ax.bar(np.arange(len(means_df.index)), means_df[score].to_numpy(), yerr=stds_df[score].to_numpy(),
               color='cyan', edgecolor='black', capsize=3.5)
        if threshold is not None:
            left, right = ax.get_xlim()
            ax.plot(np.linspace(left, right, len(means_df.index)), np.repeat(threshold, len(means_df.index)),
                    color='red', linestyle='dashed')
            ax.set_xlim(left, right)

        # Format figures.
        ax.set_ylabel(score, labelpad=15, fontsize=12, fontweight='bold')
        ax.set_xticks(np.arange(len(means_df.index)))
        ax.set_xticklabels(group_names, rotation=90)
        
        return fig, ax

@dataclass
class ClfScores(CVScores):
    
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    f_score: List[float]
    roc_auc: List[float]
    brier_score: List[float]
    cv_results: Dict[int, np.ndarray]
    group_dict: Dict[str, np.ndarray] = None
    indices: List[int] = None
    
    def confusion_matrix(self, display='percentage'):
        """Method to plot the confusion matrix of a set of cross-validated results.

        Args:
            display (str, optional): Whether to show percentages or total numbers when representing the number
            of true postitives, false negatives etc. Defaults to 'percentage'.

        Raises:
            ValueError: Raised when the input display is invalid. Must be either "percentage" or "total_count".

        Returns:
            figure, axis: Figure and axis of the confusion matrix.
        """
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([result[0, :] for result in asdict(self)['cv_results']])
        y_pred = np.hstack([result[1, :] for result in asdict(self)['cv_results']])
        
        # Compute confusion matrix.
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Format according to input display.
        if display == 'percentage':
            conf_matrix = np.around((conf_matrix/conf_matrix.sum()), 2)
        elif display == 'total_count':
            pass
        else:
            raise ValueError('Display must be "percentage" or "total_count"')
        
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

    def roc_curve(self):
        """Method to plot the ROC curve and compute the area under the ROC curve of a set of cross-validation results.

        Returns:
            figure, axis: Figure and axis of the ROC curve plot.
        """
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([result[0, :] for result in asdict(self)['cv_results']])
        y_prob = np.hstack([result[2, :] for result in asdict(self)['cv_results']])
        
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

    def det_curve(self):
        """Method to plot the DET curve and compute the error rate for different probability thresholds
        of a set of cross-validation results.

        Returns:
            figure, axis: Figure and axis of the DET curve plot.
        """
        
        # Get arrays with all the predictions made during cross-validation and their true values.
        y_true = np.hstack([result[0, :] for result in asdict(self)['cv_results']])
        y_prob = np.hstack([result[2, :] for result in asdict(self)['cv_results']])
        
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

@dataclass
class RegScores(CVScores):
    
    mae: List[float]
    rmse: List[float]
    mape: List[float]
    p_correlation: List[float]
    r2: List[float]
    s_correlation: List[float]
    kendall_tau: List[float]
    cv_results: Dict[int, np.ndarray]
    group_dict: Dict[str, np.ndarray] = None
    indices: List[int] = None

    def draw_cv(self, group=None, sel_size=1.0, threshold=None, **kwargs):
        """Method to plot predictions made during cross-validation versus true values.

        Args:
            group (int, optional): If given, onlye the predictions for this group will be ploted. Defaults to None.
            sel_size (float, optional): If less than one, only that portion of the data will be plotted. Defaults to 1.0.
            threshold (float, optional): If given, dashed lines will be ploted above and below the regression line. Then, the
            points with error below that distance will be drawn in green and the points with errors above that distance in red. 
            Defaults to None.

        Returns:
            figure, axis: Figure of the ploted predictions versus true values.
        """
        
        # If a group is given as input, then only that group is plotted.
        # Here we remove data not belonging to that group if group is given
        # and then calculate performance metrics.
        if group is None:
            data = np.hstack([result[:2, :] for result in self.cv_results])
            rmse = np.around(self.rmse.mean(), 2)
        else:
            data = np.hstack([result for result in self.cv_results])
            data = data[:2, data[2, :] == group]
            rmse = np.around(self.group_dict[f'group {group}'][1, :].mean(), 2)

        # If selection size is smaller than 1, then only plot a randomly chosen selection
        # of points. In case there is too much data and plot looks cluttered.
        if sel_size < 1.0:
            sample_num = int(data.shape[1]*sel_size)
            mask = np.random.choice(np.arange(data.shape[1]), sample_num, replace=False)
            data = data[:, mask]
            rmse = np.around(self.rmse.mean(), 2)
        
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
                    label=f'RMSE: {rmse}; Ratio: {ratio}')
            ax.plot(space, over_space, space, under_space, color='black', linestyle='dashed')
            
            ax.legend(loc='best')

        # If threshold is not given, simply plot data with colorless points.
        else:
            fig, ax = plt.subplots()

            ax.scatter(data[0, :], data[1, :], facecolor='none', edgecolor='grey', **kwargs)
            
            space = np.linspace(np.min(data[0, :]), np.max(data[0, :]), 20)            
            
            ax.plot(space, space, color='black', linewidth=3, label=f'RMSE: {rmse}')
            
            ax.legend(loc='best')

        return fig, ax
