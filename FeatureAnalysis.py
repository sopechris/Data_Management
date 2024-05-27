# Functions to evaluate the quality of features for a machine learning model.

import numpy as np
import pandas as pd
import xarray as xr
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold, RepeatedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector
import Utils
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


cpu_num = Utils.cpu_num

class FeatureList:
    
    def __init__(self, feature_df, response_df):
        
        self.feature_df = feature_df
        self.response = response_df
        self.feature_names = np.array(feature_df.columns)
    
    
    # Function to add features to FeatureList object.
    def add_features(self, add_features):
        
        self.feature_df = pd.concat([self.feature_df, add_features], axis=1)
        self.feature_names = np.array(self.feature_df.columns)
        
    
    # Function to remove features from FeatureList object.
    def remove_features(self, rm_features):
    
        self.feature_df = self.feature_df.drop(columns=rm_features)
        self.feature_names = np.array(self.feature_df.columns)

    
    # Function to draw correlation matrix.
    def draw_correlation(self, feature_subset=None, correlation_method='pearson',
                         include_response=False, cmap='binary'):
        
        # Chech that input correlation method is valid.
        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError('Correlation method must be one of: "pearson", "spearman" or "kendall".')
        
        # If the reponse feature is going to be included in correlation matrix, merge it first.
        if include_response:
            plot_df = pd.merge(self.feature_df, self.response, left_index=True, right_index=True)
        else:
            plot_df = self.feature_df.copy()

        # Plot correlation matrix.
        fig, ax = plt.subplots()
        
        if feature_subset is None:
            sns.heatmap(plot_df.corr(method=correlation_method).abs(), vmin=0, vmax=1, cmap=cmap,
                        annot=True, fmt='.2f', ax=ax)
        else:
            sns.heatmap(plot_df[feature_subset].corr(method=correlation_method).abs(), vmin=0, vmax=1, cmap=cmap,
                        annot=True, fmt='.2f', ax=ax)
        ax.set_xlabel('Features', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=15)
        
        return fig, ax

    
    # Function to cluster the features in a FeatureList with hierarchical clustering.
    def cluster_features(self, linkage='ward', correlation_method='pearson',
                         threshold=0.7, colormap=cm.rainbow):
        
        # Chech that input correlation method is valid.
        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError('Correlation method must be one of: "pearson", "spearman" or "kendall".')
        
        # Set threshold and correlation method as class attribute.
        self.cluster_threshold = threshold
        self.correlation_method = correlation_method
        
        # Cluster features.
        distances = 1 - np.absolute(self.feature_df.corr(method=correlation_method))
        dist_array = distance.squareform(distances)
        hier = hierarchy.linkage(dist_array, method=linkage)
        idx = hierarchy.fcluster(hier, 1.0 - threshold, criterion="distance")
        
        # Format cluster indices.
        # Features that don't belong to any cluster will be assigned 0 as index.
        cluster_idx = np.zeros((len(idx),))
        j = 1
        for i in np.unique(idx):
            if len(idx[idx == i]) > 1:
                cluster_idx[np.where(idx == i)[0]] = j
                j += 1
            else:
                cluster_idx[np.where(idx == i)[0]] = 0
        
        self.cluster_idx = cluster_idx
        self.hier = hier
        
        # Create list of DataFrames, one for each cluster.
        self.cluster_dfs = []

        for i in range(1, int(np.max(cluster_idx) + 1)):
            cols = self.feature_names[cluster_idx == i]
            self.cluster_dfs.append(self.feature_df[cols])
        
        # Create colors list. First set black as the color for all features not assigned to a cluster.
        # colors_list = np.flip(['black' if c == 'C0' else c for c in den_dict['leaves_color_list']])
        colors_arr = np.zeros((len(cluster_idx), )).astype('str')
        self.colors_list = [colors.to_hex(color, keep_alpha=True) for color in 
                            colormap(np.linspace(0, 1, int(np.max(cluster_idx))))]
        # Change colors assigned by scipy for a matplolib colormap.    
        for idx, color in zip(np.unique(cluster_idx)[1:], self.colors_list):
            colors_arr[np.where(cluster_idx == idx)] = color
        
        # Get the dendrogram figure and the objects neccessary to draw them.
        hierarchy.set_link_color_palette(self.colors_list)
        den_dict = hierarchy.dendrogram(hier, labels=self.feature_names, color_threshold=1.0-threshold,
                                        orientation='left', no_plot=True)
        
        # Get objects to draw heatmap from dendrogram attributes, so that dendrogram and heatmap match.
        axis = np.array([int(np.where(self.feature_names == feature)[0]) for feature in den_dict['ivl']])
        labels = np.flip(self.feature_names[axis])
        dend_colors = np.flip(['black' if c == 'C0' else c for c in den_dict['leaves_color_list']])
                
        # Add array that defines cluster indices and their colors as a class attribute.
        # First reorder cluster indices so they match the order of the feature names.
        cluster_idx = np.flip(self.cluster_idx[axis]).astype('int32')
        self.cluster_colors = xr.DataArray(np.stack([labels, cluster_idx, dend_colors], axis=1),
                                     dims=['Index', 'Columns'],
                                     coords={'Columns': ['Features', 'Cluster Indices', 'Colors']})


    # Function to summarize feature clustering results.
    def analyize_clusters(self, correlation_threshold=None, color_clusters=False):
        
        # Check that the feature list has been clustered before drawing clusters.
        if not hasattr(self, 'cluster_threshold'):
            raise ValueError('Must cluster FeatureList before drawing clusters. Use .cluster_features() method.')
        
        if correlation_threshold is None:
            correlation_threshold = self.cluster_threshold
        
        cluster_series = []

        # Create DataFrame for every cluster with cluster info.
        for i in range(1, int(self.cluster_idx.max()) + 1):
            
            # Names of features in this cluster.
            feature_names = self.feature_names[self.cluster_idx == i]
            fmt_names = ', '.join(feature_names)
            
            # Number of features in this cluster.
            num_features = len(feature_names)
            
            # Mean value of feature correlations within this cluster.
            # Calculated by taking the bottom triangle of the correlation matrix and calculating the mean.
            tmp_df = self.feature_df[feature_names]
            diag = np.tril(tmp_df.corr().to_numpy(), k=-1)
            mean_corr = np.absolute(diag[diag != 0]).mean()
            
            # Merge the data and append to df list.
            cluster_series.append(pd.DataFrame([f'Cluster {i}', num_features, mean_corr, fmt_names],
                                               index=['Index', 'Number of Features', 'Mean Correlation', 'Feature Names']))

        # Concatenate all DataFrames into single DataFrame, then set column 'Index' as df index and sort by number of features
        # and then mean correlation.
        cluster_info = pd.concat(cluster_series, axis=1).transpose()
        cluster_info.set_index('Index', inplace=True)
        cluster_info.sort_values(by=['Number of Features', 'Mean Correlation'],
                                 axis=0, ascending=False, inplace=True)
        
        # Calculate number of clusters.
        num_clusters = len(cluster_info.index)
        
        # Calculate mean within-cluster correlation, weighed by number of features in that cluster.
        mean_cluster_corr = np.average(cluster_info['Mean Correlation'].to_numpy().astype('float64'),
                                       weights=cluster_info['Number of Features'].to_numpy().astype('int32'))
        
        # Get array with all the unclustered correlation in the correlation matrix.
        # First, calculate correlation matrix.
        corr_matrix = self.feature_df.corr(method=self.correlation_method).abs()
        
        # Eliminate all clustered correlations.
        for feature_names_list in [l.split(', ') for l in cluster_info['Feature Names']]:
            for a, b in list(permutations(feature_names_list, 2)):
                corr_matrix.loc[a, b] = 0
        
        # Select bottom triangle from correlation matrix, then eliminate NaN values (as they as clustered).
        unclustered = np.tril(corr_matrix, k=-1)
        unclustered = unclustered[unclustered != 0]
        
        # Calculate out of cluster correlation mean and standard deviation, weight by number of features in that cluster.
        mean_out_of_cluster_corr = np.mean(unclustered)
        std_out_of_cluster_corr = np.std(unclustered)
        
        # Calculate number of times an unclustered correlation is above the correlation threshold.
        num_above_threshold = len(unclustered[unclustered > correlation_threshold])
        
        # Create summary DataFrame.
        summary = pd.DataFrame([[num_clusters, mean_cluster_corr, mean_out_of_cluster_corr, std_out_of_cluster_corr, num_above_threshold]],
                               index=['Summary'], columns=['Number of Clusters', 'Mean Within-Cluster Correlation',
                                                           'Mean Out-of-Cluster Correlation', 'Std Out-of-Cluster Correlation',
                                                           'Number of Unclustered Correlations Above Threshold'])
        
        if color_clusters:
            
            # Array to map cluster indices to their respective colors.
            colors_list = self.cluster_colors.loc[:, ['Cluster Indices', 'Colors']].to_numpy()
            
            # Function to calculate the intensity of a color based on its hexcode.
            def _get_intensity(hexcolor):
    
                fmt_hex = hexcolor.lstrip('#')
                red, green, blue =  tuple(int(fmt_hex[i:i+2], 16) for i in (0, 2, 4))
                return red*0.299 + green*0.587 + blue*0.114
            
            # Function for mapping the text in the DataFrame to a color. Color is either black or white,
            # based on which will be more visible for a given cluster color.
            def _letter_mapper(df):
                letter_map = []
                for i in [int(idx.split()[1]) for idx in df.index]:
                    color = colors_list[:, 1][colors_list[:, 0] == i][0]
                    if _get_intensity(color) > 150:
                        letter_map.append('color: black')
                    else:
                        letter_map.append('color: white')
                return letter_map
            
            # Function for color mapping. Returns a list of colors for each row, based on the color of the cluster in that row.
            def _color_mapper(df):
                color_map = []
                for i in [int(idx.split()[1]) for idx in df.index]:
                    color = colors_list[:, 1][colors_list[:, 0] == i][0]
                    color_map.append(f'background-color: {color}')
                return color_map
            
            # Style the DataFrame by setting the background color of each row to the color of its cluster.
            cluster_info = cluster_info.style.apply(_color_mapper).apply(_letter_mapper)
        
        return summary, cluster_info
              
    
    # Function to compute pca components of features clusters.
    def cluster_pca(self, min_cluster_len=2, components=1):
        
        if not hasattr(self, 'cluster_dfs'):
            raise ValueError('Must cluster FeatureList before calculating cluster PCA components.\
                  Use .cluster_features() method.')
        
        # Principal components analysis object.
        pca = PCA(n_components=components)
        
        # Find principal components for each cluster.
        pca_list = []
        explained_variances = []
        
        for i, cluster_df in enumerate(self.cluster_dfs):
            if len(cluster_df.columns) > min_cluster_len:
                X_pca = pca.fit_transform(cluster_df)
                cols = [f'PCA_Cluster_{i+1}.{j+1}' for j in range(components)]
                pca_list.append(pd.DataFrame(X_pca, columns=cols))
                explained_variances.append(pd.DataFrame(np.array([f'{round(100*var, 2)}%'
                                                        for var in pca.explained_variance_ratio_]).reshape(1, -1),
                                                        columns=cols))
        
        pca_df = pd.concat(pca_list, axis=1)
        explained_var = pd.concat(explained_variances, axis=1)
        
        return pca_df, explained_var
        
    
    # Function to draw clusers of a set of features according to their correlation matrix
    # using hierarchical clustering.
    def draw_clusters(self, correlation_method=None):
        
        # Check that the feature list has been clustered before drawing clusters.
        if not hasattr(self, 'cluster_threshold'):
            raise ValueError('Must cluster FeatureList before drawing clusters. Use .cluster_features() method.')
        
        # If no input correlation method is given, use the one used to create the clusters.
        if correlation_method is None:
            correlation_method = self.correlation_method
        # If input correlation method is given, chech that it is valid.
        else:
            if correlation_method not in ['pearson', 'spearman', 'kendall']:
                raise ValueError('Correlation method must be one of: "pearson", "spearman" or "kendall".')
        
        # Create figure.
        fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 2], 'height_ratios': [1, 0.05]})
        fig.subplots_adjust(wspace=0)
        
        # Get the dendrogram figure and the objects neccessary to draw them.
        hierarchy.set_link_color_palette(self.colors_list)
        den_dict = hierarchy.dendrogram(self.hier, labels=self.feature_names, color_threshold=1.0-self.cluster_threshold,
                                        ax=axes[0, 0], orientation='left', no_labels=True)
        
        # Get objects to draw heatmap from dendrogram attributes, so that dendrogram and heatmap match.
        axis = np.array([int(np.where(self.feature_names == feature)[0]) for feature in den_dict['ivl']])
        ticks = np.arange(len(axis))
        corr_matrix = np.flip(np.array(np.absolute(self.feature_df.corr(method=correlation_method)))[axis, :][:, axis])
        features_list = self.cluster_colors.loc[:, 'Features'].to_numpy()
        colors_list = self.cluster_colors.loc[:, 'Colors'].to_numpy()
        
        # Plot the dendrogram.
        axes[0, 0].set_xlabel('Distance', fontweight='bold', labelpad=14)
        axes[0, 0].plot(np.repeat(1.0 - self.cluster_threshold, len(axes[0, 0].get_xticks())),
                        axes[0, 0].get_xticks(), '--', color='black')
        
        # Plot threshold line over dendrogram.
        lineticks = np.arange(axes[0, 0].get_ylim()[0], axes[0, 0].get_ylim()[1])
        axes[0, 0].plot(np.repeat(1.0 - self.cluster_threshold, len(lineticks)), lineticks,
                        color='black', linestyle='dashed')
        
        # Plot the heatmap.
        axes[0, 1].yaxis.tick_right()
        axes[0, 1].yaxis.set_label_position("right")
        axes[0, 1].set_xticks(ticks)
        axes[0, 1].set_yticks(ticks)
        axes[0, 1].set_xticklabels(features_list, rotation=90, fontsize=8)      
        axes[0, 1].set_yticklabels(features_list)
        for i, c in zip(ticks, colors_list):
            axes[0, 1].get_xticklabels()[i].set_color(c)
            axes[0, 1].get_yticklabels()[i].set_color(c)
        pos = axes[0, 1].imshow(corr_matrix, cmap='binary', aspect='auto')
        
        # Add a colorbar.
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        cbar = fig.colorbar(pos, ax=axes[1 ,1], orientation='horizontal', aspect=100, shrink=25)
        cbar.set_label('Correlation', fontweight='bold', labelpad=14)
        
        # Draw rectangles indicating the clusters in the correlation matrix.
        # For that, first create dictionary with info about the size of each cluster based on its color.
        colors, counts = np.unique(colors_list, return_counts=True)
        color_counts = dict(zip(colors, counts))
        
        # Initialize variable used to store info about the color in the previous iteration, so that only
        # a single square will be drawn for each color, starting with the first instance of that color.
        prev_color = None
        
        for i, c in zip(ticks, colors_list):
            # If the color is the same as previous one, don't draw square, as the square for this cluster
            # has already been drawn. If the color is black, this feature is not in a cluster, so ignore.
            if c == prev_color or c == 'black':
                prev_color = c
            # If this is the first instance of this color, draw a square with size equal to the number of
            # feautures in this cluster.
            else:
                rect_size = color_counts[c]
                rect = Rectangle((i-0.5, i-0.5), rect_size, rect_size, linewidth=3, edgecolor=c, facecolor='none')
                axes[0, 1].add_patch(rect)
                prev_color = c
        
        return fig, axes
    
    
    # Function to plot feature importances for a model.
    # Possible strategies to calculate these importances are: permutation importance,
    # impurity importance and linear model weights.
    def feature_importance(self, model, X, y, strategy=['permutation'],
                           scoring=None, perms=100, top=None):

        # If no scoring metric is inputed, decide which to use based on model type (classifier or regressor).
        if scoring is None and is_regressor(model):
            scoring = 'neg_mean_squared_error'
        elif scoring is None and is_classifier(model):
            scoring = 'roc_auc'

        # Create figure object. We will then add axes to the figure one by one.
        fig = plt.figure()
        axes = ()
    
        # Make sure input strategies are valid.
        if 'weights' in strategy and not hasattr(model, 'coef_'):
            raise ValueError('Weights is not a valid strategy for computing feature importances for this model.')
        elif 'impurity' in strategy and not hasattr(model, 'feature_importances_'):
            raise ValueError('Impurity is not a valid strategy for computing feature importances for this model.')

        if 'permutation' in strategy:
            # Calculate permutation feature importance.
            result = permutation_importance(model, X, y,
                                            scoring=scoring,
                                            n_repeats=perms,
                                            n_jobs=cpu_num)
            perm_result = result.importances_mean
            perm_std = result.importances_std
            
            # Get indices that order from highest to lowest.
            perm_idx = np.flip(np.argsort(perm_result))
            if top is not None:
                perm_idx = perm_idx[np.arange(top)]
            
            # Set position of ax.
            if len(strategy) == 1:
                pos = 111
            else:
                pos = 121
            
            # Plot data
            ax1 = fig.add_subplot(pos)
            axes += (ax1,)
            ax1.set_xlabel('Permutation Importances', fontsize=12, fontweight='bold', labelpad=12)
            ax1.set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            ax1.set_xlim(0.0, np.max(perm_result) + np.max(perm_std))
            kwargs = {'xerr': perm_std[perm_idx]}
            sns.barplot(x=result.importances_mean[perm_idx], y=[self.feature_names[i] for i in perm_idx], 
                        orient='h', ax=ax1, **kwargs)
        
        if 'weights' in strategy:
            # Calculate weights.
            weights = np.absolute(model.coef_)
            
            # Get indices that order from highest to lowest.
            weight_idx = np.flip(np.argsort(weights))
            if top is not None:
                weight_idx = weight_idx[np.arange(top)]
            
            # Set position of ax.
            if len(strategy) == 1:
                pos = 111
            else:
                pos = 122
            
            # Plot data.
            ax2 = fig.add_subplot(pos)
            axes += (ax2,)
            ax2.set_xlabel('Weights Importances', fontsize=12, fontweight='bold', labelpad=12)
            ax2.set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            ax2.set_xlim(0.0, np.max(weights) + 0.05*np.max(weights))
            sns.barplot(x=weights[weight_idx], y=[self.feature_names[i] for i in weight_idx], 
                        orient='h', ax=ax2)
        
        if 'impurity' in strategy:
            # Calculate impurity based score.
            impurity = model.feature_importances_
            
            # Get indices that order from highest to lowest.
            imp_idx = np.flip(np.argsort(impurity))
            if top is not None:
                imp_idx = imp_idx[np.arange(top)]
            
            # Set position of ax.
            if len(strategy) == 1:
                pos = 111
            else:
                pos = 122
            
            # Plot data.
            ax3 = fig.add_subplot(pos)
            axes += (ax3,)
            ax3.set_xlabel('Impurity Importances', fontsize=12, fontweight='bold', labelpad=12)
            ax3.set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            ax3.set_xlim(0.0, np.max(impurity) + 0.05*np.max(impurity))
            sns.barplot(x=impurity[imp_idx], y=[self.feature_names[i] for i in imp_idx], 
                        orient='h', ax=ax3)
        
        return fig, axes
        

    # Function to evaluate how useful a set of features might be for a regression model
    # based on their F score and mutual information.
    def eval_reg_features(self, discrete_mask='auto', color_clusters=False, top=None, MI_neighbors=3):
        
        # Make sure that clustering has been performed beforehand if color_clusters is True.
        if color_clusters and not hasattr(self, 'cluster_colors'):
            print('ERROR: clustering must be performed before enabling cluster based colors.')
            color_clusters = False
        
        # Calculate F-test and mutual information for all features with respect to the response feature.
        F, _ = f_regression(self.feature_df.to_numpy(), self.response.to_numpy().reshape(-1,))
        MI = mutual_info_regression(self.feature_df.to_numpy(), self.response.to_numpy().reshape(-1,),
                                    discrete_features=discrete_mask, n_neighbors=MI_neighbors)

        # Create figure and axes.
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        # Plot F-test.
        f_idx = np.argsort(F)
        axes[0].set_yticks(np.arange(len(self.feature_names)))
        axes[0].set_yticklabels(self.feature_names[f_idx])
        axes[0].set_xlabel('F Score', fontsize=12, fontweight='bold', labelpad=12)
        axes[0].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
        if top is None:
            barplot = axes[0].barh(np.arange(len(self.feature_names)), F[f_idx], color='red')
        else:
            barplot = axes[0].barh(np.arange(len(self.feature_names))[:top], F[f_idx][:top], color='red')
        
        # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
        if color_clusters:
            for i in self.cluster_colors.coords['Index']:
                ylabels = np.array([x.get_text() for x in axes[0].get_yticklabels()])
                j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
        
        # Plot mutual infromation.
        mi_idx = np.argsort(MI)
        axes[1].set_yticks(np.arange(len(self.feature_names)))
        axes[1].set_yticklabels(self.feature_names[mi_idx])
        axes[1].set_xlabel('MI Score', fontsize=12, fontweight='bold', labelpad=12)
        axes[1].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
        if top is None:
            barplot = axes[1].barh(np.arange(len(self.feature_names)), MI[mi_idx], color='green')
        else:
            barplot = axes[1].barh(np.arange(len(self.feature_names))[:top], MI[mi_idx][:top], color='green')
        
        # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
        if color_clusters:
            for i in self.cluster_colors.coords['Index']:
                ylabels = np.array([x.get_text() for x in axes[1].get_yticklabels()])
                j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))

        return fig, axes
    
    
    # Function to evaluate how useful a set of features might be for a regression model
    # based on their F score, Chi score and mutual information.
    def eval_clf_features(self, discrete_mask='auto', color_clusters=False, top=None, use_chi2=False, MI_neighbors=3):

        # Make sure that clustering has been performed beforehand if color_clusters is True.
        if color_clusters and not hasattr(self, 'cluster_colors'):
            print('ERROR: clustering must be performed before enabling cluster based colors.')
            color_clusters = False
        
        # Calculate F-test and mutual information for all features with respect to the response feature.
        F, _ = f_classif(self.feature_df.to_numpy(), self.response.to_numpy().reshape(-1,))
        MI = mutual_info_classif(self.feature_df.to_numpy(), self.response.to_numpy().reshape(-1,),
                                 discrete_features=discrete_mask, n_neighbors=MI_neighbors)
        if use_chi2:
            Chi2, _ = chi2(self.feature_df.to_numpy(), self.response.to_numpy().reshape(-1,))
                
        # Create figure and axes.
        if use_chi2:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            
            # Plot F-test.
            f_idx = np.argsort(F)
            axes[0, 0].set_yticks(np.arange(len(self.feature_names)))
            axes[0, 0].set_yticklabels(self.feature_names[f_idx])
            axes[0, 0].set_xlabel('F Score', fontsize=12, fontweight='bold', labelpad=12)
            axes[0, 0].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            if top is None:
                barplot = axes[0, 0].barh(np.arange(len(self.feature_names)), F[f_idx], color='red')
            else:
                barplot = axes[0, 0].barh(np.arange(len(self.feature_names))[:top], F[f_idx][:top], color='red')
            
            # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
            if color_clusters:
                for i in self.cluster_colors.coords['Index']:
                    ylabels = np.array([x.get_text() for x in axes[0, 0].get_yticklabels()])
                    j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                    barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
            
            # Plot chi squared stats.
            chi_idx = np.argsort(Chi2)
            axes[1, 0].set_yticks(np.arange(len(self.feature_names)))
            axes[1, 0].set_yticklabels(self.feature_names[chi_idx])
            axes[1, 0].set_xlabel('\N{GREEK SMALL LETTER CHI}\N{SUPERSCRIPT TWO} Statistic', fontsize=12, fontweight='bold', labelpad=12)
            axes[1, 0].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            if top is None:
                barplot = axes[1, 0].barh(np.arange(len(self.feature_names)), Chi2[chi_idx], color='cyan')
            else:
                barplot = axes[1, 0].barh(np.arange(len(self.feature_names))[:top], Chi2[chi_idx][:top], color='cyan')
            
            # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
            if color_clusters:
                for i in self.cluster_colors.coords['Index']:
                    ylabels = np.array([x.get_text() for x in axes[1, 0].get_yticklabels()])
                    j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                    barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
            else:
                axes[1, 0].axis('off')
            
            # Plot mutual infromation.
            mi_idx = np.argsort(MI)
            axes[0, 1].set_yticks(np.arange(len(self.feature_names)))
            axes[0, 1].set_yticklabels(self.feature_names[mi_idx])
            axes[0, 1].set_xlabel('MI Score', fontsize=12, fontweight='bold', labelpad=12)
            axes[0, 1].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            if top is None:
                barplot = axes[0, 1].barh(np.arange(len(self.feature_names)), MI[mi_idx], color='green')
            else:
                barplot = axes[0, 1].barh(np.arange(len(self.feature_names))[:top], MI[mi_idx][:top], color='green')
            
            # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
            if color_clusters:
                for i in self.cluster_colors.coords['Index']:
                    ylabels = np.array([x.get_text() for x in axes[0, 1].get_yticklabels()])
                    j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                    barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
            
            # Hide unused axis.
            axes[1, 1].set_axis_off()
        
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2)
        
            # Plot F-test.
            f_idx = np.argsort(F)
            axes[0].set_yticks(np.arange(len(self.feature_names)))
            axes[0].set_yticklabels(self.feature_names[f_idx])
            axes[0].set_xlabel('F Score', fontsize=12, fontweight='bold', labelpad=12)
            axes[0].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            if top is None:
                barplot = axes[0].barh(np.arange(len(self.feature_names)), F[f_idx], color='red')
            else:
                barplot = axes[0].barh(np.arange(len(self.feature_names))[:top], F[f_idx][:top], color='red')
            
            # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
            if color_clusters:
                for i in self.cluster_colors.coords['Index']:
                    ylabels = np.array([x.get_text() for x in axes[0].get_yticklabels()])
                    j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                    barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
            
            # Plot mutual infromation.
            mi_idx = np.argsort(MI)
            axes[1].set_yticks(np.arange(len(self.feature_names)))
            axes[1].set_yticklabels(self.feature_names[mi_idx])
            axes[1].set_xlabel('MI Score', fontsize=12, fontweight='bold', labelpad=12)
            axes[1].set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=14)
            if top is None:
                barplot = axes[1].barh(np.arange(len(self.feature_names)), MI[mi_idx], color='green')
            else:
                barplot = axes[1].barh(np.arange(len(self.feature_names))[:top], MI[mi_idx][:top], color='green')
            
            # If color_clusters is True, color yticklabels with the color corresponding to its cluster.
            if color_clusters:
                for i in self.cluster_colors.coords['Index']:
                    ylabels = np.array([x.get_text() for x in axes[1].get_yticklabels()])
                    j = np.where(ylabels == str(self.cluster_colors.loc[i, 'Features'].to_numpy()))[0]
                    barplot[int(j)].set_color(str(self.cluster_colors.loc[i, 'Colors'].to_numpy()))
     
        return fig, axes

# Funtion to run sequential feature selection algorithm to find the features that maximize performance of a model.


    def sequential_feature_selection(self, model, top, scoring=None, strategy='k-fold', cv=10, iters=1,
                                     shuffle=False, direction='forward', floating=False):
        # Determine the appropriate scoring metric if not provided
        if scoring is None:
            if is_regressor(model):
                scoring = 'neg_mean_squared_error'
            elif is_classifier(model):
                scoring = 'roc_auc'
            else:
                raise ValueError('Model type not recognized.')

        # Prepare data for feature selection
        X = self.feature_df.to_numpy()
        y = self.response.to_numpy()

        # Set up cross-validation strategy
        if strategy == 'k-fold':
            cv_model = KFold(n_splits=cv, shuffle=shuffle)
        elif strategy == 'repeated_k-fold':
            if iters <= 1:
                raise ValueError('Iters must be greater than 1 for repeated k-fold.')
            cv_model = RepeatedKFold(n_splits=cv, n_repeats=iters)
        else:
            raise ValueError('Strategy must be one of: k-fold or repeated_k-fold.')

        # Initialize the Sequential Feature Selector
        if direction == 'backward':
            sfs = SequentialFeatureSelector(model, k_features=top, forward=False, floating=floating,
                                            scoring=scoring, cv=cv_model, n_jobs=-1)
        elif direction == 'forward':
            sfs = SequentialFeatureSelector(model, k_features=top, forward=True, floating=floating,
                                            scoring=scoring, cv=cv_model, n_jobs=-1)
        else:
            raise ValueError('Direction must be one of forward or backward.')

        # Fit the model
        sfs = sfs.fit(X, y)

        # Retrieve selected and rejected features
        selected_features = self.feature_names[np.array(sfs.subsets_[top]['feature_idx'])]
        rejected_features = self.feature_names[np.setdiff1d(np.arange(len(self.feature_names)),
                                                            np.array(sfs.subsets_[top]['feature_idx']))]
        
        # Format and print results
        selected_features = ', '.join(selected_features)
        rejected_features = ', '.join(rejected_features)
        
        print(f'Features analyzed with {direction} sequential feature selection.')
        print(f'\033[1mSelected best {top} features\033[0m: ' + selected_features)
        print(f'\033[1mRejected worst {len(self.feature_names) - top} features\033[0m: ' + rejected_features)
    
    def draw_sfs(self, model, scoring=None, strategy='k-fold', cv=10, iters=1, shuffle=True, direction='backward'):
    # Determine the appropriate scoring metric if not provided
        if scoring is None:
            if is_regressor(model):
                scoring = 'neg_root_mean_squared_error'
            elif is_classifier(model):
                scoring = 'roc_auc'
            else:
                raise ValueError('Model type not recognized.')

        # Prepare data for feature selection
        X = self.feature_df.to_numpy()
        y = self.response.to_numpy()

        # Set up cross-validation strategy
        if strategy == 'holdout':
            cv_model = StratifiedShuffleSplit(n_splits=cv, test_size=0.3, random_state=42)
            for train_index, test_index in cv_model.split(X, y):
                X_train, X_test = X[train_index], X[test_index]  # Get training and testing data
                y_train, y_test = y[train_index], y[test_index]
                y_train_series = pd.Series(y_train)
        elif strategy == 'k-fold':
            cv_model = KFold(n_splits=cv, shuffle=shuffle, random_state=42)
        elif strategy == 'repeated_k-fold':
            if iters <= 1:
                raise ValueError('Iters must be greater than 1 for repeated k-fold.')
            cv_model = RepeatedKFold(n_splits=cv, n_repeats=iters, random_state=42)
        else:
            raise ValueError('Strategy must be one of: k-fold or repeated_k-fold.')
        
        smallest_class_count = y_train_series.value_counts().min()
        smote_neighbors = max(smallest_class_count - 1, 1)
        
        smote = BorderlineSMOTE(random_state=42, k_neighbors=smote_neighbors)
        pipeline = ImbPipeline(steps=[('smote', smote), ('model', model)])

        # Initialize the Sequential Feature Selector
        sfs = SequentialFeatureSelector(pipeline, k_features='best', forward=(direction == 'forward'), scoring=scoring, cv=cv_model, n_jobs=-1)
        sfs = sfs.fit(X, y)

                # Store information about cv iteration means, standard deviations and the order in which the
        # features where added or substracted from the model in this array.
        info = np.empty((len(sfs.subsets_.keys()), 3))
        info[:, :] = np.nan

        for i, j in enumerate(sorted(sfs.subsets_.keys())):
            info[i, 0] = np.setdiff1d(list(sfs.subsets_[j]['feature_idx']), info[:, 0])
            info[i, 1] = sfs.subsets_[j]['cv_scores'].mean()
            info[i, 2] = sfs.subsets_[j]['cv_scores'].std()

        # If sklearn uses negative version of metric, correct sign and name of metric for plotting.
        if scoring.startswith('neg_'):
            info[:, 1] = -info[:, 1]
            ylabel = ' '.join([word.capitalize() for word in scoring.replace('neg_', '').split('_')])
        else:
            ylabel = ' '.join([word.capitalize() for word in scoring.split('_')])
        


        
        feature_indices = list(range(len(self.feature_names)))
        scores_mean = np.array([sfs.subsets_[i]['avg_score'] for i in sorted(sfs.subsets_.keys())])
        scores_std = np.array([np.std(sfs.subsets_[i]['cv_scores']) for i in sorted(sfs.subsets_.keys())])

        
        ylabel = scoring.replace('_', ' ').capitalize()

        
                
        fig, ax = plt.subplots()

        title = f'Sequential {direction.capitalize()} Selection'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
        ax.set_xticks(np.arange(info.shape[0]))
        ax.set_xticklabels(self.feature_names[info[:, 0].astype('int32')], rotation=90)
        ax.set_xlabel('Features', fontsize=12, fontweight='bold', labelpad=15)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=15)
        ax.plot(np.arange(info.shape[0]), info[:, 1], color='cyan')
        ax.fill_between(np.arange(info.shape[0]), info[:, 1] + info[:, 2], info[:, 1] - info[:, 2],
                        color='cyan', alpha=0.15, lw=2)
        
        # Feature names by cluster
        for i, x in enumerate(ax.get_xticklabels()):
            j = np.where(self.cluster_colors.loc[:, 'Features'] == x.get_text())[0]
            ax.get_xticklabels()[i].set_color(self.cluster_colors.loc[j, 'Colors'].to_numpy()[0])

        # Find the maximum metric value and corresponding feature set
        max_idx = np.argmax(scores_mean)
        max_metric = scores_mean[max_idx]
        max_std = scores_std[max_idx]
        best_features_idx = sfs.subsets_[sorted(sfs.subsets_.keys())[max_idx]]['feature_idx']
        n_iteration = max_idx + 1

        return fig, ax, max_metric, max_std, best_features_idx, n_iteration


    def compute_confusion_matrix(self, model, best_features_idx):
        # Prepare data with selected features
        best_features = self.feature_df.iloc[:, list(best_features_idx)]
        X = best_features.to_numpy()
        y = self.response.to_numpy()
    
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
    
        
        # Lists to store the confusion matrix and classification report for each fold
        cms = []
        reports = []
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        # Perform stratified shuffle split
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #For smote, not to let k neghbors > underclassed event counts (error)
            y_train_series = pd.Series(y_train)
            smallest_class_count = y_train_series.value_counts().min()
            smote_neighbors = max(smallest_class_count - 1, 1)
            
            smote = BorderlineSMOTE(random_state=42, k_neighbors=smote_neighbors)
            # Apply Borderline-SMOTE to the training data
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
            # Fit the model and predict
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            
            # Compute confusion matrix and classification report for the current fold
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            
            cms.append(cm)
            reports.append(cr)
    
    
        return average_cm, cr
        

        
# def PartialDependence
