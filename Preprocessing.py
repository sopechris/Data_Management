# Functions to preprocess pandas dataframes to use in machine learning models.

import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split

# Function to create a train and test set from pandas dataframe.
def GenerateTrainTest(df, resp, excpt=[], test_size=0.2):
    feature_names = df.drop(columns=resp).drop(columns=excpt).columns.to_list()
    feature_array = df[feature_names].to_numpy()
    response_array = df[resp].to_numpy().reshape(-1,)
    
    if test_size is None:
        X_train, y_train = feature_array, response_array
        X_test, y_test = [], []
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_array, response_array, test_size=test_size)
    return X_train, X_test, y_train, y_test, feature_names


# Function to create stratified train and test set from pandas dataframe.
def GenerateStratifiedTrainTest(df, resp, group, excpt=[], test_size=0.2, shuffle=False):
    
    # Calculate neccessary amount of split to create desired test_size and the number of clusters.
    if test_size is not None:
        n_splits = int(1/test_size)
    n_clusters = int(df[group].max()) + 1
    
    # Retrieve feature, response and cluster group arrays from dataframe.
    feature_names = df.drop(columns=resp).drop(columns=excpt).drop(columns=group).columns.to_list()
    feature_array = df[feature_names].to_numpy()
    response_array = df[resp].to_numpy().reshape(-1,)
    group_array = df[group].to_numpy().reshape(-1,)
    
    # If no test set is desired, simply return all feature as arrays.
    if test_size is None:
        X_train, y_train = feature_array, response_array
        X_test, y_test, g_test = [], [], []
        g_train = group_array
        distribution = pd.DataFrame()
    else:
        # Perform random shuffling of input data before separating train and test sets.
        if shuffle:
            rand = np.arange(len(group_array))
            np.random.shuffle(rand)
            feature_array = feature_array[rand, :]
            response_array = response_array[rand]
            group_array = group_array[rand]
        
        # Obtain n_split amount of stratified train test splits and choose one at random.
        skf = StratifiedKFold(n_splits=n_splits)
        rand_int = random.randint(0, n_splits-1)
        for n, (train_index, test_index) in enumerate(skf.split(feature_array, group_array)):
            if n==rand_int:
                X_train = feature_array[train_index, :]
                y_train = response_array[train_index]
                X_test = feature_array[test_index, :]
                y_test = response_array[test_index]
                g_train = group_array[train_index]
                g_test = group_array[test_index]
                break

        # Save info about data distribution in the performed random split in a dataframe.
        g_dict = {'total': dict.fromkeys(range(1, n_clusters + 1), 0),
                  'train': dict.fromkeys(range(1, n_clusters + 1), 0),
                  'test': dict.fromkeys(range(1, n_clusters + 1), 0)}

        for key, g_arr in zip(['train', 'test'], (g_train, g_test)):
            for g in g_arr:
                g_dict['total'][g+1] += 1
                g_dict[key][g+1] += 1
        
        totals = np.array(list(g_dict['total'].values())).astype('int32')
        train_total = np.array(list(g_dict['train'].values())).astype('int32')
        train_ratio = np.around(train_total.astype('float64')/totals.astype('float64'), 2)
        test_total = np.array(list(g_dict['test'].values())).astype('int32')
        test_ratio = np.around(test_total.astype('float64')/totals.astype('float64'), 2)
        
        distribution = pd.DataFrame([totals, train_total, train_ratio, test_total, test_ratio],
                                    index=['Total', 'Train Total', 'Train Ratio', 'Test Total', 'Test Ratio'],
                                    columns=[f'Cluster {n}' for n in g_dict['total'].keys()],
                                    dtype='object')

    return X_train, y_train, X_test, y_test, g_train, g_test, feature_names, distribution


# Function to standardize all features (except 'exceptions') from a dataframe.
def StandardizeDF(df, exceptions):
    
    out_df = df.copy()

    for feature in out_df.drop(columns=exceptions).columns:
        out_df[feature] = StandardScaler().fit_transform(out_df[feature].to_numpy().reshape(-1, 1))

    return out_df


# Function to add polynomial features of degree 'deg' from a set of initial features to a dataframe.
def AddPolynomialFeaturesDF(df, features, deg):

    copy_df = df.copy()

    X_poly = copy_df[features].to_numpy()

    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X_poly)[:, len(features)+1:]
    features_poly = poly.get_feature_names(features)[len(features)+1:]
    features_poly = [feature.replace(' ', '*') for feature in features_poly]

    df_poly = pd.DataFrame(X_poly, columns=features_poly)

    return pd.merge(copy_df, df_poly, left_index=True, right_index=True), df_poly
