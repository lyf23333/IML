# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    train_df_ori = train_df
    test_df_ori = test_df
    # configs
    add_noise_and_expand = True
    iterative_imputer = False
    # data points which have no labels are not used for learning
    y_train = train_df['price_CHF'].to_numpy()
    nan_inds = np.isnan(y_train)
    non_nan_inds = np.logical_not(np.isnan(y_train))
    # Nan to values
    if not iterative_imputer:
        train_mean = train_df.drop(['season'],axis=1).mean()
        train_df = train_df.fillna(train_mean)
        test_mean = test_df.drop(['season'],axis=1).mean()
        test_df = test_df.fillna(test_mean)
    # to numpy
    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()
    # process seasons
    X_train = process_seasons(X_train)
    X_test = process_seasons(X_test)
    # iterative imputer
    if iterative_imputer:
        imp = IterativeImputer(max_iter=100, random_state=0)
        imp = imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test)
        # X_train = iterative_learning_impultation(X_train, train_df)
    # normalize data
    min_max_scaler_train = MinMaxScaler()
    X_train_scaled = min_max_scaler_train.fit_transform(X_train)
    X_test_scaled = min_max_scaler_train.fit_transform(X_test)
    # only use data with clear labels
    y_train_labeled = y_train[non_nan_inds]
    X_train_scaled_labeled = X_train_scaled[non_nan_inds]
    X_train_scaled_unlabeled = X_train_scaled[nan_inds]

    if add_noise_and_expand:
        # add noise train
        a = train_df_ori.drop(['price_CHF'],axis=1).to_numpy()
        a[:,0] = 0
        nan_poses = np.isnan(a.astype(np.float32))
        X_train_expand = np.copy(X_train_scaled_labeled)
        
        expand_nums = 10
        for i in range(expand_nums):
            X = np.copy(X_train_scaled)
            X[nan_poses] += np.random.normal(0, 0.2, X_train_scaled.shape)[nan_poses]
            X = X[non_nan_inds]
            X_train_expand = np.concatenate((X_train_expand, X), axis=0)
        y_train_expanded = np.tile(y_train_labeled, expand_nums+1)
        X_train_scaled_labeled = X_train_expand
        y_train_labeled = y_train_expanded

        # add noise test
        expand_nums = 10
        a = test_df_ori.to_numpy()
        a[:,0] = 0
        nan_poses = np.isnan(a.astype(np.float32))
        X_test_expand = np.copy(X_test_scaled)

        for i in range(expand_nums):
            X = np.copy(X_test_scaled)
            X[nan_poses] += np.random.normal(0, 0.3, X_test_scaled.shape)[nan_poses]
            X_test_expand = np.concatenate((X_test_expand, X), axis=0)
        X_test_scaled = X_test_expand

    # assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train_scaled_labeled, y_train_labeled, X_train_scaled_unlabeled, X_test_scaled


def iterative_learning_impultation(X_train, train_df):
    gpr_models = [GaussianProcessRegressor(kernel=DotProduct(), alpha=1e-9)] * X_train.shape[1]

    # fit iteratively
    X_train_all = train_df.to_numpy()
    X_train_all = process_seasons(X_train_all)
    train_mean = train_df.drop(['season'],axis=1).median()
    train_mean = train_mean.to_numpy()
    X_train_o = np.copy(X_train_all)
    X_train_all[:,1:] = np.where(np.isnan(X_train_all[:,1:]), train_mean, X_train_all[:,1:])
    
    for j in range(3):
        if j == 0:
            learned_columns = 0
        else:
            learned_columns = 100
        for column in range(1, X_train_o.shape[1]):
            non_nan_rows = ~np.isnan(X_train_o[:,column])
            X_train_nonan = X_train_all[non_nan_rows]
            for i in range(1, np.clip(np.max((column, learned_columns)), None, X_train_o.shape[1])):
                nan_rows_i = np.isnan(X_train_o[non_nan_rows,i])
                if not np.any(nan_rows_i):
                    continue
                source = np.concatenate((X_train_nonan[:,:i], X_train_nonan[:,i+1:]), axis=1)
                X_train_nonan[nan_rows_i, i] = gpr_models[i-1].predict(source[nan_rows_i])
            X = np.concatenate((X_train_nonan[:,:column], X_train_nonan[:,column+1:]), axis=1)
            y = X_train_nonan[:,column]
            gpr = gpr_models[column-1]
            gpr_models[column-1] = gpr.fit(X, y)

    for i in range(X_train_all.shape[1]):
        nan_rows_i = np.isnan(X_train_o[:,i])
        if not np.any(nan_rows_i):
            continue
        source = np.concatenate((X_train_all[nan_rows_i,:i], X_train_all[nan_rows_i,i+1:]), axis=1)
        X_train_all[nan_rows_i, i] = gpr_models[i-1].predict(source)

    X_train[:]= np.delete(X_train_all, 2, 1)
    return X_train


def process_seasons(X):
    X = np.where(X=="spring", 0.25, X)
    X = np.where(X=="summer", 0.5, X)
    X = np.where(X=="autumn", 0.75, X)
    X = np.where(X=="winter", 1., X)
    # # one-hot coding
    # seasons = np.zeros((X_train.shape[0], 4))
    # seasons[X_train[:,0] == "spring", 0] = 1
    # seasons[X_train[:,0] == "summer", 1] = 1
    # seasons[X_train[:,0] == "autumn", 2] = 1
    # seasons[X_train[:,0] == "winter", 3] = 1
    # X_train = np.concatenate((seasons, X_train[:,1:]), axis=1)

    # seasons = np.zeros((X_test.shape[0], 4))
    # seasons[X_test[:,0] == "spring", 0] = 1
    # seasons[X_test[:,0] == "summer", 1] = 1
    # seasons[X_test[:,0] == "autumn", 2] = 1
    # seasons[X_test[:,0] == "winter", 3] = 1
    # X_test = np.concatenate((seasons, X_test[:,1:]), axis=1)
    return X.astype(np.float32)


def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    print("pre modeling process-------------------")
    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    n_folds = 10
    gpr = GaussianProcessRegressor(kernel=ConstantKernel() * RationalQuadratic() + WhiteKernel(noise_level_bounds=(1e-6, 1e+1)), alpha=1e-6)

    start_time = time.time()
    score_mat = np.zeros(n_folds)
    kf = KFold(n_splits=n_folds, shuffle=True)
    for j, (train_index, test_index) in enumerate(kf.split(X_train)):
        gpr = gpr.fit(X_train[train_index], y_train[train_index])
        y_p = gpr.predict(X_train[test_index])
        y_label = y_train[test_index]
        score_mat[j] = r2_score(y_label, y_p)
        end_time = time.time()
        print(f"{j}th round took {end_time - start_time}s")
        print(f"r2 score for round {j}: {score_mat[j]}")
    #     if j ==0:
    #         break

    print("score matri: ", score_mat)
    print("average score: ", np.mean(score_mat))
    end_time = time.time()
    print(f"Training took {end_time - start_time}s")
    y_pred = gpr.predict(X_test)
    if y_pred.shape[0] > 100:
        y_pred = np.mean(y_pred.reshape(-1, 100), axis = 0)
    return y_pred, gpr


def modeling_and_prediction_post(gpr, X_train_labeled, y_train_labeled, X_train_unlabeled, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    print("post modeling process-------------------")
    y_label = gpr.predict(X_train_unlabeled)
    y_train = np.concatenate((y_train_labeled, y_label))
    X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    n_folds = 5
    gpr = GaussianProcessRegressor(kernel=RationalQuadratic())

    score_mat = np.zeros(n_folds)
    kf = KFold(n_splits=n_folds, shuffle=True)
    for j, (train_index, test_index) in enumerate(kf.split(X_train)):
        gpr = gpr.fit(X_train[train_index], y_train[train_index])
        y_p = gpr.predict(X_train[test_index])
        y_label = y_train[test_index]
        score_mat[j] = r2_score(y_label, y_p)
    print("score matri: ", score_mat)
    print("average score: ", np.mean(score_mat))

    y_pred = gpr.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred, gpr


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train_labeled, y_train_labeled, X_train_unlabeled, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred, gpr=modeling_and_prediction(X_train_labeled, y_train_labeled, X_test)

    # y_pred, gpr=modeling_and_prediction_post(gpr, X_train_labeled, y_train_labeled, X_train_unlabeled, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

