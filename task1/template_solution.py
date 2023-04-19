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
from sklearn.impute import IterativeImputer, SimpleImputer
import time

def generate_validation_dataset(train_df, ratio=0.2):
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Original data:")
    print("Shape:", train_df.shape)

    split_train_df = train_df.sample(frac = 1-ratio)
    split_valid_df = train_df.drop(split_train_df.index)
    print("Train split shape: ", split_train_df.shape)
    print("Valid split shape: ", split_valid_df.shape)
    return split_train_df, split_valid_df

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

    train_df, valid_df = generate_validation_dataset(train_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    # remove y_nan
    y_train = train_df['price_CHF']
    non_nan_inds = y_train.notnull()
    X_train = train_df[non_nan_inds].drop(['price_CHF'], axis=1)
    y_train = y_train[non_nan_inds]

    y_valid = valid_df['price_CHF']
    non_nan_inds = y_valid.notnull()
    X_valid = valid_df[non_nan_inds].drop(['price_CHF'], axis=1)
    y_valid= y_valid[non_nan_inds]

    X_test = test_df

    # to numpy
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_valid = X_valid.to_numpy()
    y_valid = y_valid.to_numpy()
    X_test = X_test.to_numpy()
    # process seasons
    X_train = process_seasons(X_train)
    X_valid = process_seasons(X_valid)
    X_test = process_seasons(X_test)
    # iterative imputer
    X_train = data_processing(X_train, X_train)
    X_valid = data_processing(X_valid, X_train)
    X_test = data_processing(X_test, X_train)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_valid, y_valid, X_test


def process_seasons(X):
    X = np.where(X=="spring", 0.25, X)
    X = np.where(X=="summer", 0.5, X)
    X = np.where(X=="autumn", 0.75, X)
    X = np.where(X=="winter", 1., X)
    return X.astype(np.float32)

def data_processing(X, X_train, iterative_imputer=False, normalize=False):
    # X_copy = X.copy()
    # X_train_copy = X_train.copy()
    print("Imputer: ", iterative_imputer, "Normalize: ", normalize)
    if iterative_imputer:
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X_train)
        X = imp.transform(X)
    else:
        # col_mean = np.nanmean(X, axis=0)
        # inds = np.where(np.isnan(X))
        # X[inds] = np.take(col_mean, inds[1])
        # np.where(np.isnan(X_copy), np.ma.array(X_copy, mask=np.isnan(X_copy)).mean(axis=0), X_copy)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_train)
        X = imp.transform(X)
    # normalize data
    if normalize:
        min_max_scaler_train = MinMaxScaler()
        X = min_max_scaler_train.fit_transform(X)
    return X


def modeling_and_prediction(X_train, y_train, X_valid, y_valid, X_test):
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

    # y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    kernel_list = []
    kernel_list.append(RBF() + ConstantKernel() +  WhiteKernel())
    kernel_list.append(Matern() + ConstantKernel() +  WhiteKernel())
    # kernel_list.append(ConstantKernel() * ExpSineSquared() + WhiteKernel())
    kernel_list.append(RationalQuadratic() + ConstantKernel() + WhiteKernel())
    # kernel_list.append(ConstantKernel() * DotProduct())

    # record the best model
    best_gpr = None
    best_valid_score = -np.inf
    best_result = [-np.inf, -np.inf, {}]   # best_train_score, best_valid_score, best_params
    for kernel in kernel_list:
        gpr = GaussianProcessRegressor(kernel)
        gpr.fit(X_train, y_train)
        train_score = gpr.score(X_train, y_train)
        valid_score = gpr.score(X_valid, y_valid)
        result = [train_score, valid_score, gpr.kernel_.get_params()]
        print("result: ", result)
        if valid_score > best_valid_score:
            best_gpr = gpr
            best_valid_score = valid_score
            best_result = [train_score, valid_score, gpr.kernel_.get_params()]
        print("best result: ", best_result)
        print("\n")
    
    y_pred = best_gpr.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_valid, y_valid, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_valid, y_valid, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

