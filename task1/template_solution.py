# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
    # Nan to values
    train_df_ori = train_df
    test_df_ori = test_df

    y_train = train_df['price_CHF'].to_numpy()
    non_nan_inds = np.logical_not(np.isnan(y_train))

    train_mean = train_df.drop(['season'],axis=1).mean()
    train_df = train_df.fillna(train_mean)
    test_mean = test_df.drop(['season'],axis=1).mean()
    test_df = test_df.fillna(test_mean)
    # to numpy
    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()
    # process seasons
    X_train = np.where(X_train=="spring", 0.25, X_train)
    X_train = np.where(X_train=="summer", 0.5, X_train)
    X_train = np.where(X_train=="autumn", 0.75, X_train)
    X_train = np.where(X_train=="winter", 1., X_train)
    X_test = np.where(X_test=="spring", 0.25, X_test)
    X_test = np.where(X_test=="summer", 0.5, X_test)
    X_test = np.where(X_test=="autumn", 0.75, X_test)
    X_test = np.where(X_test=="winter", 1., X_test)

    # imp = IterativeImputer(max_iter=1000, random_state=0)
    # X_train = X_train.astype(np.float32)
    # X_test = X_test.astype(np.float32)
    # imp = imp.fit(X_train)
    # X_train = imp.transform(X_train)
    # X_test = imp.transform(X_test)

    # normalize data
    min_max_scaler_train = MinMaxScaler()
    X_train_scaled = min_max_scaler_train.fit_transform(X_train)
    X_test_scaled = min_max_scaler_train.fit_transform(X_test)

    y_train = y_train[non_nan_inds]
    X_train_scaled = X_train_scaled[non_nan_inds]

    # assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train_scaled, y_train, X_test_scaled

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

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    n_folds = 10
    gpr = GaussianProcessRegressor(kernel=RationalQuadratic())

    score_mat = np.zeros(n_folds)
    kf = KFold(n_splits=n_folds)
    for j, (train_index, test_index) in enumerate(kf.split(X_train)):
        gpr = gpr.fit(X_train[train_index], y_train[train_index])
        y_p = gpr.predict(X_train[test_index])
        y_label = y_train[test_index]
        score_mat[j] = r2_score(y_label, y_p)
    print("score matri: ", score_mat)
    print("average score: ", np.mean(score_mat))

    y_pred = gpr.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

