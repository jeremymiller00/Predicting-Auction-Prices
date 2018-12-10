import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import src.model as m

def rmsle(y_true, y_pred): 
    """Compute the Root Mean Squared Log Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred): 
    """Compute the Root Mean Squared Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

##########################################################################
if __name__ == "__main__":
    
    # load the data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    X_sid = pd.read_csv('data/processed/X_sid.csv')

    # Train Baseline model (log target)
    pipeline = Pipeline([
        ('scalar', StandardScaler()), 
        ('linear', LinearRegression())]  
        )
    
    pipeline.fit(X_train, np.log(y_train))

    # Predict
    y_pred = pipeline.predict(X_test)

    # Floor predictions at mean (unlog target)
    y_pred[y_pred < 0] = y_train.mean()
    y_pred = np.exp(y_pred)

    # Scoring
    rmse_score = np.sqrt(mean_squared_error(y_pred, y_test))
    rmsle_score = np.sqrt(mean_squared_log_error(y_pred, y_test))

    # print(X_train.columns)
    print('RMSE: {}'.format(rmse_score))
    print('RMSLE: {}'.format(rmsle_score))

    '''Recursive Feature Elimination with Cross-Validation'''

    # Scoring functions
    msle_func = make_scorer(mean_squared_log_error)
    mse_func = make_scorer(mean_squared_error)
    
    # Recursive Feature Elimination
    estimator = LinearRegression()
    selector = RFECV(estimator, cv=10)
    # selector = RFE(estimator, n_features_to_select=20)
    selector = selector.fit(X_train, np.log(y_train))
    y_pred = selector.predict(X_test)
    
    # Floor predictions at zero
    y_pred[y_pred < 0] = y_train.mean()
    y_pred = np.exp(y_pred)
    
    # Scoring
    rmse_score = np.sqrt(mean_squared_error(y_pred, y_test))
    rmsle_score = np.sqrt(mean_squared_log_error(y_pred, y_test))
    
    print('Selected features: {}'.format(X_train.columns[selector.support_]))
    print('\nRMSE: {}'.format(rmse_score))
    print('RMSLE: {}'.format(rmsle_score))