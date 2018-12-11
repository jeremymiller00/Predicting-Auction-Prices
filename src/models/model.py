import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error
from sklearn.linear_model import Ridge
import statsmodels.api as sm

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

    ''' Baseline Model'''
    # Train model (log target)
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

    # Model Evaluation
    residuals = y_test - y_pred
    stud_resid = residuals / np.std(residuals, ddof=1)
    fig, ax = plt.subplots(3,1, figsize = (12,30))

    ax[0].scatter(X_test['YearMade'], stud_resid)
    ax[0].axhline(y=0, c='k', ls='--')
    ax[0].set_xlabel('Year Made')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title("Linear Regression Residuals")

    ax[1].scatter(X_test['ModelID'], stud_resid)
    ax[1].axhline(y=0, c='k', ls='--')
    ax[1].set_xlabel('Model ID')
    ax[1].set_ylabel('Residuals')

    ax[2].scatter(y_pred, stud_resid)
    ax[2].axhline(y=0, c='k', ls='--')
    ax[2].set_xlabel('Predicted Sale Price')
    ax[2].set_ylabel('Residuals')

    f_statistic, p_value, _ = sm.stats.diagnostic.het_goldfeldquandt(
    y_test, X_test, idx=1, alternative='two-sided')
    print(p_value)

    fig = sm.graphics.qqplot(stud_resid, line='45')

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

    '''Ridge Regression'''
    # Set the range of hyper-parameters to search
    params = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}

    # Perform grid search
    ridge = Ridge()
    g = GridSearchCV(ridge, params, cv=10)
    g.fit(X_train, np.log(y_train))

    # Predictions
    g_pred = g.predict(X_test)

    # Floor predictions at zero
    g_pred[g_pred < 0] = y_train.mean()
    g_pred = np.exp(g_pred)

    # Scoring
    rmse_score = np.sqrt(mean_squared_error(g_pred, y_test))
    rmsle_score = np.sqrt(mean_squared_log_error(g_pred, y_test))

    print('Best Params: {}'.format(g.best_params_))
    print('Best Score: {}'.format(g.best_score_))
    print('\nRMSE: {}'.format(rmse_score))
    print('RMSLE: {}'.format(rmsle_score))

    # Train ridge regressor using optimal alpha value
    ridge = Ridge(alpha=1)
    ridge.fit(X_train, np.log(y_train))
    ridge_pred = ridge.predict(X_test)

    # Floor predictions at mean
    ridge_pred[ridge_pred < 0] = y_train.mean()
    ridge_pred = np.exp(ridge_pred)

    # Scoring
    rmse_score = np.sqrt(mean_squared_error(ridge_pred, y_test))
    rmsle_score = np.sqrt(mean_squared_log_error(ridge_pred, y_test))

    # print(X_train.columns)
    print('RMSE: {}'.format(rmse_score))
    print('RMSLE: {}'.format(rmsle_score))
    # ridge.coef_

    # Ridge Model Evaluation
    ridge_residuals = y_test - ridge_pred
    stud_ridge_resid = ridge_residuals / np.std(ridge_residuals, ddof=1)

    fig, ax = plt.subplots(3,1, figsize = (12,30))

    ax[0].scatter(X_test['YearMade'], stud_ridge_resid)
    ax[0].axhline(y=0, c='k', ls='--')
    ax[0].set_xlabel('Year Made')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Ridge Regression Residuals')

    ax[1].scatter(X_test['ModelID'], stud_ridge_resid)
    ax[1].axhline(y=0, c='k', ls='--')
    ax[1].set_xlabel('Model ID')
    ax[1].set_ylabel('Residuals')

    ax[2].scatter(y_pred, stud_ridge_resid)
    ax[2].axhline(y=0, c='k', ls='--')
    ax[2].set_xlabel('Predicted Sale Price')
    ax[2].set_ylabel('Residuals')

    fig = sm.graphics.qqplot(stud_ridge_resid, line='45')


    '''Lasso Regression'''
    # Train model
    lasso = LassoCV()
    lasso.fit(X_train, np.log(y_train))
    lasso_pred = lasso.predict(X_test)

    # Floor predictions at mean
    lasso_pred[lasso_pred < 0] = y_train.mean()
    lasso_pred = np.exp(lasso_pred)

    # Scoring
    rmse_score = np.sqrt(mean_squared_error(lasso_pred, y_test))
    rmsle_score = np.sqrt(mean_squared_log_error(lasso_pred, y_test))

    # print(X_train.columns)
    print('RMSE: {}'.format(rmse_score))
    print('RMSLE: {}'.format(rmsle_score))
    lasso.coef_

    # Lasso Evaluation
    lasso_residuals = y_test - lasso_pred
    stud_lasso_resid = lasso_residuals / np.std(lasso_residuals, ddof=1)

    fig, ax = plt.subplots(3,1, figsize = (12,30))

    ax[0].scatter(X_test['YearMade'], stud_lasso_resid)
    ax[0].axhline(y=0, c='k', ls='--')
    ax[0].set_xlabel('Year Made')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Lasso Regression Residuals')

    ax[1].scatter(X_test['ModelID'], stud_lasso_resid)
    ax[1].axhline(y=0, c='k', ls='--')
    ax[1].set_xlabel('Model ID')
    ax[1].set_ylabel('Residuals')

    ax[2].scatter(y_pred, stud_lasso_resid)
    ax[2].axhline(y=0, c='k', ls='--')
    ax[2].set_xlabel('Predicted Sale Price')
    ax[2].set_ylabel('Residuals')

    fig = sm.graphics.qqplot(stud_lasso_resid, line='45')
