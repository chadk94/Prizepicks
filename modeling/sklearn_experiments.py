# standard imports
import os.path as op

# 3rd party imports
import numpy as np
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import (LassoCV, ElasticNetCV, ElasticNet)

# local imports
from ..dataset.datawork import dataload, clean


def propbet(data):
    scaler = StandardScaler()

    X, y = data.drop(columns=['Name', 'Shots']), data.Shots
    print(X)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=.25, random_state=2)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.fit_transform(X_val.values)

    _LassoCV(X_train_scaled, X_train, y_train, X_val_scaled, y_val)
    _GridSearchCV(X_train, y_train)
    _ElasticNet(X, X_train, y_train, X_val, y_val, X_test, y_test)


def _LassoCV(X_train_scaled, X_train, y_train, X_val_scaled, y_val):

    alphavec = 10 ** np.linspace(-2, 2, 200)

    lasso_cv = LassoCV(alphas=alphavec, cv=5)
    lasso_cv.fit(X_train_scaled, y_train)

    lasso_cv.alpha_

    for col, coef in zip(X_train.columns, lasso_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    print(f'R2 for LassoCV Model on train set: {lasso_cv.score(X_train_scaled, y_train)}')
    val_set_preds = lasso_cv.predict(X_val_scaled)
    print(f'R2 for LassoCV Model on validation set: {lasso_cv.score(X_val_scaled, y_val)}')
    mae = mean_absolute_error(y_val, val_set_preds)
    print(f'Mean absolute error for LassoCV model on validation set: {mae}')


def _GridSearchCV(X_train, y_train):

    alpha = np.logspace(-4, 2, 100)  # np.logspace(-4, -.1, 20)
    param_grid = dict(alpha=alpha)
    grid_en = GridSearchCV(ElasticNet(), param_grid=param_grid,
                           scoring='neg_mean_absolute_error', cv=5)
    grid_result_en = grid_en.fit(X_train, y_train)

    print(f'Best Score: {grid_result_en.best_score_}')
    print(f'Best Param: {grid_result_en.best_params_}')


def _ElasticNet(X, X_train, y_train, X_val, y_val, X_test, y_test):
    elastic_cv = ElasticNetCV(alphas=[0.0021544346900318843],
                              cv=5,
                              random_state=0)
    elastic_cv.fit(X_train, y_train)
    print(f'ElasticNet Mean R Squared Score on training data: {elastic_cv.score(X_train, y_train)}')
    print(f'ElasticNet Mean R Squared Score on validation data: {elastic_cv.score(X_val, y_val)}')
    val_set_preds = elastic_cv.predict(X_val)
    mae = mean_absolute_error(y_val, val_set_preds)
    print(f'Mean absolute error for ElasticNet model on validation set: {mae}')
    rmse = mean_squared_error(y_val, val_set_preds, squared=False)
    print(f'Root mean squared error for ElasticNet model on validation set: {mae}')
    for col, coef in zip(X_test.columns, elastic_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    elastic_preds = elastic_cv.predict(X)
    data['Model Predictions'] = elastic_preds


if __name__ == '__main__':
    raw_data = dataload(op.join('data', 'MLSdata.csv'))
    data = clean(raw_data)
    propbet(data)
