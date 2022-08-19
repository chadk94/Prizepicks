# 3rd party imports
import numpy as np
from keras.models import Sequential
from keras.layers import (Dense, Dropout)
from keras import backend as K
from keras import metrics as M
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import (
    cross_val_score, KFold, train_test_split, GridSearchCV)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   PolynomialFeatures)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, mean_squared_error,
                             mean_absolute_error)
from sklearn.linear_model import (LinearRegression, LassoCV, Lasso, RidgeCV, Ridge,
                                  ElasticNetCV, ElasticNet, BayesianRidge,
                                  LogisticRegression, SGDRegressor)
import pandas as pd
import pickle


Inputs = 43


def neuralnet(x, y):
    model = Sequential(name="DeepNN", layers=[
        Dense(name="h1", input_dim=Inputs,
              units=int(round((Inputs + 1) / 2)),
              activation='relu'),
        Dropout(name="drop1", rate=0.2),

        Dense(name="h2", units=int(round((Inputs + 1) / 4)),
              activation='relu'),
        Dropout(name="drop2", rate=0.2),

        Dense(name="output", units=1, activation='sigmoid')
    ])
    model.summary()
    return model


def R2(y, y_hat):
    ss_res = K.sum(K.square(y - y_hat))
    ss_tot = K.sum(K.square(y - K.mean(y)))
    return (1 - ss_res/(ss_tot + K.epsilon()))


def playerprojection(data):
    x, y = data.drop(columns=['Name', 'Shots']), data.Shots
    print(f"variance in shots is {np.var(y)}")
    print(x.columns)
    model = neuralnet(x, y)
    model.compile(optimizer='adam', loss='mean_absolute_error',
                  metrics=M.mean_squared_error)
    training = model.fit(x=x, y=y, batch_size=32, epochs=100,
                         shuffle=True, verbose=0, validation_split=0.3)
    metrics = [k for k in training.history.keys() if (
        "loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    # TODO IMPROVE PLOTTING/DISPLAY
    plt.show()


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
    alphavec = 10 ** np.linspace(-2, 2, 200)

    lasso_cv = LassoCV(alphas=alphavec, cv=5)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_cv.alpha_
    for col, coef in zip(X_train.columns, lasso_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    print(
        f'R2 for LassoCV Model on train set: {lasso_cv.score(X_train_scaled, y_train)}')
    val_set_preds = lasso_cv.predict(X_val_scaled)
    print(
        f'R2 for LassoCV Model on validation set: {lasso_cv.score(X_val_scaled, y_val)}')
    mae = mean_absolute_error(y_val, val_set_preds)
    print(f'Mean absolute error for LassoCV model on validation set: {mae}')

    alpha = np.logspace(-4, 2, 100)  # np.logspace(-4, -.1, 20)
    param_grid = dict(alpha=alpha)
    grid_en = GridSearchCV(ElasticNet(), param_grid=param_grid,
                           scoring='neg_mean_absolute_error', cv=5)
    grid_result_en = grid_en.fit(X_train, y_train)

    print(f'Best Score: {grid_result_en.best_score_}')
    print(f'Best Param: {grid_result_en.best_params_}')
    elastic_cv = ElasticNetCV(
        alphas=[0.0021544346900318843], cv=5, random_state=0)
    elastic_cv.fit(X_train, y_train)
    print(
        f'ElasticNet Mean R Squared Score on training data: {elastic_cv.score(X_train, y_train)}')
    print(
        f'ElasticNet Mean R Squared Score on validation data: {elastic_cv.score(X_val, y_val)}')
    val_set_preds = elastic_cv.predict(X_val)
    mae = mean_absolute_error(y_val, val_set_preds)
    print(f'Mean absolute error for ElasticNet model on validation set: {mae}')
    rmse = mean_squared_error(y_val, val_set_preds, squared=False)
    print(
        f'Root mean squared error for ElasticNet model on validation set: {mae}')
    for col, coef in zip(X_test.columns, elastic_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    elastic_preds = elastic_cv.predict(X)
    data['Model Predictions'] = elastic_preds
    return
