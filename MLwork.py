import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV, Ridge, ElasticNetCV, ElasticNet, BayesianRidge, LogisticRegression, SGDRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

Inputs=1069
def simplemodel():
    model=Sequential()
    model.add(Dense(20,input_dim=Inputs,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def multiplelayers():
    model=Sequential()
    model.add(Dense(20,input_dim=Inputs,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(.5))#any useless data? avoid overfitting
    model.add(Dense(10,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(5,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def widelayer():
    model = Sequential()
    model.add(Dense(40, input_dim=Inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def evaluate(x,y,modeltype,playername,opp,home):
    estimators=[]
    estimators.append(('standardize',StandardScaler())) #scale
    estimators.append(('mlp', KerasRegressor(build_fn=modeltype, epochs=100, batch_size=5, verbose=0)))
    pipeline=Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, x, y, cv=kfold)
    pipeline.fit(x, y)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    gamedata=[playername,opp,home]
    print("Projected shots is ", pipeline.predict(gamedata))

def playerprojection(data,playername,opponent, home,modeltype):
    data=data.dropna()
    y = data['Shots']
    x = data.drop(['Shots'], axis=1)
    print("variance in shots is", np.var(y))
    basic(x, y)#, modeltype,playername,opponent,home)
def basic(x,y):
    datasets=train_test_split(x,y,test_size=.2)
    train_data, test_data, train_labels, test_labels = datasets
    scaler=StandardScaler()
    scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    test_data=scaler.transform(test_data)
    mlp=MLPClassifier(hidden_layer_sizes=(10,5),max_iter=1000)
    mlp.fit(train_data,train_labels)
    predictions_train=mlp.predict(train_data)
    print(accuracy_score(predictions_train,train_labels))
    predictions_test=mlp.predict(test_data)
    print(accuracy_score(predictions_test,test_labels))
    confusion_matrix(predictions_train, train_labels)
    print(classification_report(predictions_test, test_labels))
def propbet(data):
    scaler = StandardScaler()

    X,y=data.drop(columns=['Name','Shots']),data.Shots
    print (X)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.25, random_state=2)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.fit_transform(X_val.values)
    alphavec = 10 ** np.linspace(-2, 2, 200)

    lasso_cv = LassoCV(alphas=alphavec, cv=5)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_cv.alpha_
    for col, coef in zip(X_train.columns, lasso_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    print('R2 for LassoCV Model on train set: ' + str(lasso_cv.score(X_train_scaled, y_train)))
    val_set_preds = lasso_cv.predict(X_val_scaled)
    print('R2 for LassoCV Model on validation set: ' + str(lasso_cv.score(X_val_scaled, y_val)))
    mae = mean_absolute_error(y_val, val_set_preds)
    print('Mean absolute error for LassoCV model on validation set: ' + str(mae))

    alpha = np.logspace(-4, 2, 100)  # np.logspace(-4, -.1, 20)
    param_grid = dict(alpha=alpha)
    grid_en = GridSearchCV(ElasticNet(), param_grid=param_grid,
                           scoring='neg_mean_absolute_error', cv=5)
    grid_result_en = grid_en.fit(X_train, y_train)

    print('Best Score: ', grid_result_en.best_score_)
    print('Best Param: ', grid_result_en.best_params_)
    elastic_cv = ElasticNetCV(alphas = [0.0021544346900318843], cv=5, random_state=0);
    elastic_cv.fit(X_train, y_train)
    print('ElasticNet Mean R Squared Score on training data: ', elastic_cv.score(X_train, y_train))
    print('ElasticNet Mean R Squared Score on validation data: ', elastic_cv.score(X_val, y_val))
    val_set_preds = elastic_cv.predict(X_val)
    mae = mean_absolute_error(y_val, val_set_preds)
    print('Mean absolute error for ElasticNet model on validation set: ' + str(mae))
    rmse = mean_squared_error(y_val, val_set_preds, squared=False)
    print('Root mean squared error for ElasticNet model on validation set: ' + str(rmse))
    for col, coef in zip(X_test.columns, elastic_cv.coef_):
        print(f"{col:<16}: {coef:>12,.7f}")
    elastic_preds=elastic_cv.predict(X)
    data['Model Predictions'] = elastic_preds;
    return
