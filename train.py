# standard imports
import os.path as op

# 3rd party imports
import numpy as np
from keras import metrics as M

# local imports
from modeling.util import plot_history
from modeling.prizenet import PrizeNet
from dataset.datawork import dataload, clean


def train(data):
    x, y = data.drop(columns=['Name', 'Shots']), data.Shots
    print(f"variance in shots is {np.var(y)}")
    print(x.columns)

    model = PrizeNet(x.columns.shape)

    model.compile(optimizer='adam', loss='mean_absolute_error',
                  metrics=M.mean_squared_error)
    training = model.fit(x=x,
                         y=y,
                         batch_size=32,
                         epochs=100,
                         shuffle=True,
                         verbose=0,
                         validation_split=0.3)

    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]

    plot_history(training.history, metrics)


if __name__ == '__main__':
    raw_data = dataload(op.join('data', 'MLSdata.csv'))
    data = clean(raw_data)
    train(data)
