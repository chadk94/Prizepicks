import matplotlib.pyplot as plt
from keras import backend as K


def R2(y, y_hat):
    ss_res = K.sum(K.square(y - y_hat))
    ss_tot = K.sum(K.square(y - K.mean(y)))
    return (1 - ss_res/(ss_tot + K.epsilon()))


def plot_history(history, metrics):
    _, ax = plt.subplots(nrows=1,
                           ncols=2,
                           sharey=True,
                           figsize=(15, 3))
    _plot_training(ax, history, metrics)
    _plot_validation(ax, history, metrics)
    plt.show()


def _plot_training(ax, history, metrics):
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()


def _plot_validation(ax, history, metrics):
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    # TODO IMPROVE PLOTTING/DISPLAY
