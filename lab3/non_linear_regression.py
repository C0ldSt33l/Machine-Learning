import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
from keras import activations
from keras import optimizers

from helpers.read_csv import get_data_from_csv
from helpers.point import Point

from visualization import setup_fig

def setup_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(units=20, activation=activations.relu))
    model.add(Dense(units=10, activation=activations.relu))
    model.add(Dense(units=5, activation=activations.leaky_relu))
    model.add(Dense(units=1, activation=activations.linear))

    model.compile(
        loss=keras.losses.MeanSquaredError,
        optimizer=optimizers.Adam(0.1)
    )

    return model

def non_linear_regression_test():
    data = get_data_from_csv(r'data/non-linear regression/sample.csv', Point)
    xs, ys = [p.x for p in data], [p.y for p in data]

    model = setup_model()

    print("Origin weights")
    print(model.get_weights())

    log = model.fit(np.array(xs), np.array(ys), epochs=300, verbose=False)

    print("New weights")
    print(model.get_weights())

    setup_fig(data, model, log.history['loss'], 'Non-Linear Regression', False)

    plt.show()


if __name__ == '__main__':
    non_linear_regression_test()