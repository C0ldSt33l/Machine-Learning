import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
from keras import activations
from keras import optimizers

from helpers.point import MarkedPoint
from helpers.read_csv import get_data_from_csv

from visualization import setup_fig

def setup_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=1, activation=activations.sigmoid))

    model.compile(
        loss=keras.losses.MeanSquaredError,
        optimizer=optimizers.Adam(0.1)
    )

    return model

def linear_classification_test():
    data = get_data_from_csv(r"data/linear classification/learn sample.csv", MarkedPoint)

    inputs = np.array([(p.x, p.y) for p in data])
    targets = np.array([p.mark for p in data])
 
    model = setup_model()

    print("Origin weights")
    print(model.get_weights())

    log = model.fit(inputs, targets, epochs=300, verbose=False)

    print("New weights")
    print(model.get_weights())

    setup_fig(data, model, log.history['loss'], 'Linear Classification')
    plt.show()


if __name__ == "__main__":
    linear_classification_test()