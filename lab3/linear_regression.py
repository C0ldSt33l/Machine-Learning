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


def setup_plot(ax, points: list[Point], x_lims: list[float], y_lims: list[float]):
    """Рисует точки на заданных осях."""
    ax.set_title('Regression')
    ax.set_xlabel('X')
    ax.set_xlim(*x_lims)
    ax.set_ylabel('Y')
    ax.set_ylim(*y_lims)
    ax.scatter([p.x for p in points], [p.y for p in points], color='red')


def regression_test():
    data = get_data_from_csv(r'data/linear regression/learn sample (big).csv', Point)

    xs = np.array([p.x for p in data])
    ys = np.array([p.y for p in data])
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)

    lim_range = 2
    # Границы для отображения
    x_lims = [x_min, x_max + lim_range]
    y_lims = [y_min - lim_range, y_max + lim_range]

    # Модель
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(units=1, activation=activations.linear))

    model.compile(
        loss=keras.losses.MeanSquaredError,
        optimizer=optimizers.Adam(0.1)
    )

    print("Origin weights")
    print(model.get_weights())

    log = model.fit(xs, ys, epochs=50, verbose=False)

    print("New weights")
    print(model.get_weights())

    setup_fig(data, model, log.history['loss'], 'Linear Regression', False)

    plt.show()


if __name__ == '__main__':
    regression_test()