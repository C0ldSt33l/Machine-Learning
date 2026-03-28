import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
from keras import activations

from helpers.line import *
from helpers.point import *
from helpers.read_csv import get_data_from_csv


def split_points(
    points: list[MarkedPoint],
) -> tuple[list[MarkedPoint], list[MarkedPoint]]:
    first = list(filter(lambda el: el.mark == -1, points))
    second = list(filter(lambda el: el.mark == 1, points))
    return (first, second)


def setup_plot(
    ax,
    red_points: PointList,
    blue_points: PointList,
    x_lims: list[float],
    y_lims: list[float],
):
    ax.set_title("Non-linear Binary Classification")
    ax.set_xlabel("X")
    ax.set_xlim(*x_lims)
    ax.set_ylabel("Y")
    ax.set_ylim(*y_lims)
    ax.scatter([p.x for p in red_points], [p.y for p in red_points], color="red")
    ax.scatter([p.x for p in blue_points], [p.y for p in blue_points], color="blue")


def linear_classification_test():
    data = get_data_from_csv(r"data/linear classification/learn sample.csv", MarkedPoint)

    xs = [p.x for p in data]
    ys = [p.y for p in data]
    inputs = np.array([(p.x, p.y) for p in data])
    targets = np.array([p.mark for p in data])
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)

    # Модель
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=1, activation=activations.sigmoid))

    model.compile(
        loss=keras.losses.MeanSquaredError,
        optimizer=keras.optimizers.Adam(0.1)
    )

    print("Origin weights")
    print(model.get_weights())

    log = model.fit(inputs, targets, epochs=300, verbose=False)

    print("New weights")
    print(model.get_weights())

    print("Guesses")
    print(model.predict(inputs, verbose=False))

    # -------- Создаём два подграфика рядом --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. График ошибки (loss)
    ax1.plot(log.history["loss"])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")

    # 2. Карта классификации с фоном вероятностей
    margin = 0.5
    x_min_plot = x_min - margin
    x_max_plot = x_max + margin
    y_min_plot = y_min - margin
    y_max_plot = y_max + margin

    resolution = 200
    x_grid = np.linspace(x_min_plot, x_max_plot, resolution)
    y_grid = np.linspace(y_min_plot, y_max_plot, resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z = model.predict(grid_points, verbose=False)
    Z = Z.reshape(X_grid.shape)

    # Заливка фона
    im = ax2.contourf(X_grid, Y_grid, Z, levels=50, cmap='coolwarm', alpha=0.8)
    fig.colorbar(im, ax=ax2, label='Probability of class 1')

    # Исходные точки через setup_plot
    red_points, blue_points = split_points(data)
    setup_plot(
        ax2,
        red_points,
        blue_points,
        [x_min_plot, x_max_plot],
        [y_min_plot, y_max_plot]
    )
    ax2.set_title("Binary Classification (Probability background)")  # переопределяем заголовок

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    linear_classification_test()