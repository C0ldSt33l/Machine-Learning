import numpy as np
from keras import Sequential
import matplotlib.pyplot as pl

from helpers.point import MarkedPoint, Point

def setup_fig(data: list[Point], model: Sequential, loss, titel: str, is_classification: bool = True):
    fig, (loss_graph, predict_graph) = pl.subplots(1, 2, figsize=(14, 6))
    _setup_loss_graph(loss_graph, loss)

    if is_classification:
        _setup_classification_graph(
            fig, predict_graph, titel,
            data, model,
        )
    else:
        _setup_regression_graph(
            predict_graph, titel,
            data, model
        )

    pl.tight_layout()

def _setup_loss_graph(ax, loss):
    ax.plot(loss)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")

def _split_points(points: list[MarkedPoint]) -> tuple[list[MarkedPoint], list[MarkedPoint]]:
    first = list(filter(lambda el: el.mark == -1, points))
    second = list(filter(lambda el: el.mark == 1, points))
    return (first, second)

def _get_coords_and_lims(data: list[Point]) -> tuple[list[float], list[float], tuple[float, float], tuple[float]]:
    xs = [p.x for p in data]
    ys = [p.y for p in data]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return (xs, ys, (x_min, x_max), (y_min, y_max))

type _Persent = float

def _get_margin_value(max: float, min: float, persent: _Persent) -> float:
    return (max - min) / 100.0 * persent

def _setup_classification_graph(
        fig, ax, titel: str,
        data: list[MarkedPoint], model: Sequential,
        lim_margin: _Persent = 5.0,
        resolution: int = 200
    ):

    _, _, (x_max, x_min), (y_max, y_min) = _get_coords_and_lims(data)
    x_margin = _get_margin_value(x_max, x_min, lim_margin)
    y_margin = _get_margin_value(y_max, y_min, lim_margin)

    ax.set_title(titel)
    ax.set_xlabel("X")
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylabel("Y")
    ax.set_ylim(y_min, y_max)

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

    Z = model \
        .predict(grid_points, verbose=False) \
        .reshape(X_grid.shape)

    im = ax.contourf(X_grid, Y_grid, Z, levels=50, cmap='coolwarm', alpha=0.8)
    fig.colorbar(im, ax=ax, label='Probability of class 1')

    red_points, blue_points = _split_points(data)
    ax.scatter([p.x for p in red_points], [p.y for p in red_points], color="red")
    ax.scatter([p.x for p in blue_points], [p.y for p in blue_points], color="blue")

def _setup_regression_graph(ax, titel: str, data: list[Point], model: Sequential):
    xs, ys, x_lims, y_lims = _get_coords_and_lims(data)

    ax.set_title(titel)
    ax.set_xlabel('X')
    ax.set_xlim(*x_lims)
    ax.set_ylabel('Y')
    ax.set_ylim(*y_lims)

    ax.scatter(xs, ys, color='red')

    count = int(x_lims[1] - x_lims[0] + 1) * 100
    x_plot = np.linspace(x_lims[0], x_lims[1], count)
    y_plot = model.predict(x_plot, verbose=False)
    ax.plot(x_plot, y_plot, 'b-', label='Prediction')
    ax.legend()

