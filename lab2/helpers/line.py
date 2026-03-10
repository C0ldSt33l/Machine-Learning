from neuron import Neuron


class Line:
    xs: list[float]
    ys: list[float]

    def __init__(self, xs: list[float], ys: list[float]):
        self.xs = xs
        self.ys = ys


def calc_y(x: float, a: float, b: float):
    return a * x + b


def get_line(xs: list[float], a: float, b: float) -> Line:
    ys = [calc_y(x, a, b) for x in xs]
    return Line(xs, ys)


def get_a_and_b_classification(neuron: Neuron) -> tuple[float, float]:
    x, y, offset = neuron.weights[0], neuron.weights[1], neuron.bias
    return (-x / y, -offset / y)


def get_a_and_b_regression(neuron: Neuron) -> tuple[float, float]:
    return (neuron.weights[0], neuron.bias)
