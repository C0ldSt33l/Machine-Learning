import matplotlib.animation as animator
import matplotlib.pyplot as plt

import activation_funcs as af
from helpers.line import *
from helpers.point import *
from helpers.read_csv import get_data_from_csv
from layer import Layer
from neuron_net import NeuronNet


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


def learn_func(nn: NeuronNet, inputs: list[MarkedPoint]) -> bool:
    is_learned = True
    for i in inputs:
        guess = nn.guess([i.x, i.y])
        if guess != i.mark:
            is_learned = False
            nn.backpropagation(i.mark)

    return is_learned


def nonlinear_classification_test():
    data = get_data_from_csv(r"data/straight_xor.csv", MarkedPoint)

    hidden = Layer(2, 2, af.relu, af.relu_derivate)
    output = Layer(1, 2, af.logistic, af.logistic_derivate)

    nn = NeuronNet([hidden, output], max_iter=1000)
    nn.set_learn_func(learn_func)

    xs = [p.x for p in data]
    ys = [p.y for p in data]
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)
    min_max_xs = [x_min, x_max]

    sep_lines: list[tuple[Line, Line]] = []
    fig, ax = plt.subplots()
    (line1,) = ax.plot([], [], color="green")
    (line2,) = ax.plot([], [], color="green")
    lim_range = 0.5
    setup_plot(
        ax,
        *split_points(data),
        [x_min - lim_range, x_max + lim_range],
        [y_min - lim_range, y_max + lim_range],
    )
    iter = 0
    while (iter := iter + 1) <= nn.max_iter:
        is_learned = nn.process_learn(data)

        last_layer = nn.layers[-1]
        a1, b1 = get_a_and_b_classification(last_layer.neurons[0])
        a2, b2 = get_a_and_b_classification(last_layer.neurons[1])
        sep_lines.append((get_line(min_max_xs, a1, b1), get_line(min_max_xs, a2, b2)))

        nn.print_weights()
        nn.print_gradients()

        if is_learned:
            print(f"ALL POINTS ARE GUESSED CORRECTLLY AT {iter}TH ATTEMPT")
            break

    [print(ls[0], ls[1], end="\n") for ls in sep_lines]

    def animate(i):
        ax.set_title(f"Iter {i}")
        lines = sep_lines[i]
        line1.set_data(lines[0].xs, lines[0].ys)
        line2.set_data(lines[1].xs, lines[1].ys)
        if i == len(sep_lines) - 1:
            line1.set_color("grey")
            line2.set_color("grey")
        return line1, line2

    ani = animator.FuncAnimation(
        fig, animate, frames=len(sep_lines), interval=50, blit=True, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    nonlinear_classification_test()
