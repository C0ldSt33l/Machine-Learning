import matplotlib.pyplot as plt
import matplotlib.animation as animator

from helpers.read_csv import get_data_from_csv
from helpers.point import *
from neuron import ClassificationNeuron
from helpers.line import Line, get_line

def split_points(points: PointList) -> tuple[PointList, PointList]:
    first = list(filter(lambda el: el.mark == -1, points))
    second = list(filter(lambda el: el.mark == 1, points))
    return (first, second)

def setup_plot(ax, red_points: PointList, blue_points: PointList, x_lims: list[float], y_lims: list[float]):
    ax.set_title('Linear Binary Classification')

    ax.set_xlabel('X')
    ax.set_xlim(*x_lims)
    ax.set_ylabel('Y')
    ax.set_ylim(*y_lims)

    ax.scatter([p.x for p in red_points], [p.y for p in red_points], color='red')
    ax.scatter([p.x for p in blue_points], [p.y for p in blue_points], color='blue')

def linear_binary_classification_test():
    data = get_data_from_csv(r'data/classification/line with angle.csv', MarkedPoint)
    neuron = ClassificationNeuron(learnin_speed=0.1)

    xs = [p.x for p in data]
    ys = [p.y for p in data]
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)

    sep_lines: list[Line] = []
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='green')
    lim_range = 2
    setup_plot(
        ax, *split_points(data),
        [x_min, x_max + lim_range],
        [y_min - lim_range, y_max + lim_range]
    )
    iter = 0
    while iter := iter + 1:
        is_learned = neuron.process_learn(data)

        a, b = neuron.get_a_and_b()
        sep_lines.append(get_line(xs, a, b))
        if is_learned:
            print(f'ALL POINTS ARE GUESSED CORRECTLLY AT {iter}TH ATTEMPT')
            break
    
    def animate(i):
        sep_line = sep_lines[i]
        line.set_data(sep_line.xs, sep_line.ys)
        if i == len(sep_lines)-1:
            line.set_color('grey')
        return line,

    ani = animator.FuncAnimation(fig, animate, frames=len(sep_lines), interval=50, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    linear_binary_classification_test()