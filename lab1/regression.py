import matplotlib.pyplot as plt
import matplotlib.animation as animator

from helpers.read_csv import get_data_from_csv
from helpers.point import *
from neuron import Neuron
from helpers.line import Line, get_line, calc_y

def guess(input: float) -> float:
    return input

def process_learn(neuron: Neuron, points: list[MarkedPoint]) -> bool:
    def get_avg_offset(points: list[Point], a: float, b: float) -> float:
        s = 0
        for p in points:
            y = calc_y(p.x, a, b)
            s += (y - p.y) ** 2
        return  s / len(points)

    for p in points:
        guess = neuron.guess([p.x])
        if guess != p.y:
            neuron.learn(p.y, guess, [p.x])

    cur_offset = get_avg_offset(points, *neuron.get_a_and_b())
    is_learned = abs(neuron.last_offset - cur_offset) < neuron.precision
    neuron.last_offset = cur_offset
    return is_learned

def get_a_and_b(neuron: Neuron) -> tuple[float, float]:
    return (neuron.weights[0], neuron.bias)



def regression_test():
    data = get_data_from_csv(r'data/regression/learn sample.csv', Point)
    neuron = Neuron(
        weights=[0],
        learnin_speed=0.001,
        guess_func=guess,
        learn_func=process_learn,
        get_a_and_b_func=get_a_and_b,
        last_offset=100,
        precision=0.001
    )

    xs = [p.x for p in data]
    ys = [p.y for p in data]
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)

    reg_lines: list[Line] = []
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='green')
    lim_range = 2
    setup_plot(ax, data, [x_min, x_max + lim_range], [y_min - lim_range, y_max + lim_range])
    iter = 0
    # last_offset = 100
    while iter := iter + 1:
        is_learned = neuron.process_learn(data)
        # last_offset = neuron.get_avg_offset(data)

        a, b = neuron.get_a_and_b()
        reg_lines.append(get_line(xs, a, b))
        if is_learned:
            print(f'ALL POINTS ARE GUESSED CORRECTLLY AT {iter}TH ATTEMPT')
            break
    
    def animate(i):
        sep_line = reg_lines[i]
        line.set_data(sep_line.xs, sep_line.ys)
        if i == len(reg_lines)-1:
            line.set_color('black')
        return line,

    frames = len(reg_lines)
    ani = animator.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True, repeat=False)
    plt.show()

def setup_plot(ax, points: list[Point], x_lims: list[float], y_lims: list[float]):
    ax.set_title('Regression')

    ax.set_xlabel('X')
    ax.set_xlim(*x_lims)
    ax.set_ylabel('Y')
    ax.set_ylim(*y_lims)

    ax.scatter([p.x for p in points], [p.y for p in points], color='red')

if __name__ == '__main__':
    regression_test()
