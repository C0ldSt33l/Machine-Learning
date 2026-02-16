import matplotlib.pyplot as plt
import matplotlib.animation as animator

from read_csv import get_data_from_csv
from point import Point
from neuron import RegressionNeuron 
from line import Line, get_seperate_line

def regression_test():
    data = get_data_from_csv(r'data/regression/learn sample (big).csv', Point)
    neuron = RegressionNeuron(learnin_speed=0.001)

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
    last_offset = 100
    while iter := iter + 1:
        is_learned = neuron.process_learn(data, last_offset, [x_min, x_max])
        last_offset = neuron.get_avg_offset(data)

        a, b = neuron.get_a_and_b()
        reg_lines.append(get_seperate_line(xs, a, b))
        if is_learned:
            print(f'ALL POINTS ARE GUESSED CORRECTLLY AT {iter}TH ATTEMPT')
            break

        if iter == 100:
            break
    
    def animate(i):
        sep_line = reg_lines[i]
        line.set_data(sep_line.xs, sep_line.ys)
        if i == len(reg_lines)-1:
            line.set_color('grey')
        return line,

    frames = len(reg_lines)
    if iter > 10:
        frames = 1
        last_line = reg_lines[-1]
        reg_lines.clear()
        reg_lines.append(last_line)

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
