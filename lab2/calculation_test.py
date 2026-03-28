from datetime import datetime
from tkinter import WRITABLE

from activation import *
from helpers.log import write_log
from helpers.point import MarkedPoint
from helpers.read_csv import get_data_from_csv
from layer import Layer
from neuron_net import NeuronNet


def learn_func(nn: NeuronNet, inputs: list[MarkedPoint]) -> bool:
    mse = 0.0
    is_learned = True
    for i in inputs:
        guess = nn.guess([i.x, i.y])
        mse += (guess[0] - i.mark) ** 2
        if guess != i.mark:
            is_learned = False
            nn.backpropagation(i.mark)
    mse /= len(inputs)
    print("MSE: ", mse)
    return is_learned


def test():
    data = get_data_from_csv(r"data/rotated_xor.csv", MarkedPoint)

    hidden = Layer(2, 2, ReluActivation())
    output = Layer(1, 2, LogisticActivation())

    nn = NeuronNet([hidden, output], max_iter=100)
    nn.set_learn_func(learn_func)

    logname = datetime.now().strftime("NN test %Y-%m-%d %H_%M_%S")
    write_log(
        f"Hidden activation: {hidden.activation.__class__}\n",
        logname,
        path="log/calc_test/",
    )
    write_log(
        f"Output activation: {output.activation.__class__}\n",
        logname,
        path="log/calc_test/",
    )

    write_log(nn.get_data_str(-1), logname, path="log/calc_test/")

    for i, p in enumerate(data):
        guess = nn.guess([p.x, p.y])
        nn.backpropagation(p.mark)
        log = nn.get_data_from_point(p, i, p.mark, guess)
        write_log(log, logname, path="log/calc_test/")


test()
