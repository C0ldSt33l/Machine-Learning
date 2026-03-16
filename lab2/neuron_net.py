from __future__ import annotations

from typing import Callable

from helpers.log import append_tab_to_multilines
from helpers.point import MarkedPoint
from layer import Layer


class NeuronNet:
    layers: list[Layer] = []
    learn_func: Callable[[NeuronNet, list[MarkedPoint]], bool]
    max_iter: int

    mse: float = 0.0

    def __init__(self, layers: list[Layer], max_iter: int = 100):
        for l in reversed(layers):
            self.add_layer(l)
        self.max_iter = max_iter

    def set_learn_func(
        self, learn_func: Callable[[NeuronNet, list[MarkedPoint]], bool]
    ):
        self.learn_func = learn_func

    def process_learn(self, inputs: list[MarkedPoint]) -> bool:
        is_learned = self.learn_func(self, inputs)
        return is_learned

    def guess(self, input: list[float]) -> list[float]:
        input = input
        for l in reversed(self.layers):
            input = l.get_neurons_output(input)
            # print(f"Output: {input}")
        return input

    def backpropagation(self, answer: float):
        for i, l in enumerate(self.layers):
            if i == 0:
                # print("Start calc loc grad for output layer in nn")
                l.learn_neurons(answer)
            else:
                l.learn_neurons()

    def add_layer(self, layer: Layer):
        layer.set_neuron_net(self)
        self.layers.append(layer)

    def print_weights(self):
        for i, l in enumerate(reversed(self.layers)):
            print(f"Layer#{i}")
            for j, n in enumerate(l.neurons):
                print(f"Nueron#{j} weights: {n.weights}; bias:{n.bias}")

    def print_gradients(self):
        for i, l in enumerate(reversed(self.layers)):
            print(f"Layer#{i}")
            for j, n in enumerate(l.neurons):
                print(f"Nueron#{j} gradient: {n.local_gradient}")

    def get_data_str(self, iter: int) -> str:
        data = f"Iter#{iter}\n"
        for i, l in enumerate(reversed(self.layers)):
            data += l.get_data_str(i)
        data += "//////////////////////////////\n"
        return data
