from helpers.point import MarkedPoint
from layer import Layer


class NeuronNet:
    layers: list[Layer] = []

    inputs: list[MarkedPoint]
    max_iter: int

    def __init__(self, layers: list[Layer], max_iter: int = 100):
        for l in reversed(layers):
            self.add_layer(l)

        self.max_iter = max_iter

    def start_learn(self):
        for i in range(self.max_iter):
            for input in self.inputs:
                pass

    def add_layer(self, layer: Layer):
        layer.set_neuron_net(self)
        self.layers.append(layer)
