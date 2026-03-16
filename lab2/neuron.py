from __future__ import annotations

from random import uniform
from typing import TYPE_CHECKING, Callable, cast

from activation import BaseActivation
from activation_funcs import ActivationFunc

if TYPE_CHECKING:
    from layer import Layer


class Neuron:
    weights: list[float]
    bias: float
    learning_speed: float

    prev_layer: Layer | None

    activation: BaseActivation | None

    local_gradient: float
    net: float
    input: list[float]
    output: float

    def __init__(
        self,
        weight_count: int,
        bias: float | None = None,
        learnin_speed: float = 0.1,
        activation: BaseActivation | None = None,
        prev_layer: Layer | None = None,
    ):
        self.weights = [round(uniform(0.0, 10.0), 3) for _ in range(weight_count)]
        self.bias = bias if bias is not None else round(uniform(0.0, 10.0), 3)
        self.learning_speed = learnin_speed
        self.activation = activation
        self.prev_layer = prev_layer

    def guess(self, input: list[float]) -> float:
        if len(self.weights) != len(input):
            raise Exception("Weight count != input count")
        if self.activation is None:
            raise Exception("Activation func is not set")

        self.input = input
        self.net = self._net(input)
        self.output = self.activation.activate(self.net)
        return self.output

    def _net(self, input: list[float]) -> float:
        def get_prod(enum) -> float:
            return enum[0] * enum[1]

        sumprod = sum(map(get_prod, zip(self.weights, input)))
        return sumprod + self.bias

    def learn(self):
        for i in range(len(self.weights)):
            self.weights[i] = self._modify_weight(self.weights[i], self.input[i])
        self.bias = self._modify_weight(self.bias, 1)

    def _modify_weight(self, weight: float, input: float) -> float:
        return weight - self.learning_speed * self.local_gradient * input

    def calc_local_gradient_output(self, answer: float):
        if self.activation is None:
            raise Exception("Activation is not set")

        # print("Prev layer is ", self.prev_layer)
        delta = self.output - answer
        derivate_res = self.activation.derivate(self.net)
        # print("Output")
        # print("Delta: ", delta, "Derivate res: ", derivate_res)
        self.local_gradient = delta * derivate_res

    def calc_local_gradient_hidden(self, weight_idx: int):
        if self.activation is None:
            raise Exception("Activation is not set")
        if self.prev_layer is None:
            raise Exception("Prev layer is not set")

        # print("Prev layer neurons count: ", len(self.prev_layer.neurons))
        sumprod_gradients = self.prev_layer.get_local_gradient_sum(weight_idx)
        derivate_res = self.activation.derivate(self.net)
        # print("Hiddent")
        # print("Sumprod grads: ", sumprod_gradients, "Derivate res: ", derivate_res)
        self.local_gradient = derivate_res * sumprod_gradients

    def get_data_str(self, id: int) -> str:
        return f"""
        Neuron#{id}
        weights: {self.weights}
        bias: {self.bias}
        net: {self.net}
        local gradient: {self.local_gradient}

        input: {self.input}
        output: {self.output}
"""
