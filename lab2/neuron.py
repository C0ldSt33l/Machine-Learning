from __future__ import annotations

from random import uniform
from typing import TYPE_CHECKING, Callable, cast

from activation_funcs import ActivationFunc

if TYPE_CHECKING:
    from layer import Layer


class Neuron:
    weights: list[float]
    bias: float
    learning_speed: float

    prev_layer: Layer | None

    activation: ActivationFunc | None
    activation_derivate: ActivationFunc | None

    local_gradient: float
    net: float
    input: list[float]
    output: float

    def __init__(
        self,
        weight_count: int,
        bias: float | None = None,
        learnin_speed: float = 0.1,
        activation: Callable | None = None,
        activation_derivate: Callable | None = None,
        prev_layer: Layer | None = None,
    ):
        self.weights = [uniform(-100.0, 100.0) for _ in range(weight_count)]
        self.bias = bias if bias is not None else uniform(-100.0, 100.0)
        self.learning_speed = learnin_speed
        self.activation = activation
        self.activation_derivate = activation_derivate
        self.prev_layer = prev_layer

    def guess(self, input: list[float]) -> float:
        if len(self.weights) != len(input):
            raise Exception("Weight count != input count")
        if self.activation is None:
            raise Exception("Activation func is not set")

        self.input = input
        self.net = self._net(input)
        self.output = self.activation(self.net)
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
        if self.activation_derivate is None:
            raise Exception("Activation derivate is not set")

        delta = answer - self.output
        self.local_gradient = delta * self.activation_derivate(self.net)

    def calc_local_gradient_hidden(self, weight_idx: int):
        if self.activation_derivate is None:
            raise Exception("Activation derivate is not set")
        if self.prev_layer is None:
            raise Exception("Prev layer is not set")

        sumprod_gradients = self.prev_layer.get_local_gradient_sum(weight_idx)
        self.local_gradient = self.activation_derivate(self.net) * sumprod_gradients
