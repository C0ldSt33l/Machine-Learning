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
        learnin_speed: float = 0.01,
        activation: Callable | None = None,
        activation_derivate: Callable | None = None,
        prev_layer: Layer | None = None,
    ):
        self.weights = [uniform(0.0, 100.0) for _ in range(weight_count)]
        self.bias = bias if bias is not None else uniform(0.0, 100.0)
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
        net = self._net(input)
        return self.activation(net)

    def _net(self, input: list[float]) -> float:
        def get_prod(enum) -> float:
            return enum[0] * enum[1]

        sumprod = sum(map(get_prod, zip(self.weights, input)))

        return sumprod + self.bias

    def learn(self, input: list[float]):
        for i in range(len(self.weights)):
            self.weights[i] = self._modify_weight(self.weights[i], input[i])
        self.bias = self._modify_weight(self.bias, 1)

    def _modify_weight(self, weight: float, input: float) -> float:
        return weight - self.learning_speed * self.local_gradient * input

    def get_local_gradient_output(
        self, answer: float, guess: float, net: float
    ) -> float:
        if self.activation_derivate is None:
            raise Exception("Activation derivate is not set")

        delta = answer - guess
        self.local_gradient = delta * self.activation_derivate(net)
        return self.local_gradient

    def get_local_gradient_hidden(self, net: float, weight_idx: int) -> float:
        if self.activation_derivate is None:
            raise Exception("Activation derivate is not set")

        sumprod_gradients = sum(
            map(
                lambda n: n.local_gradient * n.weights[weight_idx],
                cast(Layer, self.prev_layer).neurons,
            )
        )
        self.local_gradient = self.activation_derivate(net) * sumprod_gradients
        return self.local_gradient
