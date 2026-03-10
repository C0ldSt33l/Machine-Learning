from __future__ import annotations

from typing import TYPE_CHECKING

from activation_funcs import ActivationFunc
from neuron import Neuron

if TYPE_CHECKING:
    from neuron_net import NeuronNet

# def local_gradient_hidden(neuron: Neuron, next_layer: Layer) -> float:
#     pass

# def local_gradient_output(guess: float, anwser: float, net: float, activation_derivate: Callable) -> float:
#     delta = guess - anwser
#     return delta * activation_derivate(net)


class Layer:
    neurons: list[Neuron]
    prev_layer: Layer | None
    neuron_net: NeuronNet | None

    activation: ActivationFunc
    activation_derivate: ActivationFunc

    predicts: list[float]

    def __init__(
        self,
        neuron_count: int,
        weights_per_neuron: int,
        activation: ActivationFunc,
        activation_derivate: ActivationFunc,
        prev_layer: Layer | None = None,
        neuron_net: NeuronNet | None = None,
    ):
        self.prev_layer = prev_layer
        self.neuron_net = neuron_net

        self._create_neurons(neuron_count, weights_per_neuron)
        self.set_activation(activation, activation_derivate)

    def _create_neurons(self, count: int, weights_per_neuron: int):
        self.neurons = [Neuron(weights_per_neuron) for _ in range(count)]
        # self.predicts = [0.0] * count

    def activate(self, input: list[float]) -> list[float]:
        return [n.guess(input) for n in self.neurons]

    def set_neuron_net(self, neuron_net: NeuronNet):
        self.neuron_net = neuron_net
        prev_layer = neuron_net.layers[-1] if len(neuron_net.layers) > 0 else None
        for n in self.neurons:
            n.prev_layer = prev_layer

    def set_activation(
        self, activation: ActivationFunc, activation_derivate: ActivationFunc
    ):
        self.activation = activation
        self.activation_derivate = activation_derivate
        for n in self.neurons:
            n.activation = activation
            n.activation_derivate = activation_derivate

    def get_local_gradient(self, weight_idx) -> float:
        sumprod = sum(
            map(lambda n: n.local_gradient * n.weights[weight_idx], self.neurons)
        )
        return sumprod
