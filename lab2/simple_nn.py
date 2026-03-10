import activation_funcs as af
from layer import Layer
from neuron_net import NeuronNet

hidden = Layer(
    2,
    2,
    af.relu,
    af.relu_derivate,
)
output = Layer(1, 2, af.logistic, af.logistic_derivate)

nn = NeuronNet([hidden, output])
