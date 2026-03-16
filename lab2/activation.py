from abc import ABC, abstractmethod

import activation_funcs as af
from activation_funcs import ActivationFunc, DerivateFunc, soft_sign_derivate


class BaseActivation:
    _activation: ActivationFunc
    _derivation: DerivateFunc

    def __init__(self, activation: ActivationFunc, derivation: DerivateFunc) -> None:
        self._activation = activation
        self._derivation = derivation

    def activate(self, x: float) -> float:
        return self._activation(x)

    def derivate(self, x: float) -> float:
        return self._derivation(x)


class LogisticActivation(BaseActivation):
    def __init__(self) -> None:
        super().__init__(af.logistic, af.logistic_derivate)


class HyperbolicTangentActivation(BaseActivation):
    def __init__(self) -> None:
        super().__init__(af.hyperbolic_tangent, af.hyperbolic_tangent_derivate)


class SoftSighActivation(BaseActivation):
    def __init__(self) -> None:
        super().__init__(af.soft_sign, soft_sign_derivate)


class ReluActivation(BaseActivation):
    def __init__(self) -> None:
        super().__init__(af.relu, af.relu_derivate)


class LeakyReluActivation(BaseActivation):
    mult: float

    def __init__(self, mult: float = 0.1) -> None:
        super().__init__(af.leaky_relu, af.leaky_relu_derivate)

    def activate(self, x: float) -> float:
        return self._activation(x, self.mult)

    def derivate(self, x: float) -> float:
        return self._derivation(x, self.mult)
