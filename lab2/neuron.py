from typing import Callable

from helpers.point import MarkedPoint

class Neuron:
    weights: list[float]
    bias: float
    learning_speed: float

    guess_func: Callable
    learn_func: Callable
    get_a_and_b_func: Callable

    def __init__(
            self,
            weights: list[float],
            bias: float=1.0,
            learnin_speed: float=0.01,
            guess_func: Callable=None,
            learn_func: Callable=None,
            get_a_and_b_func: Callable=None,
            **kwargs
    ):
        self.weights = weights
        self.bias = bias
        self.learning_speed = learnin_speed
        self.guess_func = guess_func
        self.learn_func = learn_func
        self.get_a_and_b_func = get_a_and_b_func

        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    def guess(self, input: list[float]) -> float:
        if len(self.weights) != len(input):
            raise Exception("Weight count != input count")

        net = self._net(input)
        return self.guess_func(net)

    def _net(self, input: list[float]) -> float:
        def get_prod(enum) -> float:
            idx, val = enum
            return val * input[idx]

        sumprod = sum(
            map(
                get_prod,
                enumerate(self.weights)
            )
        )

        return sumprod + self.bias

    def learn(self, answer: float, guess: float, input: list[float]):
        delta = answer - guess
        for i in range(len(self.weights)):
            self.weights[i] = self._modify_weight(self.weights[i], delta, input[i])
        self.bias = self._modify_weight(self.bias, delta, 1)

    def process_learn(self, points: list[MarkedPoint]) -> bool:
        is_learned = self.learn_func(self, points)
        return is_learned

    def _modify_weight(self, weight: float, delta: int, input: float) -> float:
        return weight + self.learning_speed * delta * input

    def get_a_and_b(self) -> tuple[float, float]:
        return self.get_a_and_b_func(self)