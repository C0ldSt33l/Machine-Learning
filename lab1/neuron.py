import matplotlib.pyplot as plt
from helpers.point import *
from helpers.line import calc_y, get_line

class ClassificationNeuron:
    weights: tuple[float, float]
    bias: float

    learning_speed: float

    def __init__(
            self,
            weights: tuple[float, float]=(0, 0),
            bias: float=1,
            learnin_speed: float=0.01
    ):
        self.weights = weights
        self.bias = bias
        self.learning_speed = learnin_speed
        
    def guess(self, inputs: tuple[float, float]) -> int:
        return 1 if self._net(inputs) >= 0 else -1

    def _net(self, inputs: tuple[float, float]) -> float:
        return self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.bias

    def learn(self, answer: int, guess: int, inputs: tuple[float, float]):
        delta = answer - guess
        self.weights = (
            self._modify_weight(self.weights[0], delta, inputs[0]),
            self._modify_weight(self.weights[1], delta, inputs[1]),
        )
        self.bias = self._modify_weight(self.bias, delta, 1)

    def process_learn(self, points: list[MarkedPoint]) -> bool:
        is_learned = True
        for p in points:
            coords = p.get_coords()
            guess = self.guess(coords)
            if guess != p.mark:
                is_learned = False
                self.learn(p.mark, guess, coords)
        return is_learned

    def _modify_weight(self, weight: float, delta: int, input: float) -> float:
        return weight + self.learning_speed * delta * input

    def get_a_and_b(self) -> tuple[float, float]:
        return (
            -self.weights[0] / self.weights[1],
            -self.bias / self.weights[1]
        )

class RegressionNeuron:
    weight: float
    bias: float

    learning_speed: float

    def __init__(
            self,
            weight: float=0,
            bias: float=0,
            learnin_speed: float=0.1
    ):
        self.weight = weight
        self.bias = bias
        self.learning_speed = learnin_speed
        
    def guess(self, input: float) -> float:
        return self._net(input)

    def _net(self, input: float) -> float:
        return self.weight * input + self.bias

    def learn(self, answer: float, guess: float, input: float):
        delta = answer - guess
        self.weight = self._modify_weight(self.weight, delta, input)
        self.bias = self._modify_weight(self.bias, delta, 1)

    def process_learn(self, points: list[Point], last_offset: float, precision: float = 0.01) -> bool:
        for p in points:
            guess = self.guess(p.x)
            if guess != p.y:
                self.learn(p.y, guess, p.x)

        cur_offset = self.get_avg_offset(points)
        return abs(last_offset - cur_offset) < precision

    def get_avg_offset(self, points: list[Point]) -> float:
        s = 0
        for p in points:
            y = calc_y(p.x, *self.get_a_and_b())
            s += (y - p.y) ** 2
        return  s / len(points)

    def _modify_weight(self, weight: float, delta: int, input: float) -> float:
        return weight + self.learning_speed * delta * input

    def get_a_and_b(self) -> tuple[float, float]:
        return (self.weight, self.bias)

