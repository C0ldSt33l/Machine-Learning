from point import MarkedPoint

class Neuron:
    weights: tuple[float, float]
    bias: float

    learning_speed: float

    def __init__(
            self,
            weights: tuple[float, float]=(0, 0),
            t: float=0,
            learnin_speed: float=0.1
    ):
        self.weights = weights
        self.bias = t
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
