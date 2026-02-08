class Neuron:
    weights: tuple[float, float]

    def __init__(self, init_weights: tuple[float, float]=(0, 0)):
        self.weights = init_weights

    def output (self, params: tuple[int, int]) -> int:
        return 1 if self.net(params) > 0 else -1

    def net(self, params: tuple[int, int]) -> float:
        return self.weights[0] * params[0] + self.weights[1] * params[1]

    def learn():
        pass