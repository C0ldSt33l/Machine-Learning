class Line:
    xs: list[float]
    ys: list[float]

    def __init__(self, xs: list[float], ys: list[float]):
        self.xs = xs
        self.ys = ys

def calc_y(x: float, a: float, b: float):
    return a * x + b

def get_seperate_line(xs: list[float], a: float, b: float) -> Line:
    ys = [calc_y(x, a, b) for x in xs]
    return Line(xs, ys)