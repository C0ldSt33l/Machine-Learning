from typing import TypeAlias

class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f'P({self.x}; {self.y})'

    def get_coords(self) -> tuple[float, float]:
        return (self.x, self.y)

class MarkedPoint(Point):
    mark: int

    def __init__(self, x, y, mark):
        super().__init__(x, y)
        self.mark = int(mark)

    def __str__(self):
        return f'MP({self.x}; {self.y}; {self.mark})'

    def to_point(self) -> Point:
        return Point(self.x, self.y)

PointList: TypeAlias = list[Point]