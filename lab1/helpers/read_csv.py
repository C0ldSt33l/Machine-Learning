from typing import Type, TypeVar
from inspect import signature

T = TypeVar('T')

def _create_point(type: Type[T], args: list) -> T:
    return type(*args)

def get_data_from_csv(file: str, type: Type[T]) -> list[T]:
    with open(file) as f:
        lines = f.readlines()

    points = []
    arg_count = len(signature(type.__init__).parameters) - 1
    for l in lines[1:]:
        l = l.strip()
        p = _create_point(
            type,
            list(map(lambda s: float(s), l.split(',')))[:arg_count]
        )
        points.append(p)

    return points