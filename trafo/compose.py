from typing import Union
from .trafo import Trafo, Transformable


class Compose(Trafo):
    pipeline: list[Trafo]

    def __init__(self, *steps: Trafo) -> None:
        if len(steps) < 1:
            ValueError("A ComposedTrafo must consist of at least 1 Trafo.")
        self.pipeline = list(steps)

    def __call__(self, *transformands: Transformable) \
            -> Union[Transformable, tuple[Transformable, ...]]:
        for step in self.pipeline[:-1]:
            transformands = step(*transformands)
            if not isinstance(transformands, tuple):
                transformands = (transformands,)
        return self.pipeline[-1](*transformands)
