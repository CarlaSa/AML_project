from typing import Union, Tuple, List

from .trafo import Trafo, Transformable


class Compose(Trafo):
    pipeline: List[Trafo]

    def __init__(self, *steps: Trafo, accept_empty: bool = False) -> None:
        if len(steps) < 1 and accept_empty is not True:
            raise ValueError("A ComposedTrafo must consist of 1 Trafo.")
        self.pipeline = list(steps)

    def __call__(self, *transformands: Transformable) \
            -> Union[Transformable, Tuple[Transformable, ...]]:
        if len(self.pipeline) == 0:
            if len(transformands) == 1:
                return transformands[0]
            else:
                return transformands
        for step in self.pipeline[:-1]:
            transformands = step(*transformands)
            if not isinstance(transformands, tuple):
                transformands = (transformands,)
        return self.pipeline[-1](*transformands)

    def _json_serializable(self) -> dict:
        return {
            "class": self.__class__.__qualname__,
            "pipeline": [step._json_serializable() for step in self.pipeline]
        }
