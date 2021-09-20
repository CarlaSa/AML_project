from typing import Union, Tuple, List, Optional

from .trafo import Trafo, Transformable


class Compose(Trafo):
    pipeline: List[Trafo]

    def __init__(self, *steps: Trafo, accept_empty: bool = False,
                 max_transformands: Optional[int] = None) -> None:
        if len(steps) < 1 and accept_empty is not True:
            raise ValueError("A ComposedTrafo must consist of 1 Trafo.")
        self.pipeline = list(steps)
        self.max_transformands = max_transformands

    def __call__(self, *transformands: Transformable) \
            -> Union[Transformable, Tuple[Transformable, ...]]:
        if self.max_transformands is None:
            _max = len(transformands)
        else:
            _max = self.max_transformands
        if len(self.pipeline) == 0:
            if len(transformands) == 1:
                return transformands[0]
            else:
                return transformands
        for step in self.pipeline:
            transformands = list(transformands)
            partially_transformed = step(*transformands[:_max])
            if not isinstance(partially_transformed, tuple):
                partially_transformed = (partially_transformed,)
            transformands[:_max] = partially_transformed
        if len(transformands) > 1:
            return tuple(transformands)
        return transformands[0]

    def _json_serializable(self) -> dict:
        return {
            "class": self.__class__.__qualname__,
            "pipeline": [step._json_serializable() for step in self.pipeline]
        }
