import torch
import random
from typing import Callable, Tuple, Dict, List

from ..trafo import Trafo, TrafoMeta, Transformable
from ..rules import preserve_bounding_boxes as _box_preserving


def custom_gauss(mu: float, sigma: float) -> float:
    x = random.gauss(mu, sigma)
    if mu == 0:
        return x
    if x > 0:
        return x
    return 0.


def randomize(functional_trafo: Callable[..., torch.Tensor],
              *parameter_names: str, preserve_bounding_boxes: bool = False) \
        -> TrafoMeta:
    class DerivedRandomizedTrafo(Trafo):
        kwargs: List[Tuple[float, float]]
        random_function: Callable[[float, float], float]

        parameters: List[str] = parameter_names

        def __init__(self, random_function: Callable[[float, float], float]
                     = custom_gauss, **kwargs: Tuple[float, float]) -> None:
            self.kwargs = kwargs
            self.random_function = random_function
            if not set(kwargs.keys()) == set(self.parameters):
                raise KeyError("We need exactly the following parameters: "
                               + str(self.parameters))

        def compute_parameters(self, *transformands: Transformable) -> \
                Dict[str, float]:
            return {name: self.random_function(a, b)
                    for name, (a, b) in self.kwargs.items()}

    if preserve_bounding_boxes is True:
        DerivedRandomizedTrafo = _box_preserving(DerivedRandomizedTrafo)

    @DerivedRandomizedTrafo.transform.register
    def _(self, transformand: torch.Tensor, **kwargs: float) \
            -> torch.Tensor:
        return functional_trafo(transformand, **kwargs)
    return DerivedRandomizedTrafo
