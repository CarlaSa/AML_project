from __future__ import annotations

import torch
import numpy as np
from typing import Union, Any

from utils import singledispatchmethod

Transformable = Union[np.ndarray, torch.Tensor]


class TrafoMeta(type):
    def __new__(cls, clsname, bases, attrs) -> TrafoMeta:
        @singledispatchmethod
        def transform(self, transformand: Transformable, **parameters) \
                -> Transformable:
            raise NotImplementedError(f"{transformand.__class__.__name__}s "
                                      + "cannot be handled by "
                                      + f"{self.__class__.__name__} Trafos.")
        attrs["transform"] = transform
        return super().__new__(cls, clsname, bases, attrs)


class Trafo(metaclass=TrafoMeta):
    """
    Generic class for transformations on labeled data.
    """

    def __call__(self, *transformands: Transformable) \
            -> Union[Transformable, tuple[Transformable, ...]]:
        """
        Args:
            *transformands (Transformable): Every transformand can be of
                individual type of course, but these types are conserved on
                output and must be can be further restricted by the
                implementing classes.

        Return:
            Union[Transformable, tuple[Transformable, ...]]: transformed
                transformands (data types induced from input)
        """
        parameters = self.compute_parameters(*transformands)
        transformed = tuple(self.transform(transformand, **parameters)
                            for transformand in transformands)
        if len(transformed) == 1:
            return transformed[0]
        return transformed

    def compute_parameters(self, *transformands: Transformable) -> dict[str,
                                                                        Any]:
        """
        This computes all the parameters which differ for each transformation.
        New values are computed on each call of the Trafo and are used to
        transform every transformand.

        This is empty by default, but may be overridden.
        """
        return {}
