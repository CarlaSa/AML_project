from utils.bounding_boxes import BoundingBoxes
import torch
import numpy as np
from typing import Union, Any
from functools import singledispatchmethod

Transformable = Union[np.ndarray, torch.Tensor]


class TrafoDispatch(object):
    preserve_bounding_box: bool
    numpy_as_3d_tensor: bool

    def __init__(self, preserve_bounding_box: bool = False,
                 numpy_as_3d_tensor: bool = False):
        self.preserve_bounding_box = preserve_bounding_box
        self.numpy_as_3d_tensor = numpy_as_3d_tensor

    def __call__(self, original_class: type) -> type:
        @singledispatchmethod
        def transform(self, transformand: Transformable, **parameters) \
                -> Transformable:
            raise NotImplementedError()

        if self.preserve_bounding_box is True:
            @transform.register
            def _(self, boxes: BoundingBoxes, **parameters) -> BoundingBoxes:
                return boxes

        if self.numpy_as_3d_tensor is True:
            @transform.register
            def _(self, transformand: np.ndarray, **parameters) -> np.ndarray:
                tensor = torch.from_numpy(transformand).reshape(
                    (1, *transformand.shape))
                tensor = self.transform(tensor, **parameters)
                array = tensor[0].numpy()
                return array

        original_class.transform = transform
        return original_class


class Trafo():
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
