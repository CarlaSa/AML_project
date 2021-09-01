from typing import Union, Protocol
from typeguard import typechecked


class ShapedData(Protocol):
    shape: tuple[int, ...]


@typechecked
class NDShape:
    shape: tuple[Union[int, None], ...]

    def __init__(self, *dimensions: Union[int, None]) -> None:
        self.shape = tuple(dimensions)

    def check(self, value: ShapedData) -> None:
        if len(value.shape) != len(self.shape):
            raise TypeError("Dimensions do not match", self.shape)
        for candidate, gold in zip(value.shape, self.shape):
            if gold is None:
                continue
            if candidate != gold:
                raise TypeError("Shape does not match", self.shape)
