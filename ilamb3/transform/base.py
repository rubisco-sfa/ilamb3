"""
An abstract class for implementing transform functions used in ILAMB.
"""

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr


class ILAMBTransform(ABC):
    """
    The ILAMB transform base class.

    A transform is just a function that takes in and returns a dataset.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any):
        """
        Initialize the transform with any keyword arguments.
        """
        raise NotImplementedError()

    @abstractmethod
    def required_variables(self) -> list[str]:
        """
        Return the variables that this transform needs to have in the dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        The body of the transform function.
        """
        raise NotImplementedError()
