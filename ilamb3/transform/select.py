"""
ILAMB transforms for selecting data.
"""

from typing import Any

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class select_depth(ILAMBTransform):
    """
    Select a depth if the dimension exists.

    Parameters
    ----------
    value : float
        The value at which to select the nearest depth.
    min : float
        The minimum depth to slice.
    max : float
        The maximum depth to slice.

    Note
    ----
    Either `value` should be specified or `min` and `max` but not both.
    """

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        raise ValueError(
            f"This transform allows for a single depth selection with `value` or a range with both `min` and `max`, but I found {kwargs}"
        )

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Select a depth or depth range of the input dataset, if a depth dimension is found.
        """
        kwargs = self.kwargs
        if not dset.is_layered(ds):
            return ds
        depth_name = dset.get_dim_name(ds, "depth")
        if "value" in kwargs:
            return ds.sel({depth_name: kwargs["value"]}, method="nearest", drop=True)
        if "min" in kwargs and "max" in kwargs:
            return ds.sel({depth_name: slice(kwargs["min"], kwargs["max"])})
        return ds
