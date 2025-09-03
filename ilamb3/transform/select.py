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
    vmin : float
        The minimum depth to slice.
    vmax : float
        The maximum depth to slice.

    Note
    ----
    Either `value` should be specified or `vmin` and `vmax` but not both.
    """

    def __init__(
        self,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        ok = True
        if value is None:
            ok = vmin is not None and vmax is not None
        if vmin is None:
            ok = vmax is None and value is not None
        if vmax is None:
            ok = vmin is None and value is not None
        if not ok:
            raise ValueError(
                "This transform allows for a single depth selection with `value` or a range with "
                f"both `vmin` and `vmax`, but I found {value=} {vmin=} {vmax=} and these other keywords {kwargs}"
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
        if not dset.is_layered(ds):
            return ds
        depth_name = dset.get_dim_name(ds, "depth")
        if self.value is not None:
            return ds.sel({depth_name: self.value}, method="nearest", drop=True)
        else:
            return ds.sel({depth_name: slice(self.vmin, self.vmax)})
        return ds


class select_time(ILAMBTransform):
    """
    Select a time if the dimension exists.

    Parameters
    ----------
    vmin : timestamp
        The minimum depth to slice.
    vmax : timestamp
        The maximum depth to slice.

    """

    def __init__(self, vmin: str, vmax: str, **kwargs: Any):
        self.vmin = vmin
        self.vmax = vmax

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Select a depth or depth range of the input dataset, if a depth dimension is found.
        """
        if not dset.is_temporal(ds):
            return ds
        time_name = dset.get_dim_name(ds, "time")
        return ds.sel({time_name: slice(self.vmin, self.vmax)})
