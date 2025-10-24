"""
ILAMB transforms for selecting data.
"""

from typing import Any

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class select_dim(ILAMBTransform):
    """
    Select a depth if the dimension exists.

    Parameters
    ----------
    dim : str
        The dimension.
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
        dim: str,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        self.dim = dim
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
                f"This transform allows for a single {dim} selection with `value` or a range with "
                f"both `vmin` and `vmax`, but I found {value=} {vmin=} {vmax=} and these other keywords {kwargs}"
            )

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Select a dim or dim range of the input dataset, if the dimension is found.
        """
        try:
            dim_name = dset.get_dim_name(ds, self.dim)
        except KeyError:
            return ds
        ds = ds.sortby(dim_name)
        if self.value is not None:
            return ds.sel({dim_name: self.value}, method="nearest", drop=True)
        else:
            return ds.sel({dim_name: slice(self.vmin, self.vmax)})


class select_time(select_dim):
    def __init__(
        self,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        super().__init__("time", value, vmin, vmax, **kwargs)


class select_depth(select_dim):
    def __init__(
        self,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        super().__init__("depth", value, vmin, vmax, **kwargs)


class select_lat(select_dim):
    def __init__(
        self,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        super().__init__("lat", value, vmin, vmax, **kwargs)


class select_lon(select_dim):
    def __init__(
        self,
        value: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ):
        super().__init__("lon", value, vmin, vmax, **kwargs)
