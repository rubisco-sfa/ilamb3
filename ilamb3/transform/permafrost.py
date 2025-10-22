"""
An ILAMB transform for
"""

from typing import Any, Literal

import numpy as np
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


def _permafrost_extent_slater2013(
    ds: xr.Dataset, depth_max: float = 3.5, temperature_threshold: float = 0.0
) -> xr.Dataset:
    """Section 3a of Slater2013"""

    # this input data must have depth and bounds
    depth_dim = dset.get_dim_name(ds, "depth")
    if depth_dim not in ds.cf.bounds:
        ds = ds.cf.add_bounds(depth_dim)
    dsl = dset.sel(ds, depth_dim, 0, depth_max)

    # a cell is frozen if the annual maximum temperature is lower than a
    # threshold
    frozen = (
        dset.convert(dsl.groupby("time.year").max()["tsl"], "degC")
        < temperature_threshold
    )

    # a cell is considered permafrosted if this is true for 2 consecutive years
    ds["permafrost_extent"] = (frozen & frozen.shift(year=-1, fill_value=False)).any(
        dim=depth_dim
    )

    # the active layer is then the non-frozen thickness over permafrosted areas
    depth_bnds = ds[ds.cf.bounds[depth_dim][0]]
    bnd_dim = [d for d in depth_bnds.dims if d != depth_dim]
    assert len(bnd_dim) == 1
    layer_thickness = depth_bnds.diff(bnd_dim[0]).squeeze()
    ds["active_layer_thickness"] = xr.where(
        ds["permafrost_extent"], (~frozen * layer_thickness).sum(dim=depth_dim), np.nan
    )

    return ds


EXTENT_METHODS = {"slater2013": _permafrost_extent_slater2013}


class permafrost_extent(ILAMBTransform):
    """
    An ILAMB transform for .

    Parameters
    ----------
    """

    def __init__(self, method: Literal["slater2013"] = "slater2013", **kwargs: Any):
        self.method = method

    def required_variables(self) -> list[str]:
        """
        Return the required variables for the transform.
        """
        return ["tsl"]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Compute
        """
        if "permafrost_extent" in ds:
            return ds
        if not set(self.required_variables()).issubset(ds):
            return ds
        ds = EXTENT_METHODS[self.method](ds)
        ds["permafrost_extent"].attrs = {
            "long_name": f"Permafrost extent based on {self.method}",
            "units": "1",
        }
        ds = ds["permafrost_extent"].any(dim="year").squeeze().to_dataset()
        ds["permafrost_extent"] = xr.where(
            ds["permafrost_extent"] > 0, ds["permafrost_extent"], np.nan
        )
        return ds
