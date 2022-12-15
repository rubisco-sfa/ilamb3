"""Functions for preparing datasets for comparison."""


import datetime
import warnings
from typing import Tuple

import numpy as np
import xarray as xr

# pylint: disable=no-name-in-module
from . import dataset as dset


def nest_spatial_grids(*args):
    """returns the arguments interpolated to a nested grid"""
    lat = np.empty(0)
    lon = np.empty(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for arg in args:
            lat = np.union1d(lat, arg[dset.get_latitude_name(arg)])
            lon = np.union1d(lon, arg[dset.get_longitude_name(arg)])
        out = [
            arg.interp(lat=lat, lon=lon, method="nearest").pint.quantify(arg.pint.units)
            for arg in args
        ]
    return out


def is_spatially_aligned(dsa: xr.Dataset, dsb: xr.Dataset) -> bool:
    """Are the lats and lons of dsa and dsb aligned?"""
    alat_name = dset.get_latitude_name(dsa)
    blat_name = dset.get_latitude_name(dsb)
    alon_name = dset.get_longitude_name(dsa)
    blon_name = dset.get_longitude_name(dsb)
    if alat_name is None or blat_name is None:
        return False
    if alon_name is None or blon_name is None:
        return False
    if dsa[alat_name].size != dsb[blat_name].size:
        return False
    if dsa[alon_name].size != dsb[blon_name].size:
        return False
    if not np.allclose(dsa[alat_name], dsb[blat_name]):
        return False
    if not np.allclose(dsa[alon_name], dsb[blon_name]):
        return False
    return True


def pick_grid_aligned(
    ref0: xr.Dataset, com0: xr.Dataset, ref: xr.Dataset = None, com: xr.Dataset = None
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Pick variables for ref and com such that they are grid aligned without
    recomputing if not needed."""
    if is_spatially_aligned(ref0, com0):
        return ref0, com0
    if (ref is not None) and (com is not None):
        if is_spatially_aligned(ref, com):
            return ref, com
    return nest_spatial_grids(ref0, com0)


def trim_time(dsa: xr.Dataset, dsb: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """When comparing dsb to dsa, we need the maximal amount of temporal
    overlap."""
    if "time" not in dsa.dims:
        return dsa, dsb
    if "time" not in dsb.dims:
        return dsa, dsb
    # pylint: disable=no-member
    at0, atf = dset.time_extent(dsa)
    bt0, btf = dset.time_extent(dsb)
    tol = datetime.timedelta(days=1)  # add some padding
    tm0 = max(at0, bt0) - tol
    tmf = min(atf, btf) + tol
    dsa = dsa.sel(time=slice(tm0, tmf))
    dsb = dsb.sel(time=slice(tm0, tmf))
    return dsa, dsb


def adjust_lon(dsa: xr.Dataset, dsb: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """When comparing dsb to dsa, we need their longitudes uniformly in
    [-180,180) or [0,360)."""
    alon_name = dset.get_longitude_name(dsa)
    blon_name = dset.get_longitude_name(dsb)
    if alon_name is None or blon_name is None:
        return dsa, dsb
    a360 = (dsa[alon_name].min() >= 0) * (dsa[alon_name].max() <= 360)
    b360 = (dsb[blon_name].min() >= 0) * (dsb[blon_name].max() <= 360)
    if a360 and not b360:
        dsb[blon_name] = dsb[blon_name] % 360
        dsb = dsb.sortby(blon_name)
    elif not a360 and b360:
        dsb[blon_name] = (dsb[blon_name] + 180) % 360 - 180
        dsb = dsb.sortby(blon_name)
    return dsa, dsb


def make_comparable(ref: xr.Dataset, com: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """."""
    ref, com = trim_time(ref, com)
    ref = ref.pint.quantify()
    com = com.pint.quantify()
    com["gpp"] = com["gpp"].pint.to(ref["gpp"].pint.units)
    ref, com = adjust_lon(ref, com)
    return ref, com
