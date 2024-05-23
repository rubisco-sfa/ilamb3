"""Functions for preparing datasets for comparison."""

import datetime
from typing import Union

import numpy as np
import xarray as xr

from ilamb3 import dataset as dset
from ilamb3.exceptions import NoSiteDimension


def nest_spatial_grids(*args):
    """Return the arguments interpolated to a nested grid.

    In order to avoid loss of information to interpolation, when comparing two or more
    datasets, ILAMB uses nearest neighbor interpolation on a grid composed of the union
    of the lat/lon breaks of all sources. See this
    [figure](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018MS001354#jame20779-fig-0002)
    and surrounding discussion for more information.

    Parameters
    ----------
    *args
        Any number of `xr.Datasets` to be included in creating a nested grid.

    Returns
    -------
    *args
        The input *args interpolated to the nested grid.

    """

    def _return_breaks(ds: xr.Dataset, dim_name: str):
        dim = ds[dim_name]
        if "bounds" in dim.attrs and dim.attrs["bounds"] in ds:
            dim = ds[dim.attrs["bounds"]]
        else:
            dim = ds.cf.add_bounds(dim_name)[f"{dim_name}_bounds"]
        return dim.to_numpy().flatten()

    # find the union of all the breaks, and then the centroids of this irregular grid
    lat = np.empty(0)
    lon = np.empty(0)
    for arg in args:
        arg = arg.to_dataset() if isinstance(arg, xr.DataArray) else arg
        lat = np.union1d(lat, _return_breaks(arg, dset.get_dim_name(arg, "lat")))
        lon = np.union1d(lon, _return_breaks(arg, dset.get_dim_name(arg, "lon")))
    lat = 0.5 * (lat[:-1] + lat[1:])
    lon = 0.5 * (lon[:-1] + lon[1:])
    out = []
    for arg in args:
        lat_name = dset.get_dim_name(arg, "lat")
        lon_name = dset.get_dim_name(arg, "lon")
        # the interp function will strip the units off the args, so we do some pint
        # gymnastics to avoid warnings
        iarg = (
            arg.pint.dequantify()
            .interp({lat_name: lat, lon_name: lon}, method="nearest")
            .pint.quantify()
        )
        # if 'bounds' existed, they will now be interpolated and incorrect
        for dim_name in [lat_name, lon_name]:
            dim = iarg[dim_name]
            try:
                iarg = iarg.drop_vars(dim.attrs["bounds"])
            except Exception:
                pass
        out.append(iarg)
    return out


def is_spatially_aligned(dsa: xr.Dataset, dsb: xr.Dataset) -> bool:
    """Check that the lats and lons of dsa and dsb close to each other.

    Parameters
    ----------
    dsa, dsb
        The datasets to compare.
    """
    alat_name = dset.get_dim_name(dsa, "lat")
    blat_name = dset.get_dim_name(dsb, "lat")
    alon_name = dset.get_dim_name(dsa, "lon")
    blon_name = dset.get_dim_name(dsb, "lon")
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
    ref0: xr.Dataset,
    com0: xr.Dataset,
    ref: Union[xr.Dataset, None] = None,
    com: Union[xr.Dataset, None] = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return a reference and comparison dataset that is spatially grid-aligned.

    This routine implements some logic to minimize the computational burden of
    interpolating when applying possibly many analysis routines to a given reference and
    comparison.

    - If the original versions are on the same grid, then return them.
    - If interpolated versions are passed in, then check they are in fact the same and
      return.
    - Otherwise compute a nesting of the original versions and return.

    Parameters
    ----------
    ref0
        The original reference dataset
    com0
        The original comparison dataset
    ref
        The optional interpolated reference
    com
        The optional interpolated comparison

    Returns
    -------
    ref, com
        A reference and comparison obtained with minimal additional computational burden

    """
    if is_spatially_aligned(ref0, com0):
        return ref0, com0
    if (ref is not None) and (com is not None):
        if is_spatially_aligned(ref, com):
            return ref, com
    return nest_spatial_grids(ref0, com0)


def trim_time(dsa: xr.Dataset, dsb: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Return the datasets trimmed to maximal temporal overlap."""
    if "time" not in dsa.dims:
        return dsa, dsb
    if "time" not in dsb.dims:
        return dsa, dsb
    at0, atf = dset.get_time_extent(dsa)
    bt0, btf = dset.get_time_extent(dsb)
    tol = datetime.timedelta(days=1)  # add some padding
    tm0 = max(at0, bt0) - tol  # type: ignore
    tmf = min(atf, btf) + tol  # type: ignore
    dsa = dsa.sel(time=slice(tm0, tmf))
    dsb = dsb.sel(time=slice(tm0, tmf))
    return dsa, dsb


def adjust_lon(dsa: xr.Dataset, dsb: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """When comparing dsb to dsa, we need their longitudes uniformly in
    [-180,180) or [0,360)."""
    try:
        dset.get_lon_span(dsa)
        dset.get_lon_span(dsb)
    except ValueError:
        return dsa, dsb

    alon_name = dset.get_dim_name(dsa, "lon")
    blon_name = dset.get_dim_name(dsb, "lon")
    if alon_name is None or blon_name is None:
        return dsa, dsb
    a360 = (dsa[alon_name].min() >= 0) * (dsa[alon_name].max() <= 360)
    b360 = (dsb[blon_name].min() >= 0) * (dsb[blon_name].max() <= 360)
    if a360 and not b360:
        dsb[blon_name] = dsb[blon_name] % 360
        if "bounds" in dsb[blon_name].attrs and dsb[blon_name].attrs["bounds"] in dsb:
            dsb[dsb[blon_name].attrs["bounds"]] = (
                dsb[dsb[blon_name].attrs["bounds"]] % 360
            )
        dsb = dsb.sortby(blon_name)
    elif not a360 and b360:
        dsb[blon_name] = (dsb[blon_name] + 180) % 360 - 180
        if "bounds" in dsb[blon_name].attrs and dsb[blon_name].attrs["bounds"] in dsb:
            dsb[dsb[blon_name].attrs["bounds"]] = (
                dsb[dsb[blon_name].attrs["bounds"]] + 180
            ) % 360 - 180
        dsb = dsb.sortby(blon_name)
    return dsa, dsb


def make_comparable(
    ref: xr.Dataset, com: xr.Dataset, varname: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return the datasets in a form where they are comparable."""
    # trim away time
    ref, com = trim_time(ref, com)

    # ensure longitudes are uniform
    ref, com = adjust_lon(ref, com)

    # pick just the sites
    try:
        com = extract_sites(ref, com, varname)
    except NoSiteDimension:
        pass

    # convert units, will read in memory so do this last
    ref = ref.pint.quantify()
    com = dset.convert(com, ref[varname].pint.units, varname=varname)

    return ref, com


def extract_sites(
    ds_site: xr.Dataset, ds_spatial: xr.Dataset, varname: str
) -> xr.Dataset:
    """
    Extract the `ds_site` lat/lons from `ds_spatial choosing the nearest site.

    Parameters
    ----------
    ds_site : xr.Dataset
        The dataset which contains sites.
    ds_spatial : xr.Dataset
        The dataset from which we will make the extraction.
    varname : str
        The name of the variable of interest.

    Returns
    -------
    xr.Dataset
        `ds_spatial` at the sites defined in `da_site`.
    """
    # If this throws an exception, this isn't a site data array
    dset.get_dim_name(ds_site[varname], "site")

    # Get the lat/lon dim names
    lat_site = [dim for dim in ["lat", "latitude", "Latitude", "y"] if dim in ds_site]
    lon_site = [dim for dim in ["lon", "longitude", "Longitude", "x"] if dim in ds_site]
    assert len(lat_site) == 1
    assert len(lon_site) == 1
    lat_site = lat_site[0]
    lon_site = lon_site[0]
    lat_spatial = dset.get_dim_name(ds_spatial, "lat")
    lon_spatial = dset.get_dim_name(ds_spatial, "lon")

    # Store the mean model resolution
    model_res = np.sqrt(
        ds_spatial[lon_spatial].diff(dim=lon_spatial).mean() ** 2
        + ds_spatial[lat_spatial].diff(dim=lat_spatial).mean() ** 2
    )

    # Choose the spatial grid to the nearest site
    ds_spatial = ds_spatial.sel(
        {lat_spatial: ds_site[lat_site], lon_spatial: ds_site[lon_site]},
        method="nearest",
    )

    # Check that these sites are 'close enough'
    dist = np.sqrt(
        (ds_site[lat_site] - ds_spatial[lat_spatial]) ** 2
        + (ds_site[lon_site] - ds_spatial[lon_spatial]) ** 2
    )
    assert (dist < model_res).all()
    return ds_spatial
