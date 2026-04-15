"""Functions for preparing datasets for comparison."""

from typing import Any

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3
from ilamb3 import dataset as dset
from ilamb3.compare.neighborhood import SITE_EXTRACT
from ilamb3.exceptions import TemporalOverlapIssue


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

    # are the arguments all spatial?
    if not all([dset.is_spatial(arg) for arg in args]):
        return args

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
        iarg = arg.interp({lat_name: lat, lon_name: lon}, method="nearest")
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
    ref: xr.Dataset | None = None,
    com: xr.Dataset | None = None,
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


def trim_time(*args: xr.Dataset, **kwargs: xr.Dataset) -> tuple[xr.Dataset]:
    """Return the datasets trimmed to maximal temporal overlap."""

    def _to_tuple(da: xr.DataArray) -> tuple[int]:
        if da.size != 1:
            raise ValueError("Single element conversions only")
        return (int(da.dt.year), int(da.dt.month))

    def _stamp(ymd: tuple[int]):
        return f"{ymd[0]:4d}-{ymd[1]:02d}"

    # If all the data is monthly, then skip the time bounds in computing the
    # time extents as they tend to mess up the logic with arbitrary calendars.
    # However, if the data is a mean over a long time span (as in biomass or
    # soil carbon), we need the time bounds in the extent to compute the mean of
    # the appropriate quantity.
    time_frequency = [
        float(dset.compute_time_measures(arg).mean().values)
        for arg in list(args) + [arg for _, arg in kwargs.items()]
    ]
    inc_bounds = True
    if np.allclose(time_frequency, 30, atol=3):
        inc_bounds = False
    if np.allclose(time_frequency, 365, atol=3):
        inc_bounds = False

    # Get the time extents in the original calendars
    t0 = []
    tf = []
    for arg in args:
        tbegin, tend = dset.get_time_extent(arg, include_bounds=inc_bounds)
        t0.append(tbegin)
        tf.append(tend)
    for _, arg in kwargs.items():
        tbegin, tend = dset.get_time_extent(arg, include_bounds=inc_bounds)
        t0.append(tbegin)
        tf.append(tend)

    # Convert to a date tuple (year, month, day) and find the maximal overlap
    tmin = max(*[_to_tuple(t) for t in t0])
    tmax = min(*[_to_tuple(t) for t in tf])
    if tmax < tmin:
        raise TemporalOverlapIssue(
            f"Minimum final time {tmax} is after maximum initial time {tmin}"
        )

    # Recast back into native calendar objects and select
    tslice = slice(_stamp(tmin), _stamp(tmax))
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg.sel({dset.get_dim_name(arg, "time"): tslice})
    for key, arg in kwargs.items():
        kwargs[key] = arg.sel({dset.get_dim_name(arg, "time"): tslice})

    # Conditional returns based on what was passed in
    if args and not kwargs:
        return args
    if kwargs and not args:
        return kwargs
    return args, kwargs


def same_spatial_grid(
    grid: xr.DataArray | xr.Dataset, *args: xr.Dataset, **kwargs: xr.Dataset
) -> tuple[xr.Dataset]:
    """."""

    def _drop_bounds(ds: xr.Dataset, dim: str) -> xr.Dataset:
        var = ds[dim]
        if "bounds" not in var.attrs:
            return ds
        if var.attrs["bounds"] not in ds:
            return ds
        ds = ds.drop_vars(var.attrs["bounds"])
        ds[dim].attrs.pop("bounds")
        return ds

    args = list(args)
    lat = grid[dset.get_dim_name(grid, "lat")].values
    lon = grid[dset.get_dim_name(grid, "lon")].values
    for i, arg in enumerate(args):
        _, arg = adjust_lon(grid, arg)
        lat_name = dset.get_dim_name(arg, "lat")
        lon_name = dset.get_dim_name(arg, "lon")
        args[i] = arg.interp(
            {lat_name: lat, lon_name: lon},
            method="nearest",
        )
        args[i] = _drop_bounds(args[i], lat_name)
        args[i] = _drop_bounds(args[i], lon_name)
    for key, arg in kwargs.items():
        _, arg = adjust_lon(grid, arg)
        lat_name = dset.get_dim_name(arg, "lat")
        lon_name = dset.get_dim_name(arg, "lon")
        kwargs[key] = arg.interp(
            {lat_name: lat, lon_name: lon},
            method="nearest",
        )
        kwargs[key] = _drop_bounds(kwargs[key], lat_name)
        kwargs[key] = _drop_bounds(kwargs[key], lon_name)

    # Conditional returns based on what was passed in
    if args and not kwargs:
        return args
    if kwargs and not args:
        return kwargs
    return args, kwargs


def adjust_lon(dsa: xr.Dataset, dsb: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """When comparing dsb to dsa, we need their longitudes uniformly in
    [-180,180) or [0,360)."""

    def is_0_to_360(lon0, lonf) -> bool:
        err180 = np.abs(lon0 + 180) + np.abs(lonf - 180)
        err360 = np.abs(lon0) + np.abs(lonf - 360)
        if err360 < err180:
            return True
        return False

    alon_name = dset.get_coord_name(dsa, "lon")
    blon_name = dset.get_coord_name(dsb, "lon")
    if alon_name is None or blon_name is None:
        return dsa, dsb
    a360 = is_0_to_360(dsa[alon_name].min(), dsa[alon_name].max())
    b360 = is_0_to_360(dsb[blon_name].min(), dsb[blon_name].max())
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


def handle_timescale_mismatch(
    ref: xr.Dataset, com: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    # Initially, we are only going to handle the case where our reference data
    # is a single time entry representing a span significantly larger than a
    # month.
    dt_ref = dset.get_mean_time_frequency(ref)
    dt_com = dset.get_mean_time_frequency(com)
    if np.allclose(dt_ref, dt_com, atol=3):
        return ref, com
    if len(ref[dset.get_dim_name(ref, "time")]) == 1:
        t0, tf = dset.get_time_extent(ref, include_bounds=True)
        com = com.sel(
            {
                dset.get_dim_name(com, "time"): slice(
                    f"{t0.dt.year:04d}-{t0.dt.month:02d}",
                    f"{tf.dt.year:04d}-{tf.dt.month:02d}",
                )
            }
        )
        return ref, com
    if np.allclose(dt_ref, 30, atol=3) and dt_com < 28:
        com = dset.compute_monthly_mean(com)
        return ref, com
    if np.allclose(dt_ref, 365, atol=6) and dt_com < 350:
        com = dset.compute_annual_mean(com)
        return ref, com
    raise NotImplementedError(
        f"We encountered a time scale mismatch ({dt_ref=:.1f} vs. {dt_com=:.1f}) that we have no logic to handle. {ref} {com}"
    )


def make_comparable(
    ref: xr.Dataset, com: xr.Dataset, varname: str, **kwargs: Any
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return the datasets in a form where they are comparable."""

    # sometimes our data represents a single time step over a long span
    if dset.is_temporal(ref):
        ref, com = handle_timescale_mismatch(ref, com)

    # trim away time, assuming data is monthly
    if dset.is_temporal(ref):
        ref, com = trim_time(ref, com)

    # latlon needs to be 1D arrays
    if dset.is_latlon2d(com[varname]):
        com = dset.latlon2d_to_1d(ref, com[varname]).to_dataset(name=varname)

    # ensure longitudes are uniform
    ref, com = adjust_lon(ref, com)

    # ensure the lat/lon dims are sorted
    if dset.is_spatial(ref):
        ref = ref.sortby([dset.get_dim_name(ref, "lat"), dset.get_dim_name(ref, "lon")])
    if dset.is_spatial(com):
        com = com.sortby([dset.get_dim_name(com, "lat"), dset.get_dim_name(com, "lon")])

    # pick the sites
    site_extraction_opts = ilamb3.conf["site_extraction"].copy()
    for key, value in kwargs.get("site_extraction", {}).items():
        site_extraction_opts[key] = value
    extraction_fcn = SITE_EXTRACT[site_extraction_opts["method"]]
    if dset.is_site(com[varname]):
        if dset.is_site(ref[varname]):
            # If the comparison dataset is sites, we assume that users ran their
            # models at distinct locations and want to see the nearest
            # reference. So if the reference is also sites, we need to match the
            # reference to the comparison.
            ref = extraction_fcn(ref, com, **site_extraction_opts)
        else:
            # Just extract sites from the reference like usual.
            com = extraction_fcn(com, ref, **site_extraction_opts)
    elif dset.is_site(ref[varname]):
        # Just extract sites from the reference like usual.
        com = extraction_fcn(com, ref, **site_extraction_opts)

    # convert units
    com = dset.convert(com, ref[varname].attrs["units"], varname=varname)

    # load into memory
    ref.load()
    com.load()
    return ref, com


def rename_dims(*args):
    """
    Rename the dimension to a uniform canonical name.

    Parameters
    ----------
    *args
        Any number of `xr.Dataset` or `xr.DataArray` objects for which we will change
        the dimension names.

    Returns
    -------
    *args
        The input *args with the dimension names changed.
    """

    def _populate_renames(ds):
        out = {}
        for dim in ["time", "lat", "lon"]:
            try:
                out[dset.get_dim_name(ds, dim)] = dim
            except KeyError:
                pass
        return out

    for arg in args:
        assert isinstance(arg, xr.DataArray | xr.Dataset)
    return [arg.rename(_populate_renames(arg)) for arg in args]


def convert_calendar_monthly_noleap(
    ds: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """
    Convert the dataset calendar to a monthly noleap.

    Note
    ----
    At some point, we need the index of time to be the same to make comparisons.
    When this is required, we will convert everything assuming that we are
    dealing with monthly data.
    """
    time_name = dset.get_dim_name(ds, "time")
    ds[time_name] = [
        cf.DatetimeNoLeap(t.dt.year, t.dt.month, 15) for t in ds[time_name]
    ]
    if "bounds" not in ds[time_name].attrs:
        return ds
    # Handle time bounds if present
    bounds_name = ds[time_name].attrs["bounds"]
    ds[bounds_name] = np.array(
        [
            [cf.DatetimeNoLeap(t.dt.year, t.dt.month, 1) for t in ds[time_name]],
            [
                cf.DatetimeNoLeap(
                    t.dt.year if t.dt.month < 12 else (t.dt.year + 1),
                    (t.dt.month + 1) if t.dt.month < 12 else 1,
                    1,
                )
                for t in ds[time_name]
            ],
        ]
    ).T
    return ds
