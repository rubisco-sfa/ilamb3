"""Functions for preparing datasets for comparison."""

import cftime as cf
import numpy as np
import xarray as xr

from ilamb3 import dataset as dset
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

    # Get the time extents in the original calendars
    t0 = []
    tf = []
    for arg in args:
        tbegin, tend = dset.get_time_extent(arg, include_bounds=False)
        t0.append(tbegin)
        tf.append(tend)
    for _, arg in kwargs.items():
        tbegin, tend = dset.get_time_extent(arg, include_bounds=False)
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
        args[i] = arg.sel({"time": tslice})
    for key, arg in kwargs.items():
        kwargs[key] = arg.sel({"time": tslice})

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
    alon_name = dset.get_coord_name(dsa, "lon")
    blon_name = dset.get_coord_name(dsb, "lon")
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
    try:
        ref, com = trim_time(ref, com)
    except KeyError:
        pass  # no time dimension

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

    # pick just the sites
    if dset.is_site(ref[varname]):
        com = extract_sites(ref, com, varname)

    # convert units
    com = dset.convert(com, ref[varname].attrs["units"], varname=varname)

    # load into memory
    ref.load()
    com.load()
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
    lat_site = dset.get_coord_name(ds_site, "lat")
    lon_site = dset.get_coord_name(ds_site, "lon")
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

    # Set these are coordinates though so that we can tell this is site data. We
    # also set them equal to the sites lat/lon so when comparing to other data
    # that the coords stay on
    ds_spatial = ds_spatial.set_coords([lat_spatial, lon_spatial])
    ds_spatial[lat_spatial] = ds_site[lat_site]
    ds_spatial[lon_spatial] = ds_site[lon_site]

    return ds_spatial


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
