"""Convenience functions which operate on datasets."""

from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator

import ilamb3.regions as ilreg
from ilamb3.exceptions import NoSiteDimension, NoUncertainty


def get_dim_name(
    dset: xr.Dataset | xr.DataArray,
    dim: Literal["time", "lat", "lon", "depth", "site"],
) -> str:
    """
    Return the name of the `dim` dimension from the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    dim : str, one of {`time`, `lat`, `lon`, `depth`, `site`}
        The dimension to find in the dataset/dataarray.

    Returns
    -------
    str
        The name of the dimension.

    See Also
    --------
    get_coord_name : A variant when the coordinate is not a dimension.

    Notes
    -----
    This function is meant to handle the problem that not all data calls the dimensions
    the same things ('lat', 'Lat', 'latitude', etc). We could replace this with
    cf-xarray functionality. My concern is that we want this to work even if the
    datasets are not CF-compliant (e.g. raw model output).
    """
    dim_names = {
        "time": ["time", "TIME", "month"],
        "lat": ["lat", "latitude", "Latitude", "y", "lat_", "Lat", "LATITUDE"],
        "lon": ["lon", "longitude", "Longitude", "x", "lon_", "Lon", "LONGITUDE"],
        "depth": ["depth", "lev"],
    }
    # Assumption: the 'site' dimension is what is left over after all others are removed
    if dim == "site":
        try:
            get_dim_name(dset, "lat")
            get_dim_name(dset, "lon")
            raise NoSiteDimension("Dataset/dataarray is spatial")
        except KeyError:
            pass
        possible_names = list(
            set(dset.dims) - set([d for _, dims in dim_names.items() for d in dims])
        )
        if len(possible_names) == 1:
            return possible_names[0]
        msg = f"Ambiguity in locating a site dimension, found: {possible_names}"
        raise NoSiteDimension(msg)
    possible_names = dim_names[dim]
    dim_name = set(dset.dims).intersection(possible_names)
    if len(dim_name) != 1:
        msg = f"{dim} dimension not found: {dset.dims} "
        msg += f"not in [{','.join(possible_names)}]"
        raise KeyError(msg)
    return str(dim_name.pop())


def get_coord_name(
    dset: xr.Dataset | xr.DataArray,
    coord: Literal["lat", "lon"],
) -> str:
    """
    Return the name of the `coord` coordinate from the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    coord : str, one of {`lat`, `lon`}
        The coordinate to find in the dataset/dataarray.

    Returns
    -------
    str
        The name of the coordinate.

    See Also
    --------
    get_dim_name : A variant when the coordinate is a dimension.
    """
    coord_names = {
        "lat": ["lat", "latitude", "Latitude", "y", "lat_", "Lat", "LATITUDE"],
        "lon": ["lon", "longitude", "Longitude", "x", "lon_", "Lon", "LONGITUDE"],
    }
    possible_names = coord_names[coord]
    coord_name = set(dset.coords).intersection(possible_names)
    if len(coord_name) != 1:
        msg = f"{coord} coordinate not found: {dset.coords} "
        msg += f"not in [{','.join(possible_names)}]"
        raise KeyError(msg)
    return str(coord_name.pop())


def is_temporal(da: xr.DataArray) -> bool:
    """
    Return if the dataarray is temporal.

    Parameters
    ----------
    da : xr.DataArray
        The input dataarray.

    Returns
    -------
    bool
        True if time dimension is present, False otherwise.
    """
    try:
        get_dim_name(da, "time")
        return True
    except KeyError:
        pass
    return False


def is_spatial(da: xr.DataArray) -> bool:
    """
    Return if the dataarray is spatial.

    Parameters
    ----------
    da : xr.DataArray
        The input dataarray.

    Returns
    -------
    bool
        True if latitude and longitude dimensions are present, False otherwise.
    """
    try:
        get_dim_name(da, "lat")
        get_dim_name(da, "lon")
        return True
    except KeyError:
        pass
    # Ocean grids have 2D lat/lons sometimes stored in their coordinates
    if is_latlon2d(da):
        return True
    return False


def is_site(da: xr.DataArray) -> bool:
    """
    Return if the dataarray is a collection of sites.

    Parameters
    ----------
    da : xr.DataArray
        The input dataarray.

    Returns
    -------
    bool
        True if sites, False otherwise.
    """
    try:
        dim_lat = get_dim_name(da, "lat")
        dim_lon = get_dim_name(da, "lon")
    except KeyError:
        dim_lat = dim_lon = None
    try:
        coord_lat = get_coord_name(da, "lat")
        coord_lon = get_coord_name(da, "lon")
    except KeyError:
        return False
    if (
        dim_lat is None
        and coord_lat is not None
        and dim_lon is None
        and coord_lon is not None
    ):
        return True
    return False


def is_layered(da: xr.DataArray) -> bool:
    """
    Return if the dataarray is layered.

    Parameters
    ----------
    da : xr.DataArray
        The input dataarray.

    Returns
    -------
    bool
        True if a depth dimension is present, False otherwise.
    """
    try:
        get_dim_name(da, "depth")
        return True
    except KeyError:
        pass
    return False


def get_integer_dims(da: xr.DataArray) -> list[str]:
    """
    Return which dimensions are integers and therefore likely indices.
    """
    return [d for d in da.dims if d in da.coords and da[d].dtype.kind == "i"]


def is_latlon2d(da: xr.DataArray) -> bool:
    """
    Return if the dataarray has 2D latitudes and longitudes.
    """
    try:
        lat_coord = get_coord_name(da, "lat")
        lon_coord = get_coord_name(da, "lon")
    except KeyError:
        return False
    index_dims = get_integer_dims(da)
    if not index_dims:
        return False
    # Are the index dimensions of da also the dimensions of the the coordinates?
    if set(da[lon_coord].dims) == set(da[lat_coord].dims) == set(index_dims):
        return True
    return False


def latlon2d_to_1d(grid: xr.Dataset | xr.DataArray, da: xr.DataArray) -> xr.DataArray:
    """
    Return the da with 2D lat/lon interpolated at the grid resolution.
    """
    if not is_latlon2d(da):
        raise ValueError("Input dataarray is not a 2D lat/lon.")
    lat_dim = get_dim_name(grid, "lat")
    lon_dim = get_dim_name(grid, "lon")
    lat_coord = get_coord_name(da, "lat")
    lon_coord = get_coord_name(da, "lon")
    index_dims = get_integer_dims(da)
    da[lon_coord] = xr.where(da[lon_coord] > 180, da[lon_coord] - 360, da[lon_coord])
    # Create an interpolator that maps lat,lon --> i,j
    index_map = NearestNDInterpolator(
        np.array([da[lat_coord].values.flatten(), da[lon_coord].values.flatten()]).T,
        np.array(
            [
                (da[index_dims[0]] * xr.ones_like(da[index_dims[1]])).values.flatten(),
                (xr.ones_like(da[index_dims[0]]) * da[index_dims[1]]).values.flatten(),
            ]
        ).T,
    )
    # Then use the interpolator to find index arrays at the input regular grid.
    ids = index_map(
        (grid[lat_dim] * xr.ones_like(grid[lon_dim])).values.flatten(),
        (xr.ones_like(grid[lat_dim]) * grid[lon_dim]).values.flatten(),
    ).astype(int)
    # Create a new dataarray
    not_index_dims = [d for d in da.dims if d not in index_dims]
    coords = {c: da[c] for c in not_index_dims if c in da.coords}
    coords.update({"lat": grid["lat"], "lon": grid["lon"]})
    shp = (grid[lat_dim].size, grid[lon_dim].size)
    out = xr.DataArray(
        data=da.to_numpy()[..., ids[:, 0].reshape(shp), ids[:, 1].reshape(shp)],
        coords=coords,
        dims=not_index_dims + [lat_dim, lon_dim],
        attrs=da.attrs,
    )
    return out


def get_time_extent(
    dset: xr.Dataset | xr.DataArray, include_bounds: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Return the time extent of the dataset/dataarray.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset.

    Returns
    -------
    tuple of xr.DataArray
        The minimum and maximum time.

    Notes
    -----
    The function will prefer the values in the 'bounds' array if present.
    """
    time_name = get_dim_name(dset, "time")
    time = dset[time_name]
    if "bounds" in time.attrs and include_bounds:
        if time.attrs["bounds"] in dset:
            time = dset[time.attrs["bounds"]]
    time.load()
    return time.min(), time.max()


def compute_time_measures(dset: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Return the length of each time interval.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset.

    Returns
    -------
    xr.DataArray
        The time measures, the length of the time intervals.

    Notes
    -----
    In order to integrate in time, we need the time measures. While this function is
    written for greatest flexibility, the most accurate time measures will be computed
    when a dataset is passed in where the 'bounds' on the 'time' dimension are labeled
    and part of the dataset.
    """

    def _measure1d(time, time_name):  # numpydoc ignore=GL08
        if time.size == 1:
            msg = "Cannot estimate time measures from single value without bounds"
            raise ValueError(msg)
        delt = time.diff(dim=time_name).to_numpy().astype(float) * 1e-9 / 3600 / 24
        delt = np.hstack([delt[0], delt, delt[-1]])
        msr = xr.DataArray(
            0.5 * (delt[:-1] + delt[1:]), coords=[time], dims=[time_name]
        )
        msr.attrs["units"] = "d"
        return msr

    time_name = get_dim_name(dset, "time")
    time = dset[time_name]
    timeb_name = time.attrs["bounds"] if "bounds" in time.attrs else None
    if timeb_name is None or timeb_name not in dset:
        return _measure1d(time, time_name)
    # compute from the bounds
    delt = dset[timeb_name]
    nbnd = delt.dims[-1]
    delt = delt.diff(nbnd).squeeze().compute()
    measure = delt.astype("float") * 1e-9 / 86400  # [ns] to [d]
    measure.attrs["units"] = "d"
    return measure


def compute_cell_measures(dset: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Return the area of each spatial cell.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset.

    Returns
    -------
    xr.DataArray
        The cell areas.

    Notes
    -----
    It would be better to get these from the model data itself, but they are not always
    provided, particularly in reference data.
    """
    earth_radius = 6.371e6  # [m]
    lat_name = get_dim_name(dset, "lat")
    lon_name = get_dim_name(dset, "lon")
    lat = dset[lat_name]
    lon = dset[lon_name]
    latb_name = lat.attrs["bounds"] if "bounds" in lat.attrs else None
    lonb_name = lon.attrs["bounds"] if "bounds" in lon.attrs else None
    # we prefer to compute your cell areas from the lat/lon bounds if they are
    # part of the dataset...
    if (
        latb_name is not None
        and latb_name in dset
        and lonb_name is not None
        and lonb_name in dset
    ):
        delx = dset[lonb_name] * np.pi / 180
        dely = np.sin(dset[latb_name] * np.pi / 180)
        other_dims = delx.dims[-1]
        delx = earth_radius * delx.diff(other_dims).squeeze()
        dely = earth_radius * dely.diff(other_dims).squeeze()  # type: ignore
        msr = dely * delx
        msr.attrs["units"] = "m2"
        return msr
    # ...and if they aren't, we assume the lat/lon we have is a cell centroid
    # and compute the area.
    lon = lon.values
    lat = lat.values
    delx = 0.5 * (lon[:-1] + lon[1:])
    dely = 0.5 * (lat[:-1] + lat[1:])
    delx = np.vstack(
        [
            np.hstack([lon[0] - 0.5 * (lon[1] - lon[0]), delx]),
            np.hstack([delx, lon[-1] + 0.5 * (lon[-1] - lon[-2])]),
        ]
    ).T
    dely = np.vstack(
        [
            np.hstack([lat[0] - 0.5 * (lat[1] - lat[0]), dely]),
            np.hstack([dely, lat[-1] + 0.5 * (lat[-1] - lat[-2])]),
        ]
    ).T
    delx = delx * np.pi / 180
    dely = np.sin(dely * np.pi / 180)
    delx = earth_radius * np.diff(delx, axis=1).squeeze()
    dely = earth_radius * np.diff(dely, axis=1).squeeze()
    delx = xr.DataArray(
        data=np.abs(delx), dims=[lon_name], coords={lon_name: dset[lon_name]}
    )
    dely = xr.DataArray(
        data=np.abs(dely), dims=[lat_name], coords={lat_name: dset[lat_name]}
    )
    msr = dely * delx
    msr.attrs["units"] = "m2"
    return msr


def coarsen_dataset(dset: xr.Dataset, res: float = 0.5) -> xr.Dataset:
    """
    Return the mass-conversing spatially coarsened dataset.

    Parameters
    ----------
    dset : xr.Dataset
        The input dataset.
    res : float, optional
        The target resolution in degrees.

    Returns
    -------
    xr.Dataset
        The coarsened dataset.

    Notes
    -----
    Coarsens the source dataset to the target resolution while conserving the
    overall integral and apply masks where all values are nan.
    """
    lat_name = get_dim_name(dset, "lat")
    lon_name = get_dim_name(dset, "lon")
    fine_per_coarse = int(
        round(res / np.abs(dset[lat_name].diff(lat_name).mean().values))  # type: ignore
    )
    # To spatially coarsen this dataset we will use the xarray 'coarsen'
    # functionality. However, if we want the area weighted sums to be the same,
    # we need to integrate over the coarse cells and then divide through by the
    # new areas. We also need to keep track of nan's to apply a mask to the
    # coarsened dataset.
    if "cell_measures" not in dset:
        dset["cell_measures"] = compute_cell_measures(dset)
    nll = (
        dset.notnull()
        .any(dim=[d for d in dset.dims if d not in [lat_name, lon_name]])
        .coarsen({"lat": fine_per_coarse, "lon": fine_per_coarse}, boundary="pad")
        .sum()  # type: ignore
        .astype(int)
    )
    dset_coarse = (
        (dset.drop_vars("cell_measures") * dset["cell_measures"])
        .coarsen({"lat": fine_per_coarse, "lon": fine_per_coarse}, boundary="pad")
        .sum()  # type: ignore
    )
    cell_measures = compute_cell_measures(dset_coarse)
    dset_coarse = dset_coarse / cell_measures
    dset_coarse["cell_measures"] = cell_measures
    dset_coarse = xr.where(nll == 0, np.nan, dset_coarse)
    return dset_coarse


def integrate_time(
    dset: xr.Dataset | xr.DataArray,
    varname: str | None = None,
    mean: bool = False,
) -> xr.DataArray:
    """
    Return the time integral or mean of the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    varname : str, optional
        The variable to integrate, must be given if a dataset is passed in.
    mean : bool, optional
        Enable to divide the integral by the integral of the measures, returning the
        mean in a functional sense.

    Returns
    -------
    integral
        The integral or mean.

    Notes
    -----
    This interface is useful in our analysis as many times we want to report the total
    of a quantity (total mass of carbon) and other times we want the mean value (e.g.
    temperature). This allows the analysis code to read the same where a flag can be
    passed to change the behavior. We could consider replacing with xarray.integrate.
    However, as of `v2023.6.0`, this does not handle the `pint` units correctly, and can
    only be done in a single dimension at a time, leaving the spatial analog to be hand
    coded. It also uses trapezoidal rule which should return the same integration, but
    could have small differences depending on how endpoints are interpretted.
    """
    time_name = get_dim_name(dset, "time")
    if isinstance(dset, xr.Dataset):
        assert varname is not None
        var = dset[varname]
        msr = (
            dset["time_measures"]
            if "time_measures" in dset
            else compute_time_measures(dset)
        )
    else:
        var = dset
        msr = compute_time_measures(dset)
    if mean:
        return var.weighted(msr.fillna(0)).mean(dim=time_name)
    var = var.pint.quantify()
    msr = msr.pint.quantify()
    out = var.weighted(msr.fillna(0)).sum(dim=time_name)
    out = out.pint.dequantify()
    return out


def accumulate_time(
    dset: xr.Dataset | xr.DataArray,
    varname: str | None = None,
) -> xr.DataArray:
    """
    Return the time accumulation of the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    varname : str, optional
        The variable to integrate, must be given if a dataset is passed in.

    Returns
    -------
    xr.DataArray
        The accumulation.
    """
    time_name = get_dim_name(dset, "time")
    if isinstance(dset, xr.Dataset):
        assert varname is not None
        var = dset[varname]
        msr = (
            dset["time_measures"]
            if "time_measures" in dset
            else compute_time_measures(dset)
        )
    else:
        var = dset
        msr = compute_time_measures(dset)
    var = (var.pint.quantify() * msr.pint.quantify()).pint.dequantify()
    var = var.cumsum(dim=time_name)
    return var


def std_time(
    dset: xr.Dataset | xr.DataArray, varname: str | None = None
) -> xr.DataArray:
    """
    Return the standard deviation of a variable in time.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    varname : str, optional
        The variable, must be given if a dataset is passed in.

    Returns
    -------
    xr.DataArray
        The weighted standard deviation.
    """
    time_name = get_dim_name(dset, "time")
    if isinstance(dset, xr.Dataset):
        var = dset[varname]
        msr = (
            dset["time_measures"]
            if "time_measures" in dset
            else compute_time_measures(dset)
        )
    else:
        var = dset
        msr = compute_time_measures(dset)
    return var.weighted(msr.fillna(0)).std(dim=time_name)


def integrate_space(
    dset: xr.DataArray | xr.Dataset,
    varname: str,
    region: None | str = None,
    mean: bool = False,
    weight: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Return the space integral or mean of the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    varname : str, optional
        The variable to integrate, must be given if a dataset is passed in.
    region : str, optional
        The region label, one of `ilamb3.Regions.regions` or `None` to indicate that the
        whole spatial domain should be used.
    mean : bool, optional
        Enable to divide the integral by the integral of the measures, returning the
        mean in a functional sense.
    weight : xr.DataArray, optional
        Optional weight for the spatial integral. Used when mass weighting.

    Returns
    -------
    xr.Dataset
        The integral or mean.

    Notes
    -----
    This interface is useful in our analysis as many times we want to report the total
    of a quantity (total mass of carbon) and other times we want the mean value (e.g.
    temperature). This allows the analysis code to read the same where a flag can be
    passed to change the behavior. We could consider replacing with xarray.integrate.
    However, as of `v2023.6.0`, this does not handle the `pint` units correctly, and can
    only be done in a single dimension at a time.
    """
    dset = dset.pint.dequantify()
    if region is not None:
        regions = ilreg.Regions()
        dset = regions.restrict_to_region(dset, region)
    space = [get_dim_name(dset, "lat"), get_dim_name(dset, "lon")]
    if not isinstance(dset, xr.Dataset):
        dset = dset.to_dataset()
    var = dset[varname]
    msr = (
        dset["cell_measures"]
        if "cell_measures" in dset
        else compute_cell_measures(dset)
    )
    if weight is not None:
        assert isinstance(weight, xr.DataArray)
        msr = (msr.pint.quantify() * weight.pint.quantify()).pint.dequantify()
    out = var.weighted(msr.fillna(0))
    if mean:
        out = out.mean(dim=space)
        out.attrs["units"] = var.attrs["units"]
    else:
        out = out.sum(dim=space)
        out.attrs["units"] = f"({var.attrs['units']})*({msr.attrs['units']})"
    return out


def sel(dset: xr.Dataset, coord: str, cmin: Any, cmax: Any) -> xr.Dataset:
    """
    Return a selection of the dataset.

    Parameters
    ----------
    dset : xr.Dataset
        The input dataset.
    coord : str
        The coordinate to slice.
    cmin :  Any
        The minimum coordinate value.
    cmax : Any
        The maximum coordinate value.

    Returns
    -------
    xr.Dataset
        The selected dataset.

    Notes
    -----
    The behavior of xarray.sel does not work for us here. We want to pick a slice of the
    dataset but where the value lies in between the coord bounds. Then we clip the min
    and max to be the limits of the slice.
    """

    def _get_interval(dset, dim, value, side):  # numpydoc ignore=GL08
        coord = dset[dim]
        if "bounds" in coord.attrs:
            if coord.attrs["bounds"] in dset:
                coord = dset[coord.attrs["bounds"]]
                ind = ((coord[:, 0] <= value) & (coord[:, 1] >= value)).to_numpy()
                ind = np.where(ind)[0]
                assert len(ind) <= 2
                assert len(ind) > 0
                if len(ind) == 2:
                    if side == "low":
                        return ind[1]
                return ind[0]
        raise NotImplementedError(f"Must have a bounds {coord}")

    dset = dset.isel(
        {
            coord: slice(
                _get_interval(dset, coord, cmin, "low"),
                _get_interval(dset, coord, cmax, "high") + 1,
            )
        }
    )
    # adjust the bounds and coord values
    bnds = dset[coord].attrs["bounds"]
    dset[bnds][0, 0] = cmin
    dset[bnds][-1, 1] = cmax
    dim = dset[coord].to_numpy()
    dim[0] = dset[bnds][0, 0].values + 0.5 * (
        dset[bnds][0, 1].values - dset[bnds][0, 0].values
    )
    dim[-1] = dset[bnds][-1, 0].values + 0.5 * (
        dset[bnds][-1, 1].values - dset[bnds][-1, 0].values
    )
    attrs = dset[coord].attrs
    dset[coord] = dim
    dset[coord].attrs = attrs
    return dset


def integrate_depth(
    dset: xr.Dataset | xr.DataArray,
    varname: str | None = None,
    mean: bool = False,
) -> xr.DataArray:
    """
    Return the depth integral or mean of the dataset.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The dataset of dataarray to integrate.
    varname : str, optional
        If dset is a dataset, the variable name to integrate.
    mean : bool, optional
        Enable to take a depth mean.

    Returns
    -------
    xr.DataArray
        The depth integral or sum.
    """
    if isinstance(dset, xr.DataArray):
        varname = dset.name
        dset = dset.to_dataset(name=varname)
    else:
        assert varname is not None
    var = dset[varname]

    # do we have a depth dimension
    if "depth" not in dset.dims:
        raise ValueError("Cannot integrate in depth without a depth dimension.")

    # does depth have bounds?
    if "bounds" not in dset["depth"].attrs or dset["depth"].attrs["bounds"] not in dset:
        dset = dset.cf.add_bounds("depth")

    # compute measures
    msr = dset[dset["depth"].attrs["bounds"]]
    msr = msr.diff(dim=msr.dims[-1])
    msr.attrs["units"] = (
        dset["depth"].attrs["units"] if "units" in dset["depth"].attrs else "m"
    )

    # integrate
    out = var.weighted(msr.fillna(0))
    if mean:
        out = out.mean(dim="depth")
        out.attrs["units"] = var.attrs["units"]
    else:
        out = out.sum(dim="depth")
        out.attrs["units"] = f"({var.attrs['units']})*({msr.attrs['units']})"
    return out


def scale_by_water_density(da: xr.DataArray, target: str) -> xr.DataArray:
    """
    Conditionally scale the dataarray by density if needed in the conversion.

    Parameters
    ----------
    da : xr.DataArray
        The pint quantified input data array.
    target : str
        The target conversion unit as a string.

    Returns
    -------
    xr.DataArray
        The potentially water density scaled data array.

    Notes
    -----
    Most modeled hydrologic quantities tend to be output as a mass flux rate. However, a
    linear speed is often the preferred unit. For example a conversion such as `kg m-2
    s-1` to `mm d-1`. This is possible if scaled by the density of water which we do
    here conditionally if needed. Used by our `convert()` routine.
    """
    da = da.pint.quantify()
    ureg = da.pint.registry
    water_density = 998.2071 * ureg.kilogram / ureg.meter**3
    src = 1.0 * da.pint.units
    tar = ureg(target)
    if (src / tar).check("[mass] / [length]**3"):
        return da / water_density
    if (tar / src).check("[mass] / [length]**3"):
        return da * water_density
    return da


def convert(
    dset: xr.Dataset | xr.DataArray,
    unit: str,
    varname: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Convert the units of the dataarray.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        The dataset (specify varname) or dataarray who units you wish to convert.
    unit : str
        The unit to which we will convert.
    varname : str, optional
        If dset is a dataset, give the variable name to convert.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The converted dataset.
    """
    dset = dset.pint.quantify()
    if isinstance(dset, xr.DataArray):
        da = dset
    else:
        assert varname is not None
        da = dset[varname]
    da = scale_by_water_density(da, unit)
    da = da.pint.to(unit)
    if isinstance(dset, xr.DataArray):
        return da.pint.dequantify()
    dset[varname] = da
    dset = dset.pint.dequantify()
    return dset


def coarsen_annual(dset: xr.Dataset) -> xr.Dataset:
    """
    Return the dataset coarsened to annual.

    Parameters
    ----------
    dset : xr.Dataset
        The input dataset/dataarray.

    Returns
    -------
    xr.Dataset
        The coarsened dataset.
    """
    # Can't sum time objects so find them and remove
    time_name = get_dim_name(dset, "time")
    if "bounds" in dset[time_name].attrs:
        bounds = dset[time_name].attrs["bounds"]
        if bounds in dset:
            dset = dset.drop_vars(bounds)
    msr = compute_time_measures(dset).pint.dequantify()
    ann = (dset * msr).groupby(f"{time_name}.year").sum() / msr.groupby(
        f"{time_name}.year"
    ).sum()
    return ann


def shift_lon(dset: xr.Dataset) -> xr.Dataset:
    """
    Return the dataset with longitudes shifted to [-180,180].

    Parameters
    ----------
    dset : xr.Dataset
        The input dataset.

    Returns
    -------
    xr.Dataset
        The longitude-shifted dataset.
    """
    lon_name = get_coord_name(dset, "lon")
    if (dset[lon_name].min() >= 0) * (dset[lon_name].max() <= 360):
        dset[lon_name] = (dset[lon_name] + 180) % 360 - 180
        if "bounds" in dset[lon_name].attrs and dset[lon_name].attrs["bounds"] in dset:
            dset[dset[lon_name].attrs["bounds"]] = (
                dset[dset[lon_name].attrs["bounds"]] + 180
            ) % 360 - 180
        dset = dset.sortby(lon_name)
    return dset


def get_scalar_uncertainty(ds: xr.Dataset, varname: str) -> xr.DataArray:
    """
    Get a scalar uncertainty from the variable.

    Note
    ----
    Uncertainty is indicated either by the presence of the `ancillary_variables`
    attribute or the `bounds` attribute. In the case of the latter, we will
    return the harmonic mean as a scalar measure of uncertainty.
    """
    var = ds[varname]
    da = None
    if "ancillary_variables" in var.attrs and var.attrs["ancillary_variables"] in ds:
        da = ds[var.attrs["ancillary_variables"]]
    if "bounds" in var.attrs and var.attrs["bounds"] in ds:
        da = ds[var.attrs["bounds"]]
        bnd_dim = set(da.dims) - set(var.dims)
        if len(bnd_dim) > 1:
            raise ValueError(
                f"Ambiguity in determinging the `bounds` dimension, found: {bnd_dim}"
            )
        bnd_dim = list(bnd_dim)[0]
        da = np.sqrt(
            (var - da.isel({bnd_dim: 0})) ** 2 + (da.isel({bnd_dim: 1}) - var) ** 2
        )
    if da is None:
        raise NoUncertainty()
    da.attrs["units"] = var.attrs["units"]
    return da


def fix_missing_bounds_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Add a `bounds` attribute if variable appears to exist.
    """

    def _fix(ds: xr.Dataset, dim_name: str) -> xr.Dataset:
        dim = ds[dim_name]
        if "bounds" in dim.attrs:
            if dim.attrs["bounds"] in ds:
                return ds  # all good, just return
            else:
                ds[dim_name].attrs.pop("bounds")
                return ds
        # is there a likely bounds variable and the attr is missing?
        for possible_bnds in [f"{dim_name}_bnds", f"{dim_name}_bounds"]:
            if possible_bnds in ds:
                ds[dim_name].attrs["bounds"] = possible_bnds
        return ds

    for d in ds.dims:
        ds = _fix(ds, d)
    return ds
