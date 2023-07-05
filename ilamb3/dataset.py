"""Dataset functions for ILAMB"""

from typing import Literal, Union

import numpy as np
import xarray as xr

from ilamb3.regions import Regions


def get_dim_name(
    dset: Union[xr.Dataset, xr.DataArray], dim: str = Literal["time", "lat", "lon"]
) -> str:
    """Return the name of the `dim` dimension from the dataset.

    Parameters
    ----------
    dset
        The input dataset/dataarray.
    dim
        The dimension to find in the dataset/dataarray

    Note
    ----
    This function is meant to handle the problem that not all data calls the dimensions
    the same things ('lat', 'Lat', 'latitude', etc). We could replace this with
    cf-xarray functionality. My concern is that we want this to work even if the
    datasets are not CF-compliant (e.g. raw CESM model output).

    """
    dim_names = {
        "time": ["time"],
        "lat": ["lat", "latitude", "Latitude", "y"],
        "lon": ["lon", "longitude", "Longitude", "x"],
    }
    possible_names = dim_names[dim]
    dim_name = set(dset.dims).intersection(possible_names)
    if len(dim_name) != 1:
        msg = f"{dim} dimension not found: [{','.join(dset.dims)}] "
        msg += f"not in [{','.join(possible_names)}]"
        raise KeyError(msg)
    return dim_name.pop()


def get_time_extent(
    dset: Union[xr.Dataset, xr.DataArray]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the time extent of the dataset/dataarray.

    The function will prefer the values in the 'bounds' array if present.

    Returns
    -------
    tmin
        The minimum time.
    tmax
        The maxmimum time.

    """
    time_name = get_dim_name(dset, "time")
    time = dset[time_name]
    if "bounds" in time.attrs:
        if time.attrs["bounds"] in dset:
            time = dset[time.attrs["bounds"]]
    return time.min(), time.max()


def compute_time_measures(dset: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
    """Return the length of each time interval.

    Note
    ----
    In order to integrate in time, we need the time measures. While this function is
    written for greatest flexibility, the most accurate time measures will be computed
    when a dataset is passed in where the 'bounds' on the 'time' dimension are labeled
    and part of the dataset.

    """

    def _measure1d(time):
        if time.size == 1:
            msg = "Cannot estimate time measures from single value without bounds"
            raise ValueError(msg)
        delt = time.diff(dim="time").to_numpy().astype(float) * 1e-9 / 3600 / 24
        delt = np.hstack([delt[0], delt, delt[-1]])
        msr = xr.DataArray(0.5 * (delt[:-1] + delt[1:]), coords=[time], dims=["time"])
        msr = msr.pint.quantify("d")
        return msr

    time_name = get_dim_name(dset, "time")
    time = dset[time_name]
    timeb_name = time.attrs["bounds"] if "bounds" in time.attrs else None
    if timeb_name is None or timeb_name not in dset:
        return _measure1d(time)
    # compute from the bounds
    delt = dset[timeb_name]
    nbnd = delt.dims[-1]
    delt = delt.diff(nbnd).squeeze()
    delt *= 1e-9 / 86400  # [ns] to [d]
    measure = delt.astype("float")
    measure = measure.pint.quantify("d")
    return measure


def compute_cell_measures(dset: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
    """Return the area of each cell.

    Note
    ----
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
        dely = earth_radius * dely.diff(other_dims).squeeze()
        msr = dely * delx
        msr.attrs["units"] = "m2"
        msr = msr.pint.quantify()
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
    msr = msr.pint.quantify()
    return msr


def coarsen_dataset(dset: xr.Dataset, res: float = 0.5) -> xr.Dataset:
    """Return the mass-conversing spatially coarsened dataset.

    Coarsens the source dataset to the target resolution while conserving the
    overall integral and apply masks where all values are nan.

    Parameters
    ----------
    dset
        The input dataset.
    res
        The target resolution in degrees.

    """
    lat_name = get_dim_name(dset, "lat")
    lon_name = get_dim_name(dset, "lon")
    fine_per_coarse = int(
        round(res / np.abs(dset[lat_name].diff(lat_name).mean().values))
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
            .sum()
            .astype(int)
        )
    dset_coarse = (
        (dset.drop("cell_measures") * dset["cell_measures"])
        .coarsen({"lat": fine_per_coarse, "lon": fine_per_coarse}, boundary="pad")
        .sum()
    )
    cell_measures = compute_cell_measures(dset_coarse)
    dset_coarse = dset_coarse / cell_measures
    dset_coarse["cell_measures"] = cell_measures
    dset_coarse = xr.where(nll == 0, np.nan, dset_coarse)
    return dset_coarse


def integrate_time(
    dset: Union[xr.Dataset, xr.DataArray], varname: str = None, mean: bool = False
) -> xr.DataArray:
    """Return the time integral or mean of the dataset.

    Parameters
    ----------
    dset
        The input dataset/dataarray.
    varname
        The variable to integrate, must be given if a dataset is passed in.
    mean
        Enable to divide the integral by the integral of the measures, returning the
        mean in a functional sense.

    Returns
    -------
    integral
        The integral or mean.

    Note
    ----
    This interface is useful in our analysis as many times we want to report the total
    of a quantity (total mass of carbon) and other times we want the mean value (e.g.
    temperature). This allows the analysis code to read the same where a flag can be
    passed to change the behavior.

    We could consider replacing with xarray.integrate. However, as of `v2023.6.0`, this
    does not handle the `pint` units correctly, and can only be done in a single
    dimension at a time, leaving the spatial analog to be hand coded. It also uses
    trapezoidal rule which should return the same integration, but could have small
    differences depending on how endpoints are interpretted.

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
    var = var.pint.quantify()
    if mean:
        return var.weighted(msr).mean(dim=time_name)
    return var.weighted(msr).sum(dim=time_name)


def std_time(dset: Union[xr.Dataset, xr.DataArray], varname: str = None):
    """Return the standard deviation of a variable in time.

    Parameters
    ----------
    dset
        The input dataset/dataarray.
    varname
        The variable, must be given if a dataset is passed in.

    Returns
    -------
    std
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
    var = var.pint.quantify()
    return var.weighted(msr).std(dim=time_name)


def integrate_space(
    dset: xr.Dataset,
    varname: str = None,
    region: Union[None, str] = None,
    mean: bool = False,
):
    """Return the space integral or mean of the dataset.

    Parameters
    ----------
    dset
        The input dataset/dataarray.
    varname
        The variable to integrate, must be given if a dataset is passed in.
    region
        The region label, one of `ilamb3.Regions.regions` or `None` to indicate that the
        whole spatial domain should be used.
    mean
        Enable to divide the integral by the integral of the measures, returning the
        mean in a functional sense.

    Returns
    -------
    integral
        The integral or mean.

    Note
    ----
    This interface is useful in our analysis as many times we want to report the total
    of a quantity (total mass of carbon) and other times we want the mean value (e.g.
    temperature). This allows the analysis code to read the same where a flag can be
    passed to change the behavior.

    We could consider replacing with xarray.integrate. However, as of `v2023.6.0`, this
    does not handle the `pint` units correctly, and can only be done in a single
    dimension at a time.

    """
    if region is not None:
        regions = Regions()
        dset = regions.restrict_to_region(dset, region)
    space = [get_dim_name(dset, "lat"), get_dim_name(dset, "lon")]
    if isinstance(dset, xr.Dataset):
        var = dset[varname]
        msr = (
            dset["cell_measures"]
            if "cell_measures" in dset
            else compute_cell_measures(dset)
        )
    else:
        var = dset
        msr = compute_cell_measures(dset)
    var = var.pint.quantify()
    # As of v2023.6.0, weighted sums drop units from pint if the weights are
    # over *all* the dimensions of the dataarray. Will do some pint gymnastics
    # to avoid the issue.
    var = var.pint.dequantify()
    msr = msr.pint.dequantify()
    out = var.weighted(msr)
    if mean:
        out = out.mean(dim=space)
        out.attrs["units"] = var.attrs["units"]
    else:
        out = out.sum(dim=space)
        out.attrs["units"] = f"({var.attrs['units']})*({msr.attrs['units']})"
    return out.pint.quantify()
