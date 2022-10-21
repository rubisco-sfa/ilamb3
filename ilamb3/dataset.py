"""Dataset functions for ILAMB"""

import numpy as np
import xarray as xr


def get_time_name(dset: xr.Dataset) -> str:
    """times"""
    possible_names = ["time"]
    time_name = set(dset.dims).intersection(possible_names)
    if not time_name:
        # pylint: disable=consider-using-f-string
        raise KeyError(
            "Time dimension not found: dataset dims [%s] not in [%s]"
            % (",".join(dset.dims), ",".join(possible_names))
        )
    if len(time_name) > 1:
        raise ValueError(f"Unsure which dimension {time_name} is time")
    return time_name.pop()


def get_latitude_name(dset: xr.Dataset) -> str:
    """latitudes"""
    possible_names = ["lat", "latitude", "Latitude", "y"]
    lat_name = set(dset.dims).intersection(possible_names)
    if not lat_name:
        # pylint: disable=consider-using-f-string
        raise KeyError(
            "Latitude dimension not found: dataset dims [%s] not in [%s]"
            % (",".join(dset.dims), ",".join(possible_names))
        )
    if len(lat_name) > 1:
        raise ValueError(f"Unsure which dimension {lat_name} is latitude")
    return lat_name.pop()


def get_longitude_name(dset: xr.Dataset) -> str:
    """longitudes"""
    possible_names = ["lon", "longitude", "Longitude", "x"]
    lon_name = set(dset.dims).intersection(possible_names)
    if not lon_name:
        # pylint: disable=consider-using-f-string
        raise KeyError(
            "Longitude dimension not found: dataset dims [%s] not in [%s]"
            % (",".join(dset.dims), ",".join(possible_names))
        )
    if len(lon_name) > 1:
        raise ValueError(f"Unsure which dimension {lon_name} is longitude")
    return lon_name.pop()


def time_extent(dset: xr.Dataset):
    """Find the beginning and ending of this dataset, preferring the
    time bounds if present."""
    if "time" not in dset.dims:
        raise KeyError("The 'time' dimension is not part of this dataset")
    time = dset["time"]
    if "bounds" in time.attrs:
        if time.attrs["bounds"] in dset:
            time = dset[time.attrs["bounds"]]
    return time.min(), time.max()


def compute_time_measures(dset: xr.Dataset) -> xr.DataArray:
    """In order to integrate in time, we need the time measures."""
    time = dset["time"]
    timeb_name = time.attrs["bounds"] if "bounds" in time.attrs else None
    # we prefer to compute your time lengths from the time bounds if they are part of the dataset
    if timeb_name is not None and timeb_name in dset:
        delt = dset[timeb_name]
        nbnd = delt.dims[-1]
        delt = delt.diff(nbnd).squeeze()
        delt *= 1e-9 / 86400  # [ns] to [d]
        measure = delt.astype("float")
    else:
        # FIX
        if time.size == 1:
            raise ValueError(
                "Cannot estimate time measures from single value times with no time bounds"
            )
        delt = float(time.diff(dim="time").mean()) * 1e-9 / 3600 / 24
        measure = dset["time"].copy(data=[delt] * dset["time"].size)
    return measure


def compute_cell_measures(dset: xr.Dataset) -> xr.DataArray:
    """In order to integrate (area weighted sums), we need the cell measures."""
    earth_radius = 6.371e6  # [m]
    lat_name = get_latitude_name(dset)
    lon_name = get_longitude_name(dset)
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
        return dely * delx
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
    return dely * delx


def coarsen_dataset(dset: xr.Dataset, res: float = 0.5) -> xr.Dataset:
    """Coarsens the source dataset to the target resolution while conserving the overall integral"""
    lat_name = get_latitude_name(dset)
    lon_name = get_longitude_name(dset)
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
        # pylint: disable=singleton-comparison
        (dset.isnull() == False)
        .all(dim=set(dset.dims) - set([lat_name, lon_name]))
        .coarsen({"lat": fine_per_coarse, "lon": fine_per_coarse}, boundary="pad")
    ).sum()
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
