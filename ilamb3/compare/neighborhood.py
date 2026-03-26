"""
Functions for converting a data source into target point data.
"""

import warnings
from typing import Literal

import numpy as np
import xarray as xr

import ilamb3.dataset as dset


def extract_neighbors_by_window(
    ds_source: xr.Dataset,
    ds_target: xr.Dataset,
    window_half_size: float,
    window_shape: Literal["box", "circle"] = "circle",
) -> list[xr.Dataset]:

    # Assumption checks (FIX: convert to raise errors)
    assert dset.is_spatial(ds_source) or dset.is_site(ds_source)
    assert dset.is_site(ds_target)
    assert window_half_size > 0

    # Extract dim/coord names
    lat_name_src = dset.get_coord_name(ds_source, "lat")
    lon_name_src = dset.get_coord_name(ds_source, "lon")
    site_name_tar = dset.get_dim_name(ds_target, "site")
    lat_name_tar = dset.get_coord_name(ds_target, "lat")
    lon_name_tar = dset.get_coord_name(ds_target, "lon")

    # Loop over sites and handle discretely
    ds_neighborhood = []
    for site in range(len(ds_target[site_name_tar])):
        ds_site = ds_target.isel({site_name_tar: site})
        if dset.is_spatial(ds_source):
            # A computational efficiency if spatial to reduce the number of
            # points for which we will compute a distance
            ds_hood = ds_source.sel(
                {
                    lat_name_src: slice(
                        ds_site[lat_name_tar] - window_half_size,
                        ds_site[lat_name_tar] + window_half_size,
                    ),
                    lon_name_src: slice(
                        ds_site[lon_name_tar] - window_half_size,
                        ds_site[lon_name_tar] + window_half_size,
                    ),
                }
            )
        else:
            # Otherwise assume the site data is small and just use the whole
            # dataset (sel does not work on coords that are not dims) but mask
            # outside the window
            condition = (
                np.abs(ds_source[lat_name_src] - ds_site[lat_name_tar])
                < window_half_size
            ) * (
                np.abs(ds_source[lon_name_src] - ds_site[lon_name_tar])
                < window_half_size
            )
            ds_hood = xr.Dataset(
                {
                    var: (
                        xr.where(condition, da, np.nan)
                        if set(condition.dims).issubset(da.dims)
                        else da
                    )
                    for var, da in ds_source.items()
                }
            )

        # Store the distance from the site for later use
        ds_hood["distance"] = np.sqrt(
            (ds_hood[lat_name_src] - ds_site[lat_name_tar]) ** 2
            + (ds_hood[lon_name_src] - ds_site[lon_name_tar]) ** 2
        )
        # Mask outside of the circle if being used
        if window_shape == "circle":
            condition = ds_hood["distance"] <= window_half_size
            for var, da in ds_hood.items():
                if set(condition.dims).issubset(da.dims):
                    ds_hood[var] = xr.where(condition, da, np.nan)
        ds_neighborhood += [ds_hood]
    return ds_neighborhood


def neighborhood_mean(
    ds_neighborhood: list[xr.Dataset], ds_target: xr.Dataset, weighted: bool = False
) -> xr.Dataset:

    def _weights(ds: xr.Dataset, weighted: bool, min_dist: float = 0.1) -> xr.DataArray:
        if weighted:
            return (1 / ds["distance"].clip(min_dist)).fillna(0)
        return xr.where(ds["distance"].isnull(), 0.0, 1.0)

    # Extract dim/coord names
    site_name_tar = dset.get_dim_name(ds_target, "site")
    lat_name_tar = dset.get_coord_name(ds_target, "lat")
    lon_name_tar = dset.get_coord_name(ds_target, "lon")

    # Assumption checks (FIX: change to raise)
    assert dset.is_site(ds_target)
    assert len(ds_neighborhood) == len(ds_target[site_name_tar])

    # Compute the mean and std for all neighborhoods
    ds_hood = next(iter(ds_neighborhood))
    op_dims = (
        [dset.get_coord_name(ds_hood, "lat"), dset.get_coord_name(ds_hood, "lon")]
        if dset.is_spatial(ds_hood)
        else [dset.get_dim_name(ds_hood, "site")]
    )
    op_vars = [v for v in ds_hood if set(op_dims).issubset(ds_hood[v].dims)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_mean = xr.concat(
            [
                ds[op_vars].weighted(_weights(ds, weighted)).mean(dim=op_dims)
                for ds in ds_neighborhood
            ],
            dim=site_name_tar,
        )
        ds_std = xr.concat(
            [
                ds[op_vars].weighted(_weights(ds, weighted)).std(dim=op_dims)
                for ds in ds_neighborhood
            ],
            dim=site_name_tar,
        )

    # Add the std as an ancillary variable when appropriate
    for varname in set(ds_mean) and set(ds_std):
        sdname = f"{varname}_sd"
        ds_mean[sdname] = ds_std[varname]
        anc_vars = ds_mean[varname].attrs.get("ancilliary_variables", "")
        anc_vars = anc_vars.split() + [sdname]
        ds_mean[varname].attrs["ancilliary_variables"] = " ".join(anc_vars)
        ds_mean[sdname].attrs["standard_name"] = f"{varname} standard deviation"

    # Restore the variables not part of the mean
    for v in set(ds_hood) - set(op_vars):
        ds_mean[v] = ds_hood[v]

    # Assign the target's coordinates to the output
    ds_mean = ds_mean.assign_coords(
        {c: ds_target[c] for c in [lat_name_tar, lon_name_tar]}
    )
    ds_mean = ds_mean.drop_vars(["distance", "distance_sd"], errors="ignore")
    return ds_mean


def neighborhood_closest(
    ds_neighborhood: list[xr.Dataset],
    ds_target: xr.Dataset,
) -> xr.Dataset:

    # Extract dim/coord names
    site_name_tar = dset.get_dim_name(ds_target, "site")
    lat_name_tar = dset.get_coord_name(ds_target, "lat")
    lon_name_tar = dset.get_coord_name(ds_target, "lon")

    # Assumption checks
    assert dset.is_site(ds_target)
    assert len(ds_neighborhood) == len(ds_target[site_name_tar])
    ds_hood = next(iter(ds_neighborhood))
    lat_name_hood = dset.get_coord_name(ds_hood, "lat")
    lon_name_hood = dset.get_coord_name(ds_hood, "lon")
    assert "distance" in ds_hood

    # Pass the mask to the distance so we pick the closest non-null value
    for i, ds_hood in enumerate(ds_neighborhood):
        for _, da in ds_hood.items():
            if not set(ds_hood["distance"].dims).issubset(da.dims):
                continue
            dim_other = set(da.dims) - set(ds_hood["distance"].dims)
            mask = da.isnull().all(dim=dim_other) + ds_hood["distance"].isnull()
            ds_neighborhood[i]["distance"] = xr.where(
                ~mask, ds_hood["distance"], np.nan
            )

    # Pick the closest value from each neighborhood
    op_dims = (
        [lat_name_hood, lon_name_hood] if dset.is_spatial(ds_hood) else [site_name_tar]
    )
    ds_close = xr.concat(
        [ds.isel(ds["distance"].argmin(dim=op_dims)) for ds in ds_neighborhood],
        dim=site_name_tar,
        coords="different",
        compat="equals",
    )
    ds_close = ds_close.drop_vars("distance", errors="ignore")

    # Assign the target's coordinates to the output
    ds_close = ds_close.assign_coords(
        {c: ds_target[c] for c in [lat_name_tar, lon_name_tar]}
    )
    return ds_close


def match_label(
    ds_neighborhood: list[xr.Dataset], ds_target: xr.Dataset, label: str
) -> list[xr.Dataset]:

    site_name_tar = dset.get_dim_name(ds_target, "site")
    assert dset.is_site(ds_target)
    assert len(ds_neighborhood) == len(ds_target[site_name_tar])
    ds_hood = next(iter(ds_neighborhood))
    assert label in ds_hood
    assert label in ds_target

    for site in range(len(ds_target[site_name_tar])):
        label_value = ds_target.isel({site_name_tar: site})[label].values
        ds = ds_neighborhood[site]
        ds_neighborhood[site]["distance"] = xr.where(
            ds[label] == label_value, ds["distance"], np.nan
        )
    return ds_neighborhood
