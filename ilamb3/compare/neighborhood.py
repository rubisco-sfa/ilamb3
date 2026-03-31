"""
Functions for converting a data source into target point data.
"""

import warnings
from functools import partial
from typing import Any, Literal

import numpy as np
import xarray as xr

import ilamb3.dataset as dset


def extract_sites_closest(
    ds: xr.Dataset, ds_sites: xr.Dataset, **kwargs: Any
) -> xr.Dataset:
    width = kwargs.get("window_half_width", 2.0)
    scale = kwargs.get("scale_width_by_grid_resolution", None)
    if scale is not None and dset.is_spatial(ds):
        width = scale * dset.get_mean_spatial_resolution(ds)
    ds_neighborhood = extract_neighbors_by_window(
        ds,
        ds_sites,
        window_half_width=width,
        window_shape=kwargs.get("window_shape", "circle"),
    )
    ds_out = neighborhood_closest(ds_neighborhood, ds_sites)
    return ds_out


def extract_sites_mean(
    ds: xr.Dataset, ds_sites: xr.Dataset, weighted: bool, **kwargs: Any
) -> xr.Dataset:
    width = kwargs.get("window_half_width", 2.0)
    scale = kwargs.get("scale_width_by_grid_resolution", None)
    if scale is not None and dset.is_spatial(ds):
        width = scale * dset.get_mean_spatial_resolution(ds)
    ds_neighborhood = extract_neighbors_by_window(
        ds,
        ds_sites,
        window_half_width=width,
        window_shape=kwargs.get("window_shape", "circle"),
    )
    ds_out = neighborhood_mean(ds_neighborhood, ds_sites, weighted=weighted)
    return ds_out


SITE_EXTRACT = {
    "closest": extract_sites_closest,
    "mean": partial(extract_sites_mean, weighted=False),
    "weighted_mean": partial(extract_sites_mean, weighted=True),
}


def extract_neighbors_by_window(
    ds_source: xr.Dataset,
    ds_target: xr.Dataset,
    window_half_width: float,
    window_shape: Literal["box", "circle"] = "circle",
) -> list[xr.Dataset]:
    """
    Return closest neighbors from the source dataset to each site in the target
    dataset.

    Parameters
    ----------
    ds_source: xr.Dataset
        The dataset from which we will extract neighborhoods. Can be gridded or
        a collection of sites.
    ds_target: xr.Dataset
        The dataset around which we will extract neighborhoods. This must be
        site or site collection data.
    window_half_width: float
        If the `window_shape='circle'`, this will be the radius inside of which
        a cell from `ds_source` will be considered in the neighborhood. If a box
        shape is used, this is half the box width to be consistent with the
        circle.
    window_shape: 'box' or 'circle'
        The shape of the window used to define the neighborhood.

    Returns
    -------
    list[xr.Dataset]
        The neighborhoods, one for each site in the `ds_target`.
    """
    if not (dset.is_spatial(ds_source) or dset.is_site(ds_source)):
        raise ValueError(
            f"The source dataset must be gridded or site data.\n{ds_source=}"
        )
    if not dset.is_site(ds_target):
        raise ValueError(f"The target dataset must be site data.\n{ds_target=}")
    if window_half_width < 0:
        raise ValueError(
            f"The window_half_size must be non-negative {window_half_width=}"
        )

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
                        ds_site[lat_name_tar] - window_half_width,
                        ds_site[lat_name_tar] + window_half_width,
                    ),
                    lon_name_src: slice(
                        ds_site[lon_name_tar] - window_half_width,
                        ds_site[lon_name_tar] + window_half_width,
                    ),
                }
            )
        else:
            # Otherwise assume the site data is small and just use the whole
            # dataset (sel does not work on coords that are not dims) but mask
            # outside the window
            condition = (
                np.abs(ds_source[lat_name_src] - ds_site[lat_name_tar])
                < window_half_width
            ) * (
                np.abs(ds_source[lon_name_src] - ds_site[lon_name_tar])
                < window_half_width
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
            condition = ds_hood["distance"] <= window_half_width
            for var, da in ds_hood.items():
                if set(condition.dims).issubset(da.dims):
                    ds_hood[var] = xr.where(condition, da, np.nan)
        ds_neighborhood += [ds_hood]
    return ds_neighborhood


def neighborhood_mean(
    ds_neighborhood: list[xr.Dataset], ds_target: xr.Dataset, weighted: bool = False
) -> xr.Dataset:
    """
    Reduce the neighborhood by taking a mean of the non-null cells.

    Parameters
    ----------
    ds_neighborhood: list[xr.Dataset]
        A neighborhood of cells, one for each site in the target dataset.
        Generated from routines like `extract_neighbors_by_window`.
    ds_target: xr.Dataset
        The target dataset. Used here to verify the neighborhood as well as pass
        along its coordinates upon exit.
    weighted: bool
        If enabled, a reciprocal distance to the target site is used to weight
        the mean. Otherwise, an arithemtic mean is returned.

    Returns
    -------
    xr.Dataset
        The mean and standard deviation of each neighborhood, now matching the
        target dataset's sites.
    """

    def _weights(ds: xr.Dataset, weighted: bool, min_dist: float = 0.1) -> xr.DataArray:
        if weighted:
            return (1 / ds["distance"].clip(min_dist)).fillna(0)
        return xr.where(ds["distance"].isnull(), 0.0, 1.0)

    # Extract dim/coord names
    site_name_tar = dset.get_dim_name(ds_target, "site")
    lat_name_tar = dset.get_coord_name(ds_target, "lat")
    lon_name_tar = dset.get_coord_name(ds_target, "lon")

    # Assumption checks
    if not dset.is_site(ds_target):
        raise ValueError(f"The target dataset must be site data.\n{ds_target=}")
    if not (len(ds_neighborhood) == len(ds_target[site_name_tar])):
        raise ValueError(
            f"Inconsistent neighborhood and target, {len(ds_neighborhood)=} != {len(ds_target[site_name_tar])=}"
        )

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
    """
    Reduce the neighborhood by selecting the closest non-null cell.

    Parameters
    ----------
    ds_neighborhood: list[xr.Dataset]
        A neighborhood of cells, one for each site in the target dataset.
        Generated from routines like `extract_neighbors_by_window`.
    ds_target: xr.Dataset
        The target dataset. Used here to verify the neighborhood as well as pass
        along its coordinates upon exit.

    Returns
    -------
    xr.Dataset
        The closest non-null cell of each neighborhood, now matching the target
        dataset's sites.
    """

    # Extract dim/coord names
    site_name_tar = dset.get_dim_name(ds_target, "site")
    lat_name_tar = dset.get_coord_name(ds_target, "lat")
    lon_name_tar = dset.get_coord_name(ds_target, "lon")

    # Assumption checks
    if not dset.is_site(ds_target):
        raise ValueError(f"The target dataset must be site data.\n{ds_target=}")
    if not (len(ds_neighborhood) == len(ds_target[site_name_tar])):
        raise ValueError(
            f"Inconsistent neighborhood and target, {len(ds_neighborhood)=} != {len(ds_target[site_name_tar])=}"
        )

    ds_hood = next(iter(ds_neighborhood))
    lat_name_hood = dset.get_coord_name(ds_hood, "lat")
    lon_name_hood = dset.get_coord_name(ds_hood, "lon")
    if "distance" not in ds_hood:
        raise ValueError(
            f"The dataarray 'distance' is not in the neighborhood but should be.\n{ds_hood=}"
        )

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
    """

    Parameters
    ----------
    ds_neighborhood: list[xr.Dataset]
        A neighborhood of cells, one for each site in the target dataset.
        Generated from routines like `extract_neighbors_by_window`.
    ds_target: xr.Dataset
        The target dataset.
    label: str
        The name of a DataArray in both datasets, where the value from
        `ds_target` will be used to mask non-matching values in each
        neighborhood.

    Returns
    -------
    list[xr.Dataset]
        The neighborhood where the `label` from each site in `ds_target` was
        used to mask non-matching values in `ds_neighborhood`.
    """
    site_name_tar = dset.get_dim_name(ds_target, "site")
    # Assumption checks
    if not dset.is_site(ds_target):
        raise ValueError(f"The target dataset must be site data.\n{ds_target=}")
    if not (len(ds_neighborhood) == len(ds_target[site_name_tar])):
        raise ValueError(
            f"Inconsistent neighborhood and target, {len(ds_neighborhood)=} != {len(ds_target[site_name_tar])=}"
        )
    ds_hood = next(iter(ds_neighborhood))
    if not (label in ds_hood and label in ds_target):
        raise ValueError(
            f"The {label=} should be a DataArray in both the neighborhood and target.\n{ds_hood=}\n{ds_target=}"
        )

    for site in range(len(ds_target[site_name_tar])):
        label_value = ds_target.isel({site_name_tar: site})[label].values
        ds = ds_neighborhood[site]
        ds_neighborhood[site]["distance"] = xr.where(
            ds[label] == label_value, ds["distance"], np.nan
        )
    return ds_neighborhood
