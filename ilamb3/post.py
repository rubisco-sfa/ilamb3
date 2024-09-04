"""Post-processing functions used in the ILAMB system."""

from typing import Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.regions import Regions


def get_plots(ds_ref: xr.Dataset, dsd_com: dict[str, xr.Dataset]) -> list[str]:
    """
    Return the union of dataset variables that among all inputs.

    Parameters
    ----------
    ds_ref : xr.Dataset
        The reference dataset.
    dsd_com : dictionary of xr.Dataset
        A dictionary of xr.Datasets whose keys are the comparisons (models).

    Returns
    -------
    list
        A list of keys (variables) found in any dataset.
    """
    # Which plots do we find in both?
    plots = set(ds_ref)
    plots = plots.union(*[set(ds) for _, ds in dsd_com.items()])
    return list(plots)


def get_plot_limits(
    ds_ref: xr.Dataset,
    dsd_com: dict[str, xr.Dataset],
    plots: Union[list[str], None] = None,
    outlier_fraction: float = 0.02,
) -> dict[str, np.typing.ArrayLike]:
    """
    Return the limits for each plot across all datasets.

    Parameters
    ----------
    ds_ref : xr.Dataset
        The reference dataset.
    dsd_com : dictionary of xr.Dataset
        A dictionary of xr.Datasets whose keys are the comparisons (models).
    plots : list, optional
        The list of variables (plots) among which to take limits. Defaults to all of
        them.
    outlier_fraction : float, optional
        The fraction of the non-null data across all datasets to consider as outliers.
        In other words, we will return limits defined by
        [outlier_fraction,1-outlier_fraction].

    Returns
    -------
    dict[str,array]
        A dictionary who keys are the variable (plot) and entries are the lower and
        upper plotting limit.
    """

    def _append_finite(values, da):  # numpydoc ignore=GL08
        return values + [da.to_numpy().flatten()[np.isfinite(da.to_numpy().flatten())]]

    if plots is None:
        plots = get_plots(ds_ref, dsd_com)

    limits = {key: [] for key in plots}
    limits.update({key: _append_finite(limits[key], ds_ref[key]) for key in ds_ref})
    limits.update(
        {
            key: _append_finite(limits[key], com[key])
            for _, com in dsd_com.items()
            for key in com
        }
    )
    limits = {
        key: np.quantile(np.hstack(arr), [outlier_fraction, 1 - outlier_fraction])
        for key, arr in limits.items()
    }
    return limits


def _plot_extents(da: xr.DataArray) -> np.typing.ArrayLike:
    """
    Find the extent of the non-null data.

    Parameters
    ----------
    da : xr.DataArray
        The input data array.

    Returns
    -------
    np.typing.ArrayLike
        The extents as [lonmin, lonmax, latmin, latmax].
    """
    lat = xr.where(da.notnull(), da[dset.get_dim_name(da, "lat")], np.nan)
    lon = xr.where(da.notnull(), da[dset.get_dim_name(da, "lon")], np.nan)
    return np.array(
        [
            float(lon.min()),
            float(lon.max()),
            float(lat.min()),
            float(lat.max()),
        ]
    )


def _plot_projection(extents) -> tuple[ccrs.Projection, float]:
    """
    Given plot extents choose a projection.

    Parameters
    ----------
    extents : np.typing.ArrayLike
        The extents as [lonmin, lonmax, latmin, latmax].

    Returns
    -------
    tuple
        The project and the figure aspect ratio.
    """
    if (extents[1] - extents[0]) > 300:
        aspect_ratio = 2.0
        proj = ccrs.Robinson(central_longitude=0)
        if (extents[2] > 0) and (extents[3] > 75):
            aspect_ratio = 1.0
            proj = ccrs.Orthographic(central_latitude=+90, central_longitude=0)
        if (extents[2] < -75) and (extents[3] < 0):
            aspect_ratio = 1.0
            proj = ccrs.Orthographic(central_latitude=-90, central_longitude=0)
    else:
        aspect_ratio = max(extents[1], extents[0]) - min(extents[1], extents[0])
        aspect_ratio /= max(extents[3], extents[2]) - min(extents[3], extents[2])
        proj = ccrs.PlateCarree(central_longitude=extents[:2].mean())
    return proj, aspect_ratio


def _plot_initialize(
    da: xr.DataArray,
) -> tuple[np.typing.ArrayLike, float, ccrs.Projection]:
    """
    Return initial plot information based on the non-null data range.

    Parameters
    ----------
    da : xr.DataArray
        The input data array.

    Returns
    -------
    tuple
        The plot extents, aspect ratio, and projection.
    """
    extents = _plot_extents(da)
    proj, aspect_ratio = _plot_projection(extents)
    return extents, aspect_ratio, proj


def _plot_finalize(ax: plt.Axes) -> plt.Axes:
    """
    Add final touches to the plot.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes object.

    Returns
    -------
    plt.Axes
        The finalized matplotlib object.
    """
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", edgecolor="face", facecolor="0.875"
        ),
        zorder=-1,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m", edgecolor="face", facecolor="0.750"
        ),
        zorder=-1,
    )
    return ax


def plot_space(
    ds: Union[xr.Dataset, xr.DataArray],
    varname: Union[str, None] = None,
    region: Union[str, None] = None,
    **kwargs,
):
    """
    Plot the spatial quantity.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The input dataset/dataarray.
    varname : str, optional
        The variable to plot, must be given if a dataset is passed in.
    region : str, optional
        The name of the region over which to plot.
    **kwargs : dict
        Additional keyword arguments passed to xr.dataarray.plot().

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure.
    ax : plt.Axes
        The matplotlib axes.
    """
    # Mask away values outside region
    ilamb_regions = Regions()
    if isinstance(ds, xr.Dataset):
        if varname is None:
            raise ValueError("Must provide varname if a dataset is passed.")
        da = ds[varname]
        if "cell_measures" in ds:
            da = xr.where(ds["cell_measures"] > 1, da, np.nan, keep_attrs=True)
    else:
        da = ds
    da = ilamb_regions.restrict_to_region(da, region)

    # Plot
    _, aspect_ratio, proj = _plot_initialize(da)
    fig, ax = plt.subplots(
        figsize=(6 * 1.03, 6 / aspect_ratio),
        subplot_kw={"projection": proj},
        tight_layout=True,
        dpi=200,
    )
    da.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    ax = _plot_finalize(ax)
    return fig, ax
