import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm

import ilamb3.dataset as dset
from ilamb3.regions import Regions


def get_extents(da: xr.DataArray) -> list[float]:
    """Find the extent of the non-null data."""
    lat = xr.where(da.notnull(), da[dset.get_coord_name(da, "lat")], np.nan)
    lon = xr.where(da.notnull(), da[dset.get_coord_name(da, "lon")], np.nan)
    extents = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    # if a da is all nan, then the extents cause trouble downstream in plotting.
    if np.isnan(extents).any():
        return [-180.0, 180.0, -90.0, 90.0]
    return extents


def pick_projection(extents: list[float]) -> tuple[ccrs.Projection, float]:
    """Given plot extents choose projection and aspect ratio."""
    if (extents[1] - extents[0]) > 300:
        aspect_ratio = 2.0
        proj = ccrs.Robinson(central_longitude=0)
        if (extents[2] > 0) and (extents[3] > 60):
            aspect_ratio = 1.0
            proj = ccrs.Orthographic(central_latitude=+90, central_longitude=0)
        if (extents[2] < -60) and (extents[3] < 0):
            aspect_ratio = 1.0
            proj = ccrs.Orthographic(central_latitude=-90, central_longitude=0)
    else:
        aspect_ratio = max(extents[1], extents[0]) - min(extents[1], extents[0])
        aspect_ratio /= max(extents[3], extents[2]) - min(extents[3], extents[2])
        proj = ccrs.PlateCarree(central_longitude=np.array(extents)[:2].mean())
    return proj, aspect_ratio


def finalize_plot(ax: plt.Axes, extents: list[float]) -> plt.Axes:
    """Add some final features to our plots."""
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
    # cleanup plotting extents
    percent_pad = 0.1
    if (extents[1] - extents[0]) > 300:
        extents[:2] = [-180, 180]  # set_extent doesn't like (0,360)
        extents[2:] = [-90, 90]
    else:
        dx = percent_pad * (extents[1] - extents[0])
        dy = percent_pad * (extents[3] - extents[2])
        extents = [
            max(extents[0] - dx, -180),
            min(extents[1] + dx, 180),
            max(extents[2] - dy, -90),
            min(extents[3] + dy, 90),
        ]
    ax.set_extent(extents, ccrs.PlateCarree())
    return ax


def plot_map(da: xr.DataArray, **kwargs):
    # Process some options
    kwargs["cmap"] = plt.get_cmap(kwargs["cmap"] if "cmap" in kwargs else "viridis", 9)
    title = kwargs.pop("title") if "title" in kwargs else ""

    # Process region if given
    ilamb_regions = Regions()
    region = kwargs.pop("region") if "region" in kwargs else None
    da = ilamb_regions.restrict_to_region(da, region)

    # Setup figure and its projection
    extents = get_extents(da)
    proj, aspect = pick_projection(extents)
    figsize = kwargs.pop("figsize") if "figsize" in kwargs else (6 * 1.03, 6 / aspect)
    _, ax = plt.subplots(
        dpi=200,
        tight_layout=(kwargs.pop("tight_layout") if "tight_layout" in kwargs else True),
        figsize=figsize,
        subplot_kw={"projection": proj},
    )

    # Setup colorbar arguments
    cba = {"label": da.attrs["units"]}
    if "cbar_kwargs" in kwargs:
        cba.update(kwargs.pop("cbar_kwargs"))

    # We can't make temporal maps
    if dset.is_temporal(da):
        raise ValueError("Cannot make spatio-temporal plots")

    # Space or sites?
    if dset.is_spatial(da):
        da.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cba, **kwargs)
    elif dset.is_site(da):
        xr.plot.scatter(
            da.to_dataset(),
            x=dset.get_coord_name(da, "lon"),
            y=dset.get_coord_name(da, "lat"),
            hue=da.name,
            lw=0,
            s=15,
            cbar_kwargs=cba,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
    else:
        raise ValueError("plotting error")
    ax.set_title(title)
    ax = finalize_plot(ax, extents)
    return ax


def plot_distribution(da: xr.DataArray, **kwargs):
    _, ax = plt.subplots(dpi=200, tight_layout=True, figsize=(6, 5.25))
    da.plot(
        ax=ax,
        norm=LogNorm(),
        vmin=1e-4,
        vmax=1e-1,
        cmap="plasma",
        cbar_kwargs={"label": "Fraction of total data"},
    )
    ax.set_title(kwargs.pop("title") if "title" in kwargs else "")
    return ax


def plot_response(
    ref_mean: xr.DataArray,
    ref_std: xr.DataArray,
    mean: xr.DataArray,
    std: xr.DataArray,
    comparison_name: str,
    **kwargs,
):
    def _plot(ax, m, s, c="k", lbl="Reference"):
        ax.fill_between(
            m[m.dims[0]].values,
            m - s,
            m + s,
            color=c,
            alpha=0.1,
            lw=0,
            label=f"{lbl} variability",
        )
        m.plot(ax=ax, color=c, label=f"{lbl} mean")
        return ax

    _, ax = plt.subplots(dpi=200, tight_layout=True, figsize=(6, 5.25))
    ax = _plot(ax, ref_mean, ref_std)
    ax = _plot(ax, mean, std, "r", comparison_name)
    ax.legend()
    ax.set_title(kwargs.pop("title") if "title" in kwargs else "")
    return ax


def determine_plot_limits(
    dsd: xr.Dataset | dict[str, xr.Dataset],
    percent_pad: float = 1.0,
    symmetrize: list[str] = ["bias"],
) -> pd.DataFrame:
    """Return a dataframe with the plot minima and maxima."""
    if isinstance(dsd, xr.Dataset):
        dsd = {"key": xr.Dataset}
    plots = set([var for _, ds in dsd.items() for var in ds.data_vars])
    out = []
    for plot in plots:
        # Score maps are always [0,1]
        if "score" in plot:
            out.append({"name": plot, "low": 0, "high": 1})
            continue
        # Otherwise pick a quantile across all data sources
        try:
            data = np.quantile(
                np.hstack(
                    [
                        np.ma.masked_invalid(ds[plot].to_numpy()).compressed()
                        for _, ds in dsd.items()
                        if plot in ds
                    ]
                ),
                [percent_pad * 0.01, 1 - percent_pad * 0.01],
            )
        except IndexError:
            # In rare instances no model has any valid data and so it doesn't
            # matter what we make the plot limits
            data = np.asarray([0.0, 1.0])
        # symmetrize
        if [tag for tag in symmetrize if tag in plot]:
            vmax = max(np.abs(data))
            data[0] = -vmax
            data[1] = vmax
        out.append({"name": plot, "low": data[0], "high": data[1]})
    return pd.DataFrame(out)
