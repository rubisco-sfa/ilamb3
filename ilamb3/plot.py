import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm

import ilamb3
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


def compute_extent_area(extents):
    return (extents[1] - extents[0]) * (extents[3] - extents[2])


def compute_overlap_fracs(
    extents_a: list[float], extents_b: list[float]
) -> tuple[float, float]:
    """Return the fractions of overlap."""

    overlap = [op(a, b) for op, a, b in zip([max, min, max, min], extents_a, extents_b)]
    area_O = (
        compute_extent_area(overlap)
        if (
            (extents_a[0] < extents_b[1])
            & (extents_a[1] > extents_b[0])
            & (extents_a[2] < extents_b[3])
            & (extents_a[3] > extents_b[2])
        )
        else 0.0
    )
    area_A = compute_extent_area(extents_a)
    area_B = compute_extent_area(extents_b)
    frac_A = area_O / area_A if area_A > 0 else 0.0
    frac_B = area_O / area_B if area_B > 0 else 0.0
    return frac_A, frac_B


def pick_projection(
    extents: list[float], fraction_threshold: float = 0.85
) -> tuple[ccrs.Projection, float]:
    """Given plot extents choose projection and aspect ratio."""
    lon = ilamb3.conf["plot_central_longitude"]
    if compute_overlap_fracs([-180, 180, 60, 90], extents)[1] > fraction_threshold:
        return ccrs.Orthographic(central_latitude=+90, central_longitude=lon), 1.0
    if compute_overlap_fracs([-180, 180, -90, -60], extents)[1] > fraction_threshold:
        return ccrs.Orthographic(central_latitude=-90, central_longitude=lon), 1.0
    if compute_overlap_fracs([-125, -66.5, 20, 50], extents)[1] > fraction_threshold:
        return ccrs.LambertConformal(), 2.05  # USA
    if compute_extent_area(extents) / compute_extent_area([-180, 180, -90, 90]) > 0.5:
        return ccrs.Robinson(central_longitude=lon), 2.0  # Global
    # If none of above, use cyclindrical
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
    ncolors = kwargs.pop("ncolors") if "ncolors" in kwargs else 9
    ticklabels = kwargs.pop("ticklabels") if "ticklabels" in kwargs else None
    kwargs["cmap"] = plt.get_cmap(
        kwargs["cmap"] if "cmap" in kwargs else "viridis", ncolors
    )
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
        out_plot = da.plot(
            ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cba, **kwargs
        )
        if ticklabels is not None:
            out_plot.colorbar.set_ticklabels(ticklabels)
    elif dset.is_site(da):
        out_plot = xr.plot.scatter(
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
        if ticklabels is not None:
            out_plot.colorbar.set_ticklabels(ticklabels)
    else:
        raise ValueError("plotting error")
    ax.set_title(title)
    ax = finalize_plot(ax, extents)
    return ax


def plot_curve(dsd: dict[str, xr.Dataset], varname: str, **kwargs):
    # Parse some options
    vmin = kwargs.pop("vmin") if "vmin" in kwargs else None
    vmax = kwargs.pop("vmax") if "vmax" in kwargs else None
    xticks = kwargs.pop("xticks") if "xticks" in kwargs else None
    xticklabels = kwargs.pop("xticklabels") if "xticklabels" in kwargs else None
    title = kwargs.pop("title") if "title" in kwargs else ""

    # Setup figure
    ASPECT = 1.618
    figsize = kwargs.pop("figsize") if "figsize" in kwargs else (6, 6 / ASPECT)
    _, ax = plt.subplots(
        dpi=200,
        tight_layout=(kwargs.pop("tight_layout") if "tight_layout" in kwargs else True),
        figsize=figsize,
    )

    # Convert to single calendar for plotting
    dsd = {
        source: ds.convert_calendar("noleap") if "time" in ds else ds
        for source, ds in dsd.items()
    }
    ref = dsd.pop("Reference")

    # Plot curves
    ref[varname].plot(ax=ax, color="k", label="Reference")
    for source, ds in dsd.items():
        ds[varname].plot(ax=ax, color=get_model_color(source), label=source)

    ax.legend()
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
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


def plot_taylor_diagram(df: pd.DataFrame):
    """
    Create a Taylor diagram.

    This is adapted from the code by Yannick Copin found here:

    https://gist.github.com/ycopin/3342888
    """
    import mpl_toolkits.axisartist.floating_axes as FA
    import mpl_toolkits.axisartist.grid_finder as GF
    from matplotlib.projections import PolarAxes

    # correlation ticks and labels
    rlocs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    tlocs = np.arccos(rlocs)
    gl1 = GF.FixedLocator(tlocs)
    tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

    # add a curvilinear grid helper
    smax = max(
        2, 1.1 * df[df["name"] == "Normalized Standard Deviation [1]"]["value"].max()
    )
    tr = PolarAxes.PolarTransform()
    ghelper = FA.GridHelperCurveLinear(
        tr,
        extremes=(0, np.pi / 2, 0, smax),
        grid_locator1=gl1,
        tick_formatter1=tf1,
    )

    # create figure
    with mpl.rc_context({"font.size": 13}):
        fig = plt.figure(dpi=200, figsize=(6, 6))
        ax = FA.FloatingSubplot(fig, 111, grid_helper=ghelper)
        fig.add_subplot(ax)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Normalized standard deviation")
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        ax.grid(True)

        # plot data
        ax = ax.get_aux_axes(tr)
        for source, grp in df[
            df["name"].isin(["Normalized Standard Deviation [1]", "Correlation [1]"])
        ].groupby("source"):
            std = grp[grp["name"] == "Normalized Standard Deviation [1]"].iloc[0][
                "value"
            ]
            corr = grp[grp["name"] == "Correlation [1]"].iloc[0]["value"]
            ax.plot(
                np.arccos(corr.clip(-1, 1)),
                std,
                "o",
                color=get_model_color(source),
                mew=0,
                ms=8,
            )

        # Add reference point and stddev contour
        ax.plot([0], 1, "k*", ms=12, mew=0)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + 1
        ax.plot(t, r, "k--")

        # centralized rms contours
        rs, ts = np.meshgrid(np.linspace(0, smax), np.linspace(0, np.pi / 2))
        rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
        contours = ax.contour(ts, rs, rms, 5, colors="k", alpha=0.4)
        ax.clabel(contours, fmt="%1.1f")

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
        # Which datasets have this plot, the data type must be a float
        plot_ds = {
            key: ds
            for key, ds in dsd.items()
            if plot in ds and (np.issubdtype(ds[plot].dtype, np.floating))
        }
        if not plot_ds:
            continue
        # Stack the data together into a single array
        data = np.hstack(
            [
                np.ma.masked_invalid(ds[plot].to_numpy()).compressed()
                for _, ds in plot_ds.items()
            ]
        )
        # If we have data, then take the quantiles, otherwise it doesn't matter.
        if data.size > 0:
            data = np.quantile(data, [percent_pad * 0.01, 1 - percent_pad * 0.01])
        else:
            data = np.asarray([0.0, 1.0])
        # Symmetrize
        if [tag for tag in symmetrize if tag in plot]:
            vmax = max(np.abs(data))
            data[0] = -vmax
            data[1] = vmax
        out.append({"name": plot, "low": data[0], "high": data[1]})
    return pd.DataFrame(out)


def get_model_color(
    model: str, base_cmap: str = "rainbow"
) -> tuple[float, float, float, float]:
    if model == "Reference":
        return (0.0, 0.0, 0.0, 1.0)
    MODEL_PREFIXES = np.array(
        [
            "ACC",
            "AWI",
            "BCC",
            "CAM",
            "CAS",
            "CES",
            "CIE",
            "CMC",
            "CNR",
            "CAN",
            "E3S",
            "EC",
            "FGO",
            "FIO",
            "GFD",
            "GIS",
            "HAD",
            "ICO",
            "IIT",
            "INM",
            "IPS",
            "KAC",
            "KIO",
            "MCM",
            "MIR",
            "MPI",
            "MRI",
            "NES",
            "Nor",
            "SAM",
            "TAI",
            "UKE",
        ]
    )
    cmap = plt.get_cmap(base_cmap, len(MODEL_PREFIXES))
    return cmap(MODEL_PREFIXES.searchsorted(model.upper()))
