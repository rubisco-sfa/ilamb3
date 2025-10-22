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


def coerce_to_cf_compliance(da: xr.DataArray) -> xr.DataArray:
    """Try to make the units CF-compliant using the pint/cf-xarray formatter."""
    try:
        da.attrs["units"] = f"{da.pint.quantify().pint.units:~cf}"
    except Exception:
        pass
    return da


def get_extents(da: xr.DataArray) -> list[float]:
    """Find the extent of the non-null data."""
    lat = xr.where(da.notnull(), da[dset.get_coord_name(da, "lat")], np.nan)
    lon = xr.where(da.notnull(), da[dset.get_coord_name(da, "lon")], np.nan)
    extents = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    # if a da is all nan, then the extents cause trouble downstream in plotting.
    if np.isnan(extents).any():
        return [-180.0, 180.0, -90.0, 90.0]
    return extents


def add_extents_pad(extents: list[float], pad: float = 0.05) -> list[float]:
    delta = pad * (max(extents[1], extents[0]) - min(extents[1], extents[0]))
    extents[0] -= delta
    extents[1] += delta
    delta = pad * (max(extents[3], extents[2]) - min(extents[3], extents[2]))
    extents[2] -= delta
    extents[3] += delta
    # make sure we didn't go too far
    extents = [
        max(extents[0], -180),
        min(extents[1], 180),
        max(extents[2], -90),
        min(extents[3], 90),
    ]
    return extents


def compute_aspect_ratio(extents: list[float]) -> float:
    """
    Compute the aspect `ratio = Δlon / Δlat`.
    """
    num = max(extents[1], extents[0]) - min(extents[1], extents[0])
    den = max(extents[3], extents[2]) - min(extents[3], extents[2])
    if np.isclose(den, 0):
        return 1.0  # Not really true but for plotting we want it equal in this case
    aspect_ratio = num / den
    return aspect_ratio


def adjust_extents_for_plotting(
    extents: list[float], max_ar: float = 3.0
) -> list[float]:
    """
    Adjust extents such that the aspect ratio is `1/max_ar < ar < max_ar`.
    """
    min_ar = 1.0 / max_ar
    ar = compute_aspect_ratio(extents)
    if ar < max_ar and ar > min_ar:
        return extents
    if ar < 1:  # too tall
        give = (min_ar - ar) * (
            max(extents[3], extents[2]) - min(extents[3], extents[2])
        )
        extents[0] -= 0.5 * give
        extents[1] += 0.5 * give
    else:  # too wide
        give = ((ar - max_ar) / max_ar) * (
            max(extents[3], extents[2]) - min(extents[3], extents[2])
        )
        extents[2] -= 0.5 * give
        extents[3] += 0.5 * give
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
    extents: list[float], fraction_threshold: float = 0.75
) -> tuple[ccrs.Projection, float]:
    df = pd.DataFrame(
        [
            {
                "key": "north-pole",
                "extents": [-180, 180, 30, 90],
                "proj": ccrs.Orthographic(
                    central_latitude=+90,
                    central_longitude=ilamb3.conf["plot_central_longitude"],
                ),
                "ratio": 1.0,
            },
            {
                "key": "south-pole",
                "extents": [-180, 180, -90, -30],
                "proj": ccrs.Orthographic(
                    central_latitude=-90,
                    central_longitude=ilamb3.conf["plot_central_longitude"],
                ),
                "ratio": 1.0,
            },
            {
                "key": "conus",
                "extents": [-125, -66.5, 20, 50],
                "proj": ccrs.LambertConformal(),
                "ratio": 2.05,
            },
            {
                "key": "globe",
                "extents": [-180, 180, -90, 90],
                "proj": ccrs.Robinson(
                    central_longitude=ilamb3.conf["plot_central_longitude"],
                ),
                "ratio": 2.0,
            },
        ]
    )
    # compute and sort by how much area is shared
    df[["f_wrt_proj", "f_wrt_input"]] = df.apply(
        lambda row: compute_overlap_fracs(row["extents"], extents),
        axis=1,
        result_type="expand",
    )
    df = df.sort_values(["f_wrt_input", "f_wrt_proj"], ascending=False)
    select = df.iloc[0]
    # if the best projection is not a good fit, then lets go cylindrical
    if min(select["f_wrt_proj"], select["f_wrt_input"]) < fraction_threshold:
        extents = add_extents_pad(extents)
        extents = adjust_extents_for_plotting(extents)
        aspect_ratio = compute_aspect_ratio(extents)
        proj = ccrs.PlateCarree(central_longitude=np.array(extents)[:2].mean())
        return proj, aspect_ratio
    return select["proj"], select["ratio"]


def finalize_plot(ax: plt.Axes) -> plt.Axes:
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
    return ax


def plot_map(da: xr.DataArray, **kwargs):
    # Process some options
    ncolors = kwargs.pop("ncolors") if "ncolors" in kwargs else 9
    ticks = kwargs.pop("ticks") if "ticks" in kwargs else None
    ticklabels = kwargs.pop("ticklabels") if "ticklabels" in kwargs else None
    kwargs["cmap"] = plt.get_cmap(
        kwargs["cmap"] if "cmap" in kwargs else "viridis", ncolors
    )
    title = kwargs.pop("title") if "title" in kwargs else ""
    da = coerce_to_cf_compliance(da)

    # Process region if given
    ilamb_regions = Regions()
    region = kwargs.pop("region") if "region" in kwargs else None
    da = ilamb_regions.restrict_to_region(da, region)

    # Setup figure and its projection
    extents = get_extents(da)
    proj, aspect = pick_projection(extents)
    figsize = kwargs.pop("figsize") if "figsize" in kwargs else (6 * 1.03, 6 / aspect)
    _, ax = plt.subplots(
        dpi=ilamb3.conf["figure_dpi"],
        tight_layout=(kwargs.pop("tight_layout") if "tight_layout" in kwargs else True),
        figsize=figsize,
        subplot_kw={"projection": proj},
    )

    # Setup colorbar arguments
    cba = {"label": da.attrs["units"] if "units" in da.attrs else ""}
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
        if ticks is not None:
            out_plot.colorbar.set_ticks(ticks)
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
        if ticks is not None:
            out_plot.colorbar.set_ticks(ticks)
        if ticklabels is not None:
            out_plot.colorbar.set_ticklabels(ticklabels)
    else:
        raise ValueError("plotting error")
    if isinstance(proj, ccrs.PlateCarree):
        ax.set_extent(extents, crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax = finalize_plot(ax)
    return ax


def _pick_convert_calendar_align_on_option(da: xr.DataArray) -> None | str:
    """
    Resturn the best option based on guidance in the xarray documentation.
    """
    t = da[dset.get_dim_name(da, "time")]
    if (hasattr(t, "dt") and t.dt.calendar == "360_day") or (
        "calendar" in t.attrs and t.attrs["calendar"] == "360_day"
    ):
        dt = dset.get_mean_time_frequency(da)
        if dt < 1:
            return "year"
        else:
            return "date"
    return None


def plot_curve(dsd: dict[str, xr.Dataset], varname: str, **kwargs):
    # Parse some options
    vmin = kwargs.pop("vmin") if "vmin" in kwargs else None
    vmax = kwargs.pop("vmax") if "vmax" in kwargs else None
    xticks = kwargs.pop("xticks") if "xticks" in kwargs else None
    xticklabels = kwargs.pop("xticklabels") if "xticklabels" in kwargs else None
    title = kwargs.pop("title") if "title" in kwargs else ""
    ylabel = kwargs.pop("ylabel") if "ylabel" in kwargs else None

    # Setup figure
    ASPECT = 1.618
    figsize = kwargs.pop("figsize") if "figsize" in kwargs else (6, 6 / ASPECT)
    _, ax = plt.subplots(
        dpi=ilamb3.conf["figure_dpi"],
        tight_layout=(kwargs.pop("tight_layout") if "tight_layout" in kwargs else True),
        figsize=figsize,
    )

    # Convert to single calendar for plotting
    dad = {
        source: (
            ds[varname].convert_calendar(
                "noleap", align_on=_pick_convert_calendar_align_on_option(ds[varname])
            )
            if "time" in ds[varname].dims
            else ds[varname]
        )
        for source, ds in dsd.items()
    }

    # Coerce units
    dad = {source: coerce_to_cf_compliance(da) for source, da in dad.items()}

    # Plot curves
    ref = dad.pop("Reference")
    ref.plot(ax=ax, color="k", label="Reference")
    for source, da in dad.items():
        da.plot(ax=ax, color=ilamb3.conf["label_colors"].get(source, "k"), label=source)

    ax.legend()
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_distribution(da: xr.DataArray, **kwargs):
    da = coerce_to_cf_compliance(da)
    _, ax = plt.subplots(
        dpi=ilamb3.conf["figure_dpi"], tight_layout=True, figsize=(6, 5.25)
    )
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

    _, ax = plt.subplots(
        dpi=ilamb3.conf["figure_dpi"], tight_layout=True, figsize=(6, 5.25)
    )
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
        fig = plt.figure(dpi=ilamb3.conf["figure_dpi"], figsize=(6, 6))
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
                color=ilamb3.conf["label_colors"].get(source, "k"),
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


def set_label_colors(
    labels: list[str], base_cmap: str = "rainbow"
) -> dict[str, list[float]]:
    """
    Return a dictionary with a color per label.
    """
    cmap = plt.get_cmap(base_cmap)
    return {
        label: [float(c) for c in cmap(x)]
        for label, x in zip(labels, np.linspace(0, 1, len(labels)))
    }
