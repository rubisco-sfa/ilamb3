from itertools import chain
from pathlib import Path
from typing import Any

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as plt
from ilamb3.analysis.base import ILAMBAnalysis, scalarify

from ilamb3.plot import unify_units


def metric_maps(
    da: xr.Dataset | xr.DataArray, varname: str | None = None
) -> xr.Dataset:
    """
    Return a dataset containing Deeksha's metrics request for the ILAMB Hydro project.

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        The dataset containing the variable
    varname: str, optional
        The name of the variable if a xr.Dataset is passed.

    Returns
    -------
    xr.Dataset
        The metrics derived from the input dataset.

    """
    if isinstance(da, xr.Dataset):
        assert varname is not None
        da = da[varname]
    out = {}

    # annual
    grp = da.groupby("time.year")
    out["annual_mean"] = grp.mean().mean(dim="year")
    out["annual_std"] = grp.mean().std(dim="year")

    # seasons
    grp = da.groupby("time.season")
    mean = grp.mean()
    out.update(
        {
            f"seasonal_mean_{str(s)}": mean.sel(season=s).drop_vars("season")
            for s in mean["season"].values
        }
    )
    std = grp.std("time")
    out.update(
        {
            f"seasonal_std_{str(s)}": std.sel(season=s).drop_vars("season")
            for s in std["season"].values
        }
    )
    return xr.Dataset(out)


def score_difference(ref: xr.Dataset, com: xr.Dataset) -> xr.Dataset:
    # Compute differences and scores
    ref_, com_ = cmp.nest_spatial_grids(ref, com)
    diff = com_ - ref_
    diff = diff.rename_vars({v: f"{v}_difference" for v in diff})
    # Add scores to the means that also have std's
    diff = diff.merge(
        {
            v.replace("_difference", "_score"): np.exp(
                -np.abs(diff[v])
                / ref_[v.replace("_mean", "_std").replace("_difference", "")]
            )
            for v in diff
            if "mean" in v and v.replace("_mean_", "_std_") in diff
        }
    )
    # Set the units of these scores
    for var, da in diff.items():
        if "score" not in var:
            continue
        diff[var].attrs["units"] = "1"
    # Rename the lat dimension for merging with the comparison on return
    lat_name = dset.get_dim_name(diff, "lat")
    lon_name = dset.get_dim_name(diff, "lon")
    com = com.merge(diff.rename({lat_name: f"{lat_name}_", lon_name: f"{lon_name}_"}))
    return com


def generate_titles(qname: str) -> str:
    """
    Transform the quantity name into a display string.
    """
    tokens = [v.capitalize() if v.islower() else v for v in qname.split("_")]
    if (
        len(tokens) > 2
        and tokens[0] == "Seasonal"
        and tokens[2] in ["DJF", "MAM", "JJA", "SON"]
    ):
        tokens[0] = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Fall"}[
            tokens.pop(2)
        ]
    title = " ".join(tokens)
    print(f"{qname:30} | {title}")
    return title


class hydro_analysis(ILAMBAnalysis):
    def __init__(
        self,
        required_variable: str,
        regions: list[str] | None = None,
        output_path: Path | None = None,
        **kwargs: Any,
    ):
        self.req_variable = required_variable
        self.regions = regions if isinstance(regions, list) else [None]
        self.output_path = output_path

        # This analysis will split plots/scalars into sections as organized below
        self.sections = {
            "Annual": [f"mean_{region}" for region in self.regions]
            + [
                "annual_mean",
                "annual_mean_difference",
                "annual_mean_score",
                "annual_std",
                "annual_std_difference",
            ],
            "Winter (DJF)": [
                "seasonal_mean_DJF",
                "seasonal_mean_DJF_difference",
                "seasonal_mean_DJF_score",
                "seasonal_std_DJF",
                "seasonal_std_DJF_difference",
            ],
            "Spring (MAM)": [
                "seasonal_mean_MAM",
                "seasonal_mean_MAM_difference",
                "seasonal_mean_MAM_score",
                "seasonal_std_MAM",
                "seasonal_std_MAM_difference",
            ],
            "Summer (JJA)": [
                "seasonal_mean_JJA",
                "seasonal_mean_JJA_difference",
                "seasonal_mean_JJA_score",
                "seasonal_std_JJA",
                "seasonal_std_JJA_difference",
            ],
            "Fall (SON)": [
                "seasonal_mean_SON",
                "seasonal_mean_SON_difference",
                "seasonal_mean_SON_score",
                "seasonal_std_SON",
                "seasonal_std_SON_difference",
            ],
        }

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.req_variable]

    def _get_analysis_section(self, varname: str) -> str:
        """Given the plot/variable, return from which section it belongs."""
        section = [s for s, vs in self.sections.items() if varname in vs]
        if not section:
            raise ValueError(f"Could not find {varname} in {self.sections}.")
        return section[0]

    def _make_comparable(
        self, ref: xr.Dataset, com: xr.Dataset
    ) -> tuple[xr.Dataset, xr.Dataset]:
        if dset.is_temporal(ref):
            ref, com = cmp.trim_time(ref, com)
        # ensure longitudes are uniform
        ref, com = cmp.adjust_lon(ref, com)
        # ensure the lat/lon dims are sorted
        if dset.is_spatial(ref):
            ref = ref.sortby(
                [dset.get_dim_name(ref, "lat"), dset.get_dim_name(ref, "lon")]
            )
        if dset.is_spatial(com):
            com = com.sortby(
                [dset.get_dim_name(com, "lat"), dset.get_dim_name(com, "lon")]
            )
        # convert units
        com = dset.convert(
            com, ref[self.req_variable].attrs["units"], varname=self.req_variable
        )
        return ref, com

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.
        """
        # Initialize
        varname = self.req_variable

        # Make the variables comparable
        ref_, com_ = self._make_comparable(ref, com)

        # Run the hydro metrics, read cached reference if running in batch mode
        if (
            self.output_path is not None
            and (self.output_path / "Reference.nc").is_file()
        ):
            logger.info(
                f"Reading in cached reference data: {self.output_path / 'Reference.nc'}"
            )
            ref = xr.open_dataset(self.output_path / "Reference.nc")
        else:
            ref = metric_maps(ref_, varname)
        com = metric_maps(com_, varname)

        # Ensure that arrays are now in memory and score
        logger.info("Computing hydro metrics...")
        ref.load()
        com.load()
        com = score_difference(ref, com)

        # Create scalars
        df = []
        for source, ds in {"Reference": ref, "Comparison": com}.items():
            for vname, da in ds.items():
                if vname.startswith("mean_"):
                    continue
                for region in self.regions:
                    scalar, unit = scalarify(da, vname, region=region, mean=True)
                    df.append(
                        [
                            source,
                            str(region),
                            self._get_analysis_section(vname),
                            generate_titles(vname),
                            "score" if "score" in vname else "scalar",
                            unit,
                            scalar,
                        ]
                    )

        # Compute the regional means
        for region in self.regions:
            if f"mean_{region}" not in ref:
                ref[f"mean_{region}"] = dset.compute_monthly_mean(
                    dset.integrate_space(
                        ref_,
                        varname,
                        region=region,
                        mean=True,
                    )
                )
            com[f"mean_{region}"] = dset.compute_monthly_mean(
                dset.integrate_space(
                    com_,
                    varname,
                    region=region,
                    mean=True,
                )
            )

        # Convert to dataframe
        df = pd.DataFrame(
            df,
            columns=[
                "source",
                "region",
                "analysis",
                "name",
                "type",
                "units",
                "value",
            ],
        )
        df.attrs = self.__dict__.copy()
        return df, ref, com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        com["Reference"] = ref

        def _choose_cmap(plot_name):
            if "score" in plot_name:
                return "plasma"
            if "difference" in plot_name:
                return "bwr"
            return "viridis"


        def _set_ylabel(name:str, units:str) -> str:

            units_str = unify_units(units)
            return f"{name} {unit_str}"




        # Which plots are we handling in here? I am building this list from a
        # section layout I created in the constructor.
        plots = list(chain(*[vs for _, vs in self.sections.items()]))

        # Setup plots
        df = plt.determine_plot_limits(com, symmetrize=["difference"]).set_index("name")
        df["title"] = [generate_titles(plot) for plot in df.index]
        df["cmap"] = df.index.map(_choose_cmap)

        # Plot the maps, saving if requested on the fly
        axs = []
        logger.info("Plotting maps...")
        for plot in plots:
            for source, ds in com.items():
                if plot not in ds:
                    continue
                if not dset.is_spatial(ds[plot]):
                    continue
                for region in self.regions:
                    row = {
                        "name": plot,
                        "title": df.loc[plot, "title"],
                        "region": region,
                        "source": source,
                        "analysis": self._get_analysis_section(plot),
                    }
                    ax = plt.plot_map(
                        ds[plot],
                        region=region,
                        vmin=df.loc[plot, "low"],
                        vmax=df.loc[plot, "high"],
                        cmap=df.loc[plot, "cmap"],
                        title=source + " " + df.loc[plot, "title"],
                    )
                    if self.output_path is None:
                        row["axis"] = ax
                    else:
                        row["axis"] = False
                        fig = ax.get_figure()
                        fig.savefig(
                            self.output_path
                            / f"{row['source']}_{row['region']}_{row['name']}.png"
                        )
                        mpl.close(fig)
                    axs.append(row)

        # Plot the curves, saving if requested on the fly
        logger.info("Plotting curves...")
        for plot in [f"mean_{region}" for region in self.regions]:
            region = plot.split("_")[-1]
            for source, ds in com.items():
                if source == "Reference":
                    continue
                if plot not in ds:
                    continue
                if not dset.is_temporal(ds[plot]):
                    continue
                row = {
                    "name": "mean",
                    "title": "Regional Mean",
                    "region": region,
                    "source": source,
                    "analysis": "Annual",
                }

                ylabel = None
                try:
                    ylabel = _set_ylabel(ref[plot].attrs["long_name"], ref[plot].attrs["units"])
                except KeyError:
                    raise ValueError("Reference Dataset attributes missing 'long_name' or 'units' key")

                ax = plt.plot_curve(
                    {source: ds} | {"Reference": ref},
                    plot,
                    vmin=df.loc[plot, "low"]
                    - 0.05 * (df.loc[plot, "high"] - df.loc[plot, "low"]),
                    vmax=df.loc[plot, "high"]
                    + 0.05 * (df.loc[plot, "high"] - df.loc[plot, "low"]),
                    title=f"{source} Regional Mean",
                    ylabel=ylabel,
                )
                if self.output_path is None:
                    row["axis"] = ax
                    continue
                else:
                    row["axis"] = False
                    fig = ax.get_figure()
                    fig.savefig(
                        self.output_path
                        / f"{row['source']}_{row['region']}_{row['name']}.png"
                    )
                    mpl.close(fig)
                axs.append(row)
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
