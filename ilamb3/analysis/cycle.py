"""
The ILAMB bias methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.dataset as dset
import ilamb3.plot as ilp
from ilamb3 import compare as cmp
from ilamb3.analysis.base import (
    ILAMBAnalysis,
    get_plot_name,
    integrate_or_mean,
    scalarify,
)
from ilamb3.exceptions import AnalysisNotAppropriate

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _has_annual_cycle(ds: xr.Dataset, varname: str) -> bool:
    da = ds[varname]
    if not dset.is_temporal(da):
        return False
    if (
        np.isclose(dset.get_mean_time_frequency(ds), 30.0, atol=3)
        and len(ds[dset.get_dim_name(da, "time")]) >= 12
    ):
        return True
    return False


class cycle_analysis(ILAMBAnalysis):
    """
    The ILAMB annual cycle methodology.

    Parameters
    ----------
    required_variable : str
        The name of the variable to be used in this analysis.
    regions : list
        A list of region labels over which to apply the analysis.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        required_variable: str,
        regions: list[str | None] = [None],
        plot_unit: str | None = None,
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
        self.regions = regions
        self.plot_unit = plot_unit
        self.kwargs = kwargs

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.req_variable]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.

        Returns
        -------
        pd.DataFrame
            A dataframe with scalar and score information from the comparison.
        xr.Dataset
            A dataset containing reference grided information from the comparison.
        xr.Dataset
            A dataset containing comparison grided information from the comparison.
        """
        # Initialize
        analysis_name = "Annual Cycle"
        varname = self.req_variable

        # Make the variables comparable and force loading into memory
        ref, com = cmp.rename_dims(
            *cmp.make_comparable(ref, com, varname, **self.kwargs)
        )

        # Is the time series long enough for this to be meaningful?
        if not (_has_annual_cycle(ref, varname) & _has_annual_cycle(com, varname)):
            raise AnalysisNotAppropriate()

        # Compute the mean annual cycles
        ref = ref[varname].groupby("time.month").mean()
        com = com[varname].groupby("time.month").mean()

        # Get the timing of the maximum
        ref_tmax = xr.where(
            ref.notnull().any("month"), ref.fillna(0).argmax("month"), np.nan
        )
        com_tmax = xr.where(
            com.notnull().any("month"), com.fillna(0).argmax("month"), np.nan
        )

        # Compute the phase shift (difference in max month)
        ref_tmax_, com_tmax_ = cmp.nest_spatial_grids(ref_tmax, com_tmax)
        shift = com_tmax_ - ref_tmax_
        shift = xr.where(shift > 6, shift - 12, shift)
        shift = xr.where(shift < -6, shift + 12, shift)
        shift_score = 1 - np.abs(shift) / 6
        shift.attrs["units"] = "month"
        shift_score.attrs["units"] = "1"

        # Build output datasets
        ref_out = ref_tmax.to_dataset(name="tmax")
        com_out = shift.to_dataset(name="shift")
        com_out["cyclescore"] = shift_score
        try:
            lat_name = dset.get_dim_name(com_tmax, "lat")
            lon_name = dset.get_dim_name(com_tmax, "lon")
            com_tmax = com_tmax.rename({lat_name: "lat_", lon_name: "lon_"})
        except KeyError:
            pass
        com_out["tmax"] = com_tmax

        # Compute scalars over all regions
        dfs = []
        for region in self.regions:
            # Regional annual cycles
            ref_out[f"cycle_{region}"] = integrate_or_mean(
                ref, varname, region, mean=True
            )
            com_out[f"cycle_{region}"] = integrate_or_mean(
                com, varname, region, mean=True
            )

            # Regional mean phase shift
            val, unit = scalarify(com_out, "shift", region, mean=True)
            dfs.append(
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": analysis_name,
                    "name": "Phase Shift",
                    "type": "scalar",
                    "units": unit,
                    "value": val,
                }
            )

            # Regional cycle scores
            val, unit = scalarify(com_out, "cyclescore", region, mean=True)
            dfs.append(
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": analysis_name,
                    "name": "Seasonal Cycle Score",
                    "type": "score",
                    "units": unit,
                    "value": val,
                },
            )

        dfs = pd.DataFrame(dfs)
        return dfs, ref_out, com_out

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        # This analysis was not run and we should skip plotting entirely
        if "Annual Cycle" not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)

        # Pull the plot regions from those found in the scalars
        regions = [None if r == "None" else r for r in df["region"].unique()]
        cycles = [f"cycle_{region}" for region in regions]

        # Handle units
        da = ref[next(iter(cycles))]
        plot_unit = da.attrs["units"] if self.plot_unit is None else self.plot_unit
        com["Reference"] = ref
        for source, ds in com.items():
            for plot in cycles:
                if plot in ds:
                    com[source][plot] = dset.convert(ds[plot], plot_unit)

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "tmax", "cmap": "rainbow", "title": "Month of Maximum"},
                {"name": "shift", "cmap": "PRGn", "title": "Phase Shift"},
                {"name": "cyclescore", "cmap": "plasma", "title": "Cycle Score"},
            ]
            + [
                {"name": cycle, "cmap": None, "title": "Time Series"}
                for cycle in cycles
            ]
        ).set_index("name")
        df_limits = ilp.determine_plot_limits(com)
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = "Annual Cycle"

        # Override a few limits and set plot options
        df.loc["tmax", ["low", "high"]] = -0.5, 11.5
        df.loc["shift", ["low", "high"]] = -6, 6
        plot_options = {
            "tmax": {
                "ncolors": 12,
                "ticklabels": MONTHS,
                "cbar_kwargs": {"ticks": range(12), "label": ""},
            },
            "shift": {},
            "cyclescore": {},
        }

        # Create each plot for each source if present in the dataset
        df_plots = []
        for plot, row in df.iterrows():
            for source, ds in com.items():
                if plot not in ds:
                    continue
                # cycle plots have already been regionalized
                if plot.startswith("cycle_"):
                    # Reference cycles don't get plots of their own
                    if source == "Reference":
                        continue
                    plotname, region = plot.split("_")
                    out = row.to_dict()
                    out["name"] = plotname
                    out["source"] = source
                    out["region"] = region
                    out["path"] = get_plot_name(source, region, plotname, path)
                    ax = ilp.plot_curve(
                        {source: ds} | {"Reference": ref},
                        plot,
                        region=region,
                        vmin=row["low"],
                        vmax=row["high"],
                        title=f"{source} {row['title']}",
                        xticks=range(1, 13),
                        xticklabels=MONTHS,
                    )
                    ax.get_figure().savefig(out["path"])
                    plt.close()
                    df_plots.append(out)
                    continue
                # Maps are plot over each region
                for region in regions:
                    out = row.to_dict()
                    out["name"] = plot
                    out["source"] = source
                    out["region"] = region
                    out["path"] = get_plot_name(source, region, plot, path)
                    ax = ilp.plot_map(
                        ds[plot],
                        region=region,
                        vmin=row["low"],
                        vmax=row["high"],
                        cmap=row["cmap"],
                        title=f"{source} {row['title']}",
                        **plot_options[plot],
                    )
                    ax.get_figure().savefig(out["path"])
                    plt.close()
                    df_plots.append(out)

        df_plots = pd.DataFrame(df_plots)
        return df_plots
