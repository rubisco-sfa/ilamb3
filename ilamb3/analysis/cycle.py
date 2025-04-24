"""
The ILAMB bias methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.dataset as dset
import ilamb3.plot as plt
from ilamb3 import compare as cmp
from ilamb3.analysis.base import ILAMBAnalysis, integrate_or_mean, scalarify
from ilamb3.exceptions import AnalysisNotAppropriate


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
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
        self.regions = regions
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
        ref, com = cmp.make_comparable(ref, com, varname)

        if len(com[dset.get_dim_name(com, "time")]) < 12:
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
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        # Some initialization
        regions = [None if r == "None" else r for r in df["region"].unique()]
        com["Reference"] = ref

        # Setup plot data
        df = plt.determine_plot_limits(com).set_index("name")
        df.loc["tmax", ["cmap", "title", "low", "high"]] = [
            "rainbow",
            "Month of Maximum",
            -0.5,
            11.5,
        ]
        df.loc["shift", ["cmap", "title", "low", "high"]] = [
            "PRGn",
            "Phase Shift",
            -6,
            6,
        ]
        df.loc["cyclescore", ["cmap", "title"]] = ["plasma", "Cycle Score"]

        # Build up a dataframe of matplotlib axes
        plot_cbar_kwargs = {
            "tmax": {"ticks": range(12), "label": ""},
            "shift": {},
            "cyclescore": {},
        }
        axs = [
            {
                "name": plot,
                "title": df.loc[plot, "title"],
                "region": region,
                "source": source,
                "axis": (
                    plt.plot_map(
                        ds[plot],
                        region=region,
                        vmin=df.loc[plot, "low"],
                        vmax=df.loc[plot, "high"],
                        cmap=df.loc[plot, "cmap"],
                        title=source + " " + df.loc[plot, "title"],
                        ncolors=12 if plot == "tmax" else 9,
                        ticklabels=(
                            [
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
                            if plot == "tmax"
                            else None
                        ),
                        **{"cbar_kwargs": plot_cbar_kwargs[plot]},
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in ["tmax", "shift", "cyclescore"]
            for source, ds in com.items()
            for region in regions
        ]

        axs += [
            {
                "name": "cycle",
                "title": "Annual Cycle",
                "region": plot.split("_")[-1],
                "source": source,
                "axis": (
                    plt.plot_curve(
                        {source: ds} | {"Reference": ref},
                        plot,
                        vmin=df.loc[plot, "low"]
                        - 0.05 * (df.loc[plot, "high"] - df.loc[plot, "low"]),
                        vmax=df.loc[plot, "high"]
                        + 0.05 * (df.loc[plot, "high"] - df.loc[plot, "low"]),
                        title=f"{source} Annual Cycle",
                        xticks=range(1, 13),
                        xticklabels=[
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
                        ],
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in [f"cycle_{region}" for region in regions]
            for source, ds in com.items()
            if source != "Reference"
        ]

        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
