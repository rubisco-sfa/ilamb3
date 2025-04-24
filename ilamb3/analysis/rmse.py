"""
The ILAMB RMSE methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as plt
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis, integrate_or_mean, scalarify
from ilamb3.exceptions import AnalysisNotAppropriate, NoUncertainty


class rmse_analysis(ILAMBAnalysis):
    """
    The ILAMB RMSE methodology.

    Parameters
    ----------
    required_variable : str
        The name of the variable to be used in this analysis.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.
    regions : list
        A list of region labels over which to apply the analysis.
    use_uncertainty : bool
        Enable to utilize uncertainty information from the reference product if
        present.
    quantile_dbase : pd.DataFrame
        If using `method='RegionalQuantiles'`, the dataframe containing the
        regional quantiles to be used to score the datasets.
    quantile_threshold : int
        If using `method='RegionalQuantiles'`, the threshold values to use from
        the `quantile_dbase`.

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
        score_basis: Literal["series", "cycle"] = "series",
        regions: list[str | None] = [None],
        use_uncertainty: bool = True,
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
        self.score_basis = score_basis
        self.regions = regions
        self.use_uncertainty = use_uncertainty
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
        Apply the ILAMB RMSE methodology on the given datasets.

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
        ANALYSIS_NAME = "RMSE"
        varname = self.req_variable

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname)

        # Is the time series long enough for this to be meaningful?
        if len(com[dset.get_dim_name(com, "time")]) < 24:
            raise AnalysisNotAppropriate()

        # Before operating on these, compute spatial means
        ds_ref = {}
        ds_com = {}
        for region in self.regions:
            ds_ref[f"trace_{region}"] = integrate_or_mean(
                ref, varname, region, mean=True
            )
            ds_com[f"trace_{region}"] = integrate_or_mean(
                com, varname, region, mean=True
            )

        # Move calendars
        ref = cmp.convert_calendar_monthly_noleap(ref)
        com = cmp.convert_calendar_monthly_noleap(com)

        # Get the reference data uncertainty, only use if present and desired
        args = [ref, com]
        if self.use_uncertainty:
            try:
                uncert = dset.get_scalar_uncertainty(ref, varname)
                args.append(uncert)
            except NoUncertainty:
                self.use_uncertainty = False
                uncert = None

        # Conversions
        if self.use_uncertainty:
            ref, com, uncert = cmp.rename_dims(
                cmp.nest_spatial_grids(ref[varname], com[varname], uncert)
            )
        else:
            ref, com = cmp.rename_dims(
                *cmp.nest_spatial_grids(ref[varname], com[varname])
            )

        # Compute the RMSE and score
        rmse = np.sqrt(dset.integrate_time((com - ref) ** 2, varname, mean=True))
        ref_mean = dset.integrate_time(ref, varname, mean=True)
        com_mean = dset.integrate_time(com, varname, mean=True)
        crmse = np.sqrt(
            dset.integrate_time(
                (
                    np.abs((com - com_mean) - (ref - ref_mean))
                    - (uncert if self.use_uncertainty else 0.0)
                ).clip(min=0)
                ** 2,
                varname,
                mean=True,
            )
        )
        crms = np.sqrt(dset.integrate_time((ref - ref_mean) ** 2, varname, mean=True))
        score = np.exp(-crmse / crms)

        # Load outputs and scalars
        ds_com["rmse"] = rmse
        ds_com["rmsescore"] = score
        df = []
        for region in self.regions:
            val, unit = scalarify(rmse, varname, region, mean=True)
            df += [
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": ANALYSIS_NAME,
                    "name": "RMSE",
                    "type": "scalar",
                    "units": unit,
                    "value": val,
                },
            ]
            val, _ = scalarify(score, varname, region, mean=True)
            df += [
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": ANALYSIS_NAME,
                    "name": "RMSE Score",
                    "type": "score",
                    "units": "1",
                    "value": val,
                },
            ]

        df = pd.DataFrame(df)
        ds_ref = xr.merge([ds_ref])
        ds_com = xr.merge([ds_com])
        return df, ds_ref, ds_com

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
        df.loc["rmse", ["cmap", "title"]] = ["Oranges", "RMSE"]
        df.loc["rmsescore", ["cmap", "title"]] = ["plasma", "RMSE Score"]

        # Build up a dataframe of matplotlib axes
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
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in ["rmse", "rmsescore"]
            for source, ds in com.items()
            for region in regions
        ]

        axs += [
            {
                "name": "trace",
                "title": "Time Series",
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
                        title=f"{source} Time Series",
                        label="",
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in [f"trace_{region}" for region in regions]
            for source, ds in com.items()
            if source != "Reference"
        ]

        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
