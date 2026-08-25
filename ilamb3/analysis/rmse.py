"""
The ILAMB RMSE methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as ilp
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import (
    ILAMBAnalysis,
    get_plot_name,
    integrate_or_mean,
    scalarify,
)
from ilamb3.exceptions import AnalysisNotAppropriate, NoUncertainty


def evaluate_rmse(
    ref: xr.Dataset,
    com: xr.Dataset,
    varname: str,
    ref_uncertainty: xr.DataArray,
    method: Literal["Collier2018", "RegionalQuantiles"],
) -> xr.Dataset:
    """
    Compute the rmse and score between the reference and comparison datasets.

    Parameters
    ----------
    ref : xr.Dataset
        The reference dataset containing the variable to compare and its uncertainty.
    com : xr.Dataset
        The comparison dataset containing the variable to compare.
    varname : str
        The name of the variable to compare in both datasets.
    ref_uncertainty : xr.DataArray
        The uncertainty associated with the reference dataset variable, used to discount
        the error.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.

    Returns
    -------
    xr.Dataset
        A dataset containing "rmse_{varname}" and "score_{varname}"
    """

    # Regrid ref and com in place
    if dset.is_gridded(ref[varname]) and dset.is_gridded(com[varname]):
        ref, com, ref_uncertainty = cmp.nest_spatial_grids(ref, com, ref_uncertainty)
        # Datasets are returned, select the da
        ref_uncertainty = ref_uncertainty[next(iter(ref_uncertainty))]

    # Compute the centralized difference, and then take the difference
    ref_c = ref[varname] - dset.integrate_time(ref, varname, mean=True)
    com_c = com[varname] - dset.integrate_time(com, varname, mean=True)
    diff = com_c - ref_c

    # Calculate per-pixel rmse and score using specified method
    discounted_diff = (np.abs(diff) - ref_uncertainty).clip(0)
    rmse = np.sqrt(dset.integrate_time((com[varname] - ref[varname]) ** 2, mean=True))
    centralized_rms = np.sqrt(dset.integrate_time(ref_c**2, mean=True))
    centralized_rmse = np.sqrt(dset.integrate_time(discounted_diff**2, mean=True))
    relative_error = centralized_rmse / centralized_rms
    match method:
        case "Collier2018":
            score = np.exp(-relative_error)
        case "RegionalQuantiles":
            raise NotImplementedError()
        case _:
            raise ValueError(f"Unknown method: {method}")

    # Create the output dataset with diff scalar, score scalar
    out = xr.Dataset({f"rmse_{varname}": rmse, f"score_{varname}": score})
    out[f"rmse_{varname}"].attrs["units"] = ref[varname].attrs["units"]
    out[f"score_{varname}"].attrs["units"] = 1
    return out


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
        table_unit: str | None = None,
        plot_unit: str | None = None,
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
        self.score_basis = score_basis
        self.regions = regions
        self.use_uncertainty = use_uncertainty
        self.table_unit = table_unit
        self.plot_unit = plot_unit
        self.kwargs = kwargs

    def name(self) -> str:
        """
        Return the name of this analysis.

        Returns
        -------
        str
            The name of this analysis.
        """
        return "RMSE"

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
        varname = self.req_variable

        if not (dset.is_temporal(ref[varname]) and dset.is_temporal(com[varname])):
            raise AnalysisNotAppropriate()

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname, **self.kwargs)

        # Is the time series long enough for this to be meaningful?
        if len(ref[dset.get_dim_name(ref, "time")]) < 24:
            raise AnalysisNotAppropriate()
        if len(com[dset.get_dim_name(com, "time")]) < 24:
            raise AnalysisNotAppropriate()

        # Before operating on these, compute spatial means
        out_ref = xr.Dataset()
        out_com = xr.Dataset()
        for region in self.regions:
            out_ref[f"trace_{region}"] = integrate_or_mean(
                ref, varname, region, mean=True
            )
            out_com[f"trace_{region}"] = integrate_or_mean(
                com, varname, region, mean=True
            )

        # Unify the calendars
        ref = cmp.convert_calendar_monthly_noleap(ref)
        com = cmp.convert_calendar_monthly_noleap(com)

        # Get the reference data uncertainty, only use if present and desired
        uncert = xr.zeros_like(ref[varname])  # Default uncertainty is 0
        if self.use_uncertainty:
            try:
                uncert = dset.get_scalar_uncertainty(ref, varname)
            except (NoUncertainty, ValueError):
                self.use_uncertainty = False

        out_nested = evaluate_rmse(ref, com, varname, uncert, "Collier2018")
        out_nested = out_nested.rename(
            {k: k.split("_")[0].replace("score", "rmsescore", 1) for k in out_nested}
        )
        out_com = xr.merge([out_com, out_nested], compat="override")

        df = []
        for region in self.regions:
            val, unit = scalarify(
                out_com["rmse"], "rmse", region, mean=True, unit=self.plot_unit
            )
            df += [
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": self.name(),
                    "name": "RMSE",
                    "type": "scalar",
                    "units": unit,
                    "value": val,
                },
            ]
            val, _ = scalarify(out_com["rmsescore"], "rmsescore", region, mean=True)
            df += [
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": self.name(),
                    "name": "RMSE Score",
                    "type": "score",
                    "units": "1",
                    "value": val,
                },
            ]

        df = pd.DataFrame(df)
        return df, out_ref, out_com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        # This analysis was not run and we should skip plotting entirely
        if self.name() not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)

        # Pull the plot regions from those found in the scalars
        regions = [None if r == "None" else r for r in df["region"].unique()]
        traces = [f"trace_{region}" for region in regions]

        # Handle units
        _, ds = next(iter(com.items()))
        plot_unit = (
            ds["rmse"].attrs["units"] if self.plot_unit is None else self.plot_unit
        )
        com["Reference"] = ref
        for source, ds in com.items():
            for plot in ["rmse"] + traces:
                if plot in ds:
                    com[source][plot] = dset.convert(ds[plot], plot_unit)

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "rmse", "cmap": "Oranges", "title": "RMSE"},
                {"name": "rmsescore", "cmap": "plasma", "title": "RMSE Score"},
            ]
            + [
                {"name": trace, "cmap": None, "title": "Time Series"}
                for trace in traces
            ]
        ).set_index("name")
        df_limits = ilp.determine_plot_limits(com)
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = self.name()

        # Create each plot for each source if present in the dataset
        df_plots = []
        for plot, row in df.iterrows():
            for source, ds in com.items():
                if plot not in ds:
                    continue
                # trace plots have already been regionalized
                if plot.startswith("trace"):
                    # Reference traces don't get plots of their own
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
                        label="",
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
                    )
                    ax.get_figure().savefig(out["path"])
                    plt.close()
                    df_plots.append(out)

        df_plots = pd.DataFrame(df_plots)
        return df_plots
