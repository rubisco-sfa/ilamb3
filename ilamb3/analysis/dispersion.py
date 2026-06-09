"""
The ILAMB dispersion methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import ilamb3.plot as ilp
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name, scalarify
from ilamb3.analysis.bias import evaluate_difference
from ilamb3.exceptions import AnalysisNotAppropriate


def _compute_dispersion_stats(
    ds: xr.Dataset, varname: str, quantiles: list[float]
) -> xr.Dataset:
    time_dim = dset.get_dim_name(ds, "time")
    ds["mean"] = dset.integrate_time(ds, varname, mean=True)
    ds["stdev"] = dset.std_time(ds, varname)
    ds["qs"] = ds[varname].quantile(q=quantiles, dim=time_dim)
    ds["iqr"] = ds["qs"].sel(quantile=0.75) - ds["qs"].sel(quantile=0.25)
    ds["skewness"] = (ds["mean"] - ds["qs"].sel(quantile=0.5)) / ds["stdev"]
    ds["kurtosis"] = ((ds[varname] - ds["mean"]) ** 4).mean(dim=time_dim) / (
        ds["stdev"] ** 4
    )
    # Correct units
    for var in ["skewness", "kurtosis"]:
        ds[var].attrs["units"] = "1"
    # Expand the quantiles out into separate variables with a better name
    for q in ds["quantile"]:
        ds[f"q{round(100 * float(q), 1):g}"] = ds["qs"].sel(quantile=q)
    # Drop the variables we will not need
    ds = ds.drop_vars(
        [
            varname,
            time_dim,
            "mean",
            "stdev",
            "qs",
            ds[time_dim].attrs.get("bounds", ""),
        ],
        errors="ignore",
    )
    return ds


class dispersion_analysis(ILAMBAnalysis):
    """
    The ILAMB dispersion methodology.

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
        required_num_years: int = 10,
        quantiles: list[float] = [
            0.001,
            0.01,
            0.05,
            1 / 8,
            1 / 6,
            1 / 4,
            1 / 3,
            1 / 2,
            2 / 3,
            3 / 4,
            5 / 6,
            7 / 8,
            9 / 10,
            0.95,
            0.99,
            0.999,
            1.0,
        ],
        regions: list[str | None] = [None],
        **kwargs: Any,
    ):
        self.req_variable = required_variable
        self.regions = regions
        self.required_num_years = required_num_years
        self.quantiles = quantiles
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

    def name(self) -> str:
        return "Dispersion"

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
        varname = self.req_variable
        if not (dset.is_temporal(ref[varname]) and dset.is_temporal(com[varname])):
            raise AnalysisNotAppropriate()

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname, **self.kwargs)

        # Is the time series long enough for this to be meaningful?
        def _time_extent_years(ds) -> float:
            t0, tf = dset.get_time_extent(ds)
            return float((tf - t0).dt.total_seconds()) / 86400 / 365

        if (
            _time_extent_years(ref) < self.required_num_years
            or _time_extent_years(ref) < self.required_num_years
        ):
            raise AnalysisNotAppropriate()

        # Compute and compare maps of dispersion metrics
        ref = _compute_dispersion_stats(ref, varname, self.quantiles)
        com = _compute_dispersion_stats(com, varname, self.quantiles)
        com_nested = xr.merge(
            [
                evaluate_difference(
                    ref,
                    com,
                    str(var),
                    xr.ones_like(ref[var]),
                    xr.zeros_like(ref[var]),
                    method="RegionalQuantiles",
                )
                for var in com
                if var not in ["cell_measures"]
            ]
        )
        com_nested = com_nested.rename({v: str(v).replace("_", "") for v in com_nested})
        com = xr.merge([com, com_nested], compat="override")

        # Compute scalars
        def _plot_to_title(name) -> str:
            if name.startswith("diff"):
                return f"{name.replace('diff', '').capitalize()} Difference"
            if name.startswith("score"):
                return f"{name.replace('score', '').capitalize()} Score"
            if name == "iqr":
                return "Interquantile Range"
            return name.capitalize()

        df = []
        for region in self.regions:
            for name in ["kurtosis", "skewness", "iqr"]:
                for src, ds in zip(["Reference", "Comparison"], [ref, com]):
                    val, unit = scalarify(ds, str(name), region, mean=True)
                    df += [
                        {
                            "source": src,
                            "region": str(region),
                            "analysis": self.name(),
                            "name": _plot_to_title(name),
                            "type": "scalar",
                            "units": unit,
                            "value": val,
                        },
                    ]

        df = pd.DataFrame(df)
        return df, ref, com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        if self.name() not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)
        regions = [None if r == "None" else r for r in df["region"].unique()]
        com["Reference"] = ref

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "kurtosis", "cmap": "PuRd", "title": "Kurtosis"},
                {
                    "name": "diffkurtosis",
                    "cmap": "seismic",
                    "title": "Kurtosis Difference",
                },
                {"name": "skewness", "cmap": "YlGn", "title": "Skewness"},
                {"name": "iqr", "cmap": "YlGnBu", "title": "Interquantile Range"},
            ]
        ).set_index("name")

        df_limits = ilp.determine_plot_limits(
            com, symmetrize=["diffskewness", "diffkurtosis", "diffiqr"]
        )
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = self.name()

        # Create each plot for each source if present in the dataset
        df_plots = []
        for plot, row in df.iterrows():
            for source, ds in com.items():
                if plot not in ds:
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
