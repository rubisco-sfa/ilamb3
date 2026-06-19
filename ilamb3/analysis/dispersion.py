"""
The ILAMB dispersion methodology.

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

import ilamb3.plot as ilp
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name, scalarify
from ilamb3.exceptions import AnalysisNotAppropriate


def _compute_binned_distribution(ds: xr.Dataset, varname: str, bins) -> xr.DataArray:
    time_dim = dset.get_dim_name(ds, "time")
    var = ds[varname]
    out = xr.concat(
        [
            ((var >= bins[i]) * (var < bins[i + 1])).sum(dim=time_dim)
            for i in range(len(bins) - 1)
        ],
        dim="bin",
    )
    return out


def _hellinger_score(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
    """
    Compute the Hellinger score (1-distance) between two distributions.

    Note
    ----
    The Hellinger distance is a statistical measures used to quantify the similarity
    between two probability distributions, returning values between 0 and 1.

    https://en.wikipedia.org/wiki/Hellinger_distance

    In ILAMB we seek to synthesize performance in the reciprocal sense, where 1 is
    perfect and 0 is poor so we return 1-distance and call it a score.
    """
    if "dist" not in ds1 or "dist" not in ds2:
        raise ValueError(
            "The 'dist' variable created by `_compute_binned_distribution` must be in both input Datasets"
        )
    ds1, ds2 = cmp.nest_spatial_grids(ds1, ds2)
    da1_norm = ds1["dist"] / ds1["dist"].sum(dim="bin")
    da2_norm = ds2["dist"] / ds2["dist"].sum(dim="bin")
    score = 1 - np.sqrt(
        ((np.sqrt(da1_norm) - np.sqrt(da2_norm)) ** 2).sum(dim="bin")
    ) / np.sqrt(2)
    score.attrs["units"] = "1"
    ds2["hellingerscore"] = score
    ds2 = ds2.drop_vars(["dist", "cell_measures"], errors="ignore")
    return ds2


def _create_scalar_dataframe(
    ref: xr.Dataset,
    com: xr.Dataset,
    regions: list[str | None],
    plot_to_name: dict[str, str],
    analysis_name: str,
) -> pd.DataFrame:
    df = []
    for region in regions:
        for var_name, scalar_name in plot_to_name.items():
            for src, ds in zip(["Reference", "Comparison"], [ref, com]):
                if var_name not in ds:
                    continue
                val, unit = scalarify(ds, str(var_name), region, mean=True)
                df += [
                    {
                        "source": src,
                        "region": str(region),
                        "analysis": analysis_name,
                        "name": scalar_name,
                        "type": "score" if "score" in var_name else "scalar",
                        "units": unit,
                        "value": val,
                    },
                ]
    return pd.DataFrame(df)


def _time_extent_years(ds) -> float:
    t0, tf = dset.get_time_extent(ds)
    return float((tf - t0).dt.total_seconds()) / 86400 / 365


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
        nbins: int = 25,
        regions: list[str | None] = [None],
        **kwargs: Any,
    ):
        self.req_variable = required_variable
        self.required_num_years = required_num_years
        self.nbins = nbins
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

    def name(self) -> str:
        return "Dispersion"

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the methodology on the given datasets.

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
        if (
            _time_extent_years(ref) < self.required_num_years
            or _time_extent_years(com) < self.required_num_years
        ):
            raise AnalysisNotAppropriate()

        # Compute and compare maps of dispersion metrics
        bins = np.linspace(
            min(float(ref[varname].min()), float(com[varname].min())),
            max(float(ref[varname].max()), float(com[varname].max())),
            self.nbins + 1,
        )
        ref["dist"] = _compute_binned_distribution(ref, varname, bins)
        com["dist"] = _compute_binned_distribution(com, varname, bins)
        ref = ref.drop_vars(varname)
        com = com.drop_vars(varname)
        com_nested = _hellinger_score(ref, com)
        com = xr.merge([com, com_nested])

        # Compute scalars
        df = _create_scalar_dataframe(
            ref,
            com,
            self.regions,
            {
                "hellingerscore": "Dispersion Score",
            },
            self.name(),
        )

        return df, ref, com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        if self.name() not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)
        regions = [None if r == "None" else r for r in df["region"].unique()]

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {
                    "name": "hellingerscore",
                    "cmap": "plasma",
                    "title": "Dispersion Score",
                },
            ]
        ).set_index("name")

        com["Reference"] = ref
        df_limits = ilp.determine_plot_limits(com)
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
