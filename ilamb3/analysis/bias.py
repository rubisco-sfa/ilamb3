"""
The ILAMB bias methodology.

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
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name, scalarify
from ilamb3.analysis.quantiles import check_quantile_database, create_quantile_map
from ilamb3.exceptions import NoDatabaseEntry, NoUncertainty


def evaluate_difference(
    ref: xr.Dataset,
    com: xr.Dataset,
    varname: str,
    error_normalization: xr.DataArray,
    ref_uncertainty: xr.DataArray,
    method: Literal["Collier2018", "RegionalQuantiles"],
) -> xr.Dataset:

    # This just overwrites the inputs into their nested grid variants
    if dset.is_spatial(ref[varname]) and dset.is_spatial(com[varname]):
        ref, com, error_normalization, ref_uncertainty = cmp.nest_spatial_grids(
            ref, com, error_normalization, ref_uncertainty
        )
    # To subtract arrays, dimension names must match
    ref, com, error_normalization, ref_uncertainty = cmp.rename_dims(
        ref, com, error_normalization, ref_uncertainty.fillna(0)
    )
    # Evaluate the difference in stages
    diff = com[varname] - ref[varname]
    discounted_diff = (np.abs(diff) - ref_uncertainty).clip(0)
    relative_error = discounted_diff / np.abs(error_normalization)
    # Scores are still dependent on the method
    match method:
        case "Collier2018":
            score = np.exp(-relative_error)
        case "RegionalQuantiles":
            score = (1.0 - relative_error).clip(0, 1)
        case _:
            raise ValueError(f"Unknown method: {method}")
    score.attrs["units"] = 1
    # Build up and return the results, rename the lat's and lon's so they can be
    # merged uniquely
    out = xr.Dataset({f"diff_{varname}": diff, f"score_{varname}": score})
    if dset.is_spatial(out[f"diff_{varname}"]):
        out = out.rename(
            {
                dset.get_dim_name(out, "lat"): "lat_",
                dset.get_dim_name(out, "lon"): "lon_",
            }
        )
    return out


class bias_analysis(ILAMBAnalysis):
    """
    The ILAMB bias methodology.

    Parameters
    ----------
    required_variable : str
        The name of the variable to be used in this analysis.
    variable_cmap : str
        The colormap to use in plots of the comparison variable, optional.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.
    regions : list
        A list of region labels over which to apply the analysis.
    use_uncertainty : bool
        Enable to utilize uncertainty information from the reference product if
        present.
    spatial_sum : bool
        Enable to report a spatial sum in the period mean as opposed to a
        spatial mean. This is often preferred in carbon variables where the
        total global carbon is of interest.
    mass_weighting : bool
        Enable to weight the score map integrals by the temporal mean of the
        reference dataset.
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
        variable_cmap: str = "viridis",
        method: Literal["Collier2018", "RegionalQuantiles"] = "Collier2018",
        regions: list[str | None] = [None],
        use_uncertainty: bool = True,
        spatial_sum: bool = False,
        mass_weighting: bool = False,
        quantile_database: pd.DataFrame | None = None,
        quantile_threshold: int = 70,
        table_unit: str | None = None,
        plot_unit: str | None = None,
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
        self.cmap = variable_cmap
        self.method = method
        self.regions = regions
        self.use_uncertainty = use_uncertainty
        self.spatial_sum = spatial_sum
        self.mass_weighting = mass_weighting
        self.quantile_database = quantile_database
        self.quantile_threshold = quantile_threshold
        self.table_unit = table_unit
        self.plot_unit = plot_unit
        self.kwargs = kwargs

    def name(self) -> str:
        """
        Return the name of the analysis.

        Returns
        -------
        str
            The name of the analysis.
        """
        return "Bias"

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
            A dataset containing reference gridded information from the comparison.
        xr.Dataset
            A dataset containing comparison gridded information from the comparison.
        """
        # Checks on the quantile database if used
        varname = self.req_variable
        quantile_map = None
        if self.method == "RegionalQuantiles":
            check_quantile_database(self.quantile_database)
            try:
                quantile_map = create_quantile_map(
                    self.quantile_database, varname, "bias", self.quantile_threshold
                )
                dset.convert(quantile_map, ref[varname].attrs["units"])
            except NoDatabaseEntry:
                # Fallback if the variable/type/quantile is not in the database
                self.method = "Collier2018"
        # Never mass weight if regional quantiles are used
        if self.method == "RegionalQuantiles":
            self.mass_weighting = False

        ref, com = cmp.make_comparable(ref, com, varname, **self.kwargs)

        # Build up the output datasets as we go
        out_ref = xr.Dataset()
        out_com = xr.Dataset()

        # Temporal means across the time period
        out_ref["mean"] = (
            dset.integrate_time(ref, varname, mean=True)
            if dset.is_temporal(ref[varname])
            else ref[varname]
        )
        out_com["mean"] = (
            dset.integrate_time(com, varname, mean=True)
            if dset.is_temporal(com[varname])
            else com[varname]
        )

        # Choose what we will use to normalize the error, defaults to traditional definition
        error_norm = out_ref["mean"]
        if self.method == "RegionalQuantiles" and quantile_map is not None:
            # Use the quantile map if that is what we are doing
            error_norm = quantile_map
        elif (
            dset.is_temporal(ref[varname])
            and ref[dset.get_dim_name(ref[varname], "time")].size > 1
        ):
            # Otherwise normalize by the standard deviation of the reference
            error_norm = dset.std_time(ref, varname)

        # Get the reference data uncertainty if present and desired
        out_ref["uncert"] = xr.zeros_like(out_ref["mean"])
        if self.use_uncertainty:
            try:
                out_ref["uncert"] = dset.get_scalar_uncertainty(ref, varname)
            except (NoUncertainty, ValueError):
                self.use_uncertainty = False
        if dset.is_temporal(out_ref["uncert"]):
            out_ref["uncert"] = dset.integrate_time(out_ref, "uncert", mean=True)

        # Now score the difference and merge with the comparison output.
        out_nested = evaluate_difference(
            ref, com, varname, error_norm, out_ref["uncert"], self.method
        )
        out_nested = out_nested.rename(
            {
                k: k.split("_")[0].replace("diff", "bias").replace("score", "biasscore")
                for k in out_nested
            }
        )
        out_com = xr.merge([out_com, out_nested], compat="override")

        # We are going to integrate over regions in the next step. Values will
        # be more accurate if using the cell measures provided in the original
        # sources if present.
        if "cell_measures" in ref:
            out_ref["cell_measures"] = ref["cell_measures"]
        if "cell_measures" in com:
            out_com["cell_measures"] = com["cell_measures"]

        # If the user has selected to use mass weighting, we will need the
        # reference mean interpolated to the nested grid if sources are gridded.
        if self.mass_weighting:
            if dset.is_spatial(out_com["biasscore"]):
                weight = (
                    out_ref["mean"]
                    .rename(
                        {
                            dset.get_dim_name(out_ref, "lat"): "lat_",
                            dset.get_dim_name(out_ref, "lon"): "lon_",
                        }
                    )
                    .interp_like(out_nested, method="nearest")
                )
            else:
                weight = out_ref[
                    "mean"
                ]  # sites and therefore does not need to be interpolated

        # Compute scalars over all regions
        dfs = []
        for region in self.regions:
            # Period mean
            for src, var in zip(["Reference", "Comparison"], [out_ref, out_com]):
                val, unit = scalarify(
                    var, "mean", region, not self.spatial_sum, unit=self.table_unit
                )
                dfs.append(
                    [
                        src,
                        str(region),
                        self.name(),
                        "Period Mean",
                        "scalar",
                        unit,
                        val,
                    ]
                )
            # Bias
            val, unit = scalarify(out_com, "bias", region, True, unit=self.plot_unit)
            dfs.append(
                ["Comparison", str(region), self.name(), "Bias", "scalar", unit, val]
            )
            # Bias Score
            val, unit = scalarify(
                out_com,
                "biasscore",
                region,
                True,
                weight=weight if self.mass_weighting else None,
            )
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    self.name(),
                    "Bias Score",
                    "score",
                    unit,
                    val,
                ]
            )

        # Convert to dataframe
        dfs = pd.DataFrame(
            dfs,
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

        # Now that we have the scalars, now we can drop the cell measures
        out_ref = out_ref.drop_vars("cell_measures", errors="ignore")
        out_com = out_com.drop_vars("cell_measures", errors="ignore")
        return dfs, out_ref, out_com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:
        # This analysis was not run and we should skip plotting entirely
        if "Bias" not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)

        # Pull the plot regions from those found in the scalars
        regions = [None if r == "None" else r for r in df["region"].unique()]

        # Handle units, use the reference units if not given
        plot_unit = (
            ref["mean"].attrs["units"] if self.plot_unit is None else self.plot_unit
        )
        com["Reference"] = ref
        for source, ds in com.items():
            for plot in ["mean", "bias", "uncert"]:
                if plot in ds:
                    com[source][plot] = dset.convert(ds[plot], plot_unit)

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "mean", "cmap": self.cmap, "title": "Period Mean"},
                {"name": "bias", "cmap": "seismic", "title": "Bias"},
                {"name": "biasscore", "cmap": "plasma", "title": "Bias Score"},
                {"name": "uncert", "cmap": "Reds", "title": "Uncertainty"},
            ]
        ).set_index("name")
        df_limits = ilp.determine_plot_limits(com)
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = "Bias"

        # Create each plot for each source if present in the dataset
        df_plots = []
        for plot, row in df.iterrows():
            for source, ds in com.items():
                if plot not in ds:
                    continue
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
