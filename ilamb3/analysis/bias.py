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
    """
    Compute the difference and score between the reference and comparison datasets.

    Parameters
    ----------
    ref : xr.Dataset
        The reference dataset containing the variable to compare and its uncertainty.
    com : xr.Dataset
        The comparison dataset containing the variable to compare.
    varname : str
        The name of the variable to compare in both datasets.
    error_normalization : xr.DataArray
        The data array to use for normalizing the error, such as the reference mean,
        the reference standard deviation, or a quantile map.
    ref_uncertainty : xr.DataArray
        The uncertainty associated with the reference dataset variable, used to discount
        the error.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.

    Returns
    -------
    xr.Dataset
        A dataset containing diff_{varname} and "score_{varname}"
    """

    # Regrid ref and com in place
    if dset.is_gridded(ref[varname]) and dset.is_gridded(com[varname]):
        ref, com, error_normalization, ref_uncertainty = cmp.nest_spatial_grids(
            ref, com, error_normalization, ref_uncertainty
        )

    # Ensure the dimension names match for all inputs before getting difference
    ref, com, error_normalization, ref_uncertainty = cmp.rename_dims(
        ref, com, error_normalization, ref_uncertainty.fillna(0)
    )

    # Get per-pixel difference scalars for the variable of interest
    diff = com[varname] - ref[varname]

    # Calculate per-pixel bias score using specified method
    discounted_diff = (np.abs(diff) - ref_uncertainty).clip(0)
    relative_error = discounted_diff / np.abs(error_normalization)
    match method:
        case "Collier2018":
            score = np.exp(-relative_error)
        case "RegionalQuantiles":
            score = (1.0 - relative_error).clip(0, 1)
        case _:
            raise ValueError(f"Unknown method: {method}")

    # Create the output dataset with diff scalar, score scalar, and renamed lat/lon dims
    out = xr.Dataset({f"diff_{varname}": diff, f"score_{varname}": score})
    if dset.is_gridded(out[f"diff_{varname}"]):
        # Rename lat and lon to generic names for plotting purposes
        out = out.rename(
            {
                dset.get_dim_name(out, "lat"): "lat_nested",
                dset.get_dim_name(out, "lon"): "lon_nested",
            }
        )
    out[f"score_{varname}"].attrs["units"] = 1
    return out


def get_weights(
    ref: xr.Dataset, com: xr.Dataset, varname: str, scorename: str, nested: xr.Dataset
) -> xr.DataArray:
    """Get the weights for score spatial integration if mass weighting is enabled."""
    if dset.is_gridded(com[scorename]):
        weight = (
            ref[varname]
            .rename(
                {
                    dset.get_dim_name(ref, "lat"): "lat_nested",
                    dset.get_dim_name(ref, "lon"): "lon_nested",
                }
            )
            .interp_like(nested, method="nearest")
        )
    else:
        weight = ref[varname]  # sites and therefore does not need to be interpolated
    return weight


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
    seasons : list
        A list of season strings to calculate difference (bias) scalars. For valid
        strings, see :class:`xr.groupers.SeasonResampler`.
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
        seasons: list[str] | None = None,
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
        self.seasons = seasons
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
            The dataset that will be compared to the reference.

        Returns
        -------
        pd.DataFrame
            A dataframe with scalar and score information from the comparison.
        xr.Dataset
            A dataset containing reference gridded information from the comparison.
        xr.Dataset
            A dataset containing comparison gridded information from the comparison.
        """

        # ------------------------------------------------------------------------------
        # 1. Set up parameters
        # ------------------------------------------------------------------------------

        # Checks on the quantile database if used
        varname = self.req_variable
        quantile_map = None
        if self.method == "RegionalQuantiles":
            check_quantile_database(self.quantile_database)
            try:
                quantile_map = create_quantile_map(
                    self.quantile_database,  # type: ignore
                    varname,
                    "bias",
                    self.quantile_threshold,
                )
                quantile_map = dset.convert(quantile_map, ref[varname].attrs["units"])
            except NoDatabaseEntry:
                # Fallback if the variable/type/quantile is not in the database
                self.method = "Collier2018"

        # Never mass weight if regional quantiles are used
        if self.method == "RegionalQuantiles":
            self.mass_weighting = False

        # ------------------------------------------------------------------------------
        # 2.a. Create mean Datasets, uncertainty DataArray, & error normalizer DataArray
        # ------------------------------------------------------------------------------

        # Ensure ref and com are comparable
        ref, com = cmp.make_comparable(ref, com, varname, **self.kwargs)

        # Instantiate the two output xr.Datasets
        out_ref = xr.Dataset()
        out_com = xr.Dataset()

        # Create temporal mean xr.Datasets
        out_ref["mean"] = (
            dset.integrate_time(ref, varname, mean=True)
            if dset.is_temporal(ref[varname])
            else ref[varname]  # If no time dim, we can't take mean & that's fine
        )  # Should this raise logger warning? Users won't know if mean wasn't taken
        out_com["mean"] = (
            dset.integrate_time(com, varname, mean=True)
            if dset.is_temporal(com[varname])
            else com[varname]
        )

        # Create error normalizer xr.DataArray depending on chosen method
        error_norm = out_ref["mean"]  # Default error normalizer if no time dim

        # If RegionalQuantiles, use the quantile map on the comparison grid
        if self.method == "RegionalQuantiles" and quantile_map is not None:
            error_norm = quantile_map.rename(
                {
                    dset.get_dim_name(quantile_map, "lat"): dset.get_dim_name(
                        out_com, "lat"
                    ),
                    dset.get_dim_name(quantile_map, "lon"): dset.get_dim_name(
                        out_com, "lon"
                    ),
                }
            ).interp_like(out_com, method="nearest")

        # Otherwise, if temporal, normalize by the standard deviation of the reference
        elif (
            dset.is_temporal(ref[varname])
            and ref[dset.get_dim_name(ref[varname], "time")].size > 1
        ):
            error_norm = dset.std_time(ref, varname)

        # Get the reference data uncertainty if present and desired
        uncert = xr.zeros_like(out_ref["mean"])  # Default uncertainty is 0
        if self.use_uncertainty:
            try:
                uncert = dset.get_scalar_uncertainty(ref, varname)
            except (NoUncertainty, ValueError):
                self.use_uncertainty = False

        # Integrate uncertainty over time like we did with mean
        if dset.is_temporal(uncert):
            uncert = dset.integrate_time(uncert, mean=True)

        # Carry it into out_ref so plots() can pick it up via com["Reference"]
        if self.use_uncertainty:
            out_ref["uncert"] = uncert

        # ------------------------------------------------------------------------------
        # 2.b. Calculate scalars/score
        # ------------------------------------------------------------------------------

        # Now score the difference and merge with the comparison output
        out_nested = evaluate_difference(
            out_ref,
            out_com,
            "mean",
            error_norm,  # type: ignore
            uncert,
            self.method,  # type: ignore
        )
        # Rename diff_{varname} and score_{varname} to bias and biasscore
        out_nested = out_nested.rename(
            {
                k: str(k)
                .replace("diff_", "bias_", 1)
                .replace("score_", "biasscore_", 1)
                for k in out_nested
            }
        )
        out_com = xr.merge([out_com, out_nested], compat="override")

        # Before doing regional integration, gather cell_measures if present
        if "cell_measures" in ref:
            out_ref["cell_measures"] = ref["cell_measures"]
        if "cell_measures" in com:
            out_com["cell_measures"] = com["cell_measures"]

        # Get the weights for score spatial integration if mass weighting is enabled
        weight = None
        if self.mass_weighting:
            weight = get_weights(out_ref, out_com, "mean", "biasscore_mean", out_nested)

        # ------------------------------------------------------------------------------
        # 3.a. Create mean Datasets and uncertainty/norm DataArrays for seasons
        # ------------------------------------------------------------------------------

        # If requested, get means/norm/uncertainty for each season as well
        out_ref_season = None
        out_com_season = None

        if self.seasons:
            out_ref_season = xr.Dataset()
            out_com_season = xr.Dataset()

            # Create seasonal mean xr.Datasets
            out_ref_temp = (
                dset.compute_seasonal_climatology(ref, self.seasons, varname)
                if dset.is_temporal(ref[varname])
                else ref[varname]
            )
            out_com_temp = (
                dset.compute_seasonal_climatology(com, self.seasons, varname)
                if dset.is_temporal(com[varname])
                else com[varname]
            )
            # Make sure the season coordinates are never persisted because...
            # We'll want to merge seasonal means back into the period mean dataset later
            for season in self.seasons:
                da_ref = (
                    out_ref_temp[varname]
                    .sel(season=season)
                    .drop_vars("season", errors="ignore")
                    if isinstance(out_ref_temp, xr.Dataset)
                    and "season" in out_ref_temp.dims
                    else out_ref_temp[season]
                )
                da_com = (
                    out_com_temp[varname]
                    .sel(season=season)
                    .drop_vars("season", errors="ignore")
                    if isinstance(out_com_temp, xr.Dataset)
                    and "season" in out_com_temp.dims
                    else out_com_temp[season]
                )
                out_ref_season[f"{season}_mean"] = da_ref
                out_com_season[f"{season}_mean"] = da_com

            # Set dummy normalizer and uncertainty because they are used for score
            # And we won't bother with scores for seasonal means
            error_norm_season = xr.ones_like(out_ref_season[f"{self.seasons[0]}_mean"])
            uncert_season = xr.zeros_like(out_ref_season[f"{self.seasons[0]}_mean"])

            # --------------------------------------------------------------------------
            # 3.b. Calculate scalars
            # --------------------------------------------------------------------------

            per_season_dfs = []
            for season in self.seasons:
                out_season_nested = evaluate_difference(
                    out_ref_season,
                    out_com_season,
                    f"{season}_mean",
                    error_norm_season,
                    uncert_season,
                    self.method,  # type: ignore
                )

                # Rename diff_{varname} and score_{varname}
                out_season_nested = out_season_nested.rename(
                    {
                        k: str(k)
                        .replace("diff_", "bias_", 1)
                        .replace("score_", "biasscore_", 1)
                        for k in out_season_nested
                    }
                )

                per_season_dfs.append(out_season_nested)

            # Combine so that each season is a different variable in the same dataset
            out_season_nested = xr.merge(per_season_dfs, compat="override")
            out_com_season = xr.merge(
                [out_com_season, out_season_nested], compat="override"
            )

            # Add seasons vars to out_ref and out_com so they're available for plotting
            out_ref = xr.merge([out_ref, out_ref_season], compat="override")
            out_com = xr.merge([out_com, out_com_season], compat="override")

        # ------------------------------------------------------------------------------
        # 4. Break up results by regions
        # ------------------------------------------------------------------------------

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

            # Seasonal means
            if (
                self.seasons
                and out_ref_season is not None
                and out_com_season is not None
            ):
                for season in self.seasons:
                    for src, var in zip(
                        ["Reference", "Comparison"],
                        [out_ref_season, out_com_season],
                    ):
                        val, unit = scalarify(
                            var,
                            f"{season}_mean",
                            region,
                            not self.spatial_sum,
                            unit=self.table_unit,
                        )
                        dfs.append(
                            [
                                src,
                                str(region),
                                self.name(),
                                f"{season} Mean",
                                "scalar",
                                unit,
                                val,
                            ]
                        )

            # Bias
            val, unit = scalarify(
                out_com, "bias_mean", region, True, unit=self.plot_unit
            )
            dfs.append(
                ["Comparison", str(region), self.name(), "Bias", "scalar", unit, val]
            )

            # Seasonal Bias
            if (
                self.seasons
                and out_ref_season is not None
                and out_com_season is not None
            ):
                for season in self.seasons:
                    val, unit = scalarify(
                        out_com_season,
                        f"bias_{season}_mean",
                        region,
                        True,
                        unit=self.plot_unit,
                    )
                    dfs.append(
                        [
                            "Comparison",
                            str(region),
                            self.name(),
                            f"{season} Bias",
                            "scalar",
                            unit,
                            val,
                        ]
                    )

            # Bias Score
            val, unit = scalarify(
                out_com,
                "biasscore_mean",
                region,
                True,
                weight=weight if self.mass_weighting else None,
                unit=None,  # bias score is unitless
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

        # Now that we have the scalars, we can drop the cell measures
        out_ref = out_ref.drop_vars("cell_measures", errors="ignore")
        out_com = out_com.drop_vars("cell_measures", errors="ignore")
        return dfs, out_ref, out_com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:
        """Create plots for the bias analysis."""

        # If bias analysis was not run, we should skip plotting entirely
        if "Bias" not in df["analysis"].unique():
            return pd.DataFrame()
        path.mkdir(parents=True, exist_ok=True)

        # Pull the plot regions from those found in the scalars
        regions = [None if r == "None" else r for r in df["region"].unique()]

        # Handle units, use the reference units if not given
        plot_unit = (
            ref["mean"].attrs["units"] if self.plot_unit is None else self.plot_unit
        )

        # Gather all the variables we need for plotting and convert to common units
        plot_vars = ["mean", "bias_mean", "uncert"]
        if self.seasons:
            plot_vars += [f"{s}_mean" for s in self.seasons]
            plot_vars += [f"bias_{s}_mean" for s in self.seasons]
        com["Reference"] = ref
        for source, ds in com.items():
            for plot in plot_vars:
                if plot in ds:
                    com[source][plot] = dset.convert(ds[plot], plot_unit)

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "mean", "cmap": self.cmap, "title": "Period Mean"},
                {"name": "bias_mean", "cmap": "seismic", "title": "Bias"},
                {"name": "biasscore_mean", "cmap": "plasma", "title": "Bias Score"},
                {"name": "uncert", "cmap": "Reds", "title": "Uncertainty"},
            ]
        ).set_index("name")
        df_limits = ilp.determine_plot_limits(com)
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = "Bias"

        # Update df_meta for seasonal analyses; same color scales as period mean/bias
        if self.seasons:
            season_rows = []
            for season in self.seasons:
                if "mean" in df.index:
                    row = df.loc["mean"].to_dict()
                    row.update(name=f"{season}_mean", title=f"{season} Mean")
                    season_rows.append(row)
                if "bias_mean" in df.index:
                    row = df.loc["bias_mean"].to_dict()
                    row.update(
                        name=f"bias_{season}_mean",
                        cmap="seismic",
                        title=f"{season} Bias",
                    )
                    season_rows.append(row)
            if season_rows:
                df = pd.concat([df, pd.DataFrame(season_rows).set_index("name")])

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
                    out["path"] = get_plot_name(source, region, str(plot), path)
                    ax = ilp.plot_map(
                        ds[plot],
                        region=region,
                        vmin=row["low"],
                        vmax=row["high"],
                        cmap=row["cmap"],
                        title=f"{source} {row['title']}",
                    )
                    ax.get_figure().savefig(out["path"])  # type: ignore
                    plt.close()
                    df_plots.append(out)
        df_plots = pd.DataFrame(df_plots)
        return df_plots
