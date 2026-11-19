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
from ilamb3.analysis import scoring
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name, scalarify
from ilamb3.analysis.quantiles import check_quantile_database, create_quantile_map
from ilamb3.exceptions import NoDatabaseEntry, NoUncertainty


def nested_difference(
    ref: xr.Dataset, com: xr.Dataset, varname: str, *others: xr.DataArray
) -> tuple[xr.DataArray, xr.Dataset, list[xr.DataArray]]:
    """
    Return the difference between the datasets on their nested grid.

    Parameters
    ----------
    ref : xr.Dataset
        The reference dataset containing the variable to compare.
    com : xr.Dataset
        The comparison dataset containing the variable to compare.
    varname : str
        The name of the variable to compare in both datasets.
    *others : xr.DataArray
        Data arrays on the grid of ref or com, such as a normalizer, to put on the
        nested grid as well.

    Returns
    -------
    xr.DataArray
        The difference ``com - ref``.
    xr.Dataset
        The comparison on the nested grid, whose bounds give the area of each cell.
    list of xr.DataArray
        The other data arrays on the same grid.
    """
    # Regrid ref and com in place. The others are on the grid of ref or com, so
    # including them does not change the nested grid.
    if dset.is_gridded(ref[varname]) and dset.is_gridded(com[varname]):
        ref, com, *nested = cmp.nest_spatial_grids(ref, com, *others)
        # Datasets are returned, select the da
        others = tuple(ds[next(iter(ds))] for ds in nested)
    diff = com[varname] - ref[varname]
    diff.attrs["units"] = ref[varname].attrs["units"]
    return diff, com, list(others)


def evaluate_difference(
    ref: xr.Dataset,
    com: xr.Dataset,
    varname: str,
    error_normalization: xr.DataArray,
    ref_uncertainty: xr.DataArray,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Compute the difference between the datasets and the statistics of its error.

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

    Returns
    -------
    xr.Dataset
        A dataset containing "diff_{varname}".
    xr.Dataset
        The statistics of the error, see `ilamb3.analysis.scoring.statistics`.
    """
    diff, grid, (norm, uncert) = nested_difference(
        ref, com, varname, error_normalization, ref_uncertainty
    )
    # The error outside the uncertainty, relative to the normalizer
    error = (np.abs(diff) - uncert).clip(0).assign_attrs(diff.attrs)
    stats = scoring.statistics(error, norm, grid=grid)
    return xr.Dataset({f"diff_{varname}": diff}), stats


def get_weights(ref: xr.Dataset, varname: str, nested: xr.Dataset) -> xr.DataArray:
    """Get the weight of each cell of the nested grid, if mass weighting is enabled."""
    if dset.is_gridded(nested):
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

        # Don't do seasons if not temporal or if there are not at least a year of data
        if self.seasons and not dset.is_temporal(ref):
            self.seasons = None
        if self.seasons:
            tmin, tmax = dset.get_time_extent(ref)
            trange = tmax - tmin
            if trange <= np.timedelta64(359, "D"):
                self.seasons = None

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
        # A label for the normalizer, so the Overall Score adds only errors that share it
        norm_label = scoring.REFERENCE_MEAN

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
            norm_label = scoring.REGIONAL_QUANTILE

        # Otherwise, if temporal, normalize by the standard deviation of the reference
        elif (
            dset.is_temporal(ref[varname])
            and ref[dset.get_dim_name(ref[varname], "time")].size > 1
        ):
            error_norm = dset.std_time(ref, varname)
            norm_label = scoring.TIME_VARIANCE

        # Get the reference data uncertainty if present and desired
        uncert = xr.zeros_like(out_ref["mean"])  # Default uncertainty is 0
        if self.use_uncertainty:
            try:
                ds_unc = dset.get_scalar_uncertainty(ref, varname)
                if dset.is_temporal(ds_unc):
                    uncert = dset.integrate_time(ds_unc, varname="uncert", mean=True)
                else:
                    uncert = ds_unc["uncert"]
            except (NoUncertainty, ValueError):
                self.use_uncertainty = False

        # Carry it into out_ref so plots() can pick it up via com["Reference"]
        if self.use_uncertainty:
            out_ref["uncert"] = uncert

        # ------------------------------------------------------------------------------
        # 2.b. Calculate scalars/score
        # ------------------------------------------------------------------------------

        # Now score the difference and merge with the comparison output
        out_nested, stats = evaluate_difference(
            out_ref,
            out_com,
            "mean",
            error_norm,  # type: ignore
            uncert,
        )
        # Weight the statistics, so the saved statistics reproduce the weighted scores
        if self.mass_weighting:
            stats = scoring.weighted(stats, get_weights(out_ref, "mean", stats))
        out_nested = scoring.add_score(
            out_nested.rename({"diff_mean": "bias"}),
            "biasscore",
            stats,
            self.method,  # type: ignore
        )
        out_com = xr.merge([out_com, out_nested], compat="override")

        # Before doing regional integration, gather cell_measures if present
        if "cell_measures" in ref:
            out_ref["cell_measures"] = ref["cell_measures"]
        if "cell_measures" in com:
            out_com["cell_measures"] = com["cell_measures"]

        # ------------------------------------------------------------------------------
        # 3.a. Create mean Datasets and uncertainty/norm DataArrays for seasons
        # ------------------------------------------------------------------------------

        # If requested, get means/norm/uncertainty for each season as well
        out_ref_season = None
        out_com_season = None

        # Only do seasons if asked, if temporal, and ensure there are at least 12 months
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
                out_ref_season[f"mean{season}"] = da_ref
                out_com_season[f"mean{season}"] = da_com

            # --------------------------------------------------------------------------
            # 3.b. Calculate scalars
            # --------------------------------------------------------------------------

            per_season_dfs = []
            for season in self.seasons:
                diff, _, _ = nested_difference(
                    out_ref_season, out_com_season, f"mean{season}"
                )
                per_season_dfs.append(diff.to_dataset(name=f"bias{season}"))

            # Combine so that each season is a different variable in the same dataset
            out_season_nested = xr.merge(per_season_dfs, compat="override")
            out_com_season = xr.merge(
                [out_com_season, out_season_nested], compat="override"
            )

            # Add seasons vars to out_ref and out_com so they're available for plotting
            out_ref = xr.merge([out_ref, out_ref_season], compat="override")
            out_com = xr.merge([out_com, out_com_season], compat="override")

        else:
            self.seasons = None  # prevent plots() from using seasonal info in this case

        # ------------------------------------------------------------------------------
        # 4. Break up results by regions
        # ------------------------------------------------------------------------------

        # Compute scalars over all regions
        dfs = []
        scores = []
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
                            f"mean{season}",
                            region,
                            not self.spatial_sum,
                            unit=self.table_unit,
                        )
                        dfs.append(
                            [
                                src,
                                str(region),
                                f"{self.name()} {season}",
                                f"Mean {season}",
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

            # Seasonal Bias
            if (
                self.seasons
                and out_ref_season is not None
                and out_com_season is not None
            ):
                for season in self.seasons:
                    val, unit = scalarify(
                        out_com_season,
                        f"bias{season}",
                        region,
                        True,
                        unit=self.plot_unit,
                    )
                    dfs.append(
                        [
                            "Comparison",
                            str(region),
                            f"{self.name()} {season}",
                            f"Bias {season}",
                            "scalar",
                            unit,
                            val,
                        ]
                    )

            # Bias Score, aggregating the statistics over the region before scoring
            scores.append(
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": self.name(),
                    "name": "Bias Score",
                    **scoring.score_row(
                        scoring.aggregate(stats, region),
                        self.method,  # type: ignore
                        normalizer=norm_label,
                    ),
                }
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
        dfs = pd.concat([dfs, pd.DataFrame(scores)], ignore_index=True)

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

        # Convert plotting variables to common units
        plot_vars = ["mean", "bias", "uncert"]  # biasscore is unitless
        if self.seasons:
            plot_vars += [f"mean{s}" for s in self.seasons]  # seasonal means
            plot_vars += [f"bias{s}" for s in self.seasons]  # seasonal bias scalars
        com["Reference"] = ref
        for source, ds in com.items():
            for plot in plot_vars:
                if plot in ds:
                    com[source][plot] = dset.convert(ds[plot], plot_unit)

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {
                    "name": "mean",
                    "cmap": self.cmap,
                    "title": "Period Mean",
                    "analysis": self.name(),
                },
                {
                    "name": "bias",
                    "cmap": "seismic",
                    "title": "Bias",
                    "analysis": self.name(),
                },
                {
                    "name": "biasscore",
                    "cmap": "plasma",
                    "title": "Bias Score",
                    "analysis": self.name(),
                },
                {
                    "name": "uncert",
                    "cmap": "Reds",
                    "title": "Uncertainty",
                    "analysis": self.name(),
                },
            ]
            + [
                {
                    "name": f"mean{s}",
                    "cmap": self.cmap,
                    "title": f"Mean {s}",
                    "analysis": f"{self.name()} {s}",
                }
                for s in self.seasons or []
            ]
            + [
                {
                    "name": f"bias{s}",
                    "cmap": "seismic",
                    "title": f"Bias {s}",
                    "analysis": f"{self.name()} {s}",
                }
                for s in self.seasons or []
            ]
        ).set_index("name")
        df_limits = ilp.determine_plot_limits(
            com, symmetrize=[p for p in plot_vars if p.startswith("bias")]
        )
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)

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
