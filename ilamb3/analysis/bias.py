"""
The ILAMB bias methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as plt
import ilamb3.regions as ilr
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.analysis.quantiles import check_quantile_database, create_quantile_map
from ilamb3.exceptions import NoDatabaseEntry


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
        analysis_name = "Bias"
        varname = self.req_variable
        if self.use_uncertainty and "bounds" not in ref[varname].attrs:
            self.use_uncertainty = False

        # Checks on the quantile database if used
        if self.method == "RegionalQuantiles":
            check_quantile_database(self.quantile_database)
            try:
                quantile_map = create_quantile_map(
                    self.quantile_database, varname, "bias", self.quantile_threshold
                )
                dset.convert(quantile_map, ref[varname].attrs["units"])
            except NoDatabaseEntry:
                # fallback if the variable/type/quantile is not in the database
                self.method = "Collier2018"

        # Never mass weight if regional quantiles are used
        if self.method == "RegionalQuantiles":
            self.mass_weighting = False

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname)

        # Temporal means across the time period
        ref_mean = (
            dset.integrate_time(ref, varname, mean=True)
            if dset.is_temporal(ref[varname])
            else ref[varname]
        )
        com_mean = (
            dset.integrate_time(com, varname, mean=True)
            if dset.is_temporal(com[varname])
            else com[varname]
        )

        # Get the reference data uncertainty
        uncert = xr.zeros_like(ref_mean)
        uncert.attrs["units"] = ref[varname].attrs["units"]
        if self.use_uncertainty:
            uncert = ref[ref[varname].attrs["bounds"]]
            uncert.attrs["units"] = ref[varname].attrs["units"]
            uncert = (
                dset.integrate_time(uncert, mean=True)
                if dset.is_temporal(uncert)
                else uncert
            )

        # If temporal information is available, we normalize the error by the
        # standard deviation of the reference. If not, we revert to the traditional
        # definition of relative error.
        norm = ref_mean
        if dset.is_temporal(ref):
            time_dim = dset.get_dim_name(ref, "time")
            if ref[time_dim].size > 1:
                norm = dset.std_time(ref, varname)

        # Nest the grids for comparison, we postpend composite grid variables with "_"
        if dset.is_spatial(ref) and dset.is_spatial(com):
            ref_, com_, norm_, uncert_ = cmp.nest_spatial_grids(
                ref_mean, com_mean, norm, uncert
            )
        elif dset.is_site(ref) and dset.is_site(com):
            ref_ = ref_mean
            com_ = com_mean
            norm_ = norm
            uncert_ = uncert
        else:
            raise ValueError("Reference and comparison not uniformly site/spatial.")

        # Compute score by different methods
        ref_, com_, norm_, uncert_ = cmp.rename_dims(ref_, com_, norm_, uncert_)
        bias = com_ - ref_
        if self.method == "Collier2018":
            score = np.exp(-(np.abs(bias) - uncert_).clip(0) / norm_)
        elif self.method == "RegionalQuantiles":
            norm = quantile_map.interp(
                lat=bias["lat"], lon=bias["lon"], method="nearest"
            )
            score = (1 - (np.abs(bias) - uncert_).clip(0) / norm).clip(0, 1)
        else:
            msg = (
                "The method used to score the bias must be 'Collier2018' or "
                f"'RegionalQuantiles' but found {self.method=}"
            )
            raise ValueError(msg)

        # Build output datasets
        ref_out = ref_mean.to_dataset(name="mean")
        if self.use_uncertainty:
            ref_out["uncert"] = uncert
        com_out = bias.to_dataset(name="bias")
        com_out["biasscore"] = score
        try:
            lat_name = dset.get_dim_name(com_mean, "lat")
            lon_name = dset.get_dim_name(com_mean, "lon")
            com_mean = com_mean.rename({lat_name: "lat_", lon_name: "lon_"})
        except KeyError:
            pass
        com_out["mean"] = com_mean

        # Either integrate or average depending on the input var
        def _scalar(
            var, varname, region, mean=True, weight=False
        ):  # numpydoc ignore=GL08
            da = var
            if isinstance(var, xr.Dataset):
                da = var[varname]
            if dset.is_spatial(da):
                da = dset.integrate_space(
                    da,
                    varname,
                    region=region,
                    mean=mean,
                    weight=ref_ if (self.mass_weighting and weight) else None,
                )
            elif dset.is_site(da):
                da = ilr.Regions().restrict_to_region(da, region)
                da = da.mean(dim=dset.get_dim_name(da, "site"))
            else:
                raise ValueError(f"Input is neither spatial nor site: {da}")
            da = da.pint.quantify()
            return da

        # Compute scalars over all regions
        dfs = []
        for region in self.regions:
            # Period mean
            for src, var in zip(["Reference", "Comparison"], [ref_mean, com_mean]):
                var = _scalar(var, varname, region, not self.spatial_sum)
                dfs.append(
                    [
                        src,
                        str(region),
                        analysis_name,
                        "Period Mean",
                        "scalar",
                        f"{var.pint.units:~cf}",
                        float(var.pint.dequantify()),
                    ]
                )
            # Bias
            bias_scalar = _scalar(com_out, "bias", region, True)
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    analysis_name,
                    "Bias",
                    "scalar",
                    f"{bias_scalar.pint.units:~cf}",
                    float(bias_scalar.pint.dequantify()),
                ]
            )
            # Bias Score
            bias_scalar_score = _scalar(com_out, "biasscore", region, True, True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "divide by zero encountered in divide", RuntimeWarning
                )
                bias_scalar_score = float(bias_scalar_score.pint.dequantify())
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    analysis_name,
                    "Bias Score",
                    "score",
                    "1",
                    bias_scalar_score,
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
        dfs.attrs = self.__dict__.copy()
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
        df.loc["mean", ["cmap", "title"]] = [self.cmap, "Period Mean"]
        df.loc["bias", ["cmap", "title"]] = ["seismic", "Bias"]
        df.loc["biasscore", ["cmap", "title"]] = ["plasma", "Bias Score"]

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
            for plot in ["mean", "bias", "biasscore"]
            for source, ds in com.items()
            for region in regions
        ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])

        return axs
