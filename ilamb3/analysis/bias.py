"""
The ILAMB bias methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
import xarray as xr

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

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(self, required_variable: str):  # numpydoc ignore=GL08
        self.req_variable = required_variable

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
        method: Literal["Collier2018", "RegionalQuantiles"] = "Collier2018",
        regions: list[Union[str, None]] = [None],
        use_uncertainty: bool = True,
        spatial_sum: bool = False,
        mass_weighting: bool = False,
        quantile_dbase: Union[pd.DataFrame, None] = None,
        quantile_threshold: int = 70,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.
        method : str
            The name of the scoring methodology to use, either `Collier2018` or
            `RegionalQuantiles`.
        regions : list
            A list of region labels over which to apply the analysis.
        use_uncertainty : bool
            Enable to utilize uncertainty information from the reference product if
            present.
        spatial_sum : bool
            Enable to report a spatial sum in the period mean as opposed to a spatial
            mean. This is often preferred in carbon variables where the total global
            carbon is of interest.
        mass_weighting : bool
            Enable to weight the score map integrals by the temporal mean of the
            reference dataset.
        quantile_dbase : pd.DataFrame
            If using `method='RegionalQuantiles'`, the dataframe containing the regional
            quantiles to be used to score the datasets.
        quantile_threshold : int
            If using `method='RegionalQuantiles'`, the threshold values to use from the
            `quantile_dbase`.

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
        if use_uncertainty and "bounds" not in ref[varname].attrs:
            use_uncertainty = False

        # Checks on the database if it is being used
        if method == "RegionalQuantiles":
            check_quantile_database(quantile_dbase)
            try:
                quantile_map = create_quantile_map(
                    quantile_dbase, varname, "bias", quantile_threshold
                )
                dset.convert(quantile_map, ref[varname].pint.units)
            except NoDatabaseEntry:
                # fallback if the variable/type/quantile is not in the database
                method = "Collier2018"

        # Never mass weight if regional quantiles are used
        if method == "RegionalQuantiles":
            mass_weighting = False

        # Temporal means across the time period
        ref, com = cmp.make_comparable(ref, com, varname)
        ref.pint.dequantify().load().pint.quantify()
        com.pint.dequantify().load().pint.quantify()
        ref_mean = (
            dset.integrate_time(ref, varname, mean=True)
            if "time" in ref[varname].dims
            else ref[varname]
        )
        com_mean = (
            dset.integrate_time(com, varname, mean=True)
            if "time" in com[varname].dims
            else com[varname]
        )

        # Get the reference data uncertainty
        uncert = xr.zeros_like(ref_mean)
        if use_uncertainty:
            uncert = ref[ref[varname].attrs["bounds"]]
            uncert.attrs["units"] = ref[varname].pint.units
            uncert = (
                dset.integrate_time(uncert, mean=True)
                if "time" in uncert.dims
                else uncert
            )

        # If temporal information is available, we normalize the error by the
        # standard deviation of the reference. If not, we revert to the traditional
        # definition of relative error.
        norm = ref_mean
        if "time" in ref.dims and ref["time"].size > 1:
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
        if method == "Collier2018":
            score = np.exp(-(np.abs(bias) - uncert_).clip(0) / norm_)
        elif method == "RegionalQuantiles":
            quantile_map = quantile_map.pint.dequantify()
            norm = quantile_map.interp(
                lat=bias["lat"], lon=bias["lon"], method="nearest"
            )
            norm = norm.pint.quantify()
            score = (1 - (np.abs(bias) - uncert_).clip(0) / norm).clip(0, 1)
        else:
            msg = (
                "The method used to score the bias must be 'Collier2018' or "
                f"'RegionalQuantiles' but found {method=}"
            )
            raise ValueError(msg)

        # Build output datasets
        ref_out = ref_mean.to_dataset(name="mean")
        if use_uncertainty:
            ref_out["uncert"] = uncert
        com_out = bias.to_dataset(name="bias")
        com_out["bias_score"] = score
        try:
            lat_name = dset.get_dim_name(com_mean, "lat")
            lon_name = dset.get_dim_name(com_mean, "lon")
            com_mean = com_mean.rename(
                {lat_name: f"{lat_name}_", lon_name: f"{lon_name}_"}
            )
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
                    weight=ref_ if (mass_weighting and weight) else None,
                )
            elif dset.is_site(da):
                site_dim = dset.get_dim_name(da, "site")
                da = da.pint.dequantify()
                da = da.mean(dim=site_dim)
                da = da.pint.quantify()
            else:
                raise ValueError(f"Input is neither spatial nor site: {da}")
            return da

        # Compute scalars over all regions
        dfs = []
        for region in regions:
            # Period mean
            for src, var in zip(["Reference", "Comparison"], [ref_mean, com_mean]):
                var = _scalar(var, varname, region, not spatial_sum)
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
            bias_scalar_score = _scalar(com_out, "bias_score", region, True, True)
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
        dfs.attrs = dict(method=method)
        return dfs, ref_out, com_out
