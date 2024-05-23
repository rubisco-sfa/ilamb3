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
    def __init__(self, required_variable: str):
        self.req_variable = required_variable

    def required_variables(self) -> list[str]:
        return [self.req_variable]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
        method: Literal["Collier2018", "RegionalQuantiles"] = "Collier2018",
        regions: list[Union[str, None]] = [None],
        mass_weighting: bool = False,
        use_uncertainty: bool = True,
        quantile_dbase: Union[pd.DataFrame, None] = None,
        quantile_threshold: int = 70,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref, com
            The reference and comparison dataset.
        method
            The name of the scoring methodology to use.
        regions
            A list of region labels over which to apply the analysis.
        mass_weighting
            Enable to weight the score map integrals by the temporal mean of the
            reference dataset.
        use_uncertainty
            Enable to utilize uncertainty information from the reference product if
            present.
        quantile_dbase
            If using `method='RegionalQuantiles'`, the dataframe containing the regional
            quantiles to be used to score the datasets.
        quantile_threshold
            If using `method='RegionalQuantiles'`, the threshold values to use from the
            `quantile_dbase`.

        Returns
        -------
        df
            A dataframe with scalar and score information from the comparison.
        ref_out, com_out
            A dataset containing grided information resulting from the reference and
            comparison.
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
        ref_, com_, norm_, uncert_ = cmp.nest_spatial_grids(
            ref_mean, com_mean, norm, uncert
        )

        # Compute score by different methods
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
        lat_name = dset.get_dim_name(com_mean, "lat")
        lon_name = dset.get_dim_name(com_mean, "lon")
        com_out["mean"] = com_mean.rename(
            {lat_name: f"{lat_name}_", lon_name: f"{lon_name}_"}
        )

        # Compute scalars over all regions
        dfs = []
        for region in regions:
            # Period mean
            for src, var in zip(["Reference", "Comparison"], [ref_mean, com_mean]):
                var = dset.integrate_space(var, varname, region=region, mean=True)
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
            bias_scalar = dset.integrate_space(
                com_out, "bias", region=region, mean=True
            )
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
            bias_scalar_score = dset.integrate_space(
                com_out,
                "bias_score",
                region=region,
                mean=True,
                weight=ref_ if mass_weighting else None,
            )
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    analysis_name,
                    "Bias Score",
                    "score",
                    "1",
                    float(bias_scalar_score.pint.dequantify()),
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
