from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis


class bias_collier2018(ILAMBAnalysis):
    def __init__(self, required_variables: str):
        self.req_variables = required_variables

    def required_variables(self) -> list[str]:
        return [self.req_variables]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
        regions: list[Union[str, None]] = [None],
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        # Initialize
        analysis_name = "Bias"
        varname = self.req_variables[0]
        ref, com = cmp.make_comparable(ref, com, varname)

        # Temporal means across the time period
        ref_mean = (
            dset.integrate_time(ref, varname, mean=True) if "time" in ref.dims else ref
        )
        com_mean = (
            dset.integrate_time(com, varname, mean=True) if "time" in com.dims else com
        )

        # If temporal information is available, we normalize the error by the
        # standard deviation of the reference. If not, we revert to the traditional
        # definition of relative error.
        norm = ref_mean
        if "time" in ref.dims and ref["time"].size > 1:
            norm = dset.std_time(ref, varname)

        # Nest the grids for comparison, we postpend composite grid variables with "_"
        ref_, com_, norm_ = cmp.nest_spatial_grids(ref_mean, com_mean, norm)
        bias = com_ - ref_
        score = np.exp(-np.abs(bias) / norm_)

        # Build output datasets
        ref_out = ref_mean.to_dataset(name="mean")
        com_out = xr.Dataset({"bias": bias, "bias_score": score})

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
                com_out, "bias_score", region=region, mean=True
            )
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    analysis_name,
                    "Bias Score",
                    "score",
                    f"{bias_scalar_score.pint.units:~cf}",
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
        return dfs, ref_out, com_out
