from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis  # import the ABC from ilamb
from ilamb3.exceptions import AnalysisNotAppropriate

class absrelerror(ILAMBAnalysis):  # define a class that inherits from the ABC
    def __init__(self, required_variable: str, **kwargs: Any):
        self.varname = required_variable

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        varname = self.varname

        # limit analysis to gridded data only
        if dset.is_site(ref[varname]) or dset.is_site(com[varname]):
            raise AnalysisNotAppropriate()

        # drop non-overlapping time and ensure units/coords are aligned
        ref, com = cmp.make_comparable(ref, com, varname)

        # ensure a time dimension exists
        if dset.is_temporal(ref[varname]):
            # handle possible calendar mismatch by calculating integrated time means
            ref[varname] = dset.integrate_time(ref, varname, mean=True)
        if dset.is_temporal(com[varname]):
            com[varname] = dset.integrate_time(com, varname, mean=True)

        # nest the spatial grids as explained in Collier, et al., JAMES, 2018
        ref, com = cmp.nest_spatial_grids(ref, com)

        # compute epsilon per gridcell
        eps_gridded = np.abs(ref[varname] - com[varname]) / np.abs(ref[varname]).clip(min=1e-10)

        # compute an area-weighted mean of the per-gridcell epsilon values
        epsilon = dset.integrate_space(eps_gridded, varname, mean=True)

        # build up the output dataframe
        df = pd.DataFrame(
            [
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Relative Error",
                    "name": "Mean Absolute Relative Error",
                    "type": "scalar",
                    "units": "1",
                    "value": float(epsilon),
                }
            ]
        )

        return df, xr.Dataset(), xr.Dataset()

    def name(self) -> str:
        return "Relative Error"

    def required_variables(self) -> list[str]:
        return [self.varname]

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["name","source","region","analysis","path"])
