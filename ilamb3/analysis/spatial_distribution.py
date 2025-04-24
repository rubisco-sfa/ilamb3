"""
The ILAMB bias methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as ilplt
import ilamb3.regions as ilr
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis


class spatial_distribution_analysis(ILAMBAnalysis):
    """
    The ILAMB bias methodology.

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
        regions: list[str | None] = [None],
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.req_variable = required_variable
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
        analysis_name = "Spatial Distribution"
        varname = self.req_variable

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname)

        # Temporal means across the time period...
        ref = (
            dset.integrate_time(ref, varname, mean=True)
            if dset.is_temporal(ref[varname])
            else ref[varname]
        )
        com = (
            dset.integrate_time(com, varname, mean=True)
            if dset.is_temporal(com[varname])
            else com[varname]
        )

        # ... on the same grid
        ref, com = cmp.nest_spatial_grids(ref, com)

        # Compute scalars over all regions
        dfs = []
        ilamb_regions = ilr.Regions()
        for region in self.regions:
            # Get regional versions
            rref = ilamb_regions.restrict_to_region(ref, region)
            rcom = ilamb_regions.restrict_to_region(com, region)

            # Spatial standard deviation
            ref_std = float(rref.std())
            com_std = float(rcom.std())
            norm_std = com_std / ref_std

            # Correlation
            isnan = rref.isnull() | rcom.isnull()
            rref = np.ma.masked_invalid(
                xr.where(isnan, np.nan, rref).values
            ).compressed()
            rcom = np.ma.masked_invalid(
                xr.where(isnan, np.nan, rcom).values
            ).compressed()
            corr = float(np.corrcoef(rref, rcom)[0, 1])
            taylor_score = 4 * (1 + corr) / ((norm_std + 1 / norm_std) ** 2 * 2)

            dfs += [
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": analysis_name,
                    "name": "Normalized Standard Deviation",
                    "type": "scalar",
                    "units": "1",
                    "value": norm_std,
                },
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": analysis_name,
                    "name": "Correlation",
                    "type": "scalar",
                    "units": "1",
                    "value": corr,
                },
                {
                    "source": "Comparison",
                    "region": str(region),
                    "analysis": analysis_name,
                    "name": "Spatial Distribution Score",
                    "type": "score",
                    "units": "1",
                    "value": taylor_score,
                },
            ]

        dfs = pd.DataFrame(dfs)
        return dfs, xr.Dataset(), xr.Dataset()

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        # Add the Taylor diagram
        axs = []
        for region in df["region"].unique():
            axs += [
                {
                    "name": "taylor",
                    "title": "Spatial Distribution",
                    "region": str(region),
                    "source": None,
                    "axis": ilplt.plot_taylor_diagram(df[df["region"] == region]),
                }
            ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
