"""
The ILAMB net biome production scoring methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingVariable, TemporalOverlapIssue


class nbp_analysis(ILAMBAnalysis):
    """
    The ILAMB net biome production scoring methodology.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def required_variables(self) -> list[str]:
        """
        Return the variable names required in this analysis.

        Returns
        -------
        list
            A list of the required variables, here always [`nbp`].

        Notes
        -----
        This analysis also accepts the variable `netAtmosLandCO2Flux`. If you are
        running this routine inside an ILAMB analysis and need to use this variable,
        register it with the model as a synonym.
        """
        return ["nbp"]

    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset, evaluation_year: Union[int, None] = None
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.
        evaluation_year : int, optional
            The year at which to report a difference and score. If not given, the last
            year of the reference dataset.

        Returns
        -------
        pd.DataFrame
            A dataframe with scalar and score information from the comparison.
        xr.Dataset
            A dataset containing reference grided information from the comparison.
        xr.Dataset
            A dataset containing comparison grided information from the comparison.
        """
        # Default year to evaluate if none given
        if evaluation_year is None:
            evaluation_year = int(ref["time"][-1].dt.year)

        # Check that the comparison starts in the appropriate year
        if com["time"].dt.year > ref["time"].dt.year:
            raise TemporalOverlapIssue()
        tstart = min([t for t in com["time"] if t.dt.year == ref["time"][0].dt.year])
        com = com.sel({"time": slice(tstart, com["time"][-1])})

        # Fixes to data names and checks for required variables
        if "netAtmosLandCO2Flux" in com:
            com = com.rename_vars(dict(netAtmosLandCO2Flux="nbp"))
        if "nbp" not in com:
            msg = "`nbp` or `netAtmosLandCO2Flux` needs to be in the `com` Dataset."
            raise MissingVariable(msg)
        ref = ref.pint.quantify()
        for var in ref:
            if not var.startswith("nbp"):
                ref = ref.drop_vars(var)
            else:
                ref[var] = -ref[var]
        com["nbp"] = -com["nbp"]

        # Integrate globally
        if dset.is_spatial(com):
            com["nbp"] = dset.integrate_space(com, "nbp")

        # Load into memory, not sure this is the right place for this
        com = com.pint.dequantify()
        com.load()
        com = com.pint.quantify()

        # Accumulate
        def _cumsum(ds):  # numpydoc ignore=GL08
            for var, da in ds.items():
                if da.pint.units is None:
                    continue
                unit = 1.0 * da.pint.units
                if not unit.check("[mass] / [time]"):
                    continue
                da = dset.accumulate_time(ds, var)
                da = da.pint.to("Pg")
                ds[var] = da
            return ds

        ref = _cumsum(ref)
        com = _cumsum(com)

        # Coarsen to annual
        ref = dset.coarsen_annual(ref)
        com = dset.coarsen_annual(com)

        # Trajectory score
        out_units = f"{ref['nbp'].pint.units:~cf}"
        ref = ref.pint.dequantify()
        com = com.pint.dequantify()
        nbp_low = ref["nbp"]
        nbp_high = ref["nbp"]
        if "bounds" in ref["nbp"].attrs and ref["nbp"].attrs["bounds"] in ref:
            nbp_low = ref[ref["nbp"].attrs["bounds"]][:, 0]
            nbp_high = ref[ref["nbp"].attrs["bounds"]][:, 1]
        uncert = xr.DataArray(
            np.sqrt(
                (ref["nbp"] - nbp_low).values ** 2, (nbp_high - ref["nbp"]).values ** 2
            ),
            coords={"year": ref["year"]},
        )
        eps = (com["nbp"] - nbp_low).clip(0) + (nbp_high - com["nbp"]).clip(0)
        traj_score = np.exp(-eps / uncert)
        traj_score = float(traj_score.mean())

        # Difference score
        ref_val = float(ref["nbp"].sel(year=evaluation_year))
        try:
            com_val = float(com["nbp"].sel(year=evaluation_year))
        except KeyError:
            com_val = np.nan
        uncert_val = float(uncert.sel(year=evaluation_year))
        scale = -np.log(0.5) / 1  # outside of the uncertainty window? score < 50%
        diff_score = np.exp(-scale * np.abs(com_val - ref_val) / uncert_val)

        # Scores and scalars
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
                    "value": ref_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
                    "value": com_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"diff({evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
                    "value": com_val - ref_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": "Difference Score",
                    "type": "score",
                    "units": "1",
                    "value": diff_score,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": "Trajectory Score",
                    "type": "score",
                    "units": "1",
                    "value": traj_score,
                },
            ]
        )
        return df, ref, com
