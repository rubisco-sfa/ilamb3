import pandas as pd
import xarray as xr

from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingVariable, TemporalOverlapIssue


class nbp(ILAMBAnalysis):
    def required_variables(self) -> list[str]:
        """Return the variable names required in this analysis.

        Note
        ----
        This analysis also accepts the variable `netAtmosLandCO2Flux`. If you are
        running this routine inside an ILAMB analysis and need to use this variable,
        register it with the model as a synonym.
        """
        return ["nbp"]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        # Fixes to data names and checks for required variables
        if "netAtmosLandCO2Flux" in com:
            com = com.rename_vars(dict(netAtmosLandCO2Flux="nbp"))
        if "nbp" not in com:
            msg = "`nbp` or `netAtmosLandCO2Flux` needs to be in the `com` Dataset."
            raise MissingVariable(msg)

        # Does the comparison start when the reference does?
        ref_init, _ = dset.get_time_extent(ref)
        com_init, _ = dset.get_time_extent(com)
        if ref_init.dt.year != com_init.dt.year:
            msg = f"The reference starts in {ref_init.dt.year} but the comparison"
            msg += f" in {com_init.dt.year}"
            raise TemporalOverlapIssue(msg)
