"""
An ILAMB transform to extract AMOC out of msftmz for comparison to the RAPID data.
"""

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class msftmz_to_rapid(ILAMBTransform):
    """
    An ILAMB transform to extract AMOC out of msftmz for comparison to the RAPID data.
    """

    def __init__(self):
        pass

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return ["msftmz"]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extract AMOC strength for comparison to RAPID.
        """
        if "amoc" in ds:
            return ds
        if "msftmz" not in ds:
            return ds
        lat_name = dset.get_dim_name(ds, "lat")
        depth_name = dset.get_dim_name(ds, "depth")
        ds["amoc"] = dset.convert(
            ds["msftmz"]
            .isel(basin=0)
            .sel({lat_name: 26.5}, method="nearest")
            .max(depth_name),
            "Sv",
            "msftmz",
        )
        ds = ds.drop_vars("msftmz")
        return ds
