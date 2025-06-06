import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform import ILAMBTransform


class msftmz_to_rapid(ILAMBTransform):
    def required_variables(self) -> list[str]:
        return ["msftmz"]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
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
