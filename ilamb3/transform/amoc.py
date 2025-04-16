import xarray as xr

import ilamb3.dataset as dset


def msftmz_to_rapid(ds: xr.Dataset) -> xr.Dataset:
    if "msftmz" not in ds:
        return ds
    if "amoc" in ds:
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
