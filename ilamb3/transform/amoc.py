import xarray as xr

import ilamb3.dataset as dset


def msftmz_to_rapid(ds: xr.Dataset) -> xr.Dataset:
    print("In transform")
    print(ds)
    if "msftmz" not in ds:
        return ds
    if "amoc" in ds:
        return ds
    print("Computing AMOC")
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
    print(ds)
    return ds
