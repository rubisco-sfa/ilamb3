import xarray as xr

import ilamb3.dataset as dset


def select_depth(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    try:
        depth_name = dset.get_dim_name(ds, "depth")
    except KeyError:
        return ds
    if "value" in kwargs:
        return ds.sel({depth_name: kwargs["value"]}, method="nearest", drop=True)
    if "min" in kwargs and "max" in kwargs:
        return ds.sel({depth_name: slice(kwargs["min"], kwargs["max"])})
    raise ValueError(
        f"This transform allows for a single depth selection with `value` or a range with both `min` and `max`, but I found {kwargs}"
    )
