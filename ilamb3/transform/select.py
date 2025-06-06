from typing import Any

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class select_depth(ILAMBTransform):
    def required_variables(self) -> list[str]:
        return []

    def __call__(self, ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        if not dset.is_layered(ds):
            return ds
        depth_name = dset.get_dim_name(ds, "depth")
        if "value" in kwargs:
            return ds.sel({depth_name: kwargs["value"]}, method="nearest", drop=True)
        if "min" in kwargs and "max" in kwargs:
            return ds.sel({depth_name: slice(kwargs["min"], kwargs["max"])})
        raise ValueError(
            f"This transform allows for a single depth selection with `value` or a range with both `min` and `max`, but I found {kwargs}"
        )
