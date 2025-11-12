from typing import Any

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class integrate(ILAMBTransform):
    def __init__(
        self,
        dim: str,
        varname: str,
        mean: bool = False,
        # give trapezoidal option to use xarray built-in .integrate()
        # otherwise, use nate's method
        **kwargs: Any,
    ):
        self.dim = dim
        self.varname = varname
        self.mean = mean

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Select a dim or dim range of the input dataset, if the dimension is found.
        """
        try:
            dim_name = dset.get_dim_name(ds, self.dim)
        except KeyError:  # so if the user implements integrate but it hits this error, it'll still pass thru even tho it's wrong
            return ds
        if dim_name == "time":
            ds[self.varname] = dset.integrate_time(ds, self.varname, mean=self.mean)
        elif dim_name == "depth":
            ds[self.varname] = dset.integrate_depth(ds, self.varname, mean=self.mean)
        elif dim_name == "space":
            ds[self.varname] = dset.integrate_space(ds, self.varname, mean=self.mean)
        else:
            return ds  # if someone really wants to integrate over a different magical dim, they can implement their own transform


class integrate_time(integrate):
    def __init__(
        self,
        varname: str,
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("time", varname, mean=mean, **kwargs)


class integrate_depth(integrate):
    def __init__(
        self,
        varname: str,
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("depth", varname, mean=mean, **kwargs)


class integrate_space(integrate):
    def __init__(
        self,
        varname: str,
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("space", varname, mean=mean, **kwargs)
