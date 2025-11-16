from typing import Any

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class integrate(ILAMBTransform):
    """
    This ILAMB Transform integrates a variable over a specified dimension if that
    dimension exists. The integrated variable replaces the original variable in the
    dataset. The integration is the sum over the specified dimension weighted by
    the appropriate measure (e.g., time interval lengths for time integration).
    E.g., `ds["nbp"].weighted(ds["time_measures"].fillna(0)).sum(dim="time")`

    Parameters
    ----------
    dim : str
        The dimension over which to integrate ('time', 'depth', or 'space').
    varname : str
        The variable to integrate.
    mean : bool, optional
        If True, compute the integrated mean instead of sum (default: False).
    **kwargs : Any
        Additional keyword arguments passed to the base `ILAMBTransform` class.
    """

    def __init__(
        self,
        dim: str,
        varname: str,
        mean: bool = False,
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

        # time integration
        if self.dim == "time":
            if dset.is_temporal(ds[self.varname]):
                ds[self.varname] = dset.integrate_time(ds, self.varname, mean=self.mean)
                return ds
            else:
                return ds

        # depth integration
        elif self.dim == "depth":
            if dset.is_layered(ds[self.varname]):
                ds[self.varname] = dset.integrate_depth(
                    ds, self.varname, mean=self.mean
                )
                return ds
            else:
                return ds

        # spatial (lat/lon) integration
        elif self.dim == "space":
            if dset.is_spatial(ds[self.varname]):
                ds[self.varname] = dset.integrate_space(
                    ds, self.varname, mean=self.mean
                )
                return ds
            else:
                return ds
        else:
            return ds


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
