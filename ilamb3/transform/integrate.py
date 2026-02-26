from functools import partial
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
        varname: str | list[str],
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
        Apply the appropriate integration transform to the dataset.
        """

        # Normalize varname to list
        varnames = [self.varname] if isinstance(self.varname, str) else self.varname

        # Map dimension to integration function and validator
        integration_map = {
            "time": (dset.integrate_time, dset.is_temporal),
            "depth": (dset.integrate_depth, dset.is_layered),
            "space": (dset.integrate_space, dset.is_spatial),
        }

        integration_func, var_type_func = integration_map[self.dim]

        # Create partially applied function
        integrate = partial(integration_func, mean=self.mean)

        # Apply to all variables
        for varname in varnames:
            if varname in ds:
                if var_type_func(ds[varname]):
                    ds[varname] = integrate(ds, varname)
        return ds


class integrate_time(integrate):
    def __init__(
        self,
        varname: str | list[str],
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("time", varname, mean=mean, **kwargs)


class integrate_depth(integrate):
    def __init__(
        self,
        varname: str | list[str],
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("depth", varname, mean=mean, **kwargs)


class integrate_space(integrate):
    def __init__(
        self,
        varname: str | list[str],
        mean: bool = False,
        **kwargs: Any,
    ):
        super().__init__("space", varname, mean=mean, **kwargs)
