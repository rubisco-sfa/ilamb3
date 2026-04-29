"""
An ILAMB transform to compute quantiles.
"""

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class quantile(ILAMBTransform):
    """
    Compute quantiles of a variable over one or more dimensions.

    If `dims` is a list/tuple, the quantile is computed jointly across all
    listed dimensions in a single reduction.

    Parameters
    ----------
    quantiles: float or list
        The quantiles in [0,1].
    dims: str of list
        The meta dimension names, one of {`time`, `lat`, `lon`, `depth`, `site`}.

    """

    def __init__(
        self,
        quantiles: float | list[float],
        dims: str | list[str],
    ):
        self.quantiles = quantiles if isinstance(quantiles, list) else [quantiles]
        self.dims = dims if isinstance(dims, list) else [dims]

    def required_variables(self) -> list[str]:
        """
        This transform does not require anything in particular.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Compute quantiles of all varaibles in dataset along dimensions.
        """
        # dim names as they are known in the dataset
        reduce_dims = [dset.get_dim_name(ds, dim) for dim in self.dims]
        bounds_vars = dset.get_all_bounds_vars(ds)
        for name, da in ds.data_vars.items():
            if not set(reduce_dims).issubset(da.dims) or name in bounds_vars:
                continue
            qname = f"{name}_quantile"
            ds[qname] = da.quantile(q=self.quantiles, dim=reduce_dims)
            ds[qname].attrs["transform"] = "quantiles"
            ds[qname].attrs["reduced_dims"] = reduce_dims
        return ds
