"""
An ILAMB transform to compute quantiles.
"""

from collections.abc import Sequence

import xarray as xr

from ilamb3.transform.base import ILAMBTransform


class Quantiles(ILAMBTransform):
    """
    Compute quantiles of a variable over one or more dimensions.

    If `dims` is a list/tuple, the quantile is computed jointly across all
    listed dimensions in a single reduction.

    Examples
    --------
    dims="time"
        -> quantile over time

    dims=["time", "lat", "lon"]
        -> quantile over all time/lat/lon values together
    """

    def __init__(
        self,
        quantiles: float | Sequence[float],
        dims: str | Sequence[str],
        variable: str,
        skipna: bool = True,
        keep_attrs: bool = True,
    ):
        self.quantiles = quantiles
        self.dims = dims
        self.variable = variable
        self.skipna = skipna
        self.keep_attrs = keep_attrs

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        pass

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable not in ds:
            # raise KeyError(f"Variable '{self.variable}' not found in dataset")
            print(f"Variable '{self.variable}' not found in dataset")
            return ds

        da = ds[self.variable]

        # normalize dims
        reduce_dims = self.dims if isinstance(self.dims, list) else [self.dims]

        # validate dims
        missing = [
            d
            for d in (reduce_dims if isinstance(reduce_dims, list) else [reduce_dims])
            if d not in da.dims
        ]
        if missing:
            # raise ValueError(f"Dimension(s) not found in variable: {missing}")
            print(f"Dimension(s) not found in variable: {missing}")
            return ds

        quantile_da = da.quantile(
            q=self.quantiles,
            dim=reduce_dims,
            skipna=self.skipna,
            keep_attrs=self.keep_attrs,
        )

        quantile = quantile_da.to_dataset(name=self.variable)
        quantile.attrs.update(ds.attrs)

        quantile[self.variable].attrs["transform"] = "quantiles"
        quantile[self.variable].attrs["quantiles"] = self.quantiles
        quantile[self.variable].attrs["reduced_dims"] = reduce_dims

        return quantile
