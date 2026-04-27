"""
An ILAMB transform to compute quantiles.
"""

from collections.abc import Sequence

import xarray as xr

from ilamb3.transform.base import ILAMBTransform


class quantiles(ILAMBTransform):
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
        keep_attrs: bool = True,
    ):
        self.quantiles = quantiles
        self.dims = dims
        self.keep_attrs = keep_attrs

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        pass

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Compute quantiles of all varaibles in dataset
        along dimensions.
        """
        # normalize dims
        reduce_dims = self.dims if isinstance(self.dims, list) else [self.dims]

        out_vars: dict[str, xr.DataArray] = {}

        for name, da in ds.data_vars.items():
            dims_to_use = [d for d in reduce_dims if d in da.dims]

            if not dims_to_use:
                continue

            qda = da.quantile(
                q=self.quantiles,
                dim=dims_to_use,
                skipna=True,
                keep_attrs=self.keep_attrs,
            )

            qda.attrs["transform"] = "quantiles"
            qda.attrs["quantiles"] = self.quantiles
            qda.attrs["reduced_dims"] = dims_to_use

            out_vars[name] = qda

        ds_quantile = xr.Dataset(out_vars, attrs=ds.attrs)
        return ds_quantile
