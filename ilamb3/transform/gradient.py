"""
An ILAMB transform for computing the depth gradient.
"""

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class depth_gradient(ILAMBTransform):
    """
    Compute the depth gradient of the dataset using least squares regression.
    """

    def __init__(self):
        pass

    def required_variables(self) -> list[str]:
        """Return the variables this transform requires, none in this case."""
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Return the depth gradient of the input dataset, if a depth dimension
        exists.

        Note
        ----
        This transform will also integrate out the temporal dimension. It may be
        useful to use this transform in conjunction with the `select_depth` to
        pick a range of depths across which you want to estimate the gradient.
        """
        if not dset.is_layered(ds):
            return ds
        for var, da in ds.items():
            if dset.is_temporal(da):
                ds[var] = dset.integrate_time(ds, var, mean=True)
        grad = ds.polyfit(dset.get_dim_name(ds, "depth"), deg=1).sel(
            degree=1, drop=True
        )
        grad = grad.rename_vars(
            {
                var: var.replace("_polyfit_coefficients", "_depth_gradient")
                for var in grad
            }
        )
        for var, da in grad.items():
            if "_depth_gradient" in var:
                orig = var.replace("_depth_gradient", "")
                grad[var].attrs["units"] = f"{ds[orig].attrs['units']} m-1"
        return grad
