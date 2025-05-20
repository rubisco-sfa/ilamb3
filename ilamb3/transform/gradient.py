import xarray as xr

import ilamb3.dataset as dset


def depth_gradient(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the depth gradient of the dataset using least squares regression.
    """
    if not dset.is_layered(ds):
        return ds
    for var, da in ds.items():
        if dset.is_temporal(da):
            ds[var] = dset.integrate_time(ds, var, mean=True)
    grad = ds.polyfit(dset.get_dim_name(ds, "depth"), deg=1).sel(degree=1, drop=True)
    grad = grad.rename_vars(
        {var: var.replace("_polyfit_coefficients", "_depth_gradient") for var in grad}
    )
    for var, da in grad.items():
        if "_depth_gradient" in var:
            orig = var.replace("_depth_gradient", "")
            grad[var].attrs["units"] = f"{ds[orig].attrs['units']} m-1"
    return grad
