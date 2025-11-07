"""
An ILAMB transform for converting integrated soil moisture to a density.
"""

import xarray as xr

import ilamb3
import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class soil_moisture_to_vol_fraction(ILAMBTransform):
    """
    Convert depth integrated moisture to a volumetric fraction.
    """

    def __init__(self):
        pass

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform can convert.
        """
        return ["mrsos", "mrsol"]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Convert depth integrated moisture to a volumnetric fraction.
        """
        for var in self.required_variables():
            if var in ds:
                ds[var] = _to_vol_fraction(ds, var).squeeze()
        return ds


def _to_vol_fraction(ds: xr.Dataset, varname: str) -> xr.DataArray:
    """
    Convert the array from a mass area density to a volume fraction.

    Note
    ----
    As of xarray 2025.10.1, pint 0.25, pint-xarray 0.6.0, quantifying the
    dataarray here changes the depth dimension data yielding differences in the
    dimension on the order (1e-8). Here we avoid using quantify to circumvent
    this issue.
    """
    da = ds[varname]
    ureg = ilamb3.units
    quantity = ureg(da.attrs["units"])
    if quantity.check(""):
        # units are already compatible with volume fraction
        return da
    if not quantity.check("[mass] / [length]**2"):
        raise ValueError(
            f"Cannot convert a variable with units of this type {da.pint.units}."
        )
    if dset.is_layered(da):
        depth_dim = dset.get_dim_name(da, "depth")
        depth = ds[depth_dim]
        bnd_name = depth.attrs["bounds"] if "bounds" in depth.attrs else None
        if not dset.has_bounds(ds, depth_dim, bnd_name):
            raise ValueError(
                "Cannot convert soil mositure data when depth bounds are not coordinates."
            )
        depth = ds[bnd_name]
        interval_dim = set(depth.dims).difference([depth_dim]).pop()
        da = (
            da / depth.diff(dim=interval_dim).squeeze()
        )  # <-- this was failing, see Note
        da = da.pint.quantify() / ureg.meter
    else:
        # If not layered, we are assuming this is mrsos which is the moisture in
        # the top 10cm
        depth = 0.1 * ureg.meter
        da = da.pint.quantify() / depth
    water_density = 998.2071 * ureg.kilogram / ureg.meter**3
    da = (da / water_density).pint.to("dimensionless")
    da = da.pint.dequantify()
    return da
