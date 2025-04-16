import xarray as xr

import ilamb3.dataset as dset


def soil_moisture_to_vol_fraction(ds: xr.Dataset) -> xr.Dataset:
    for var in ["mrsos", "mrsol"]:
        if var in ds:
            ds[var] = _to_vol_fraction(ds, var).squeeze()
    return ds


def _to_vol_fraction(ds: xr.Dataset, varname: str) -> xr.DataArray:
    """
    Convert the array from a mass area density to a volume fraction.
    """
    da = ds[varname]
    da = da.pint.quantify()
    ureg = da.pint.registry
    quantity = 1.0 * da.pint.units
    if quantity.check(""):
        return da.pint.dequantify()
    if not quantity.check("[mass] / [length]**2"):
        raise ValueError("Cannot convert array of this type.")
    if dset.is_layered(da):
        depth_name = dset.get_dim_name(da, "depth")
        depth = da[depth_name]
        if "bounds" not in depth.attrs:
            raise ValueError(
                "Cannot convert soil mositure data when depth bounds are not coordinates."
            )
        depth_name = depth.attrs["bounds"]
        if depth_name not in ds:
            raise ValueError(
                "Cannot convert soil mositure data when depth bounds are not coordinates."
            )
        depth = ds[depth_name] * da.pint.registry.meter
        da = da / depth.diff(dim=depth.dims[-1])
    else:
        # If not layered, we are assuming this is mrsos which is the moisture in
        # the top 10cm
        depth = 0.1 * da.pint.registry.meter
        da = da / depth
    water_density = 998.2071 * ureg.kilogram / ureg.meter**3
    da = (da / water_density).pint.to("dimensionless")
    da = da.pint.dequantify()
    return da
