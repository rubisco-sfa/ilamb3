import numpy as np
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class stratification_index(ILAMBTransform):
    """Computes a time series of the stratification index in the Southern Ocean
    (lat -55 to -30) using EN4.2.2 temperature and salinity data."""

    def __init__(
        self,
        depth_horizon: float = 1000,
        lat_min: float = -55,
        lat_max: float = -30,
    ):
        self.depth_horizon = depth_horizon
        self.lat_min = lat_min
        self.lat_max = lat_max

    def required_variables(self):
        return ["thetao", "so"]

    def __call__(self, ds: xr.Dataset) -> xr.DataArray:
        # Just return if we do not have what it takes
        if not set(self.required_variables()).issubset(ds):
            return ds

        try:
            import gsw
        except ImportError:
            raise ImportError(
                "The use of the `stratification_index` transform requires additional dependencies. Try `uv sync --group gcb` and then run again."
            )

        # Reads the temperature and salinity and ensure units
        ds["thetao"] = ds["thetao"].pint.quantify().pint.to("degC")
        ds["so"] = ds["so"].pint.quantify().pint.to("psu")
        ds = ds.pint.dequantify()

        # Convenience definitions
        Ts = ds["thetao"]
        Ss = ds["so"]
        lat_name = dset.get_dim_name(ds, "lat")
        lon_name = dset.get_dim_name(ds, "lon")
        depth_name = dset.get_dim_name(ds, "depth")

        # Calculates the sea pressure
        Pp = gsw.p_from_z(
            -xr.ones_like(ds["thetao"]) * ds["depth"],
            xr.ones_like(ds["thetao"]) * ds[lat_name],
        )

        # Calculate Absolute Salinity and Conservative Temperature
        SA = gsw.SA_from_SP(
            Ss,
            Pp,
            xr.ones_like(ds["thetao"]) * ds[lon_name],
            xr.ones_like(ds["thetao"]) * ds[lat_name],
        )

        # Calculate insitu density (surface and 1000m) from Absolute salinity and Conservative temperature
        rho = gsw.rho(SA, gsw.CT_from_pt(SA, Ts), Pp)

        # Calculate the stratification index using density differences
        SO_SI = rho.interp(
            {depth_name: self.depth_horizon}, method="linear"
        ) - rho.isel({depth_name: 0}).sel({lat_name: slice(self.lat_min, self.lat_max)})

        # Area weights
        area_SO_SI = np.cos(np.deg2rad(SO_SI[lat_name]))
        weights_2D = xr.ones_like(SO_SI) * area_SO_SI

        ds["SI"] = SO_SI.weighted(weights_2D).mean(dim=(lat_name, lon_name))
        ds = ds.drop_vars(["thetao", "so"])
        return ds
