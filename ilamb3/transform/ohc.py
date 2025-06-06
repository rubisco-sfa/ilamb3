import cftime as cf
import xarray as xr

import ilamb3
import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class ocean_heat_content(ILAMBTransform):
    def required_variables(self) -> list[str]:
        return ["thetao", "volcello"]

    def __call__(self, ds: xr.Dataset, reference_year: int | None = None) -> xr.Dataset:
        if "ohc" in ds:
            return ds
        if not set(self.required_variables()).issubset(ds):
            return ds
        time_name = dset.get_dim_name(ds, "time")
        if reference_year is None:
            reference_year = int(ds[time_name].min().dt.year.values)
        ds = ds.sel({time_name: slice(f"{reference_year}-01", None)})
        ds = ds.pint.quantify()
        Cp = 3991.868 * ilamb3.units.J / ilamb3.units.kg / ilamb3.units.K
        rho = 1026.0 * ilamb3.units.kg / ilamb3.units.m**3
        ohc = (Cp * rho * ds["thetao"] * ds["volcello"]).sum(dim=ds["volcello"].dims)
        ohc = ohc - ohc.isel({time_name: 0})
        ohc = ohc.pint.to("ZJ")
        ohc.load()
        ohc.name = "ohc"
        ohc.attrs = {
            "standard_name": "Ocean Heat Content",
            "comment": f"Relative to {reference_year}",
        }
        ds = ds.drop_vars(["thetao", "volcello"], errors="ignore")
        ds["ohc"] = ohc
        ds = ds.pint.dequantify()
        ds = ds.groupby(f"{time_name}.year").mean()
        ds = ds.rename(year="time")
        ds["time"] = [cf.DatetimeNoLeap(y, 6, 1) for y in ds["time"]]
        return ds
