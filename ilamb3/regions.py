"""This class holds a list of all regions currently registered in the ILAMB
system via a static property of the class. It also comes with methods for
defining additional regions by lat/lon bounds or by a mask specified by a
netCDF4 file. A set of regions used in the Global Fire Emissions Database (GFED)
is included by default."""
import os
from typing import Union

import numpy as np
import xarray as xr

from . import dataset as dset


class Regions:
    """A class for unifying the treatment of regions in ILAMB."""

    _regions = {}
    _sources = {}

    @property
    def regions(self):
        """Returns a list of region identifiers."""
        return Regions._regions.keys()

    def add_latlon_bounds(
        self,
        label: str,
        name: str,
        lats: tuple[float],
        lons: tuple[float],
        source: str = "user-provided latlon bounds",
    ):
        """Add a region by lat/lon bounds."""
        assert len(lats) == 2
        assert len(lons) == 2
        rtype = 0
        Regions._regions[label] = [rtype, name, lats, lons]
        Regions._sources[label] = source

    def add_netcdf(self, netcdf: Union[str, xr.Dataset]) -> list[str]:
        """Add regions found in a netCDF file and returns a list of the labels
        found."""
        rtype = 1
        if isinstance(netcdf, str):
            dsr = xr.open_dataset(netcdf)
        else:
            dsr = netcdf
        if "ids" in dsr:
            ids = "ids"
        else:
            ids = [
                v for v in dsr.data_vars if dsr[v].ndim == 2 and dsr[v].dtype == np.int
            ]
            if len(ids) == 0:
                raise ValueError(
                    f"Found no 2d integer arrays in the region file {netcdf}"
                )
            if len(ids) > 1:
                raise ValueError(
                    f"Amiguous integer array for regions in {netcdf}: {','.join(ids)}"
                )
            ids = ids[0]
        labels = list(dsr[dsr[ids].attrs["labels"]].to_numpy())
        names = list(dsr[dsr[ids].attrs["names"]].to_numpy())
        for label, name in zip(labels, names):
            dar = xr.where(dsr[ids] == labels.index(label), 1, 0)
            Regions._regions[label] = [rtype, name, dar]
            Regions._sources[label] = (
                os.path.basename(netcdf) if isinstance(netcdf, str) else "dataset"
            )
        return labels

    def get_name(self, label: str):
        """Given the region label, return the full name."""
        return Regions._regions[label][1]

    def get_source(self, label: str):
        """Given the region label, return the source."""
        return Regions._sources[label]

    def get_mask(self, label: str, var: xr.DataArray) -> xr.DataArray:
        """Given the region label and a variable, return a mask."""
        rdata = Regions._regions[label]
        rtype = rdata[0]
        lat = var[dset.get_latitude_name(var)]
        lon = var[dset.get_longitude_name(var)]
        if rtype == 0:
            rtype, _, rlat, rlon = rdata
            out = xr.where(
                (lat >= rlat[0])
                * (lat <= rlat[1])
                * (lon >= rlon[0])
                * (lon <= rlon[1]),
                False,
                True,
            )
            return out
        if rtype == 1:
            rtype, _, dar = rdata
            out = dar.interp(lat=lat, lon=lon, method="nearest") == 0
            return out
        raise ValueError(f"Region type #{rtype} not recognized")

    def restrict_variable(self, label: str, var: xr.DataArray) -> xr.DataArray:
        """Given the region label and a variable, return the variable with nan's
        outside of the region."""
        mask = self.get_mask(label, var)
        return xr.where(mask, np.nan, var)

    def has_data(self, label: str, var: xr.DataArray) -> bool:
        """Checks if the variable has data on the given region."""
        mask = self.get_mask(label, var)
        if (mask == 0).sum() == 0:
            return False
        _ = xr.where(mask, np.nan, var)  # broken
        return True


if "global" not in Regions().regions:
    # Populate some regions
    r = Regions()
    src = "ILAMB internal"
    r.add_latlon_bounds("global", "Globe", (-89.999, 89.999), (-179.999, 179.999), src)
    r.add_latlon_bounds(
        "globe", "Global - All", (-89.999, 89.999), (-179.999, 179.999), src
    )

    # GFED regions
    src = "Global Fire Emissions Database (GFED)"
    r.add_latlon_bounds(
        "bona", "Boreal North America", (49.75, 79.75), (-170.25, -60.25), src
    )
    r.add_latlon_bounds(
        "tena", "Temperate North America", (30.25, 49.75), (-125.25, -66.25), src
    )
    r.add_latlon_bounds(
        "ceam", "Central America", (9.75, 30.25), (-115.25, -80.25), src
    )
    r.add_latlon_bounds(
        "nhsa",
        "Northern Hemisphere South America",
        (0.25, 12.75),
        (-80.25, -50.25),
        src,
    )
    r.add_latlon_bounds(
        "shsa",
        "Southern Hemisphere South America",
        (-59.75, 0.25),
        (-80.25, -33.25),
        src,
    )
    r.add_latlon_bounds("euro", "Europe", (35.25, 70.25), (-10.25, 30.25), src)
    r.add_latlon_bounds("mide", "Middle East", (20.25, 40.25), (-10.25, 60.25), src)
    r.add_latlon_bounds(
        "nhaf", "Northern Hemisphere Africa", (0.25, 20.25), (-20.25, 45.25), src
    )
    r.add_latlon_bounds(
        "shaf", "Southern Hemisphere Africa", (-34.75, 0.25), (10.25, 45.25), src
    )
    r.add_latlon_bounds("boas", "Boreal Asia", (54.75, 70.25), (30.25, 179.75), src)
    r.add_latlon_bounds("ceas", "Central Asia", (30.25, 54.75), (30.25, 142.58), src)
    r.add_latlon_bounds("seas", "Southeast Asia", (5.25, 30.25), (65.25, 120.25), src)
    r.add_latlon_bounds(
        "eqas", "Equatorial Asia", (-10.25, 10.25), (99.75, 150.25), src
    )
    r.add_latlon_bounds("aust", "Australia", (-41.25, -10.50), (112.00, 154.00), src)
