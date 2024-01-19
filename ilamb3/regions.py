"""This class holds a list of all regions registered in the ILAMB
system via a static property of the class. It also comes with methods for defining
additional regions by lat/lon bounds or by a mask specified by a netCDF4 file. A set of
regions used in the Global Fire Emissions Database (GFED) is included by default."""
import os
from typing import Union

import numpy as np
import xarray as xr

from . import dataset as dset


def restrict_to_bbox(
    var: Union[xr.Dataset, xr.DataArray],
    lat_name: str,
    lon_name: str,
    lat0: float,
    latf: float,
    lon0: float,
    lonf: float,
):
    """Return the dataset selected to the nearest bounding box.

    This is awkward because as of `v2023.6.0`, the `method` keyword cannot be used in
    slices. Note that this routine will sort the dimensions because slicing does not
    work well on unsorted indices.

    """
    var = var.sortby(list(var.dims))
    var = var.sel(
        lat=slice(
            var[lat_name].sel({lat_name: lat0}, method="nearest"),
            var[lat_name].sel({lat_name: latf}, method="nearest"),
        ),
        lon=slice(
            var[lon_name].sel({lon_name: lon0}, method="nearest"),
            var[lon_name].sel({lon_name: lonf}, method="nearest"),
        ),
    )
    return var


class Regions:
    """A class for unifying the treatment of regions in ILAMB."""

    _regions = {}
    _sources = {}

    @property
    def regions(self):
        """Return a list of region identifiers."""
        return Regions._regions.keys()

    def add_latlon_bounds(
        self,
        label: str,
        name: str,
        lats: list[float],
        lons: list[float],
        source: str = "user-provided latlon bounds",
    ) -> None:
        """Add a region by lat/lon bounds.

        Parameters
        ----------
        label
            The label by which the region will be known in the ilamb system. Note that
            this will overwrite any current region known by this label.
        name
            The region text displayed in the ilamb output.
        lats
            The lower and upper latitude bounds [-90,90] defining the region.
        lons
            The lower and upper longitude bounds [-180,180] defining the region.
        source
            An optional text description for the source of these regions.

        """
        assert len(lats) == 2
        assert len(lons) == 2
        rtype = 0
        Regions._regions[label] = [rtype, name, lats, lons]
        Regions._sources[label] = source

    def add_netcdf(self, netcdf: Union[str, xr.Dataset]) -> list[str]:
        """Add regions found in a netCDF file or dataset.

        Region formatting guidelines can be found
        [here](https://www.ilamb.org/doc/custom_regions.html).

        Parameters
        ----------
        netcdf
            The filename or dataset which contain the regions.

        Returns
        -------
        labels
            The list of region labels added to the ilamb system as a result of this
            function call.

        """
        rtype = 1
        if isinstance(netcdf, str):
            dsr = xr.open_dataset(netcdf)
        else:
            dsr = netcdf
        if "ids" in dsr:
            ids = "ids"
        else:
            ids = [
                v
                for v in dsr.data_vars
                if dsr[v].ndim == 2 and dsr[v].dtype == np.integer
            ]
            if len(ids) == 0:
                raise ValueError(
                    f"Found no 2d integer arrays in the region file {netcdf}"
                )
            if len(ids) > 1:
                raise ValueError(
                    f"Amiguous integer array for regions in {netcdf}: {ids}"
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

    def get_name(self, label: str) -> str:
        """Return the region name given its label."""
        return Regions._regions[label][1]

    def get_source(self, label: str):
        """Return the source of the region given its label."""
        return Regions._sources[label]

    def restrict_to_region(
        self,
        var: Union[xr.Dataset, xr.DataArray],
        label: Union[str, None],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Given the region label and a variable, return a mask."""
        if label is None:
            return var
        rdata = Regions._regions[label]
        rtype = rdata[0]
        lat_name = dset.get_dim_name(var, "lat")
        lon_name = dset.get_dim_name(var, "lon")
        if rtype == 0:
            _, _, rlat, rlon = rdata
            out = restrict_to_bbox(
                var, lat_name, lon_name, rlat[0], rlat[1], rlon[0], rlon[1]
            )
            return out
        if rtype == 1:
            _, _, dar = rdata
            out = xr.where(
                dar.interp(
                    {lat_name: var[lat_name], lon_name: var[lon_name]}, method="nearest"
                ),
                var,
                np.nan,
            )
            rlat_name = dset.get_dim_name(dar, "lat")
            rlon_name = dset.get_dim_name(dar, "lon")
            out = restrict_to_bbox(
                out,
                lat_name,
                lon_name,
                dar[rlat_name].min(),
                dar[rlat_name].max(),
                dar[rlon_name].min(),
                dar[rlon_name].max(),
            )
            return out
        raise ValueError(f"Region type #{rtype} not recognized")


# If no regions have been registered, then add these
if len(Regions().regions) == 0:
    r = Regions()
    regions = """
    bona, Boreal North America             , 50, 80,-170, -60
    tena, Temperate North America          , 30, 50,-125, -66
    ceam, Central America                  , 10, 30,-115, -80
    nhsa, Northern Hemisphere South America,  0, 13, -80, -50
    shsa, Southern Hemisphere South America,-60,  0, -80, -33
    euro, Europe                           , 35, 70, -10,  30
    mide, Middle East                      , 20, 40, -10,  60
    nhaf, Northern Hemisphere Africa       ,  0, 20, -20,  45
    shaf, Southern Hemisphere Africa       ,-35,  0,  10,  45
    boas, Boreal Asia                      , 55, 70,  30, 180
    ceas, Central Asia                     , 30, 55,  30, 143
    seas, Southeast Asia                   ,  5, 30,  65, 120
    eqas, Equatorial Asia                  ,-10, 10, 100, 150
    aust, Australia                        ,-41,-11, 112, 154
    """.strip().split(
        "\n"
    )
    for line in regions:
        lbl, name, lat0, latf, lon0, lonf = line.split(",")
        r.add_latlon_bounds(
            lbl.strip(),
            name.strip(),
            [float(lat0), float(latf)],
            [float(lon0), float(lonf)],
            "Global Fire Emissions Database (GFED)",
        )
