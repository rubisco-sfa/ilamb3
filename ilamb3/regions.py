"""Region definitions for use in the ILAMB system."""

import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import ilamb3.compare as ilc
import ilamb3.dataset as ild
import ilamb3.load as ill


def restrict_to_bbox(
    da: xr.DataArray, lat0: float, latf: float, lon0: float, lonf: float
):
    """Return the dataarray selected to the nearest bounding box.

    This is awkward because as of xarray `v2023.6.0`, the `method` keyword
    cannot be used in slices. Note that this routine will sort the dimensions
    because slicing does not work well on unsorted indices.
    """
    lat_name = ild.get_coord_name(da, "lat")
    lon_name = ild.get_coord_name(da, "lon")
    if ild.is_site(da):
        site_name = ild.get_dim_name(da, "site")
        da = da.sel(
            {
                site_name: (
                    (da[lat_name] >= lat0)
                    & (da[lat_name] <= latf)
                    & (da[lon_name] >= lon0)
                    & (da[lon_name] <= lonf)
                )
            }
        )
    else:
        da = da.sel(
            {
                lat_name: slice(
                    da[lat_name].sel({lat_name: lat0}, method="nearest"),
                    da[lat_name].sel({lat_name: latf}, method="nearest"),
                ),
                lon_name: slice(
                    da[lon_name].sel({lon_name: lon0}, method="nearest"),
                    da[lon_name].sel({lon_name: lonf}, method="nearest"),
                ),
            }
        )
    return da


class RegionType(ABC):
    """
    An abstract definition for how regions are encoded in the ilamb system.

    Note
    ----
    The only people who need to interact with this class and its children are
    those that wish to add a new class of region support into ilamb (e.g., support
    for shapefiles). For developers writing code that needs to use the current region
    system, simply import the `Regions` object.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the region name.

        Note
        ----
        The name is only used when displaying the output (e.g. in the region
        pulldown menu in the html dataset page output). It should be something
        human readable and may contain spaces and special characters.
        """
        ...

    @property
    @abstractmethod
    def source(self) -> str:
        """
        Return the region source.

        Note
        ----
        The source should be a string that reflects from where a region
        definition originates. It is currently only used in the CMEC
        specification.
        """
        ...

    @property
    @abstractmethod
    def bbox(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the region.

        Note
        ----
        The (lat_min, lat_max, lon_min, lon_max) limits of the non-null data in
        the region.
        """
        ...

    @abstractmethod
    def mask(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        Mask the input data to the region.

        Note
        ----
        This routine should mask out data (set to NaN) data values falling outside
        the region.
        """
        ...


class RegionLatLon(RegionType):
    """
    A region defined by latitude and longitude bounds.
    """

    def __init__(
        self,
        name: str,
        source: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        self._name = name
        self._source = source
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._source

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.lat_min, self.lat_max, self.lon_min, self.lon_max

    def mask(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        Mask the data that falls outside of this region.
        """
        if isinstance(data, xr.DataArray):
            out = restrict_to_bbox(
                data, self.lat_min, self.lat_max, self.lon_min, self.lon_max
            )
        else:
            out = xr.Dataset(
                {
                    key: restrict_to_bbox(
                        da, self.lat_min, self.lat_max, self.lon_min, self.lon_max
                    )
                    for key, da in data.items()
                    if (ild.is_gridded(da) or ild.is_site(da))
                }
            )
        return out

    @staticmethod
    def parse_yaml_region(region_yaml_file: Path) -> list[str]:
        """

        Examples
        --------
        Custom regions in YAML:

        .. code-block:: yaml

            conus:
                name: Continental United States
                lat_min: 24.3
                lat_max: 49.4
                lon_min: -125.0
                lon_max: -67.0

        """
        KEYS = ["name", "lat_min", "lat_max", "lon_min", "lon_max"]
        with open(region_yaml_file) as fin:
            regions = yaml.safe_load(fin)
        labels_added = []
        for label, region in regions.items():
            Regions.validate_label(label)
            if set(KEYS).issubset(region.keys()):
                labels_added.append(label)
                Regions().add_region_latlon(
                    label,
                    region["name"],
                    str(region_yaml_file),
                    *[region[key] for key in KEYS[1:]],
                )
        return labels_added


class RegionNetCDF(RegionType):
    def __init__(self, name: str, source: str, da: xr.DataArray):
        self._name = name
        self._source = source
        self.da = da

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._source

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        lat_name = ild.get_dim_name(self.da, "lat")
        lon_name = ild.get_dim_name(self.da, "lon")
        lats = xr.where(self.da.any(dim=lon_name), self.da[lat_name], np.nan)
        lons = xr.where(self.da.any(dim=lat_name), self.da[lon_name], np.nan)
        return (
            float(lats.min()),
            float(lats.max()),
            float(lons.min()),
            float(lons.max()),
        )

    @staticmethod
    def mask_dataarray(da: xr.DataArray, da_region: xr.DataArray) -> xr.DataArray:
        da, da_region = ilc.adjust_lon(da, da_region)  # type: ignore
        lat_name = ild.get_coord_name(da, "lat")
        lon_name = ild.get_coord_name(da, "lon")
        da_region = da_region.rename(
            {
                ild.get_dim_name(da_region, "lat"): lat_name,
                ild.get_dim_name(da_region, "lon"): lon_name,
            }
        )
        args = {lat_name: da[lat_name], lon_name: da[lon_name]}
        if ild.is_site(da):
            out = xr.where(
                da_region.astype(bool).sel(args, method="nearest", tolerance=1.0),
                da,
                np.nan,
            )
            out = out.assign_coords({c: da[c] for c in da.coords if c not in da.dims})
        else:
            out = xr.where(
                da_region.interp(
                    args, method="nearest", kwargs={"fill_value": "extrapolate"}
                ),
                da,
                np.nan,
            )
        return out

    def mask(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        if isinstance(data, xr.DataArray):
            out = self.mask_dataarray(data, self.da)
        else:
            out = xr.Dataset(
                {
                    key: self.mask_dataarray(da, self.da)
                    for key, da in data.items()
                    if (ild.is_gridded(da) or ild.is_site(da))
                }
            )
        return out

    @staticmethod
    def parse_v2_region(ds: xr.Dataset, source: str) -> list[str]:
        # Identify the variables that could be the IDs
        possible_id_vars = [v for v, da in ds.items() if ild.is_gridded(da)]
        if not possible_id_vars:
            return []
        ids_var = (
            possible_id_vars.pop(possible_id_vars.index("ids"))
            if "ids" in possible_id_vars
            else possible_id_vars[0]
        )
        # Scan the dataset for ilamb v2 format needs
        ds.load()
        da_region = ds[ids_var]
        if "labels" not in da_region.attrs:
            return []
        labels_var = da_region.attrs["labels"]
        if labels_var not in ds:
            return []
        names_var = da_region.attrs.get("names", da_region.attrs["labels"])
        if names_var not in ds:
            names_var = labels_var
        # Grab the labels and names and add the regions to the global dict
        labels = [str(lbl).lower() for lbl in ds[labels_var].to_numpy()]
        names = list(ds[names_var].to_numpy())
        for label, name in zip(labels, names):
            da = xr.where(da_region == labels.index(label), 1, 0)
            Regions.validate_label(label)
            Regions._regions[label] = RegionNetCDF(name, source, da)
        return labels

    @staticmethod
    def parse_cf_region(ds: xr.Dataset, source: str) -> list[str]:
        # Identify the variables that could be the IDs
        possible_id_vars = [v for v, da in ds.items() if ild.is_gridded(da)]
        if not possible_id_vars:
            return []
        ids_var = (
            possible_id_vars.pop(possible_id_vars.index("ids"))
            if "ids" in possible_id_vars
            else possible_id_vars[0]
        )
        # Scan the dataset for ilamb cf format needs
        ds.load()
        da_region = ds[ids_var]
        if not (
            "flag_values" in da_region.attrs
            and "flag_meanings" in da_region.attrs
            and "flag_descriptions" in da_region.attrs
        ):
            # This means the dataset does not follow the standard
            return []
        values = da_region.attrs["flag_values"]
        labels = [
            str(lbl).lower() for lbl in da_region.attrs["flag_meanings"].split(" ")
        ]
        names = da_region.attrs["flag_descriptions"].split(" ")
        for value, label, name in zip(values, labels, names):
            da = xr.where(da_region == value, 1, 0)
            Regions.validate_label(label)
            Regions._regions[label] = RegionNetCDF(name, source, da)
        return labels


class Regions:
    _regions: dict[str, RegionType] = {}

    def __init__(self):
        if not Regions._regions:
            self.register_regions_gfed()

    def __repr__(self) -> str:
        # TODO: Add behavior `ilamb run --region-source Blah.nc --regions-list`
        # which loads all regions and this prints this to the terminal
        df = (
            pd.DataFrame(
                [
                    {
                        "Label": label,
                        "Name": region.name,
                        "Type": type(region).__name__,
                        "Source": region.source,
                    }
                    for label, region in Regions._regions.items()
                ]
            )
            .sort_values(["Source", "Label"])
            .set_index(["Source", "Label"])
        )
        return df.to_string()

    @property
    def regions(self) -> list[str]:
        return list(Regions._regions.keys())

    def get_name(self, label: str) -> str:
        if label not in Regions._regions:
            raise ValueError(
                f"The region {label=} is not in the registered regions. Here is what is currently registered:\n\n{self}"
            )
        return Regions._regions[label].name

    def get_source(self, label: str) -> str:
        if label not in Regions._regions:
            raise ValueError(
                f"The region {label=} is not in the registered regions. Here is what is currently registered:\n\n{self}"
            )
        return Regions._regions[label].source

    def add_region_yaml(self, data: str | Path):
        pass

    def add_region_latlon(
        self,
        label: str,
        name: str,
        source: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        self.validate_label(label)
        Regions._regions[label] = RegionLatLon(
            name, source, lat_min, lat_max, lon_min, lon_max
        )

    def add_region_netcdf(self, data: str | Path | xr.Dataset) -> list[str]:
        if isinstance(data, xr.Dataset):
            source = data.attrs.get("title", "custom dataset")
        else:
            source = str(data)
            data = ill.load_key_or_filename(str(data))
        labels_added = []
        labels_added += RegionNetCDF.parse_v2_region(data, source)
        labels_added += RegionNetCDF.parse_cf_region(data, source)
        return labels_added

    def mask(
        self,
        data: xr.Dataset | xr.DataArray,
        label: str | None,
        trim: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        if label is None:
            return data
        if label not in Regions._regions:
            raise ValueError(
                f"The region {label=} is not in the registered regions. Here is what is currently registered:\n\n{self}"
            )
        data = Regions._regions[label].mask(data)
        if trim:
            bbox = Regions._regions[label].bbox
            if isinstance(data, xr.DataArray):
                data = restrict_to_bbox(data, *bbox)
            else:
                data = xr.Dataset(
                    {
                        key: restrict_to_bbox(da, *bbox)
                        for key, da in data.items()
                        if (ild.is_gridded(da) or ild.is_site(da))
                    }
                )
        return data

    @staticmethod
    def register_regions_gfed():
        region_data = """
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
        """.strip().split("\n")
        for line in region_data:
            lbl, name, lat0, latf, lon0, lonf = line.split(",")
            # To avoid circular references, we must add these directly
            Regions._regions[lbl.strip()] = RegionLatLon(
                name.strip(),
                "Global Fire Emissions Database",
                float(lat0),
                float(latf),
                float(lon0),
                float(lonf),
            )

    @staticmethod
    def validate_label(label: str):
        if label in Regions._regions:
            warnings.warn(
                f"The region {label=} already exists, we are overwriting with this new definition."
            )
        match = re.match("^[a-z0-9]+$", label)
        if not match:
            raise ValueError(
                f"The region {label=} can only be lowercase characters and digits with no spaces or special characters."
            )

    @staticmethod
    def region_scalars_to_map(
        grid: xr.DataArray, scalars: dict[str, float]
    ) -> xr.DataArray:
        # check that regions are part of our system
        regions = Regions()
        diff = set(scalars) - set(regions.regions)
        if diff:
            raise ValueError(
                f"Keys in the scalar dictionary aren't registered regions: {diff}"
            )
        grid = xr.ones_like(grid)
        da = xr.concat(
            [
                regions.mask(grid, region, trim=False) * value
                for region, value in scalars.items()
            ],  # type: ignore
            dim="region",
        )
        mask = da.isnull().all(dim="region")
        da = da.sum(dim="region")
        da = xr.where(~mask, da, np.nan)
        return da
