"""
An ILAMB transform for applying a label to a dataset.
"""

import os
from pathlib import Path
from typing import Any

import xarray as xr

import ilamb3
import ilamb3.compare as cmp
import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


# FIX: this should just be how all asset loading works
def _load_asset(asset_name: str) -> xr.Dataset:
    # First check each catalog
    for cat in [ilamb3.ilamb3_catalog(), ilamb3.ilamb_catalog(), ilamb3.iomb_catalog()]:
        try:
            ds = xr.open_dataset(cat.fetch(asset_name))
            return ds
        except ValueError:
            pass
    # Next treat it like an absolute path
    asset_path = Path(asset_name)
    if asset_path.is_file():
        ds = xr.open_dataset(asset_path)
        return ds
    # Finally treat it like relative to ILAMB_ROOT
    if "ILAMB_ROOT" in os.environ:
        asset_path = Path(os.environ["ILAMB_ROOT"]) / asset_path
        if asset_path.is_file():
            ds = xr.open_dataset(asset_path)
            return ds
    raise FileNotFoundError(f"Could not find {asset_name=}")


class apply_label(ILAMBTransform):
    """
    Apply a label to the input dataset by NN interpolation.
    """

    def __init__(self, label_source: str, label_key: str, **kwargs: Any):
        self.ds_label = _load_asset(label_source)
        if label_key not in self.ds_label:
            raise ValueError(f"{label_source=} not found in {self.ds_label=}")
        try:
            self.lat_name = dset.get_coord_name(self.ds_label, "lat")
            self.lon_name = dset.get_coord_name(self.ds_label, "lon")
        except KeyError:
            raise ValueError(
                f"{label_source=} does not have lat and/or lon coordinates"
            )
        self.label_key = label_key

    def required_variables(self) -> list[str]:
        """Return the variables this transform requires, none in this case."""
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolate and add the label to the input dataset.
        """
        try:
            lat_name = dset.get_coord_name(ds, "lat")
            lon_name = dset.get_coord_name(ds, "lon")
        except KeyError:
            return ds

        # rename if needed so we can add the result to the input ds
        renames = {
            key: val
            for key, val in {self.lat_name: lat_name, self.lon_name: lon_name}.items()
            if key != val
        }
        ds_label = self.ds_label.rename_dims(renames).rename_vars(renames)

        # interpolate and add to the ds
        ds_label, ds = cmp.adjust_lon(ds_label, ds)
        ds_label = ds_label.interp(
            {lat_name: ds[lat_name], lon_name: ds[lon_name]}, method="nearest"
        )
        ds[self.label_key] = ds_label[self.label_key]
        return ds
