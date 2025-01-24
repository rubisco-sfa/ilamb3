from pathlib import Path

import numpy as np
import xarray as xr

import ilamb3
from ilamb3.regions import Regions
from ilamb3.tests.test_dataset import generate_test_dset


def test_basic():
    reg = Regions()
    assert reg.get_name("euro") == "Europe"
    assert reg.get_source("euro") == "Global Fire Emissions Database (GFED)"


def test_netcdf():
    # can we add regions via a dataset?
    cat = ilamb3.ilamb_catalog()
    dsr = xr.open_dataset(cat.fetch("regions/GlobalLand.nc"))
    reg = Regions()
    lbl = reg.add_netcdf(dsr)
    assert "global" in lbl

    # can we add regions via a netcdf file?
    dsr.to_netcdf("tmp.nc")
    lbl = reg.add_netcdf("tmp.nc")
    assert "global" in lbl

    # does the region work?
    ds = generate_test_dset()
    ds = reg.restrict_to_region(ds, "global")
    assert np.isclose(ds["da"].mean(), 4.28285108e-09)

    Path("tmp.nc").unlink()
