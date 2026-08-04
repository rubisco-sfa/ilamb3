from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import ilamb3
import ilamb3.regions as ilr
from ilamb3.tests.test_dataset import generate_test_dset


def test_basic():
    reg = ilr.Regions()
    assert reg.get_name("euro") == "Europe"
    assert reg.get_source("euro") == "Global Fire Emissions Database"


def test_netcdf():
    # can we add regions via a dataset?
    cat = ilamb3.ilamb_catalog()
    dsr = xr.open_dataset(cat.fetch("regions/GlobalLand.nc"))
    reg = ilr.Regions()
    lbl = reg.add_region_netcdf(dsr)
    assert "global" in lbl

    # can we add regions via a netcdf file?
    dsr.to_netcdf("tmp.nc")
    lbl = reg.add_region_netcdf("tmp.nc")
    assert "global" in lbl

    # does the region work?
    ds = generate_test_dset()
    ds = reg.mask(ds, "global")
    assert np.isclose(ds["da"].mean(), 4.28285108e-09)

    Path("tmp.nc").unlink()


@pytest.mark.xfail
def test_validate():
    reg = ilr.Regions()
    reg.validate_label("not_a_region")
