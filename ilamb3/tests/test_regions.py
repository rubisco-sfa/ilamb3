from pathlib import Path

import intake
import numpy as np

from ilamb3.regions import Regions
from ilamb3.tests.test_dataset import generate_test_dset


def test_basic():
    reg = Regions()
    assert reg.get_name("euro") == "Europe"
    assert reg.get_source("euro") == "Global Fire Emissions Database (GFED)"


def test_netcdf():
    # can we add regions via a dataset?
    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/nocollier/intake-ilamb/main/ilamb.yaml"
    )
    dsr = cat["regions_global_land | ILAMB"].read()
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
