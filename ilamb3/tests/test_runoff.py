import numpy as np
import xarray as xr

from ilamb3.analysis.runoff_sensitivity import compute_runoff_sensitivity
from ilamb3.regions import Regions
from ilamb3.tests.test_compare import generate_test_dset


def test_runoff_sensitivity():
    ilamb_regions = Regions()
    ilamb_regions.add_latlon_bounds("test", "test", [-80, 80], [-170, 170])
    ds = xr.Dataset(
        {
            "pr": 1e3 * generate_test_dset(seed=1, ntime=240, nlat=10, nlon=20)["da"],
            "tas": 273
            + 1e9 * generate_test_dset(seed=2, ntime=240, nlat=10, nlon=20)["da"],
            "mrro": generate_test_dset(seed=3, ntime=240, nlat=10, nlon=20)["da"],
        }
    )
    ds["mrro"] = (
        ds["mrro"].mean()
        + 1e2 * (ds["pr"] - ds["pr"].mean())
        + 1e-3 * (ds["tas"] - ds["tas"].mean())
    )
    df = compute_runoff_sensitivity(ds, ["test"])
    assert np.allclose(df.loc["test"]["tas Sensitivity"], 28006.878361)
    assert np.allclose(df.loc["test"]["pr Sensitivity"], 139.850275)
