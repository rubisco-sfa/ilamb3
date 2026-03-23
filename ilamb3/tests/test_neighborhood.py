import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ilamb3.compare.neighborhood import extract_neighbors_by_window
from ilamb3.tests.test_compare import generate_test_dset


def generate_test_site_dset(ntime, nsite, seed: int = 1):
    rs = np.random.RandomState(seed)
    coords = {}
    dims = []
    if ntime is not None:
        time = pd.date_range(start="2000-01-15", periods=ntime, freq="30D")
        coords["time"] = time
        dims.append("time")
    if nsite is not None:
        lat = xr.DataArray((np.random.rand(nsite) - 0.5) * 180, dims="site")
        lon = xr.DataArray((np.random.rand(nsite) - 0.5) * 360, dims="site")
        dims.append("site")
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(rs.rand(ntime, nsite) * 1e-8, coords=coords, dims=dims),
        }
    )
    ds["da"] = ds["da"].assign_coords({"lat": lat, "lon": lon})
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


SOURCES = {
    "gridded": generate_test_dset(seed=1, ntime=12, nlat=180, nlon=360),
    "sites": generate_test_site_dset(12, 10, seed=2),
    "target": generate_test_site_dset(12, 2, seed=3),
}


# XFAIL: ds_target is not a site, negative window_size
@pytest.mark.parametrize("source", ["gridded", "sites"])
@pytest.mark.parametrize("window_size", [1.5, 3.0])
@pytest.mark.parametrize("window_shape", ["box", "circle"])
def test_extract_neighbors_by_window(source, window_size, window_shape):
    ds = next(
        iter(
            extract_neighbors_by_window(
                SOURCES[source], SOURCES["target"], window_size, window_shape
            )
        )
    )
    # print("BLAH", source, window_size, window_shape, int(ds["distance"].isnull().sum()))
    check = {
        "gridded": {1.5: {"box": 0, "circle": 2}, 3.0: {"box": 0, "circle": 9}},
        "sites": {1.5: {"box": 0, "circle": 10}, 3.0: {"box": 0, "circle": 10}},
    }
    assert check[source][window_size][window_shape] == int(
        ds["distance"].isnull().sum()
    )
