import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ilamb3.compare.neighborhood import extract_neighbors_by_window
from ilamb3.tests.test_compare import generate_test_dset


def generate_test_site_dset(ntime, nsite, seed: int = 1):
    rs = np.random.RandomState(seed)
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(ntime, nsite) * 1e-8,
                coords={
                    "time": pd.date_range(start="2000-01-15", periods=ntime, freq="30D")
                },
                dims=("time", "site"),
            ),
        }
    )
    ds["da"] = ds["da"].assign_coords(
        {
            "lat": xr.DataArray((rs.rand(nsite) - 0.5) * 180, dims="site"),
            "lon": xr.DataArray((rs.rand(nsite) - 0.5) * 360, dims="site"),
        }
    )
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


SOURCES = {
    "gridded": generate_test_dset(seed=1, ntime=12, nlat=180, nlon=360),
    "sites": generate_test_site_dset(12, 10, seed=2),
    "target": generate_test_site_dset(12, 2, seed=3),
}

TESTDATA = [
    ("gridded", "target", 1.5, "box", 9),
    ("gridded", "target", 3.0, "box", 36),
    ("gridded", "target", 1.5, "circle", 7),
    ("gridded", "target", 3.0, "circle", 28),
    ("sites", "target", 30.0, "box", 10),
    ("sites", "target", 70.0, "box", 10),
    ("sites", "target", 30.0, "circle", 1),
    ("sites", "target", 70.0, "circle", 4),
    pytest.param("sites", "target", -1.0, "circle", 10, marks=pytest.mark.xfail),
    pytest.param("gridded", "gridded", 3.0, "circle", 10, marks=pytest.mark.xfail),
]


@pytest.mark.parametrize(
    "source,target,window_size,window_shape,expected_notnull", TESTDATA
)
def test_extract_neighbors_by_window(
    source, target, window_size, window_shape, expected_notnull
):
    ds = next(
        iter(
            extract_neighbors_by_window(
                SOURCES[source], SOURCES[target], window_size, window_shape
            )
        )
    )
    notnull = int(ds["distance"].notnull().sum())
    assert notnull == expected_notnull
