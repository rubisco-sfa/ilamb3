import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ilamb3.compare.neighborhood as iln


def generate_test_dset(seed: int = 1, ntime=None, nlat=None, nlon=None):
    rs = np.random.RandomState(seed)
    coords = []
    dims = []
    if ntime is not None:
        time = pd.date_range(start="2000-01-15", periods=ntime, freq="30D")
        coords.append(time)
        dims.append("time")
    if nlat is not None:
        lat = np.linspace(-90, 90, nlat + 1)
        lat = 0.5 * (lat[1:] + lat[:-1])
        coords.append(lat)
        dims.append("lat")
    if nlon is not None:
        lon = np.linspace(-180, 180, nlon + 1)
        lon = 0.5 * (lon[1:] + lon[:-1])
        coords.append(lon)
        dims.append("lon")
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(*[len(c) for c in coords]) * 30, coords=coords, dims=dims
            ),
        }
    )
    ds["da"].attrs["units"] = "degC"
    return ds


def generate_test_site_dset(ntime, nsite, seed: int = 1):
    rs = np.random.RandomState(seed)
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(ntime, nsite) * 30.0,
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
    ds["da"].attrs["units"] = "degC"
    return ds


SOURCES = {
    "gridded": generate_test_dset(seed=1, ntime=12, nlat=180, nlon=360),
    "sites": generate_test_site_dset(12, 10, seed=2),
    "target": generate_test_site_dset(12, 2, seed=3),
}

WINDOW_TESTDATA = [
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
    "source,target,window_size,window_shape,expected_notnull", WINDOW_TESTDATA
)
def test_extract_neighbors_by_window(
    source, target, window_size, window_shape, expected_notnull
):
    ds = next(
        iter(
            iln.extract_neighbors_by_window(
                SOURCES[source], SOURCES[target], window_size, window_shape
            )
        )
    )
    notnull = int(ds["distance"].notnull().sum())
    assert notnull == expected_notnull


MEAN_TESTDATA = [
    ("gridded", "target", 3.0, False, 15.565812483412744),
    ("gridded", "target", 3.0, True, 16.15160032150377),
    ("sites", "target", 70.0, False, 13.153713684788999),
    ("sites", "target", 70.0, True, 11.739552491536935),
]


@pytest.mark.parametrize(
    "source,target,window_size,weighted,expected_mean", MEAN_TESTDATA
)
def test_neighborhood_mean(source, target, window_size, weighted, expected_mean):
    ds_hood = iln.extract_neighbors_by_window(
        SOURCES[source], SOURCES[target], window_size, "circle"
    )
    out = iln.neighborhood_mean(ds_hood, SOURCES["target"], weighted=weighted)
    validate = float(out.mean()["da"].values)
    print(f"{validate=}")
    assert np.allclose(validate, expected_mean)


CLOSEST_TESTDATA = [
    ("gridded", "target", 3.0, 17.621949545522344),
    ("sites", "target", 70.0, 8.611945290360019),
]


@pytest.mark.parametrize("source,target,window_size,expected_mean", CLOSEST_TESTDATA)
def test_neighborhood_closest(source, target, window_size, expected_mean):
    ds_hood = iln.extract_neighbors_by_window(
        SOURCES[source], SOURCES[target], window_size, "circle"
    )
    out = iln.neighborhood_closest(ds_hood, SOURCES["target"])
    validate = float(out.mean()["da"].values)
    print(f"{validate=}")
    assert np.allclose(validate, expected_mean)
