import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ilamb3.dataset as ild


@pytest.fixture
def ds_bounds() -> xr.Dataset:
    time = pd.date_range(start="2000-01-15", periods=36, freq="30D")
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    ds = xr.Dataset(
        {
            "da": xr.DataArray(
                np.random.rand(len(time), len(lat), len(lon)),
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lat, "lon": lon},
                attrs={"units": "kg m-2 s-1", "bounds": "da_bnds"},
            ),
            "da_bnds": xr.DataArray(
                0.1 * np.random.rand(len(time), len(lat), len(lon), 2),
                dims=("time", "lat", "lon", "nb"),
                coords={"time": time, "lat": lat, "lon": lon},
            ),
        }
    )
    ds["da_bnds"][..., 0] = ds["da"] - ds["da_bnds"][..., 0]
    ds["da_bnds"][..., 1] = ds["da"] + ds["da_bnds"][..., 1]
    return ds


@pytest.fixture
def ds_bounds_std(ds_bounds: xr.Dataset) -> xr.Dataset:
    ds_bounds["da_bnds"] = ds_bounds["da_bnds"].mean(dim="nb")
    return ds_bounds


@pytest.fixture
def ds_anc(ds_bounds: xr.Dataset) -> xr.Dataset:
    ds_bounds["da"].attrs = {"units": "kg m-2 s-1", "ancillary_variables": "low high"}
    ds_bounds["low"] = ds_bounds["da_bnds"].isel(nb=0)
    ds_bounds["high"] = ds_bounds["da_bnds"].isel(nb=1)
    ds_bounds = ds_bounds.drop_vars("da_bnds")
    return ds_bounds


@pytest.fixture
def ds_anc_std(ds_anc: xr.Dataset) -> xr.Dataset:
    ds_anc["da"].attrs = {"units": "kg m-2 s-1", "ancillary_variables": "std"}
    ds_anc["std"] = ds_anc["low"]
    ds_anc = ds_anc.drop_vars(["low", "high"])
    return ds_anc


@pytest.fixture
def ds_anc_cov(ds_anc_std: xr.Dataset) -> xr.Dataset:
    ds_anc_std["std"].attrs["name"] = "coefficient of variation"
    return ds_anc_std


@pytest.fixture
def ds_anc_too_many(ds_anc: xr.Dataset) -> xr.Dataset:
    ds_anc["da"].attrs = {"units": "kg m-2 s-1", "ancillary_variables": "low mid high"}
    ds_anc["mid"] = ds_anc["low"]
    return ds_anc


@pytest.mark.parametrize(
    "ds_name",
    [
        "ds_bounds",
        "ds_bounds_std",
        "ds_anc",
        "ds_anc_std",
        "ds_anc_cov",
        pytest.param("ds_anc_too_many", marks=pytest.mark.xfail),
    ],
)
def test_scalar(ds_name: str, request) -> None:
    ds = request.getfixturevalue(ds_name)
    unc = ild.get_scalar_uncertainty(ds, "da")
    assert np.allclose(unc["uncert"].shape, ds["da"].shape)


@pytest.mark.parametrize(
    "ds_name",
    [
        "ds_bounds",
        "ds_bounds_std",
        "ds_anc",
        "ds_anc_std",
        "ds_anc_cov",
        pytest.param("ds_anc_too_many", marks=pytest.mark.xfail),
    ],
)
def test_interval(ds_name: str, request) -> None:
    ds = request.getfixturevalue(ds_name)
    unc = ild.get_interval_uncertainty(ds, "da")
    bnds_dim = set(unc.dims).difference(ds["da"].dims)
    assert len(bnds_dim) == 1
    assert len(unc[bnds_dim.pop()]) == 2
