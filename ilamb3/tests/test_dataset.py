import numpy as np
import pandas as pd
import xarray as xr
from pint import application_registry as ureg  # shouldn't need this

from ilamb3 import dataset as dset


def generate_test_dset(seed: int = 1):
    rs = np.random.RandomState(seed)
    lat = [-67.5, -22.5, 22.5, 67.5]
    lon = [-135.0, -45.0, 45.0, 135.0]
    time = pd.date_range(start="2000-01-15", periods=5, freq="30D")
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(len(time), len(lat), len(lon)) * 1e-8,
                coords=[time, lat, lon],
                dims=["time", "lat", "lon"],
            ),
        }
    )
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


def test_integrate():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds, "da"), "da")
    da = da.pint.to("Pg")
    assert np.isclose(da.pint.dequantify(), 31.53474998108379)
    da = dset.integrate_time(dset.integrate_space(ds, "da", region="euro"), "da")
    da = da.pint.to("Pg")
    assert np.isclose(da.pint.dequantify(), 8.370451774151613)


def test_mean():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds["da"], mean=True), "da", mean=True)
    da = da.pint.to(ureg.Unit("g m-2 d-1"))
    assert np.isclose(da.pint.dequantify(), 0.4121668497188348)


def test_std():
    ds = generate_test_dset()
    da = dset.std_time(ds["da"])
    da = da.sum()
    assert np.isclose(da.pint.dequantify(), 4.343761009115869e-08)


def test_basic():
    ds = generate_test_dset()
    try:
        dset.get_dim_name(ds, "depth")  # not in our dimensions
    except Exception as exc:
        assert isinstance(exc, KeyError)
    t0, _ = dset.get_time_extent(ds)
    assert t0 == ds["time"].isel(time=0)
