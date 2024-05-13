import numpy as np
import pandas as pd
import xarray as xr

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


def generate_test_dset_with_depth(seed: int = 1):
    rs = np.random.RandomState(seed)
    lat = [-67.5, -22.5, 22.5, 67.5]
    lon = [-135.0, -45.0, 45.0, 135.0]
    depth = [0.5, 3.0, 7.5, 30.0]
    time = pd.date_range(start="2000-01-15", periods=5, freq="30D")
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(len(time), len(depth), len(lat), len(lon)) * 1e-8,
                coords=[time, depth, lat, lon],
                dims=["time", "depth", "lat", "lon"],
            ),
        }
    )
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


def test_get_dim_name():
    ds = generate_test_dset()
    lon_name = dset.get_dim_name(ds, "lon")
    assert lon_name == "lon"
    try:
        dset.get_dim_name(ds, "depth")  # not in our dimensions
    except Exception as exc:
        assert isinstance(exc, KeyError)


def test_get_time_extent():
    ds = generate_test_dset()
    t0, _ = dset.get_time_extent(ds)
    assert t0 == ds["time"].isel(time=0)
    ds = ds.cf.add_bounds("time")
    t0, _ = dset.get_time_extent(ds)
    assert t0.values == pd.Timestamp("1999-12-31T00:00:00.000000000")


def test_compute_time_measures():
    ds = generate_test_dset()
    try:
        dset.compute_time_measures(ds.isel(time=slice(1)))
    except Exception as exc:
        assert isinstance(exc, ValueError)
    msr = dset.compute_time_measures(ds)
    assert np.allclose(float(msr.pint.dequantify().mean().values), 30.0)
    ds = ds.cf.add_bounds("time")
    msr = dset.compute_time_measures(ds)
    assert np.allclose(float(msr.pint.dequantify().mean().values), 30.0)


def test_coarsen_dataset():
    ds = generate_test_dset()
    ds = dset.coarsen_dataset(ds, res=90.0)
    ds = ds.pint.dequantify()
    assert np.allclose(ds["da"].values[0, 0, 0], 2.51093204e-09)


def test_integrate_time_and_space():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds, "da"), "da")
    da = dset.convert(da, "Pg")
    assert np.isclose(da.pint.dequantify(), 31.53474998108379)
    da = dset.integrate_time(dset.integrate_space(ds, "da", region="euro"), "da")
    da = dset.convert(da, "Pg")
    assert np.isclose(da.pint.dequantify(), 8.370451774151613)


def test_integrate_space_weighted():
    ds = generate_test_dset()
    wgt = ds["da"].sum(dim="time")
    da = dset.integrate_space(ds, "da", weight=wgt)
    assert np.isclose(da.pint.dequantify().sum(), 0.30301439)


def test_integreate_depth():
    ds = generate_test_dset_with_depth()
    ds = dset.integrate_space(
        dset.integrate_time(dset.integrate_depth(ds, "da"), "da"), "da"
    )
    ds = ds.to_dataset(name="da")
    ds = dset.convert(ds, "Pg", varname="da")
    assert np.allclose(ds["da"].pint.dequantify().values, 1002.990371115)


def test_mean():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds["da"], mean=True), "da", mean=True)
    da = dset.convert(da, "g m-2 d-1")
    assert np.isclose(da.pint.dequantify(), 0.4121668497188348)


def test_std():
    ds = generate_test_dset()
    da = dset.std_time(ds["da"])
    da = da.pint.dequantify().sum()
    assert np.isclose(da.pint.dequantify(), 4.343761009115869e-08)
    da = dset.std_time(ds, varname="da")
    da = da.pint.dequantify().sum()
    assert np.isclose(da.pint.dequantify(), 4.343761009115869e-08)


def test_sel():
    ds = generate_test_dset()
    ds = ds.cf.add_bounds("lat")
    ds = dset.sel(ds, "lat", 0, 45)
    dlat = ds["lat_bounds"].diff(dim="bounds").values
    assert dlat.size == 1
    assert np.allclose(dlat[0, 0], 45)


def test_scale_water():
    ds = generate_test_dset()
    da = ds["da"]
    da.attrs["units"] = "kg m-2 s-1"
    da = da.pint.quantify()
    val0 = da.mean().values
    da = dset.scale_by_water_density(da, "mm d-1")
    assert np.allclose(da.mean().values, val0 / 998.2071)
