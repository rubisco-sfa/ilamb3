import numpy as np
import pandas as pd
import xarray as xr

from ilamb3 import dataset as dset


def generate_test_dset(seed: int = 1, shift: bool = False):
    rs = np.random.RandomState(seed)
    lat = [-67.5, -22.5, 22.5, 67.5]
    lon = [-135.0 + 360 * shift, -45.0 + 360 * shift, 45.0, 135.0]
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
    ds["da"].attrs["units"] = "kg m-3 s-1"
    return ds


def generate_test_site_dset(seed: int = 1):
    rs = np.random.RandomState(seed)
    lat = xr.DataArray(data=[-67.5, -22.5, 22.5, 67.5], dims=["site"])
    lon = xr.DataArray(data=[-135.0, -45.0, 45.0, 135.0], dims=["site"])
    time = pd.date_range(start="2000-01-15", periods=5, freq="30D")
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(len(time), lat.size) * 1e-8,
                coords={"time": time},
                dims=["time", "site"],
            ),
        }
    )
    ds["da"] = ds["da"].assign_coords({"lat": lat, "lon": lon})
    ds["da"].attrs["units"] = "kg m-2 s-1"
    ds["da"].attrs["coordinates"] = "lat lon"
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
    assert np.allclose(ds["da"].values[0, 0, 0], 2.51093204e-09)


def test_integrate_time_and_space():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds, "da"), "da")
    da = dset.convert(da, "Pg")
    assert np.isclose(da, 31.53474998108379)
    da = dset.integrate_time(dset.integrate_space(ds, "da", region="euro"), "da")
    da = dset.convert(da, "Pg")
    assert np.isclose(da, 8.370451774151613)


def test_integrate_space_weighted():
    ds = generate_test_dset()
    wgt = ds["da"].sum(dim="time")
    da = dset.integrate_space(ds, "da", weight=wgt)
    assert np.isclose(da.sum(), 0.30301439)


def test_integrate_depth():
    ds = generate_test_dset_with_depth()
    ds = dset.integrate_space(
        dset.integrate_time(dset.integrate_depth(ds, "da"), "da"), "da"
    )
    ds = ds.to_dataset(name="da")
    ds = dset.convert(ds, "Pg", varname="da")
    assert np.allclose(ds["da"], 1002.990371115)


def test_mean():
    ds = generate_test_dset()
    da = dset.integrate_space(dset.integrate_time(ds["da"], mean=True), "da", mean=True)
    da = dset.convert(da, "g m-2 d-1")
    assert np.isclose(da, 0.4121668497188348)


def test_std():
    ds = generate_test_dset()
    da = dset.std_time(ds["da"])
    da = da.sum()
    assert np.isclose(da, 4.343761009115869e-08)
    da = dset.std_time(ds, varname="da")
    da = da.sum()
    assert np.isclose(da, 4.343761009115869e-08)


def test_sel():
    ds = generate_test_dset()
    ds = ds.cf.add_bounds("lat")
    ds = dset.sel(ds, "lat", 0, 45)
    dlat = ds["lat_bounds"].diff(dim="bounds").values
    assert dlat.size == 1
    assert np.allclose(dlat[0, 0], 45)


def test_scale_water():
    da = xr.DataArray(1.0)
    da.attrs["units"] = "kg m-2 s-1"
    da = dset.scale_by_water_density(da, "mm d-1")
    assert np.allclose(da.pint.dequantify(), 1 / 998.2071)  # dequantify needed
    da = xr.DataArray(1.0)
    da.attrs["units"] = "mm d-1"
    da = dset.scale_by_water_density(da, "kg m-2 s-1")
    assert np.allclose(da.pint.dequantify(), 998.2071)
    da = xr.DataArray(1.0)
    da.attrs["units"] = "kg m-2"
    da = dset.scale_by_water_density(da, "mm d-1")
    assert np.allclose(da.pint.dequantify(), 1)


def test_is_spatial_or_site():
    ds = generate_test_dset()
    assert dset.is_spatial(ds)
    assert not dset.is_site(ds)
    ds = generate_test_site_dset()
    assert not dset.is_spatial(ds)
    assert dset.is_site(ds)


def test_shift_lon():
    ds = generate_test_dset(shift=True)
    ds = dset.shift_lon(ds)
    assert ds["lon"].min() < -120


def test_cell_measures():
    lats = np.linspace(-90, 90, 4)
    lons = np.linspace(-180, 180, 7)
    ds = xr.DataArray(
        np.random.rand(3, 6),
        coords={
            "lat": 0.5 * (lats[:-1] + lats[1:]),
            "lon": 0.5 * (lons[:-1] + lons[1:]),
        },
        dims=["lat", "lon"],
        name="da",
    ).to_dataset()
    ds["lat_bnds"] = (("lat", "nb"), np.array([lats[:-1], lats[1:]]).T)
    ds["lon_bnds"] = (("lon", "nb"), np.array([lons[:-1], lons[1:]]).T)
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["lon"].attrs["bounds"] = "lon_bnds"
    assert np.allclose(dset.compute_cell_measures(ds).mean(), 2.83369151e13)
