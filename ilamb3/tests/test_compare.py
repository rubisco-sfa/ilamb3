import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp


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
                rs.rand(*[len(c) for c in coords]) * 1e-8, coords=coords, dims=dims
            ),
        }
    )
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


def test_nest_spatial_grids():
    ds1 = generate_test_dset(nlat=2, nlon=3)
    ds2 = generate_test_dset(nlat=5, nlon=7)
    ds1 = ds1.cf.add_bounds("lat")
    ds1_, ds2_ = cmp.nest_spatial_grids(ds1, ds2)
    assert np.allclose((ds1_ - ds2_).sum()["da"], -6.06395917e-08)
    dsa, dsb = cmp.pick_grid_aligned(ds1, ds2, ds1_, ds2_)
    xr.testing.assert_allclose(dsa, ds1_)
    xr.testing.assert_allclose(dsb, ds2_)


def test_spatial_align():
    ds1 = generate_test_dset(nlat=2, nlon=3)
    ds2 = generate_test_dset(nlat=3, nlon=4)
    assert not cmp.is_spatially_aligned(ds1, ds2)
    ds2 = generate_test_dset(nlat=2, nlon=4)
    assert not cmp.is_spatially_aligned(ds1, ds2)
    ds2 = generate_test_dset(nlat=2, nlon=3)
    ds2["lat"] = [1, 2]
    assert not cmp.is_spatially_aligned(ds1, ds2)
    ds2 = generate_test_dset(nlat=2, nlon=3)
    ds2["lon"] = [1, 2, 3]


def test_grid_align():
    ds1 = generate_test_dset(nlat=2, nlon=3)
    ds2 = generate_test_dset(nlat=2, nlon=3)
    ds1_, ds2_ = cmp.pick_grid_aligned(ds1, ds2)
    xr.testing.assert_allclose(ds1_, ds2_)
    ds2 = generate_test_dset(nlat=3, nlon=4)
    ds1_, ds2_ = cmp.pick_grid_aligned(ds1, ds2)
    xr.testing.assert_allclose(ds1_["lat"], ds2_["lat"])
    xr.testing.assert_allclose(ds1_["lon"], ds2_["lon"])


def test_trim_time():
    ds = {"1": generate_test_dset(ntime=2), "2": generate_test_dset(ntime=3)}
    out = cmp.trim_time(**ds)
    assert len(out) == 2
    assert all([ds["time"].size == 2 for _, ds in out.items()])
    out1, out2 = cmp.trim_time(generate_test_dset(ntime=4), **ds)
    assert len(out1) == 1
    assert len(out2) == 2


def test_same_spatial_grid():
    grid = generate_test_dset(nlat=2, nlon=2)
    out = cmp.same_spatial_grid(
        grid,
        generate_test_dset(nlat=2, nlon=3),
    )
    xr.testing.assert_allclose(grid["lat"], out[0]["lat"])
    xr.testing.assert_allclose(grid["lon"], out[0]["lon"])
    out = cmp.same_spatial_grid(
        grid,
        **{"1": generate_test_dset(nlat=3, nlon=2)},
    )
    xr.testing.assert_allclose(grid["lat"], out["1"]["lat"])
    xr.testing.assert_allclose(grid["lon"], out["1"]["lon"])
    out, outkw = cmp.same_spatial_grid(
        grid,
        generate_test_dset(nlat=2, nlon=3),
        **{"1": generate_test_dset(nlat=3, nlon=2)},
    )
    xr.testing.assert_allclose(grid["lat"], out[0]["lat"])
    xr.testing.assert_allclose(grid["lon"], out[0]["lon"])
    xr.testing.assert_allclose(grid["lat"], outkw["1"]["lat"])
    xr.testing.assert_allclose(grid["lon"], outkw["1"]["lon"])
