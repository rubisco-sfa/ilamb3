import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ilamb3 import post


def generate_test_dset(seed: int = 1, shift: bool = False):
    rs = np.random.RandomState(seed)
    lat = [-67.5, -22.5, 22.5, 67.5]
    lon = [-135.0 + 360 * shift, -45.0 + 360 * shift, 45.0, 135.0]
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(len(lat), len(lon)) * 1e-8,
                coords=[lat, lon],
                dims=["lat", "lon"],
            ),
        }
    )
    ds["da"].attrs["units"] = "kg m-2 s-1"
    return ds


def generate_test_site_dset(seed: int = 1):
    rs = np.random.RandomState(seed)
    lat = xr.DataArray(data=[-67.5, -22.5, 22.5, 67.5], dims=["site"])
    lon = xr.DataArray(data=[-135.0, -45.0, 45.0, 135.0], dims=["site"])
    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(lat.size) * 1e-8,
                dims=["time", "site"],
            ),
        }
    )
    ds["da"] = ds["da"].assign_coords({"lat": lat, "lon": lon})
    ds["da"].attrs["units"] = "kg m-2 s-1"
    ds["da"].attrs["coordinates"] = "lat lon"
    return ds


def test_get_plot_limits():
    limits = post.get_plot_limits(
        generate_test_dset(seed=1).rename_vars({"da": "a"}),
        {
            "m": generate_test_dset(seed=1).rename_vars({"da": "b"}),
            "n": generate_test_dset(seed=1).rename_vars({"da": "c"}),
        },
    )
    assert not (set(limits) - set(["a", "b", "c"]))


def test_extents():
    ds = generate_test_dset()
    ext = post._plot_extents(ds["da"])
    assert np.allclose(ext, [-135.0, 135.0, -67.5, 67.5])


def test_proj():
    ds = generate_test_dset()
    proj = post._plot_projection(post._plot_extents(ds["da"]))
    assert isinstance(proj[0], ccrs.PlateCarree)
