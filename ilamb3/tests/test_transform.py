import cftime as cf
import numpy as np
import pytest
import xarray as xr

from ilamb3.tests.test_run import generate_test_dset
from ilamb3.transform import ALL_TRANSFORMS

ALL_TRANSFORMS


def gen_msftmz(seed: int = 1):
    rs = np.random.RandomState(seed)
    coords = {}
    # time, basin, lev, lat
    coords["time"] = [
        cf.DatetimeNoLeap(2000 + int(m / 12), (m % 12) + 1, 15) for m in range(1 * 12)
    ]
    coords["basin"] = ["hi", "there"]
    coords["lev"] = 0.5 * (
        np.linspace(0, 100, 10 + 1)[1:] + np.linspace(0, 100, 10 + 1)[:-1]
    )
    coords["lat"] = 0.5 * (
        np.linspace(-90, 90, 10 + 1)[1:] + np.linspace(-90, 90, 10 + 1)[:-1]
    )
    dims = [c for c in coords]
    return xr.Dataset(
        data_vars={
            "msftmz": xr.DataArray(
                rs.rand(*[len(coords[d]) for d in dims]) * 20,
                coords=coords,
                dims=dims,
                name="msftmz",
                attrs={"units": "kg s-1"},
            ),
        }
    )


DATA = {
    "soil_moisture_to_vol_fraction": generate_test_dset(
        "mrsos", "kg m-2", nyear=1, nlat=2, nlon=4
    ),
    "msftmz_to_rapid": gen_msftmz(),
    "ocean_heat_content": xr.merge(
        [
            generate_test_dset(
                "thetao",
                "K",
                nyear=10,
                nlat=2,
                nlon=4,
                ndepth=10,
                scale=20,
                shift=273.0,
            ),
            generate_test_dset("volcello", "m3", nlat=2, nlon=4, ndepth=10, scale=1e10),
        ]
    ),
    "select_depth": generate_test_dset(
        "thetao", "K", nyear=1, nlat=2, nlon=4, ndepth=10
    ),
    "depth_gradient": generate_test_dset(
        "thetao", "K", nyear=1, nlat=2, nlon=4, ndepth=10
    ),
}


@pytest.mark.parametrize(
    "name,kwargs,out,value",
    [
        ("soil_moisture_to_vol_fraction", {}, "mrsos", 0.09702962798695143),
        ("msftmz_to_rapid", {}, "amoc", 1.763701642255525e-08),
        ("ocean_heat_content", {"reference_year": 2000}, "ohc", -0.0046867534613105055),
        ("select_depth", {"value": 0}, "thetao", 9.43861150676275),
        ("select_depth", {"vmin": 1, "vmax": 40}, "thetao", 9.983843647875275),
        ("depth_gradient", {}, "thetao_depth_gradient", -0.005550578833866464),
    ],
)
def test_transform(name, kwargs, out, value):
    transform = ALL_TRANSFORMS[name](**kwargs)
    ds = transform(DATA[name])
    assert np.allclose(value, ds[out].mean().values)
