import cftime as cf
import numpy as np
import pytest
import xarray as xr

from ilamb3.tests.test_run import generate_test_dset
from ilamb3.transform import ALL_TRANSFORMS

ALL_TRANSFORMS
PYTHON_VARIABLE = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"


# Helper function to generate msftmz dataset
def gen_msftmz_dset(seed: int = 1):
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


# Helper function to generate permafrost dataset
def gen_permafrost_dset(seed: int = 1):
    rs = np.random.RandomState(seed)
    coords = {}
    # time, depth, lat, lon
    coords["time"] = [
        cf.DatetimeNoLeap(2000 + int(m / 12), (m % 12) + 1, 15) for m in range(3 * 12)
    ]
    coords["depth"] = 0.5 * (
        np.linspace(0, 5, 10 + 1)[1:] + np.linspace(0, 5, 10 + 1)[:-1]
    )
    coords["lat"] = 0.5 * (
        np.linspace(60, 90, 10 + 1)[1:] + np.linspace(60, 90, 10 + 1)[:-1]
    )
    coords["lon"] = 0.5 * (
        np.linspace(-180, 180, 20 + 1)[1:] + np.linspace(-180, 180, 20 + 1)[:-1]
    )
    dims = [c for c in coords]
    ds = xr.Dataset(
        data_vars={
            "tsl": xr.DataArray(
                rs.rand(*[len(coords[d]) for d in dims]) * 20 - 18,
                coords=coords,
                dims=dims,
                name="tsl",
                attrs={"units": "degC"},
            ),
        }
    )
    ds = ds.cf.add_bounds("depth")
    return ds


# Synthetic datasets for testing
DATA = {
    "soil_moisture_to_vol_fraction": generate_test_dset(
        "mrsos", "kg m-2", nyear=1, nlat=2, nlon=4
    ),
    "msftmz_to_rapid": gen_msftmz_dset(),
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
    "active_layer_thickness": gen_permafrost_dset(),
    "integrate_time": generate_test_dset(
        "pr", "kg m-2 s-1", nyear=3, nlat=2, nlon=4, scale=5e-4, shift=0.0
    ),
    "integrate_depth": generate_test_dset(
        "rhopoto", "kg m-3", nlat=2, nlon=4, ndepth=10, scale=4.0, shift=1023.0
    ),
    "integrate_space": generate_test_dset(
        "rlut", "W m-2", nyear=1, nlat=2, nlon=4, scale=100.0, shift=200.0
    ),
    "expression": xr.merge(
        [
            generate_test_dset(
                "rsus", "W m-2", nyear=1, nlat=2, nlon=4, scale=150.0, shift=0.0
            ),
            generate_test_dset(
                "rsds", "W m-2", nyear=1, nlat=2, nlon=4, scale=300.0, shift=0.0
            ),
        ]
    ),
}


# Test all the above transforms
@pytest.mark.parametrize(
    "name,kwargs,out,value",
    [
        ("soil_moisture_to_vol_fraction", {}, "mrsos", 0.09702962798695143),
        ("msftmz_to_rapid", {}, "amoc", 1.763701642255525e-08),
        ("ocean_heat_content", {"reference_year": 2000}, "ohc", -0.0046867534613105055),
        ("select_depth", {"value": 0}, "thetao", 9.43861150676275),
        ("select_depth", {"vmin": 1, "vmax": 40}, "thetao", 9.983843647875275),
        ("depth_gradient", {}, "thetao", -0.005550578833866464),
        (
            "active_layer_thickness",
            {"method": "slater2013"},
            "active_layer_thickness",
            2.224852071005917,
        ),
    ],
)
def test_transform(name, kwargs, out, value):
    transform = ALL_TRANSFORMS[name](**kwargs)
    ds = transform(DATA[name])
    assert np.allclose(value, ds[out].mean().values)


@pytest.mark.parametrize(
    "name,out",
    [
        ("integrate_time", "pr"),
        ("integrate_depth", "rhopoto"),
        ("integrate_space", "rlut"),
    ],
)
def test_integrate_common(name, out):
    original_ds = DATA[name]

    # Test mean=False (sum)
    transform = ALL_TRANSFORMS[name](varname=out, mean=False)
    integral = transform(original_ds.copy())

    # Test mean=True (mean)
    transform_mean = ALL_TRANSFORMS[name](varname=out, mean=True)
    integral_mean = transform_mean(original_ds.copy())

    # Dim(s) should be removed for spatial integration
    if name == "integrate_space":
        for d in ("lat", "lon"):
            if d in original_ds[out].dims:
                assert d not in integral[out].dims
                assert d not in integral_mean[out].dims
    else:
        if transform.dim in DATA[name][out].dims:
            assert transform.dim not in integral[out].dims
            assert transform.dim not in integral_mean[out].dims

    # Mean should keep original units
    assert integral_mean[out].attrs["units"] == DATA[name][out].attrs["units"]

    # Sum should NOT keep original units
    assert integral[out].attrs["units"] != DATA[name][out].attrs["units"]

    # Sum and Mean values should differ
    assert not np.allclose(integral[out].values, integral_mean[out].values)
