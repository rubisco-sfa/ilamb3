import re

import cftime as cf
import numpy as np
import pytest
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.tests.test_run import generate_test_dset
from ilamb3.transform import ALL_TRANSFORMS

ALL_TRANSFORMS
PYTHON_VARIABLE = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"


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


def gen_expression_dset(
    expr: str,
    var_meta: dict[str, dict[str, float | str]],
    nyear: int = 2,
    nlat: int = 2,
    nlon: int = 4,
    base_seed: int = 1,
) -> xr.Dataset:
    lhs, rhs_vars = _parse_expr_variables(expr)

    ds_list: list[xr.Dataset] = []
    for i, var in enumerate(rhs_vars):
        meta = var_meta.get(var, {})
        unit = meta.get("unit", "1")
        scale = meta.get("scale", 20.0)
        shift = meta.get("shift", 0.0)
        seed = base_seed + i  # different seed per variable

        ds_var = generate_test_dset(
            name=var,
            unit=unit,
            seed=seed,
            nyear=nyear,
            nlat=nlat,
            nlon=nlon,
            scale=scale,
            shift=shift,
        )
        ds_list.append(ds_var)

    return xr.merge(ds_list)


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
}


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


def _parse_expr_variables(expr: str):
    lhs, rhs = expr.split("=")
    lhs_vars = re.findall(PYTHON_VARIABLE, lhs)
    rhs_vars = re.findall(PYTHON_VARIABLE, rhs)

    assert len(lhs_vars) == 1
    lhs = lhs_vars[0]
    return lhs, rhs_vars


@pytest.mark.parametrize(
    "expr_kwargs,var_meta,value",
    [
        (
            {"expr": "albedo = rsus / rsds", "integrate_time": False},
            {
                "rsus": {"unit": "W m-2", "scale": 150.0, "shift": 0.0},
                "rsds": {"unit": "W m-2", "scale": 300.0, "shift": 100.0},
            },
            0.34482726580080203,
        ),
        (
            {"expr": "albedo = rsus / rsds", "integrate_time": True},
            {
                "rsus": {"unit": "W m-2", "scale": 150.0, "shift": 0.0},
                "rsds": {"unit": "W m-2", "scale": 300.0, "shift": 100.0},
            },
            0.30226715582441277,
        ),
    ],
)
def test_expression(expr_kwargs, var_meta, value):
    expr = expr_kwargs["expr"]

    # build a test dataset for whatever variables appear in expr
    ds = gen_expression_dset(
        expr,
        var_meta=var_meta,
        nyear=2,
        nlat=2,
        nlon=4,
        base_seed=1,
    )

    transform = ALL_TRANSFORMS["expression"](**expr_kwargs)
    ds_out = transform(ds)

    lhs, rhs_vars = _parse_expr_variables(expr)
    assert lhs in ds_out

    if not expr_kwargs.get("integrate_time", False):
        print(ds_out[lhs].mean().values)
        assert np.allclose(ds_out[lhs].mean().values, value)
    else:
        ds_expected = ds.copy()
        for v in rhs_vars:
            arr = ds_expected[v]
            if dset.is_temporal(arr):
                ds_expected[v] = dset.integrate_time(ds_expected, v, mean=True)
        print(ds_out[lhs].mean().values)
        assert np.allclose(ds_out[lhs].mean().values, value)
