import tempfile
from pathlib import Path

import cftime as cf
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ilamb3
import ilamb3.run as run


def generate_test_dset(
    name: str,
    unit: str,
    seed: int = 1,
    nyear: int | None = None,
    nlat: int | None = None,
    nlon: int | None = None,
    ndepth: int | None = None,
    latlon2d: bool = False,
):
    rs = np.random.RandomState(seed)
    coords = {}
    if nyear:
        coords["time"] = [
            cf.DatetimeNoLeap(2000 + int(m / 12), (m % 12) + 1, 15)
            for m in range(nyear * 12)
        ]
    if ndepth:
        coords["depth"] = 0.5 * (
            np.linspace(0, 100, ndepth + 1)[1:] + np.linspace(0, 100, ndepth + 1)[:-1]
        )
    if nlat:
        coords["lat"] = 0.5 * (
            np.linspace(-90, 90, nlat + 1)[1:] + np.linspace(-90, 90, nlat + 1)[:-1]
        )
    if nlon:
        coords["lon"] = 0.5 * (
            np.linspace(0, 360, nlon + 1)[1:] + np.linspace(0, 360, nlon + 1)[:-1]
        )
    if latlon2d:
        coords["j"] = np.asarray(range(nlat))
        coords["i"] = np.asarray(range(nlon))
        dims = [c for c in coords if c not in ["lat", "lon"]]
        coords["lat"], coords["lon"] = np.meshgrid(coords["lat"], coords["lon"])
        coords["lat"] = xr.DataArray(coords["lat"].T, dims=["j", "i"])
        coords["lon"] = xr.DataArray(coords["lon"].T, dims=["j", "i"])
    else:
        dims = [c for c in coords]
    ds = xr.Dataset(
        data_vars={
            name: xr.DataArray(
                rs.rand(*[len(coords[d]) for d in dims]) * 20,
                coords=coords,
                dims=dims,
                name=name,
                attrs={"units": unit},
            ),
        }
    )
    return ds


@pytest.mark.parametrize(
    "reference_key,registry_name,score",
    [
        ("test/Site/tas.nc", "ilamb", 0.1310922449190615),
        ("test/Grid/gpp.nc", "ilamb", 0.0128094224567734),
        ("test/thetao.nc", "test", 0.0622860516334433),
    ],
)
def test_run(reference_key: str, registry_name: str, score: float):
    UNITS = {"tas": "degC", "gpp": "g m-2 d-1", "thetao": "degC"}
    # setup test reference data
    variable_id = reference_key.split("/")[-1].replace(".nc", "")
    if registry_name == "ilamb":
        registry = ilamb3.ilamb_catalog()
    elif registry_name == "iomb":
        registry = ilamb3.iomb_catalog()
    elif registry_name == "test":
        registry = ilamb3.test_catalog()
    registry.fetch(reference_key)
    df_registry = run.registry_to_dataframe(registry)
    # setup temp dir and clean
    output_path = Path(tempfile.gettempdir()) / "-".join(reference_key.split("/")[:2])
    output_path.mkdir(parents=True, exist_ok=True)
    for fname in output_path.iterdir():
        fname.unlink()
    # setup some test comparison data
    com = generate_test_dset(
        variable_id,
        UNITS[variable_id],
        seed=1,
        nyear=10,
        nlat=18,
        nlon=36,
        ndepth=2 if registry_name == "test" else None,
        latlon2d=(registry_name == "test"),
    )
    com.to_netcdf(output_path / "tmp.nc")
    df_com = pd.DataFrame(
        [
            {
                "source_id": "Junk",
                "experiment_id": "historical",
                "member_id": "r1i1p1f1",
                "variable_id": variable_id,
                "grid_label": "gn",
                "path": output_path / "tmp.nc",
            }
        ]
    )
    # run the analysis
    ilamb3.conf.set(prefer_regional_quantiles=False, use_uncertainty=False)
    setup = {"sources": {variable_id: reference_key}}
    if registry_name == "test":
        setup["depth"] = 10.0
    run.run_simple(df_registry, reference_key, df_com, output_path, **setup)

    # are the scores correct?
    df = pd.read_csv(output_path / "Junk-r1i1p1f1-gn.csv")
    q = df[df["name"] == "Bias Score [1]"]
    print(q.iloc[0].value)
    assert len(q) == 1
    assert np.allclose(q.iloc[0].value, score)

    # did we get the right plot count?
    assert len(list(output_path.glob("*bias*.png"))) == 2

    # did html generate?
    with open(str(output_path / "index.html")) as fin:
        html = fin.read()
        assert '<img id="divJunk-r1i1p1f1-gn"' in html
