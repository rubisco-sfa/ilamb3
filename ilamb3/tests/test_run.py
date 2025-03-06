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
    seed: int = 1, nyear=None, nlat=None, nlon=None, name="tas", unit="degC"
):
    rs = np.random.RandomState(seed)
    coords = []
    dims = []
    if nyear is not None:
        time = [
            cf.DatetimeNoLeap(2000 + int(m / 12), (m % 12) + 1, 15)
            for m in range(nyear * 12)
        ]
        coords.append(time)
        dims.append("time")
    if nlat is not None:
        lat = np.linspace(-90, 90, nlat + 1)
        lat = 0.5 * (lat[1:] + lat[:-1])
        coords.append(lat)
        dims.append("lat")
    if nlon is not None:
        lon = np.linspace(0, 360, nlon + 1)
        lon = 0.5 * (lon[1:] + lon[:-1])
        coords.append(lon)
        dims.append("lon")
    ds = xr.Dataset(
        data_vars={
            name: xr.DataArray(
                rs.rand(*[len(c) for c in coords]) * 20,
                coords=coords,
                dims=dims,
            ),
        }
    )
    ds[name].attrs["units"] = unit
    return ds


@pytest.mark.parametrize(
    "reference_key,score",
    [
        ("test/Site/tas.nc", 0.1310922449190615),
        ("test/Grid/gpp.nc", 0.0127712918632477),
    ],
)
def test_run(reference_key: str, score: float):
    UNITS = {"tas": "degC", "gpp": "g m-2 d-1"}
    # setup test reference data
    variable_id = reference_key.split("/")[-1].replace(".nc", "")
    registry = ilamb3.ilamb_catalog()
    registry.fetch(reference_key)
    df_registry = run.registry_to_dataframe(registry)
    # setup temp dir and clean
    output_path = Path(tempfile.gettempdir()) / "-".join(reference_key.split("/")[:2])
    output_path.mkdir(parents=True, exist_ok=True)
    for fname in output_path.iterdir():
        fname.unlink()
    # setup some test comparison data
    com = generate_test_dset(
        seed=1, nyear=10, nlat=18, nlon=36, name=variable_id, unit=UNITS[variable_id]
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
    run.run_simple(
        df_registry,
        reference_key,
        df_com,
        output_path,
        sources={variable_id: reference_key},
    )

    # are the scores correct?
    df = pd.read_csv(output_path / "Junk-r1i1p1f1-gn.csv")
    q = df[df["name"] == "Bias Score [1]"]
    assert len(q) == 1
    assert np.allclose(q.iloc[0].value, score)

    # did we get the right plot count?
    assert len(list(output_path.glob("*bias*.png"))) == 2

    # did html generate?
    with open(str(output_path / "index.html")) as fin:
        html = fin.read()
        assert '<img id="divJunk-r1i1p1f1-gn"' in html
