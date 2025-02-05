import tempfile
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3
import ilamb3.run as run
from ilamb3.analysis import bias_analysis


def generate_test_dset(seed: int = 1, nyear=None, nlat=None, nlon=None):
    rs = np.random.RandomState(seed)
    coords = []
    dims = []
    if nyear is not None:
        time = [
            cf.DatetimeNoLeap(1980 + int(m / 12), (m % 12) + 1, 15)
            for m in range(nyear * 12)
        ]
        coords.append(time)
        dims.append("time")
    if nlat is not None:
        lat = np.linspace(-90, -80, nlat + 1)
        lat = 0.5 * (lat[1:] + lat[:-1])
        coords.append(lat)
        dims.append("lat")
    if nlon is not None:
        lon = np.linspace(0, 20, nlon + 1)
        lon = 0.5 * (lon[1:] + lon[:-1])
        coords.append(lon)
        dims.append("lon")
    ds = xr.Dataset(
        data_vars={
            "tas": xr.DataArray(
                rs.rand(*[len(c) for c in coords]) * 20,
                coords=coords,
                dims=dims,
            ),
        }
    )
    ds["tas"].attrs["units"] = "degC"
    return ds


def test_run():
    reg = ilamb3.ilamb_catalog()
    ref = xr.open_dataset(reg.fetch("test/Test/tas.nc"))
    com = generate_test_dset(1, nyear=35, nlat=10, nlon=20)
    tmp = Path(tempfile.gettempdir())
    ds_com = {}
    anl = {"Bias": bias_analysis("tas")}
    df, ds_ref, ds_com["Comparison"] = run.run_analyses(ref, com, anl)
    dfp = run.plot_analyses(df, ds_ref, ds_com, anl, tmp)
    html = run.generate_html_page(df, ds_ref, ds_com, dfp)
    q = df.query("type=='score'")
    assert len(q) == 1
    assert np.allclose(q.iloc[0].value, 0.138678)
    assert len(dfp) == 4
    assert '<img id="divComparison" src="Comparison_None_mean.png" width="32%">' in html
