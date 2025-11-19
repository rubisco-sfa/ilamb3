import numpy as np
import pytest
import xarray as xr

from ilamb3.analysis import nbp_analysis
from ilamb3.exceptions import TemporalOverlapIssue
from ilamb3.tests.test_compare import generate_test_dset


def test_nbp():
    grid = dict(ntime=36)
    ref = generate_test_dset(**grid).rename_vars({"da": "nbp"})
    uncert = generate_test_dset(seed=2, **grid)["da"] * 1e-2
    ref["nbp_bnds"] = xr.DataArray(
        np.array(
            [ref["nbp"].values - uncert.values, ref["nbp"].values + uncert.values]
        ).T,
        coords={"time": ref["time"]},
        dims=["time", "nb"],
    )
    ref["nbp"].attrs["bounds"] = "nbp_bnds"
    com = generate_test_dset(seed=3, **grid).rename_vars({"da": "nbp"})
    ref["nbp"].attrs["units"] = "Pg yr-1"
    ref["nbp_bnds"].attrs["units"] = "Pg yr-1"
    com["nbp"].attrs["units"] = "Pg yr-1"
    analysis = nbp_analysis()
    df, _, _ = analysis(ref, com)
    df = df[df["type"] == "score"]
    assert len(df) == 2
    assert np.allclose(
        df[df["name"] == "Difference Score"].iloc[0]["value"], 0.8815878925867227
    )
    assert np.allclose(
        df[df["name"] == "Trajectory Score"].iloc[0]["value"], 1.2568309976103773e-06
    )


def test_fail():
    grid = dict(ntime=36)
    ref = generate_test_dset(**grid).rename_vars({"da": "nbp"})
    uncert = generate_test_dset(seed=2, **grid)["da"] * 1e-2
    ref["nbp_bnds"] = xr.DataArray(
        np.array(
            [ref["nbp"].values - uncert.values, ref["nbp"].values + uncert.values]
        ).T,
        coords={"time": ref["time"]},
        dims=["time", "nb"],
    )
    ref["nbp"].attrs["bounds"] = "nbp_bnds"
    com = (
        generate_test_dset(seed=3, **grid)
        .rename_vars({"da": "nbp"})
        .isel(time=slice(24, 36))
    )
    analysis = nbp_analysis()
    with pytest.raises(TemporalOverlapIssue):
        analysis(ref, com)
