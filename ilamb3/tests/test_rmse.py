import numpy as np
import pytest

from ilamb3.analysis.rmse import rmse_analysis
from ilamb3.tests.test_compare import generate_test_dset
from ilamb3.tests.test_dataset import generate_test_site_dset


@pytest.mark.parametrize(
    "use_uncertainty,rmse,score",
    [
        (True, 4.054071846722371e-09, 0.5253398932130192),
        (False, 4.054071846722371e-09, 0.2438115935402378),
    ],
)
def test_rmse_collier2018(use_uncertainty: bool, rmse: float, score: float):
    grid = dict(ntime=36, nlat=10, nlon=20)
    ref = generate_test_dset(**grid)
    ref["da_bnds"] = generate_test_dset(seed=2, **grid)["da"] * 9e-1
    ref["da"].attrs["bounds"] = "da_bnds"
    com = generate_test_dset(seed=3, **grid)
    analysis = rmse_analysis(
        "da",
        method="Collier2018",
        use_uncertainty=use_uncertainty,
    )
    df, _, _ = analysis(ref, com)
    assert len(df) == 2
    assert np.allclose(df[df["name"] == "RMSE"].iloc[0]["value"], rmse)
    assert np.allclose(df[df["name"] == "RMSE Score"].iloc[0]["value"], score)


def test_rmse_site_collier2018():
    ref = generate_test_site_dset(ntime=36)
    com = generate_test_dset(ntime=36, nlat=10, nlon=20)
    analysis = rmse_analysis("da")
    assert set(["da"]) == set(analysis.required_variables())
    df, _, _ = analysis(ref, com)
    df = df[df["type"] == "score"]
    assert len(df) == 1
    assert np.allclose(df.iloc[0].value, 0.2527023684081725)
