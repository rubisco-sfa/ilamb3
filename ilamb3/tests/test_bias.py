import numpy as np
import pandas as pd
import pytest

from ilamb3.analysis import bias_analysis
from ilamb3.regions import Regions
from ilamb3.tests.test_compare import generate_test_dset
from ilamb3.tests.test_dataset import generate_test_site_dset


def gen_quantile_dbase(seed=1):
    rs = np.random.RandomState(seed)
    df = []
    for r in Regions().regions:
        for th in [70, 80]:
            df.append(
                {
                    "variable": "da",
                    "region": r,
                    "quantile": th,
                    "type": "bias",
                    "value": rs.rand(1)[0] * 1e-9,
                    "unit": "kg m-2 s-1",
                }
            )
    return pd.DataFrame(df)


@pytest.mark.parametrize(
    "use_uncertainty,mass_weighting,score",
    [
        (True, False, 0.5037343625713414),
        (False, False, 0.49809129117395395),
        (True, True, 0.6211524133325482),
        (False, True, 0.6162697692652096),
    ],
)
def test_bias_collier2018(use_uncertainty: bool, mass_weighting: bool, score: float):
    grid = dict(nlat=10, nlon=20)
    ref = generate_test_dset(**grid)
    ref["da_bnds"] = generate_test_dset(seed=2, **grid)["da"] * 1e-2
    ref["da"].attrs["bounds"] = "da_bnds"
    com = generate_test_dset(seed=3, **grid)
    analysis = bias_analysis(
        "da",
        method="Collier2018",
        use_uncertainty=use_uncertainty,
        mass_weighting=mass_weighting,
    )
    df, _, _ = analysis(ref, com)
    df = df[df["type"] == "score"]
    assert len(df) == 1
    assert np.allclose(df.iloc[0].value, score)


def test_bias_site_collier2018():
    ref = generate_test_site_dset().mean(dim="time")
    com = generate_test_dset(nlat=10, nlon=20)
    analysis = bias_analysis("da")
    assert set(["da"]) == set(analysis.required_variables())
    df, _, _ = analysis(ref, com)
    df = df[df["type"] == "score"]
    assert len(df) == 1
    assert np.allclose(df.iloc[0].value, 0.665078089592162)


@pytest.mark.parametrize(
    "use_uncertainty,quantile_threshold,score",
    [
        (True, 80, 0.038101086238025224),
        (False, 80, 0.03181825609556247),
        (False, 70, 0.01916371991867042),
    ],
)
def test_bias_regionalquantiles(
    use_uncertainty: bool, quantile_threshold: bool, score: float
):
    grid = dict(nlat=10, nlon=20)
    ref = generate_test_dset(seed=1, **grid)
    ref["da_bnds"] = generate_test_dset(**grid)["da"] * 1e-2
    ref["da"].attrs["bounds"] = "da_bnds"
    com = generate_test_dset(seed=2, **grid)
    analysis = bias_analysis(
        "da",
        method="RegionalQuantiles",
        use_uncertainty=use_uncertainty,
        quantile_database=gen_quantile_dbase(),
        quantile_threshold=quantile_threshold,
    )
    df, _, _ = analysis(ref, com)
    df = df[df["type"] == "score"]
    print(df.iloc[0].value)
    assert len(df) == 1
    assert np.allclose(df.iloc[0].value, score)
