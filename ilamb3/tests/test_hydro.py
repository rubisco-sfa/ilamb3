import matplotlib.pyplot as plt
import numpy as np

from ilamb3.analysis import hydro_analysis
from ilamb3.tests.test_run import generate_test_dset


def test_hydro():
    ref = generate_test_dset("tas", "degC", nyear=2, nlat=2, nlon=3, seed=1)
    com = generate_test_dset("tas", "degC", nyear=2, nlat=2, nlon=3, seed=2)
    analysis = hydro_analysis("tas")
    assert set(["tas"]) == set(analysis.required_variables())
    ds_com = {}
    df, ds_ref, ds_com["Comparison"] = analysis(ref, com)
    # check scalars
    assert len(df) == 35
    q = df[(df["analysis"] == "Seasonal SON") & (df["type"] == "score")]
    assert len(q) == 1
    assert np.allclose(q.iloc[0].value, 0.6474532320617079)
    plt.rcParams.update({"figure.max_open_warning": 0})
    dfp = analysis.plots(df, ds_ref, ds_com)
    assert len(dfp) == 35
    plt.close()
