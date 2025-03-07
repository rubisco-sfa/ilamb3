import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ilamb3.analysis import relationship_analysis
from ilamb3.tests.test_compare import generate_test_dset


def test_relationship():
    ds = xr.merge(
        [
            generate_test_dset(nlat=2, nlon=3).rename_vars({"da": "gpp"})["gpp"],
            generate_test_dset(seed=2, nlat=2, nlon=3).rename_vars({"da": "tas"})[
                "tas"
            ],
        ]
    )
    analysis = relationship_analysis("gpp", "tas")
    assert set(["gpp", "tas"]) == set(analysis.required_variables())
    ds_com = {}
    df, ds_ref, ds_com["Comparison"] = analysis(ds, ds)
    assert len(df) == 1
    assert np.allclose(df.loc[0, "value"], 1.0)
    dfp = analysis.plots(df, ds_ref, ds_com)
    assert len(dfp) == 3
    plt.close()
