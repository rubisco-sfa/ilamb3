from pathlib import Path

import numpy as np

from ilamb3.analysis.dispersion import dispersion_analysis
from ilamb3.tests.test_compare import generate_test_dset


def test_dispersion_analysis(test_path):
    ref = generate_test_dset(ntime=24, nlat=2, nlon=4, seed=1)
    com = generate_test_dset(ntime=24, nlat=2, nlon=4, seed=2)
    anl = dispersion_analysis("da", required_num_years=0, nbins=3)
    ds_com = {}
    df, ds_ref, ds_com["Comparison"] = anl(ref, com)
    assert np.allclose(df.iloc[0]["value"], 0.8705789814424281)
    df_plots = anl.plots(df, ds_ref, ds_com, test_path / "Dispersion")
    assert len(df_plots) == 1
    row = df_plots.iloc[0]
    assert row["title"] == "Dispersion Score"
    assert Path(row["path"]).is_file()
