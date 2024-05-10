import numpy as np

from ilamb3.analysis import relationship_analysis
from ilamb3.tests.test_compare import generate_test_dset


def test_relationship():
    ds1 = generate_test_dset(nlat=2, nlon=3)
    ds2 = generate_test_dset(seed=2, nlat=2, nlon=3)
    analysis = relationship_analysis("da", "da")
    assert analysis.required_variables().count("da") == 2
    df, _, _ = analysis(ds1, ds2)
    assert len(df) == 1
    assert np.allclose(df.loc[0, "value"], 0.411777817007)
