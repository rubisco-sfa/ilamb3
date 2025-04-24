import pandas as pd
import pytest

from ilamb3.analysis.quantiles import check_quantile_database, create_quantile_map
from ilamb3.exceptions import MissingRegion, NoDatabaseEntry


def test_check():
    df = None
    with pytest.raises(ValueError):
        check_quantile_database(df)
    df = pd.DataFrame([{"region": "fake_region"}, {"region": "another_fake_region"}])
    with pytest.raises(MissingRegion):
        check_quantile_database(df)


def test_map():
    df = pd.DataFrame(
        [
            {"quantile": 50, "type": "bias", "variable": "var"},
            {"quantile": 50, "type": "rmse", "variable": "var"},
        ]
    )
    with pytest.raises(NoDatabaseEntry):
        create_quantile_map(df, "var", "bias", 70)
