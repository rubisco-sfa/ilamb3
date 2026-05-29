import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import ilamb3
import ilamb3.load as ill
from ilamb3.tests.test_dataset import generate_test_dset


@pytest.fixture
def df():
    ds = generate_test_dset(seed=1)
    path = str(Path(tempfile.gettempdir()) / "tmp.nc")
    ds.to_netcdf(path)
    return pd.DataFrame(
        [
            {"path": path, "frequency": None},
            {"path": path, "frequency": "mon"},
            {"path": path, "frequency": "junk"},
        ]
    )


def test_add_frequency_column(df):
    df = ill.add_frequency_column(df)
    assert (df["frequency"] == "mon").all()


def test_load_key_or_filename():
    # setup, make sure this asset exists
    asset = Path(ilamb3.ilamb_catalog().fetch("regions/GlobalLand.nc"))
    ill.load_key_or_filename("regions/GlobalLand.nc")
    ill.load_key_or_filename(str(asset.absolute()))
    os.environ["ILAMB_ROOT"] = str(asset.parent)
    ill.load_key_or_filename("GlobalLand.nc")
    with pytest.raises(FileNotFoundError):
        ill.load_key_or_filename("not_a_file.nc")
