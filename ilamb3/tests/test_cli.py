import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from pytest import fixture

import ilamb3
import ilamb3.cli as cli


@fixture
def ilamb_catalogs():
    return [ilamb3.ilamb3_catalog(), ilamb3.ilamb_catalog()]


@fixture
def ilamb_data_key():
    cat = ilamb3.ilamb3_catalog()
    some_key = cat.registry_files[0]
    return some_key


def test_flatten_dict():
    nested_dict = {"level0": {"level1": {"level2": "hi"}}, "stuff": "cool"}
    flat_dict = cli.flatten_dict(nested_dict)
    assert len(flat_dict) == 2


def test_parse_registry_keys(ilamb_data_key):
    out = {
        "Heading1": {
            "Another1": {
                "Something": {
                    "sources": {"fake": ilamb_data_key},
                    "variable_cmap": "Greens",
                }
            }
        }
    }
    config_file = Path(tempfile.gettempdir()) / "junk.yaml"
    with open(str(config_file), "w") as f:
        f.write(yaml.dump(out))
    keys = cli.parse_registry_keys(config_file)
    assert len(keys) == 1
    assert keys[0] == ilamb_data_key


def test_get_local_path():
    os.environ["ILAMB_ROOT"] = tempfile.TemporaryDirectory().name
    catalogs = [ilamb3.ilamb_catalog()]
    local_path = cli.get_local_path("test/Site/tas.nc", catalogs)
    assert local_path is None
    cli.fetch_key("test/Site/tas.nc", catalogs)
    local_path = cli.get_local_path("test/Site/tas.nc", catalogs)
    assert local_path is not None
    assert local_path.is_file()


def test_form_reference_dataframe():
    os.environ["ILAMB_ROOT"] = tempfile.TemporaryDirectory().name
    catalogs = [ilamb3.ilamb_catalog()]
    cli.fetch_key("test/Site/tas.nc", catalogs)
    df = cli.form_reference_dataframe(["test/Site/tas.nc"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
