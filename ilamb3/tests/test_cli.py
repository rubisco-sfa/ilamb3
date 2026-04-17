import tempfile
from pathlib import Path

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
