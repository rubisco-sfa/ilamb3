from pathlib import Path

import pytest
import yaml

import ilamb3.esgf as esgf


@pytest.fixture
def conf_yaml(test_path: Path) -> Path:
    yaml_file = test_path / "esgf.yaml"
    cfg = {"Test": {"sources": {"gpp": "test/Grid/gpp.nc"}, "alternate_vars": ["GPP"]}}
    with open(yaml_file, "w") as out:
        yaml.safe_dump(cfg, out)
    return yaml_file


def test_get_configure_variables(conf_yaml: Path):
    df = esgf.get_configure_variables(conf_yaml)
    assert set(["gpp", "GPP"]) == set(df["variable_id"])
    # twice to check the caching mechanism
    df = esgf.get_configure_variables(conf_yaml)
    assert set(["gpp", "GPP"]) == set(df["variable_id"])


def test_get__and_download_esgf_catalog(conf_yaml: Path):
    df = esgf.get_configure_variables(conf_yaml)
    cat = esgf.get_esgf_catalog(df, ["MPI-ESM1-2-HR"])
    yml = esgf.download_esgf_catalog(df, cat)
    assert len(yml) == 3
    assert "CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical.r1i1p1f1.Lmon.gpp.gn" in yml
