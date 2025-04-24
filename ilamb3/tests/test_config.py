import tempfile
from pathlib import Path

import pytest

import ilamb3


def test_saveload():
    ilamb3.conf.reset()
    ilamb3.conf.set(
        build_dir="junkdir",
        regions=["nhsa"],
        prefer_regional_quantiles=True,
        use_uncertainty=True,
    )
    cfg_file = Path(tempfile.gettempdir()) / "tmp.cfg"
    ilamb3.conf.save(cfg_file)
    ilamb3.conf.reset()
    ilamb3.conf.load(cfg_file)
    cfg_str = f"{ilamb3.conf}"
    assert "junkdir" in cfg_str
    assert "- nhsa" in cfg_str
    ilamb3.conf.reset()


def test_context():
    ilamb3.conf.reset()
    ilamb3.conf.set(build_dir="outcontext")
    with ilamb3.conf.set(build_dir="incontext"):
        assert "incontext" in f"{ilamb3.conf}"
        assert "outcontext" not in f"{ilamb3.conf}"
    assert "incontext" not in f"{ilamb3.conf}"
    assert "outcontext" in f"{ilamb3.conf}"
    ilamb3.conf.reset()


def test_get():
    ilamb3.conf.reset()
    assert ilamb3.conf["regions"][0] is None
    assert ilamb3.conf.get("regions")[0] is None


def test_fail():
    with pytest.raises(ValueError):
        ilamb3.conf.set(regions=["not_a_region"])
