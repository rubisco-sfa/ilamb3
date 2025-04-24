from pathlib import Path

import ilamb3


def test_ilamb():
    reg = ilamb3.ilamb_catalog()
    fname = reg.fetch("test/Site/tas.nc")
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()


def test_iomb():
    reg = ilamb3.iomb_catalog()
    fname = reg.fetch("RAPID/amoc_mon_RAPID_BE_NA_200404-202302.nc")
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()


def test_test():
    reg = ilamb3.test_catalog()
    fname = reg.fetch("test/thetao.nc")
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()
