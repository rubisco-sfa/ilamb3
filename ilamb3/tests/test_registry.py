from pathlib import Path

import ilamb3


def test_ilamb():
    reg = ilamb3.ilamb_catalog()
    fname = reg.fetch("test/Site/tas.nc")
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()


def test_iomb():
    reg = ilamb3.ilamb3_catalog()
    fname = reg.fetch(
        "RAPID-2023-1a/obs4MIPs_NOC_RAPID-2023-1a_mon_msftmz_gm_v20250902.nc"
    )
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()


def test_test():
    reg = ilamb3.test_catalog()
    fname = reg.fetch("test/thetao.nc")
    assert str(reg.abspath).endswith(ilamb3.ILAMB_DATA_VERSION)
    assert Path(fname).is_file()
