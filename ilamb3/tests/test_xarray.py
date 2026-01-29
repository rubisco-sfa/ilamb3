import pytest
import xarray as xr

import ilamb3
from ilamb3.dataset import get_coord_name
from ilamb3.run import fix_lndgrid_coords

FILES = [
    "ELMngee4_TFSmeq2-DAT_0.01deg_GSWP3_arctic_ICB20TRCNPRDCTCBC.elm.h0.2008-01-01-00000.nc",
    "ELMngee4_TFSmeq2-DAT_0.01deg_GSWP3_arctic_ICB20TRCNPRDCTCBC.elm.h0.2009-01-01-00000.nc",
    "gpp_Lmon_NorESM2-LM_historical_r1i1p1f1_gn_200001-200912.nc",
    "gpp_Lmon_NorESM2-LM_historical_r1i1p1f1_gn_201001-201412.nc",
]


def check_coord_bounds(ds: xr.Dataset, coord: str) -> None:
    coord_name = get_coord_name(ds, coord)
    da = ds[coord_name]
    if "bounds" not in da.attrs:
        return
    bnd_name = da.attrs["bounds"]
    da = ds[bnd_name]
    if da.ndim != 2:
        print(da)
    assert da.ndim == 2


@pytest.mark.parametrize(
    "files",
    [[FILES[0]], FILES[:2], [FILES[2]], FILES[-2:]],
)
def test_open_mfdataset(files):
    """In the past, open_mfdataset would merge lat/lon bounds across time which
    is undesireable. We keep playing around with options but behavior changes
    somewhat and this check will ensure some stability."""
    cat = ilamb3.test_catalog()
    ds = xr.open_mfdataset(
        [cat.fetch(f"test/{f}") for f in files],
        preprocess=fix_lndgrid_coords,
        data_vars="minimal",
    )
    for c in ["lat", "lon"]:
        check_coord_bounds(ds, c)
