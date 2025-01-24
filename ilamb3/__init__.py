"""A library of functions used to benchmark earth system models."""

# import these packages so that units via pint will be possible once anything
# from ilamb is imported.
import importlib

import pint_xarray  # noqa
import pooch
import xarray as xr
from cf_xarray.units import units

from ilamb3._version import __version__  # noqa
from ilamb3.config import conf  # noqa

# additional units that pint/cf-xarray does not handle
units.define("kg = 1e3 * g")
units.define("Mg = 1e6 * g")
units.define("Pg = 1e15 * g")


def ilamb_catalog() -> pooch.Pooch:
    """
    Return the pooch ilamb reference data catalog.

    Returns
    -------
    pooch.Pooch
        The intake ilamb reference data catalog.
    """

    _ILAMB_DATA_VERSION = (
        "0.1"  # we don't really have data versions for the collection :/
    )
    registry = pooch.create(
        path=pooch.os_cache("ilamb3"),
        base_url="https://www.ilamb.org/ILAMB-Data/DATA",
        version=_ILAMB_DATA_VERSION,
        env="ILAMB_ROOT",
    )
    registry.load_registry(
        importlib.resources.open_binary("ilamb3.registry", "ilamb.txt")
    )
    return registry


__all__ = ["dataset", "compare", "analysis", "regions", "ilamb_catalog,", "conf"]
xr.set_options(keep_attrs=True)
