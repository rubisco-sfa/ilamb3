"""A library of functions used to benchmark earth system models."""

# import these packages so that units via pint will be possible once anything
# from ilamb is imported.
import intake
import pint_xarray  # noqa
import xarray as xr
from cf_xarray.units import units

from ilamb3._version import __version__  # noqa

# additional units that pint/cf-xarray does not handle
units.define("kg = 1e3 * g")
units.define("Mg = 1e6 * g")
units.define("Pg = 1e15 * g")


def ilamb_catalog() -> intake.Catalog:
    """
    Return the intake ilamb reference data catalog.

    Returns
    -------
    intake.Catalog
        The intake ilamb reference data catalog.
    """
    return intake.open_catalog(
        "https://raw.githubusercontent.com/nocollier/intake-ilamb/main/ilamb.yaml"
    )


__all__ = ["dataset", "compare", "analysis", "regions", "ilamb_catalog"]
xr.set_options(keep_attrs=True)
