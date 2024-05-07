"""A library of functions used to benchmark earth system models."""

# import these packages so that units via pint will be possible once anything
# from ilamb is imported.
import pint_xarray  # noqa: F401
import xarray as xr
from cf_xarray.units import units

# additional units that pint/cf-xarray does not handle
units.define("Mg = 1e6 * g")

__all__ = ["dataset", "compare", "analysis", "regions"]
xr.set_options(keep_attrs=True)
