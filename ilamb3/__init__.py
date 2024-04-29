"""A library of functions used to benchmark earth system models."""

# import these packages so that units via pint will be possible once anything
# from ilamb is imported.
import cf_xarray.units  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

__all__ = ["dataset", "compare", "analysis", "regions"]
xr.set_options(keep_attrs=True)
