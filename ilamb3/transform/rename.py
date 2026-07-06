"""
An ILAMB transform for renaming variables.
"""

import xarray as xr
from loguru import logger

from ilamb3.transform.base import ILAMBTransform


class rename(ILAMBTransform):
    """
    Rename the variable.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def required_variables(self) -> list[str]:
        """Return the variables this transform requires, none in this case."""
        return list(self.kwargs.keys())

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Return
        """
        names = {
            old: new
            for old, new in self.kwargs.items()
            if isinstance(new, str) and old in ds
        }
        if names:
            txt = ", ".join([f"{old}->{new}" for old, new in names.items()])
            logger.info(f"Renaming {txt}")
            ds = ds.rename_vars(names)
        return ds
