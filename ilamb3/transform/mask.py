import operator
from typing import Any

import numpy as np
import xarray as xr

from ilamb3.transform.base import ILAMBTransform, split_by_op


class mask_condition(ILAMBTransform):
    """
    This ILAMB Transform applies a condition that sets values to NaN where the condition
    is not met. The condition is specified as a string, e.g., 'hfls < 0', which would
    set all values of `hfls` that are greater than or equal to 0 to NaN. Chaining
    conditions with AND or OR is not currently supported, but multiple conditions can be
    applied by chaining multiple `mask_condition` transforms together in the YAML config
    file.

    Parameters
    ----------
    condition: str
        A condition of the form 'hfls < 0', where the left-hand side is a variable in
        the dataset, the operator is one of '<', '<=', '==', '!=', '>=', '>',
        and the right-hand side is a numeric value.
    **kwargs : Any
        Additional keyword arguments passed to the base `ILAMBTransform` class.
    """

    def __init__(
        self,
        condition: str,
        **kwargs: Any,
    ):
        lhs, op, rhs = split_by_op(condition)
        self.lhs = lhs
        self.rhs = float(rhs)
        self.operator = getattr(operator, op)

    def required_variables(self) -> list[str]:
        """
        Return the required variables for the transform.
        """
        return [self.lhs]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the masking condition to the dataset
        """

        # if the lhs variable is not in the dataset, we can't apply the condition
        if self.lhs not in ds:
            return ds

        # set values to NaN where the condition is not met
        ds[self.lhs] = xr.where(
            ~self.operator(ds[self.lhs], self.rhs),
            ds[self.lhs],
            np.nan,
            keep_attrs=True,
        )
        return ds
