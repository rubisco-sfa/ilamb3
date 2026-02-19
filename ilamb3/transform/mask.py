import operator
from typing import Any

import numpy as np
import xarray as xr

from ilamb3.transform.base import ILAMBTransform

# the 'operators' package has these...
OPERATORS = ["lt", "le", "eq", "ne", "ge", "gt"]

# ...but users may give the mathematical expression, so map them to operators
MATH_MAP = {"<": "lt", "<=": "le", "==": "eq", "!=": "ne", ">=": "ge", ">": "gt"}


def _split_by_op(condition: str) -> tuple[str, str, str]:
    op_fn = None
    # first check the operators package operators
    for op in OPERATORS:
        if f" {op} " in condition:
            op_fn = op
    # then check the math symbols
    for op in MATH_MAP:
        if op in condition:
            op_fn = op
    # if we couldn't find an operator, raise an error
    if op_fn is None:
        raise ValueError(f"Could not parse the condition string '{condition}'.")
    # split the condition into lhs and rhs by the operator
    lhs, rhs = condition.split(op_fn)
    # if the operator is a math symbol, map it to the operators package function name
    op_fn = MATH_MAP[op_fn] if op_fn in MATH_MAP else op_fn
    lhs = lhs.strip()
    rhs = rhs.strip()
    return lhs, op_fn, rhs


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
        lhs, op, rhs = _split_by_op(condition)
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
