import operator
from typing import Any

import numpy as np
import xarray as xr

from ilamb3.transform.base import ILAMBTransform

# the 'operators' package has these...
OPERATORS = ["lt", "le", "eq", "ne", "ge", "gt"]

# ...but users may give the mathematical expression, so map back
MATH_MAP = {"<": "lt", "<=": "le", "==": "eq", "!=": "ne", ">=": "ge", ">": "gt"}


def _split_by_op(condition: str) -> tuple[str, str, str]:
    op_fn = None
    for op in OPERATORS:
        if f" {op} " in condition:
            op_fn = op
    for op in MATH_MAP:
        if op in condition:
            op_fn = op
    if op_fn is None:
        raise ValueError(f"Could not parse the condition string '{condition}'.")
    lhs, rhs = condition.split(op_fn)
    op_fn = MATH_MAP[op_fn] if op_fn in MATH_MAP else op_fn
    lhs = lhs.strip()
    rhs = rhs.strip()
    return lhs, op_fn, rhs


class mask_condition(ILAMBTransform):
    """
    This ILAMB Transform

    Parameters
    ----------
    condition: str
        A condition of the form `hfls < 0`
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
        Return the variables this transform uses.
        """
        return [self.lhs]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the appropriate integration transform to the dataset.
        """
        if self.lhs not in ds:
            return ds
        ds[self.lhs] = xr.where(
            ~self.operator(ds[self.lhs], self.rhs),
            ds[self.lhs],
            np.nan,
            keep_attrs=True,
        )
        return ds
