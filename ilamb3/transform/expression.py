"""
An ILAMB transform for combining variables algebraically at runtime.
"""

import re
from typing import Any

import xarray as xr

from ilamb3.transform.base import ILAMBTransform


class expression(ILAMBTransform):
    """
    An ILAMB transform for combining variables algebraically at runtime.

    Parameters
    ----------
    expression : str
        An expression of the form `a = b + c`.
    """

    def __init__(self, expr: str, **kwargs: Any):
        assert "=" in expr
        self.expression = expr.split("=")[1]
        PYTHON_VARIABLE = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        lhs, rhs = expr.split("=")
        self.lhs_vars = re.findall(PYTHON_VARIABLE, lhs)
        self.rhs_vars = re.findall(PYTHON_VARIABLE, rhs)
        assert len(self.lhs_vars) == 1
        assert len(self.rhs_vars) > 0

    def required_variables(self) -> list[str]:
        """
        Return the required variables for the transform.
        """
        return self.rhs_vars

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Evaluate the expression.
        """
        if self.lhs_vars[0] in ds:
            return ds
        if not set(self.required_variables()).issubset(ds):
            return ds
        ds[self.lhs_vars[0]] = eval(
            self.expression, {}, {key: ds[key] for key in self.rhs_vars}
        )
        return ds
