import re
from typing import Any

import xarray as xr

from ilamb3.transform.base import ILAMBTransform


class expression(ILAMBTransform):
    """
    This ILAMB transform parses and applies a user-provided algebraic expression (e.g., "new_var = var1 + var2")
    to an `xarray.Dataset`. It creates a new variable in the dataset if variables on the left-hand-side (lhs) of `=`
    do not already exist and all right-hand-side (rhs) variables are present.

    Parameters
    ----------
    expr : str
        An algebraic expression of the form "a = b + c" defining how to compute a new variable.
        The lhs is the output variable name, and the rhs is a valid Python expression referencing existing dataset variables.
    **kwargs : Any
        Additional keyword arguments passed to the base `ILAMBTransform` class.
    """

    def __init__(self, expr: str, **kwargs: Any):
        # instantiate expression
        assert "=" in expr
        self.expression = expr.split("=")[1]

        # instantiate the lhs_vars, rhs_vars, and check validity
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
        # check if lhs variable already exists or if rhs variables are missing
        if self.lhs_vars[0] in ds:
            return ds
        if not set(self.required_variables()).issubset(ds):
            return ds

        # evaluate the expression and add to dataset
        ds[self.lhs_vars[0]] = eval(
            self.expression, {}, {key: ds[key].pint.quantify() for key in self.rhs_vars}
        ).pint.dequantify()
        return ds
