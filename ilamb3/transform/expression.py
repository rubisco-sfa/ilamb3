import re
from typing import Any

import xarray as xr

from ilamb3.transform.base import ILAMBTransform

"""

"""


class expression(ILAMBTransform):
    """
    Evaluate an algebraic expression and add the result as a new variable.

    This :class:`ILAMBTransform` parses a user-provided expression of the form
    ``"new_variable = <rhs>"`` and evaluates the right-hand side against variables in an
    :class:`xarray.Dataset`, assigning the result to ``new_variable``. The right-hand
    side is evaluated with unit-aware arithmetic via :mod:`pint`, so operands carrying
    CF-style units are reconciled automatically.

    If the ``new_variable`` already exists in the dataset, or if any of the right-hand
    side variables are missing, the transform will not operate, and it will return the
    dataset unchanged.

    Parameters
    ----------
    expr : str
        An algebraic expression of the form ``"a = b + c"``. The left-hand side must be
        a single valid Python identifier that names the output variable. The right-hand
        side must be a valid Python expression that references one or more existing
        dataset variables by name.
    **kwargs : Any
        Additional keyword arguments passed to the base :class:`ILAMBTransform` class.

    Attributes
    ----------
    expression : str
        The right-hand side of ``expr``, as a string passed to :func:`eval`.
    lhs_vars : list of str
        Identifiers parsed from the left-hand side. Always length 1.
    rhs_vars : list of str
        Identifiers parsed from the right-hand side; the variables that must be present
        in the dataset for the transform to operate.

    Notes
    -----
    The right-hand side of ``expr`` is passed to :func:`eval` in a namespace containing
    only the referenced dataset variables. ``expr`` should be sourced from a trusted
    run yaml, not constructed from runtime input.

    The input dataset is modified in place. After evaluation, the new
    variable's ``standard_name`` and ``long_name`` attributes are set from
    ``expr``, and any inherited ``ancillary_variables`` attribute is removed.

    Examples
    --------
    In the yaml file used by ``ilamb3 run``, you can use ``expression`` to calculate a
    new variable from existing variables. If the new variable already exists, or if any
    of the required variables are missing, the transform will skip and return the
    dataset unchanged. For example:

    .. code-block:: yaml

        Radiation and Energy Cycle:
          Fluxnet-2015:
            sources:
              rns: Fluxnet-2015/obs4MIPs_Fluxnet_Fluxnet-2015_mon_rns_site_v20260302.nc
            transforms:
            - expression:
                expr: "rns = rsds - rsus"

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
        Return the dataset variables required for the transform to operate.
        """
        return self.rhs_vars

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Evaluate the expression and assign the result to ``ds``.

        Returns ``ds`` unchanged if the output variable already exists or if any
        required variable is missing. The input dataset is modified in place.
        """

        # Check if lhs variable already exists or if rhs variables are missing
        lhs = self.lhs_vars[0]
        if lhs in ds:
            return ds
        if not set(self.required_variables()).issubset(ds):
            return ds

        # Evaluate the expression and add to dataset
        ds[lhs] = eval(
            self.expression,
            {"__builtins__": {}},
            {key: ds[key].pint.quantify() for key in self.rhs_vars},
        ).pint.dequantify()

        # Prepare attributes for the new variable
        ds[lhs].attrs["standard_name"] = lhs
        ds[lhs].attrs["long_name"] = f"{lhs} = {self.expression.strip()}"
        if "ancillary_variables" in ds[lhs].attrs:
            ds[lhs].attrs.pop("ancillary_variables")
        return ds
