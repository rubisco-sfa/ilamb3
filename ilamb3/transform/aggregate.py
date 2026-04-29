import operator
import re
from collections.abc import Callable
from typing import Any

import pint
import xarray as xr

from ilamb3.transform.base import (
    _OP_SYMBOLS,
    MIP_FREQ_TO_ALIAS,
    ILAMBTransform,
    _split_by_op,
)

_RHS_PATTERN = re.compile(
    r"\s*(?P<constant>[^\[\]]*?)\s*\[\s*(?P<unit>[^\[\]]*?)\s*\]\s*"
)

_OP_FUNCS: dict[str, Callable[[Any, Any], Any]] = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}

_VALID_AGG = ("sum", "mean")


def _unit_from_rhs(rhs: str) -> tuple[str, str]:
    """
    Parse a right-hand side string of the form ``'<constant> [<unit>]'``.

    Returns a tuple of the constant and the unit as strings. Forgiving about whitespace
    (around the number, around the brackets, and inside the brackets) but strict about
    the overall structure.
    """
    # Ensure rhs is a string & isn't empty/whitespace-only
    if not isinstance(rhs, str):
        raise TypeError(f"Expected str for rhs, got {type(rhs).__name__}: {rhs!r}.")
    stripped = rhs.strip()
    if not stripped:
        raise ValueError("Right-hand side is empty.")

    # Ensure units are enclosed in square brackets
    if "[" not in stripped or "]" not in stripped:
        for opener, closer, name in (
            ("(", ")", "parentheses"),
            ("{", "}", "curly braces"),
        ):
            if opener in stripped and closer in stripped:
                raise ValueError(
                    f"Right-hand side {rhs!r} uses {name} for the unit. "
                    f"Expected square brackets, e.g. '1 [mm/day]'."
                )
        raise ValueError(
            f"Right-hand side {rhs!r} does not contain a unit in square "
            f"brackets. Expected format like '1 [mm/day]'."
        )

    # Parse out the constant and unit
    match = _RHS_PATTERN.fullmatch(stripped)
    if match is None:
        raise ValueError(
            f"Right-hand side {rhs!r} is not in the expected format "
            f"'<number> [<unit>]'. Avoid nested or repeated brackets."
        )

    # Ensure constant and unit are non-empty and that the constant is a valid number
    constant = match.group("constant")
    unit = match.group("unit")

    if not constant:
        raise ValueError(
            f"Right-hand side {rhs!r} is missing a numeric constant before the unit."
        )
    if not unit:
        raise ValueError(
            f"Right-hand side {rhs!r} has an empty unit. "
            f"Expected format like '1 [mm/day]'."
        )

    try:
        float(constant)
    except ValueError:
        raise ValueError(
            f"Expected a numeric constant in {rhs!r}, but {constant!r} is "
            f"not a valid number."
        ) from None

    return constant, unit


def _convert_unit(da: xr.DataArray, target_unit: str) -> xr.DataArray:
    """
    Convert a DataArray to the target unit using xarray's pint integration.
    """
    try:
        return da.pint.to(target_unit).pint.dequantify()
    except (pint.DimensionalityError, pint.UndefinedUnitError) as e:
        raise ValueError(
            f"Error converting variable {da.name!r} to target unit {target_unit!r}: {e}"
        ) from e


class agg_time_on_condition(ILAMBTransform):
    """
    Aggregate a per-timestep condition along the time dimension.

    Parses ``"<lhs> <op> <rhs> [unit]"``, evaluates the condition at every timestep,
    then resamples to ``freq`` and reduces with ``agg``. This transform makes no
    assumptions about input cadence and does not pre-reduce the input. For anomaly
    indices (wet days, summer days, etc.), use the dedicated subclasses in
    :mod:`ilamb3.transform.threshold_indices`, which handle cadence mismatches.

    Parameters
    ----------
    condname : str
        Name of the output variable.
    cond : str
        A condition of the form ``"<lhs> <op> <rhs> [unit]"``.
    agg : str
        How to reduce the boolean condition mask within each resample window.
        One of ``"sum"`` (count of True values) or ``"mean"`` (fraction of True values).
    freq : str
        MIP frequency to resample to (key of
        :data:`ilamb3.transform.base.MIP_FREQ_TO_ALIAS`).

    Notes
    -----
    - Setting agg to ``sum`` counts the number of timesteps where the condition is True
    - Setting agg to ``mean`` calculates the fraction of timesteps where the condition
      is True (robust to differing window lengths and missing data)
    - ``max``/``min`` are excluded for now because booleans collapse to "any?" / "all?"

    Returns
    -------
    xr.Dataset
        Dataset with a single variable named ``condname`` containing the aggregated
        result.

    Examples
    --------

    .. code-block:: yaml
    H2O Anomalies:

    Wet Days:

        CLASS-1-1:
        sources:
            pr: CLASS-1-1/obs4MIPs_UNSW_CLASS-1-1_mon_pr_gn_v20260302.nc
        variable_cmap: Blues
        transforms:
        - agg_time_on_condition:
            condname: wet_days
            cond: "pr >= 1 [mm/day]"
            agg: sum
            freq: mon

    """

    def __init__(
        self,
        condname: str,
        cond: str,
        agg: str,
        freq: str,
    ):
        # Parse and validate attributes
        lhs, op, rhs = _split_by_op(cond)
        constant, unit = _unit_from_rhs(rhs)

        # Ensure the operator is sum or mean
        agg_lower = agg.lower()
        if agg_lower not in _VALID_AGG:
            raise ValueError(
                f"Invalid aggregation {agg!r}. Must be one of {list(_VALID_AGG)}."
            )

        # Ensure the MIP freqency used is in our mapping
        if freq not in MIP_FREQ_TO_ALIAS:
            raise ValueError(
                f"Invalid frequency {freq!r}. Must be one of "
                f"{sorted(MIP_FREQ_TO_ALIAS)}."
            )

        self.condname = condname
        self.var = lhs
        self.constant = constant
        self.unit = unit
        self.agg = agg_lower
        self.freq = freq
        self._freq_alias = MIP_FREQ_TO_ALIAS[freq]
        self._op_name = op
        self._op_func = _OP_FUNCS[op]
        self._op_symbol = _OP_SYMBOLS[op]

    def required_variables(self) -> list[str]:
        return [self.var]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:

        # Return early if the output var already exists or if the input var doesn't
        if self.condname in ds or self.var not in ds:
            return ds

        # Convert the input variable to the target unit
        da_converted = _convert_unit(ds[self.var], self.unit)

        # Evaluate the condition at every timestep, resample, and reduce
        condition_met = self._op_func(da_converted, float(self.constant))
        resampled = condition_met.resample(time=self._freq_alias)
        result = getattr(resampled, self.agg)()

        # Assign new attrs to the resultant variable, describing the transformation
        result.attrs = {
            "long_name": (
                f"{self.agg} per {self.freq} of (1 where {self.var} "
                f"{self._op_symbol} {self.constant} {self.unit}, else 0)"
            ),
            "standard_name": self.condname,
            "units": self.unit,
        }

        return ds.assign({self.condname: result})
