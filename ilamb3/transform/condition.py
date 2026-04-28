import operator
from typing import Any

import pint
import xarray as xr

from ilamb3.transform.base import ILAMBTransform

OPERATORS = ["lt", "le", "eq", "ne", "ge", "gt"]
MATH_MAP = {"<=": "le", ">=": "ge", "==": "eq", "!=": "ne", "<": "lt", ">": "gt"}
_MATH_OPS_BY_LENGTH = sorted(MATH_MAP, key=len, reverse=True)
_OP_SYMBOLS = {v: k for k, v in MATH_MAP.items()}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _split_by_op(condition: str) -> tuple[str, str, str]:
    """
    Parse a condition string into (lhs, operator_name, rhs).

    Parameters
    ----------
    condition : str
        A string of the form ``"<lhs> <op> <rhs>"``.

    Returns
    -------
    tuple of (str, str, str)
        The left-hand side, the canonical operator name (one of ``lt``, ``le``,
        ``eq``, ``ne``, ``ge``, ``gt``), and the right-hand side.

    Raises
    ------
    ValueError
        If no operator is found, or the right-hand side is not numeric.
    """
    op_fn = None
    lower = condition.lower()

    # first check the operators package operators (case-insensitive)
    for op in OPERATORS:
        if f" {op} " in lower or f".{op}." in lower:
            op_fn = op
            break

    # if we found a textual operator, split on the original (case-preserving) string
    if op_fn is not None:
        if f" {op_fn} " in lower:
            idx = lower.index(f" {op_fn} ")
            sep_len = len(op_fn) + 2
        else:
            idx = lower.index(f".{op_fn}.")
            sep_len = len(op_fn) + 2
        lhs = condition[:idx].strip()
        rhs = condition[idx + sep_len :].strip()
        if not _is_number(rhs):
            raise ValueError(
                f"The right-hand side of the condition {condition!r} must be a "
                f"number, got {rhs!r}."
            )
        return lhs, op_fn, rhs

    # otherwise check math symbols (longest-first so '>=' isn't shadowed by '>')
    for symbol in _MATH_OPS_BY_LENGTH:
        if symbol in condition:
            op_fn = MATH_MAP[symbol]
            lhs, rhs = condition.split(symbol, maxsplit=1)
            lhs, rhs = lhs.strip(), rhs.strip()
            if not _is_number(rhs):
                raise ValueError(
                    f"The right-hand side of the condition {condition!r} must be "
                    f"a number, got {rhs!r}."
                )
            return lhs, op_fn, rhs

    raise ValueError(f"Could not parse the condition string {condition!r}.")


class condition(ILAMBTransform):
    """
    Evaluate a condition and replace the variable with a boolean result.

    This :class:`ILAMBTransform` evaluates a condition of the form
    ``"<variable> <operator> <number>"`` (e.g., ``"pr >= 1"``) and replaces
    the left-hand-side variable in the dataset with a boolean array that is
    ``True`` where the condition is met and ``False`` elsewhere. Useful for
    constructing climate indicators such as wet days, frost days, or ice days,
    which can then be counted by chaining with an aggregation transform.

    If ``units`` is provided, the threshold is interpreted in those units and
    reconciled against the variable's native units via :mod:`pint`. This lets
    the same YAML work across model and observation datasets that may ship the
    same variable in different units. If ``units`` is omitted, the threshold
    is interpreted in the variable's native units and no conversion occurs.

    Parameters
    ----------
    condition : str
        A condition of the form ``"<lhs> <op> <rhs>"``, where ``lhs`` names a
        variable in the dataset, ``op`` is one of ``<``, ``<=``, ``==``, ``!=``,
        ``>=``, ``>`` (or their textual equivalents ``lt``, ``le``, ``eq``,
        ``ne``, ``ge``, ``gt``), and ``rhs`` is a numeric value.
    units : str, optional
        Units of the threshold ``rhs``. If given, the comparison is performed
        in these units regardless of the dataset variable's native units. Must
        be dimensionally compatible with the variable's units.
    **kwargs : Any
        Additional keyword arguments passed to the base :class:`ILAMBTransform`
        class.

    Attributes
    ----------
    lhs : str
        Name of the dataset variable on the left-hand side of the condition.
        This variable is replaced in place with the boolean result.
    rhs : float
        Numeric threshold from the right-hand side of the condition.
    operator : Callable
        The comparison function from the :mod:`operator` module.
    units : str or None
        Units of the threshold, or ``None`` if not specified.

    Notes
    -----
    The variable named on the left-hand side of ``condition`` is overwritten
    with the boolean result. Downstream transforms operating on this variable
    will see a boolean array, not the original quantity. In typical usage,
    ``condition`` is followed immediately by an aggregation step that counts the
    ``True`` values.

    Compound conditions (``and``, ``or``, chained inequalities) are not
    supported. Apply multiple :class:`condition` transforms and combine the
    resulting variables with :class:`expression` if needed.

    Examples
    --------
    Count the number of wet days per month at each grid cell. The threshold
    is given in ``mm/day``; the source data may be in any dimensionally
    compatible unit (e.g., ``kg m-2 s-1``):

    .. code-block:: yaml

        H2O Anomalies:
          Wet Days:
            CLASS-1-1:
              sources:
                pr: CLASS-1-1/obs4MIPs_UNSW_CLASS-1-1_day_pr_gn_v20260302.nc
              variable_cmap: Blues
              transforms:
              - condition:
                  expr: "pr >= 1"
                  units: "mm/day"
                  output: wet_days
              - aggregate_time:
                  varname: wet_days
                  agg: sum
                  freq: mon

    A bare condition without ``units`` works for cases where the threshold
    is dimensionally trivial (e.g., a sign check):

    .. code-block:: yaml

        - condition:
            expr: "hfls < 0"
            output: negative_hfls
    """

    def __init__(
        self,
        expr: str,
        output: str,
        units: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the transform with any keyword arguments.
        """
        lhs, op, rhs = _split_by_op(expr)
        self.lhs = lhs
        self.rhs = float(rhs)
        self.operator = getattr(operator, op)
        self.units = units
        self.output = output
        self._condition_str = expr
        self._op_symbol = _OP_SYMBOLS[op]

    def required_variables(self) -> list[str]:
        """
        Return the variables that this transform needs to have in the dataset.
        """
        return [self.lhs]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:

        # silently pass thru if output variable exists or if lhs variable is missing
        if self.output in ds:
            return ds
        if self.lhs not in ds:
            return ds

        # quantify with pint and apply the operator, then dequantify back
        if self.units is not None:
            da = ds[self.lhs].pint.quantify()
            threshold = pint.application_registry.Quantity(self.rhs, self.units)
            result = self.operator(da, threshold).pint.dequantify()
        else:
            result = self.operator(ds[self.lhs], self.rhs)

        # assign the result to the output variable with appropriate attrs
        ds[self.output] = result
        ds[self.output].attrs.clear()
        ds[self.output].attrs["standard_name"] = self.output
        threshold_str = f"{self.rhs} {self.units}" if self.units else f"{self.rhs}"
        ds[self.output].attrs["long_name"] = (
            f"{self.output} = {self.lhs} {self._op_symbol} {threshold_str}"
        )
        return ds
