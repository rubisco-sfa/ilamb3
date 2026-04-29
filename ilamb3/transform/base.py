"""
An abstract class for implementing transform functions used in ILAMB.
"""

import re
from abc import ABC, abstractmethod
from typing import Any

import xarray as xr


class ILAMBTransform(ABC):
    """
    The ILAMB transform base class.

    A transform is just a function that takes in and returns a dataset.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any):
        """
        Initialize the transform with any keyword arguments.
        """
        raise NotImplementedError()

    @abstractmethod
    def required_variables(self) -> list[str]:
        """
        Return the variables that this transform needs to have in the dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        The body of the transform function.
        """
        raise NotImplementedError()


MIP_FREQ_TO_ALIAS = {
    "subhr": "30min",
    "subhrPt": "30min",
    "1hr": "h",
    "1hrPt": "h",
    "3hr": "3h",
    "3hrPt": "3h",
    "6hr": "6h",
    "6hrPt": "6h",
    "day": "D",
    "mon": "MS",
    "monPt": "MS",
    "yr": "YS",
    "yrPt": "YS",
    "dec": "10YS",
}

# Expression operators mapped to their Python operator names
OPERATORS = ("lt", "le", "eq", "ne", "ge", "gt")
MATH_MAP = {"<=": "le", ">=": "ge", "==": "eq", "!=": "ne", "<": "lt", ">": "gt"}
_OP_SYMBOLS = {v: k for k, v in MATH_MAP.items()}

# Word-boundary-aware match for textual operators in either ' lt ' or '.lt.' form.
# Using \b avoids false hits on 'lt' inside identifiers like 'lt_value' or 'altitude'.
_TEXTUAL_OP_RE = re.compile(
    r"(?:\s|\.)\b(" + "|".join(OPERATORS) + r")\b(?:\s|\.)",
    re.IGNORECASE,
)

# Math symbols, longest-first so '>=' isn't shadowed by '>'.
_MATH_OP_RE = re.compile(
    "|".join(re.escape(s) for s in sorted(MATH_MAP, key=len, reverse=True))
)


def _split_by_op(condition: str) -> tuple[str, str, str]:
    """
    Parse a condition string into ``(lhs, operator_name, rhs)``.

    The operator is returned as its canonical name (``lt``, ``le``, ``eq``,
    ``ne``, ``ge``, ``gt``). The rhs is returned as a stripped string with no further
    validation. Callers are responsible for parsing the resulting rhs string.
    """
    if not isinstance(condition, str):
        raise TypeError(
            f"Expected str for condition, got {type(condition).__name__}: {condition!r}."
        )

    # first check the textual operators
    match = _TEXTUAL_OP_RE.search(condition)
    if match is not None:
        op_name = match.group(1).lower()
        lhs = condition[: match.start()].strip()
        rhs = condition[match.end() :].strip()
    # otherwise check math symbols
    else:
        match = _MATH_OP_RE.search(condition)
        if match is None:
            raise ValueError(f"Could not parse the condition string {condition!r}.")
        op_name = MATH_MAP[match.group(0)]
        lhs = condition[: match.start()].strip()
        rhs = condition[match.end() :].strip()

    # validate that we found something on both sides
    if not lhs:
        raise ValueError(f"Condition {condition!r} is missing a left-hand side.")
    if not rhs:
        raise ValueError(f"Condition {condition!r} is missing a right-hand side.")

    return lhs, op_name, rhs
