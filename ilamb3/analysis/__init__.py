"""Modular ILAMB methodology functions."""

from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.nbp import nbp_analysis
from ilamb3.analysis.relationship import relationship_analysis
from ilamb3.analysis.runoff_sensitivity import runoff_sensitivity_analysis

DEFAULT_ANALYSES = {"Bias": bias_analysis}

__all__ = [
    "bias_analysis",
    "nbp_analysis",
    "relationship_analysis",
    "runoff_sensitivity_analysis",
    "DEFAULT_ANALYSES",
]
