"""Modular ILAMB methodology functions."""

from ilamb3.analysis.base import add_overall_score
from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.hydro import hydro_analysis
from ilamb3.analysis.nbp import nbp_analysis
from ilamb3.analysis.relationship import relationship_analysis
from ilamb3.analysis.runoff_sensitivity import runoff_sensitivity_analysis
from ilamb3.analysis.timeseries import timeseries_analysis

DEFAULT_ANALYSES = {"Bias": bias_analysis}
ALL_ANALYSES = {
    "nbp": nbp_analysis,
    "Runoff Sensitivity": runoff_sensitivity_analysis,
    "Hydro": hydro_analysis,
    "timeseries": timeseries_analysis,
}
ALL_ANALYSES = ALL_ANALYSES | DEFAULT_ANALYSES

__all__ = [
    "bias_analysis",
    "nbp_analysis",
    "relationship_analysis",
    "runoff_sensitivity_analysis",
    "timeseries_analysis",
    "add_overall_score",
    "DEFAULT_ANALYSES",
    "ALL_ANALYSES",
]
