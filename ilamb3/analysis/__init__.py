"""Modular ILAMB methodology functions."""

from ilamb3.analysis.accumulate import accumulate_analysis
from ilamb3.analysis.base import add_overall_score
from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.cycle import cycle_analysis
from ilamb3.analysis.hydro import hydro_analysis
from ilamb3.analysis.nbp import nbp_analysis
from ilamb3.analysis.relationship import relationship_analysis
from ilamb3.analysis.rmse import rmse_analysis
from ilamb3.analysis.runoff_sensitivity import runoff_sensitivity_analysis
from ilamb3.analysis.spatial_distribution import spatial_distribution_analysis
from ilamb3.analysis.timeseries import timeseries_analysis

DEFAULT_ANALYSES = {
    "Bias": bias_analysis,
    "RMSE": rmse_analysis,
    "Annual Cycle": cycle_analysis,
    "Spatial Distribution": spatial_distribution_analysis,
}
ALL_ANALYSES = {
    "nbp": nbp_analysis,
    "Runoff Sensitivity": runoff_sensitivity_analysis,
    "Hydro": hydro_analysis,
    "timeseries": timeseries_analysis,
    "accumulate": accumulate_analysis,
    "relationship": relationship_analysis,
} | DEFAULT_ANALYSES

__all__ = list(ALL_ANALYSES.keys()) + ["add_overall_score"]
