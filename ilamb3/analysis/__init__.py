"""Modular ILAMB methodology functions."""

from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.nbp import nbp_analysis
from ilamb3.analysis.relationship import relationship_analysis

__all__ = ["bias_analysis", "relationship_analysis", "nbp_analysis"]
