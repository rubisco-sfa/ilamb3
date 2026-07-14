"""
Analysis Python classes are statistical comparisons between model and reference data.
An anlaysis class must contain (1) an `__init__` abstract method that contains the
arguments needed to initialize the analysis, (2) a `name` method that returns a
human-readable name for the analysis, (3) a `required_variables` method that returns the
variables needed to perform the analysis, and (4) a `__call__` method that performs the
actual anlysis and returns a dataframe of scores, an xarray dataset of
reference data, and an xarray dataset of comparison data.

Users can directly access the analysis classes in Python scripts, they can be used in
the benchmark run YAML file, and default analyses can be toggled on and off in the ILAMB
config.py.

"""

import importlib
import inspect
from importlib import resources

from ilamb3.analysis.base import ILAMBAnalysis, add_overall_score


# resources.files("numpy").joinpath("linalg")
def _get_ilamb_analyses(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return {}
    return {
        f.replace("_analysis", ""): mod.__dict__[f]
        for f in dir(mod)
        if inspect.isclass(mod.__dict__[f])
        and issubclass(mod.__dict__[f], ILAMBAnalysis)
        and f != "ILAMBAnalysis"
    }


# Programmatically load all analyses found in the analysis directory and the
# current directory. They keys of this dictionary are the valid ids that can be
# given in a configure file.
ALL_ANALYSES = {}
try:
    with resources.path("ilamb3.analysis") as path:
        for f in path.glob("*.py"):
            if f.stem == "__init__" or f.stem == "base":
                continue
            ALL_ANALYSES.update(_get_ilamb_analyses(f"ilamb3.analysis.{f.stem}"))
except Exception:
    from pathlib import Path

    path = Path(str(resources.files("ilamb3"))) / "analysis"
    for f in path.glob("*.py"):
        if f.stem == "__init__" or f.stem == "base":
            continue
        ALL_ANALYSES.update(_get_ilamb_analyses(f"ilamb3.analysis.{f.stem}"))


# set the default analyses as a subset
DEFAULT_ANALYSES = {
    key: ALL_ANALYSES[key] for key in ["bias", "rmse", "cycle", "spatial_distribution"]
}

__all__ = ["ILAMBAnalysis", "add_overall_score"]
