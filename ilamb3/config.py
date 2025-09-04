"""Configuration for ilamb3"""

import contextlib
import copy
import os
from pathlib import Path

import yaml

import ilamb3.regions as reg

defaults = {
    "regions": [None],
    "prefer_regional_quantiles": False,
    "quantile_database": "quantiles/quantiles_Whittaker_cmip5v6.parquet",
    "quantile_threshold": 70,
    "use_uncertainty": False,
    "model_name_facets": ["source_id", "member_id", "grid_label"],
    "plot_central_longitude": 0,
    "comparison_groupby": ["source_id", "member_id", "grid_label"],
    "use_cached_results": False,
    "figure_dpi": 100,
    "debug_mode": False,
    "run_mode": "interactive",  # for internal use
}


class Config(dict):
    """A global configuration object used in the package."""

    def __init__(self, filename: Path | None = None, **kwargs):
        if filename is not None:
            self.filename = Path(filename)
        elif "ILAMB3_CONFIGURATION" in os.environ:
            self.filename = Path(os.environ["ILAMB3_CONFIGURATION"])
        else:
            self.filename = Path.home() / ".config/ilamb3/conf.yaml"
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.reload_all()
        self.temp = None
        super().__init__(**kwargs)

    def __repr__(self):
        return yaml.dump(dict(self))

    def reset(self):
        """Return to defaults."""
        self.clear()
        self.update(copy.deepcopy(defaults))

    def save(self, filename: Path | None = None):
        """Save current configuration to file as YAML."""
        filename = filename or self.filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(dict(self), f)

    @contextlib.contextmanager
    def _unset(self, temp):
        yield
        self.clear()
        self.update(temp)

    def set(
        self,
        *,
        build_dir: str | None = None,
        regions: list[str] | None = None,
        prefer_regional_quantiles: bool | None = None,
        use_uncertainty: bool | None = None,
        model_name_facets: list[str] | None = None,
        plot_central_longitude: float | None = None,
        comparison_groupby: list[str] | None = None,
        use_cached_results: bool | None = None,
        figure_dpi: int | None = None,
        debug_mode: bool | None = None,
    ):
        """Change ilamb3 configuration options."""
        temp = copy.deepcopy(self)
        if build_dir is not None:
            self["build_dir"] = str(build_dir)
        if regions is not None:
            ilamb_regions = reg.Regions()
            does_not_exist = set(regions) - set(ilamb_regions._regions) - set([None])
            if does_not_exist:
                raise ValueError(
                    f"Cannot run ILAMB over these regions {list(does_not_exist)} which are not registered in our system {list(ilamb_regions._regions)}"
                )
            self["regions"] = regions
        if prefer_regional_quantiles is not None:
            self["prefer_regional_quantiles"] = bool(prefer_regional_quantiles)
        if use_uncertainty is not None:
            self["use_uncertainty"] = bool(use_uncertainty)
        if model_name_facets is not None:
            self["model_name_facets"] = model_name_facets
        if plot_central_longitude is not None:
            self["plot_central_longitude"] = float(plot_central_longitude)
        if comparison_groupby is not None:
            self["comparison_groupby"] = comparison_groupby
        if use_cached_results is not None:
            self["use_cached_results"] = bool(use_cached_results)
        if figure_dpi is not None:
            self["figure_dpi"] = int(figure_dpi)
        if debug_mode is not None:
            self["debug_mode"] = bool(debug_mode)
        return self._unset(temp)

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        elif item in defaults:
            return defaults[item]
        else:
            raise KeyError(item)

    def get(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        return default

    def reload_all(self):
        self.reset()
        self.load()

    def load(self, filename: Path | None = None):
        """Update global config from YAML file or default file if None."""
        filename = filename or self.filename
        if filename.is_file():
            with open(filename) as f:
                try:
                    self.update(yaml.safe_load(f))
                except Exception:
                    pass


conf = Config()
conf.reload_all()
