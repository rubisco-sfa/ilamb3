"""Configuration for ilamb3"""

import contextlib
import copy
from pathlib import Path

import yaml

defaults = {"build_dir": "./_build"}


class Config(dict):
    """A global configuration object used in the package."""

    def __init__(self, filename: Path | None = None, **kwargs):
        self.filename = (
            Path(filename)
            if filename is not None
            else Path.home() / ".config/ilamb3/conf.yaml"
        )
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
    ):
        """Change ilamb3 configuration options."""
        temp = copy.deepcopy(self)
        if build_dir is not None:
            self["build_dir"] = str(build_dir)
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
