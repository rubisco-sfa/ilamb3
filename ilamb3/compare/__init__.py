import inspect

import ilamb3.compare.base as base
from ilamb3.compare.base import *  # noqa

__all__ = [[f[0] for f in inspect.getmembers(base) if inspect.isfunction(f[1])]]
