"""Analysis functions used in ILAMB."""

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import xarray as xr


# An abstract base class (ABC) in python is a way to define the structure of an object
# that can be used in other parts of the system. In our case this means that in order
# for your analysis to be compatible in the ILAMB system, you need to write a class that
# has *at minimum* the following member functions with their arguments and return
# values.
class ILAMBAnalysis(ABC):
    # Your analysis must be able to provide a list of variables that are to be used.
    # This is so we can pass models your function and query which variables are to be
    # used.
    @abstractmethod
    def required_variables(self) -> Union[list[str], dict[str, list[str]]]:
        raise NotImplementedError()

    # The work of your analysis must accept a reference and comparison dataset. The user
    # will need to package everything that each needs for the analysis upon submission.
    # Inside this function you can use whatever you want to generate scalars and
    # maps/curves for plotting. Once complete, ILAMB would like results in a pandas
    # dataframe (for scalar information) as well as a dataset for the reference and
    # comparison.
    @abstractmethod
    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset, **kwargs
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        raise NotImplementedError()
