from typing import Any

import xarray as xr

from ilamb3.transform.base import ILAMBTransform


class aggregate(ILAMBTransform):
    """
    Collapse a dataset variable along one or more dimensions using a specified
    aggregation function.

    This :class:`ILAMBTransform`

    Parameters
    ----------
    dim : str or list of str
        The dimension(s) along which to aggregate (e.g., ``time``).
    varname : str
        The name of the variable to aggregate.
    agg : str
        The aggregation function to apply (e.g., ``mean``, ``sum``, ``max``, ``min``,
        ``quantile``).
    freq : str, optional
        An offset alias string specifying the resampling frequency for temporal
        aggregation (e.g., ``"M"`` for monthly). If not provided, the data will be
        aggregated along the existing time dimension without resampling. See
        `pandas offset aliases
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    climatology : bool, optional
        If True, do not use xr.resample for temporal aggregation, and instead perform a
        groupby over the appropriate time component (e.g., month) to compute a
        climatology. This option is ignored if ``freq`` is not provided. Default is False
    **kwargs : Any
        Additional keyword arguments passed to the base :class:`ILAMBTransform` class.

    Attributes
    ----------


    Notes
    -----


    Examples
    --------


    """

    def __init__(
        self,
        dim: str,
        varname: str,
        agg: str,
        freq: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the transform with any keyword arguments.
        """
        self.dim = dim
        self.varname = varname
        self.agg = agg
        self.freq = freq

    def required_variables(self) -> list[str]:
        """
        Return the variables that this transform needs to have in the dataset.
        """
        return list()

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        The body of the transform function.
        """

        return ds
