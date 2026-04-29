import warnings

import pandas as pd
import xarray as xr

from ilamb3.transform.aggregate import agg_time_on_condition

# How to reduce sub-daily input to daily before applying the threshold.
# Distinct from condition_aggregator._VALID_AGG (sum, mean) because the
# operations are different: here we're reducing physical values (mean for
# fluxes, max/min for extremes), not counting boolean hits.
_VALID_DAILY_REDUCTION = ("mean", "max", "min", "sum")
_SECONDS_PER_DAY = 86_400


def _ensure_daily(da: xr.DataArray, reduction: str) -> xr.DataArray:
    """
    Return ``da`` at daily cadence, reducing sub-daily input via ``reduction``.

    Raises if the input is coarser than daily, since you cannot recover daily extremes
    from monthly (or coarser) means.
    """
    # Validate reduction method before doing any work
    if reduction not in _VALID_DAILY_REDUCTION:
        raise ValueError(
            f"Invalid daily reduction {reduction!r}. Must be one of "
            f"{list(_VALID_DAILY_REDUCTION)}."
        )

    # Infer the input cadence. If we can't infer it, warn and pass through
    freq_str = xr.infer_freq(da.time)

    if freq_str is None:
        warnings.warn(
            f"Could not infer time frequency of {da.name!r}; passing through "
            f"unchanged. If the cadence is not daily, the resulting index "
            f"will be wrong.",
            stacklevel=2,
        )
        return da

    # If cannot convert to seconds, it's coarser than daily, so we can't compute
    try:
        seconds = pd.Timedelta(
            pd.tseries.frequencies.to_offset(freq_str)  # type: ignore
        ).total_seconds()
    except (ValueError, TypeError):
        raise ValueError(
            f"Variable {da.name!r} has frequency {freq_str!r} which is coarser "
            f"than daily; cannot compute a daily threshold index from it."
        ) from None

        # If daily, return as-is; if sub-daily, resample to daily using specified reduction
        return da
    if seconds < _SECONDS_PER_DAY:
        return getattr(da.resample(time="D"), reduction)()

    # If we get here, the input is coarser than daily, so we can't compute the index
    raise ValueError(
        f"Variable {da.name!r} has frequency {freq_str!r} ({seconds}s per step) "
        f"which is coarser than daily; cannot compute a daily threshold index."
    )


class daily_threshold_index(agg_time_on_condition):
    """
    Parent class for daily threshold indices (wet days, summer days, ice days,
    frost days, tropical nights, ...).

    Beyond what :class:`agg_time_on_condition` already does, this class handles cadence
    mismatches: sub-daily input is reduced to daily using ``daily_reduction`` before the
    threshold is applied. Monthly or coarser input raises ``ValueError`` (you cannot
    recover daily extremes from coarser means).

    Can be instantiated directly for a one-off custom index, or subclassed to give a
    named index a friendly constructor. To add a named index, subclass with an
    ``__init__`` that fixes the index-specific values (``threshold``, ``operator``,
    ``daily_reduction``) while exposing the user-configurable ones (``condname``,
    ``var``, ``freq``, ``agg``).

    Parameters
    ----------
    condname : str
        Output variable name (e.g. ``"wet_days"``).
    var : str
        Input variable name (e.g. ``"pr"``).
    threshold : str
        Threshold and unit, in ``"<number> [<unit>]"`` form (e.g. ``"1 [mm/day]"``).
    operator : str
        Comparison symbol. One of ``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``.
    daily_reduction : str
        How to reduce sub-daily input to daily before the threshold pass. For flux-like
        quantities (``pr``) typically ``"mean"``; for extremes (``tasmax``, ``tasmin``)
        typically ``"max"`` or ``"min"``.
    freq : str, optional
        MIP frequency to resample the daily mask to. Default ``"mon"``.
    agg : str, optional
        How to reduce the daily condition mask within each ``freq`` window. Default
        ``"sum"`` (count); ``"mean"`` gives the fraction of days in each window meeting
        the condition.

    Returns
    -------
    xr.Dataset
        Dataset with a single variable named ``condname`` containing the aggregated
        daily threshold index.

    Notes
    -----
    We recommend using this class only as a parent for specific named indices (e.g.,
    wet days, summer days, etc.) with friendly constructors. If you do not want to
    create a subclass, you can use this transform directly in the YAML, but you will
    need to specify all the parameters, including the more technical ones like
    ``daily_reduction``.

    Examples
    --------
    Custom daily threshold index in YAML:

    .. code-block:: yaml
        transforms:
        - daily_threshold_index:
          condname: windy_days
          var: sfcWind
          threshold: "10 [m/s]"
          operator: ">"
          daily_reduction: "max"
          freq: mon
          agg: mean

    Custom daily threshold index transform in Python:

    .. code-block:: python

        class windy_days(daily_threshold_index):
            def __init__(
                self,
                condname: str = "windy_days",
                var: str = "sfcWind",
                threshold: str = "10 [m/s]",
                operator: str = ">",
                daily_reduction: str = "max",
                freq: str = "mon",
                agg: str = "mean",
            ):
                super().__init__(
                    condname=condname,
                    var=var,
                    threshold=threshold,
                    operator=operator,
                    daily_reduction=daily_reduction,
                    freq=freq,
                    agg=agg,
                )
    """

    def __init__(
        self,
        condname: str,
        var: str,
        threshold: str,
        operator: str,
        daily_reduction: str,
        freq: str = "mon",
        agg: str = "sum",
    ):
        cond = f"{var} {operator} {threshold}"
        super().__init__(
            condname=condname,
            cond=cond,
            agg=agg,
            freq=freq,
        )
        self.daily_reduction = daily_reduction

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        # Early-return checks before doing any work
        if self.condname in ds or self.var not in ds:
            return ds

        # Reduce input to daily cadence (raises if input is coarser than daily)
        daily_da = _ensure_daily(ds[self.var], self.daily_reduction)

        # Run the parent transform on a minimal dataset, then merge the output back
        result_ds = super().__call__(xr.Dataset({self.var: daily_da}))
        return ds.assign({self.condname: result_ds[self.condname]})


class get_wet_days(daily_threshold_index):
    """
    Count of days with daily precipitation >= 1 mm/day.

    Sub-daily precipitation flux is reduced to a daily mean before the threshold is
    applied, so the transform accepts any cadence from sub-hourly through daily.
    Monthly or coarser input raises ``ValueError``.

    Parameters
    ----------
    condname : str, optional
        Output variable name. Default ``"wet_days"``.
    var : str, optional
        Input variable name. Default ``"pr"``.
    freq : str, optional
        MIP frequency to resample the count to. Default ``"mon"``.
    agg : str, optional
        ``"sum"`` (default) gives wet-day counts per ``freq`` window;
        ``"mean"`` gives the fraction of days in each window that were wet.

    Examples
    --------

    .. code-block:: yaml

        transforms:
        - get_wet_days:
          freq: mon
    """

    def __init__(
        self,
        condname: str = "wet_days",
        var: str = "pr",
        freq: str = "mon",
        agg: str = "sum",
    ):
        super().__init__(
            condname=condname,
            var=var,
            threshold="1 [mm/day]",
            operator=">=",
            daily_reduction="mean",
            freq=freq,
            agg=agg,
        )


class get_summer_days(daily_threshold_index):
    """
    Count of days with daily maximum temperature >= 25 degC.

    Sub-daily temperature input is reduced to a daily maximum before the threshold is
    applied, so the transform accepts any cadence from sub-hourly through daily. Monthly
    or coarser input raises ``ValueError``.

    Parameters
    ----------
    condname : str, optional
        Output variable name. Default ``"summer_days"``.
    var : str, optional
        Input variable name. Default ``"tasmax"``.
    freq : str, optional
        MIP frequency to resample the count to. Default ``"mon"``.
    agg : str, optional
        ``"sum"`` (default) gives summer-day counts per ``freq`` window;
        ``"mean"`` gives the fraction of days in each window that qualified.

    Examples
    --------

    .. code-block:: yaml

        transforms:
          - get_summer_days:
              freq: mon
    """

    def __init__(
        self,
        condname: str = "summer_days",
        var: str = "tasmax",
        freq: str = "mon",
        agg: str = "sum",
    ):
        super().__init__(
            condname=condname,
            var=var,
            threshold="25 [degC]",
            operator=">=",
            daily_reduction="max",
            freq=freq,
            agg=agg,
        )


class get_ice_days(daily_threshold_index):
    """
    Count of days with daily maximum temperature < 0 degC.

    Sub-daily temperature input is reduced to a daily maximum before the threshold is
    applied, so the transform accepts any cadence from sub-hourly through daily.
    Monthly or coarser input raises ``ValueError``.

    Parameters
    ----------
    condname : str, optional
        Output variable name. Default ``"ice_days"``.
    var : str, optional
        Input variable name. Default ``"tasmax"``.
    freq : str, optional
        MIP frequency to resample the count to. Default ``"mon"``.
    agg : str, optional
        ``"sum"`` (default) gives ice-day counts per ``freq`` window;
        ``"mean"`` gives the fraction of days in each window that qualified.

    Examples
    --------

    .. code-block:: yaml

        transforms:
          - get_ice_days:
              freq: mon
    """

    def __init__(
        self,
        condname: str = "ice_days",
        var: str = "tasmax",
        freq: str = "mon",
        agg: str = "sum",
    ):
        super().__init__(
            condname=condname,
            var=var,
            threshold="0 [degC]",
            operator="<",
            daily_reduction="max",
            freq=freq,
            agg=agg,
        )


class get_tropical_nights(daily_threshold_index):
    """
    Count of days with daily minimum temperature > 20 degC.

    Sub-daily temperature input is reduced to a daily minimum before the threshold is
    applied, so the transform accepts any cadence from sub-hourly through daily.
    Monthly or coarser input raises ``ValueError``.

    Parameters
    ----------
    condname : str, optional
        Output variable name. Default ``"tropical_nights"``.
    var : str, optional
        Input variable name. Default ``"tasmin"``.
    freq : str, optional
        MIP frequency to resample the count to. Default ``"mon"``.
    agg : str, optional
        ``"sum"`` (default) gives tropical-night counts per ``freq`` window;
        ``"mean"`` gives the fraction of days in each window that qualified.

    Examples
    --------

    .. code-block:: yaml

        transforms:
          - get_tropical_nights:
              freq: mon
    """

    def __init__(
        self,
        condname: str = "tropical_nights",
        var: str = "tasmin",
        freq: str = "mon",
        agg: str = "sum",
    ):
        super().__init__(
            condname=condname,
            var=var,
            threshold="20 [degC]",
            operator=">",
            daily_reduction="min",
            freq=freq,
            agg=agg,
        )


class get_frost_days(daily_threshold_index):
    """
    Count of days with daily minimum temperature < 0 degC.

    Sub-daily temperature input is reduced to a daily minimum before the threshold is
    applied, so the transform accepts any cadence from sub-hourly through daily.
    Monthly or coarser input raises ``ValueError``.

    Parameters
    ----------
    condname : str, optional
        Output variable name. Default ``"frost_days"``.
    var : str, optional
        Input variable name. Default ``"tasmin"``.
    freq : str, optional
        MIP frequency to resample the count to. Default ``"mon"``.
    agg : str, optional
        ``"sum"`` (default) gives frost-day counts per ``freq`` window;
        ``"mean"`` gives the fraction of days in each window that qualified.

    Examples
    --------

    .. code-block:: yaml

        transforms:
          - get_frost_days:
              freq: mon
    """

    def __init__(
        self,
        condname: str = "frost_days",
        var: str = "tasmin",
        freq: str = "mon",
        agg: str = "sum",
    ):
        super().__init__(
            condname=condname,
            var=var,
            threshold="0 [degC]",
            operator="<",
            daily_reduction="min",
            freq=freq,
            agg=agg,
        )
