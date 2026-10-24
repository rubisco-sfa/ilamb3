"""
Scores computed from additive statistics.

A score maps a relative error onto the unit interval with a nonlinear function, so a
map of scores must not be averaged. Each score is instead computed in three steps:

1. `statistics` reduces the error in each cell to two additive statistics, the mean
   squared error ``mse`` and the mean squared error of the reference ``mse_ref``,
   which is the square of the normalizer.
2. `aggregate` takes the mean of the statistics over a region or over dimensions.
3. `score` maps the relative error ``sqrt(mse / mse_ref)`` onto a score.

A score that is linear in its error, such as the Seasonal Cycle score, uses the mean
absolute error ``mae`` and ``mae_ref`` instead, for which aggregating the error and
averaging the scores agree.

Scoring statistics that were not aggregated gives a local score, such as a map, which
is flagged so that `ilamb3` refuses to aggregate it. `add_score` stores a score map
with the statistics and method it was computed from, and `regional_score` recomputes
a regional score exactly from an analysis output.

The same rule applies across analyses and datasets. A score row in the results table
carries the normalized mean squared error ``nmse`` it was computed from, and
`add_overall_score` aggregates those errors and scores the result once.

See Also
--------
ilamb3.dataset.mark_local_score : Flag a local score.
"""

import operator
from functools import reduce
from types import EllipsisType
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.dataset as dset
import ilamb3.regions as ilr
from ilamb3.analysis.base import integrate_or_mean
from ilamb3.exceptions import MissingVariable

Method = Literal["Collier2018", "RegionalQuantiles", "Linear"]
Power = Literal[1, 2]

# The names of the statistics of an error raised to each power, and of its normalizer
_NAMES = {2: ("mse", "mse_ref"), 1: ("mae", "mae_ref")}

# Statistics stored with a score map are named after it with this suffix
STATISTICS_SUFFIX = "_statistics"

# Normalizer labels. The Overall Score adds the NMSEs of scores that share a label,
# such as Bias and RMSE, which divide by the variance of the reference in time and so
# are components of one error.
TIME_VARIANCE = "variance of the reference in time"
ANNUAL_VARIANCE = "variance of the annual means of the reference"
REFERENCE_MEAN = "squared mean of the reference"
REFERENCE_UNCERTAINTY = "squared uncertainty of the reference"
REGIONAL_QUANTILE = "squared regional quantile"

OVERALL_SCORE = "Overall Score [1]"


def statistics(
    error: xr.DataArray,
    norm: xr.DataArray | float | None = None,
    grid: xr.Dataset | None = None,
    *,
    power: Power = 2,
) -> xr.Dataset:
    """
    Return the additive statistics of an error.

    Parameters
    ----------
    error : xr.DataArray
        The error in each cell, such as a bias or a centralized error.
    norm : xr.DataArray or float, optional
        The normalizer in each cell, or one for all cells, in the units of `error`,
        such as the reference standard deviation. Omit it to get only the statistic of
        the error.
    grid : xr.Dataset, optional
        The dataset on whose grid the error is defined. The bounds of that grid are
        kept, so that `aggregate` weights each cell by its true area and each time step
        by its true length.
    power : int
        2 for the mean squared error ``mse`` and its normalizer ``mse_ref``, or 1 for
        the mean absolute error ``mae`` and ``mae_ref``.

    Returns
    -------
    xr.Dataset
        The statistics of the error and, if `norm` is given, of the normalizer, on the
        cells where every statistic is defined.
    """
    numerator, denominator = _NAMES[power]
    stats = {numerator: _power(error, power)}
    if norm is not None:
        if not isinstance(norm, xr.DataArray):
            norm = xr.full_like(error, norm).assign_attrs(units=error.attrs.get("units"))
        stats[denominator] = _power(norm, power)
    return _with_bounds(_paired(xr.Dataset(stats)), grid)


def weighted(stats: xr.Dataset, weight: xr.DataArray) -> xr.Dataset:
    """
    Return the statistics with each cell weighted, such as by the reference mass.

    A score depends only on the ratio of the aggregated statistics. Multiplying every
    statistic by the weight therefore gives the ratio of their weighted means. The
    weight is divided by its mean, so the statistics keep their units.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics of each cell, see `statistics`.
    weight : xr.DataArray
        The weight of each cell, on the grid of the statistics.

    Returns
    -------
    xr.Dataset
        The weighted statistics.
    """
    weight = weight / weight.mean()
    out = xr.Dataset(
        {name: (da * weight).assign_attrs(da.attrs) for name, da in stats.items()}
    )
    # The same weight multiplies every statistic, so they stay paired
    return _with_bounds(out, stats)


def aggregate(
    stats: xr.Dataset,
    region: str | None = None,
    *,
    dim: str | list[str] | EllipsisType | None = None,
) -> xr.Dataset:
    """
    Return the mean of each statistic over a region or over dimensions.

    The statistics are additive, so their mean over many cells is the statistic of
    those cells pooled. The statistics must be defined on the same cells, as
    `statistics` returns them.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics, see `statistics`.
    region : str, optional
        The region label over which to take the area-weighted mean, or `None` for the
        whole domain.
    dim : str, list of str, or ..., optional
        Dimensions over which to take the mean instead of a region, such as years, or
        ``...`` for all dimensions. Time steps are weighted by their length.

    Returns
    -------
    xr.Dataset
        The aggregated statistics.
    """
    if dim is not None and region is not None:
        raise ValueError("Aggregate over a region or over dimensions, not both.")
    dset.check_aggregable(stats)
    if dim is not None:
        return _mean_over(stats, dim)
    # Mask the region once for all statistics, rather than once for each
    stats = ilr.Regions().mask(stats, region)
    return xr.Dataset(
        {
            name: integrate_or_mean(stats, str(name), None, mean=True)
            for name in stats.data_vars
        }
    )


def nmse(stats: xr.Dataset) -> xr.DataArray:
    """
    Return the normalized mean squared error ``mse / mse_ref`` of the statistics.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics of a squared error, aggregated or not.

    Returns
    -------
    xr.DataArray
        The normalized mean squared error.
    """
    if "mse" not in stats:
        raise MissingVariable(
            "The statistics lack 'mse', so they have no mean squared error. Compute "
            "them with scoring.statistics(error, norm), with power=2."
        )
    return _ratio(stats, 2)


def score_nmse(nmse: Any, method: Method = "Collier2018") -> Any:
    """
    Score a normalized mean squared error.

    Parameters
    ----------
    nmse : float, array, pd.Series, or xr.DataArray
        The normalized mean squared error, aggregated before it is scored.
    method : str
        The name of the scoring methodology to use, one of `Collier2018`,
        `RegionalQuantiles`, or `Linear`.

    Returns
    -------
    float, array, pd.Series, or xr.DataArray
        The score of the relative error ``sqrt(nmse)``.
    """
    return _score_function(np.sqrt(nmse), method)


def score(stats: xr.Dataset, method: Method = "Collier2018") -> xr.DataArray:
    """
    Score the relative error of the statistics.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics, aggregated for a regional score or not for a local score.
    method : str
        The name of the scoring methodology to use, one of `Collier2018`,
        `RegionalQuantiles`, or `Linear`.

    Returns
    -------
    xr.DataArray
        The score of the relative error, ``sqrt(mse / mse_ref)`` or
        ``mae / mae_ref``. A score that still varies over cells is flagged as a local
        score. Where the normalizer is zero the score is 0, or NaN if the error is also
        zero.
    """
    power = 2 if "mse" in stats else 1
    out = _score_function(_ratio(stats, power) ** (1 / power), method)
    out.attrs = {"units": "1"}
    return dset.mark_local_score(out) if out.ndim else out


def score_row(
    stats: xr.Dataset, method: Method = "Collier2018", normalizer: str | None = None
) -> dict[str, Any]:
    """
    Return the entries of a row of the results table for a regional score.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics, aggregated over the region.
    method : str
        The name of the scoring methodology to use, one of `Collier2018`,
        `RegionalQuantiles`, or `Linear`.
    normalizer : str, optional
        A label for the normalizer, given only if the score is a component of the
        error of its dataset. Scores of one dataset with the same label are components
        of one error, see `add_overall_score`. Without a label the score is a
        diagnostic, which the Overall Score leaves out.

    Returns
    -------
    dict
        The ``type``, ``units``, and ``value`` of the row and, for a component, the
        ``nmse`` that was scored and the ``normalizer`` label.
    """
    row = {"type": "score", "units": "1", "value": float(score(stats, method))}
    if normalizer is not None:
        row |= {"nmse": float(nmse(stats)), "normalizer": normalizer}
    return row


def add_score(
    ds: xr.Dataset, name: str, stats: xr.Dataset, method: Method = "Collier2018"
) -> xr.Dataset:
    """
    Return the dataset with a local score map and the statistics it was computed from.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to add the score to.
    name : str
        The name of the score map.
    stats : xr.Dataset
        The statistics of each cell, see `statistics`.
    method : str
        The name of the scoring methodology to use, one of `Collier2018`,
        `RegionalQuantiles`, or `Linear`.

    Returns
    -------
    xr.Dataset
        The dataset with the score map `name`, and its statistics stacked along a
        ``statistic`` dimension in `name` followed by `STATISTICS_SUFFIX`, with the
        method in their attributes.
    """
    units = next(iter(stats.data_vars.values())).attrs["units"]
    ds = ds.assign(
        {
            name: score(stats, method),
            name + STATISTICS_SUFFIX: stats.to_dataarray(dim="statistic").assign_attrs(
                units=units, method=method
            ),
        }
    )
    return _with_bounds(ds, stats)


def get_statistics(ds: xr.Dataset, name: str) -> xr.Dataset:
    """
    Return the statistics stored with a score map by `add_score`.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset holding the score map, such as an analysis output.
    name : str
        The name of the score map.

    Returns
    -------
    xr.Dataset
        The statistics of each cell, ready to `aggregate`.
    """
    if name + STATISTICS_SUFFIX not in ds:
        raise MissingVariable(
            f"The dataset has no '{name}{STATISTICS_SUFFIX}', the statistics that "
            f"scoring.add_score saves with a score map named '{name}'. For a score map "
            "saved without them, aggregate the error map that the score was computed "
            "from, then score the result."
        )
    stacked = ds[name + STATISTICS_SUFFIX]
    stats = stacked.to_dataset(dim="statistic").drop_vars("statistic", errors="ignore")
    for da in stats.data_vars.values():
        da.attrs = {"units": stacked.attrs["units"]}
    return _with_bounds(stats, ds)


def regional_score(ds: xr.Dataset, name: str, region: str | None = None) -> float:
    """
    Return the regional score of a score map, from the statistics stored with it.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset holding the score map, such as an analysis output.
    name : str
        The name of the score map.
    region : str, optional
        The region label, or `None` for the whole domain.

    Returns
    -------
    float
        The score of the statistics aggregated over the region, with the method that
        `add_score` stored.
    """
    stats = get_statistics(ds, name)
    method = ds[name + STATISTICS_SUFFIX].attrs.get("method", "Collier2018")
    return float(score(aggregate(stats, region), method))


def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the results table with the Overall Score of each source and region.

    The Overall Score aggregates the normalized mean squared errors of the scores,
    then scores the result once, so that the ranking of sources does not depend on
    the score function. Scores whose errors share a normalizer label are components
    of one error, so their NMSEs add. The sums for different normalizers are averaged
    with equal weights, as the dashboard averages datasets. Scores without an NMSE are
    diagnostics and are left out, so a source and region without component scores has
    no Overall Score.

    Parameters
    ----------
    df : pd.DataFrame
        The results table, with the ``nmse`` and ``normalizer`` columns of
        `score_row`.

    Returns
    -------
    pd.DataFrame
        The results table with a row for each Overall Score.
    """
    if "nmse" not in df.columns:
        return df
    components = df[(df["type"] == "score") & df["nmse"].notna()]
    overall = (
        components.groupby(["source", "region", "normalizer"])["nmse"]
        .sum()
        .groupby(["source", "region"])
        .mean()
        .reset_index()
    )
    overall = overall.assign(
        name=OVERALL_SCORE, type="score", units="1", value=score_nmse(overall["nmse"])
    )
    return pd.concat([df, overall]).reset_index(drop=True)


def _score_function(relative_error: Any, method: Method) -> Any:
    """Map a relative error onto a score with the function of the method."""
    match method:
        case "Collier2018":
            return np.exp(-relative_error)
        case "RegionalQuantiles" | "Linear":
            return np.clip(1.0 - relative_error, 0, 1)
        case _:
            methods = ", ".join(f"'{m}'" for m in get_args(Method))
            raise ValueError(f"Unknown method '{method}'. Use one of {methods}.")


def _ratio(stats: xr.Dataset, power: Power) -> xr.DataArray:
    """Return the statistic of the error divided by that of the normalizer."""
    numerator, denominator = _NAMES[power]
    if denominator not in stats:
        raise MissingVariable(
            f"The statistics lack '{denominator}', the statistic of the normalizer, so "
            "the error cannot be made relative. Compute them with "
            "scoring.statistics(error, norm), giving the normalizer."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return stats[numerator] / stats[denominator]


def _power(da: xr.DataArray, power: Power) -> xr.DataArray:
    """Raise the magnitude of the input to a power, labelling the units to match."""
    units = da.attrs.get("units")
    if power == 2:
        out = da**2
        out.attrs = {"units": "1" if units is None else f"({units})**2"}
    else:
        out = np.abs(da)
        out.attrs = {"units": "1" if units is None else units}
    return out


def _mean_over(
    stats: xr.Dataset, dim: str | list[str] | EllipsisType
) -> xr.Dataset:
    """Return the mean over dimensions, weighting each time step by its length."""
    dims = (
        list(stats.dims) if dim is ... else [dim] if isinstance(dim, str) else list(dim)
    )
    if dset.is_temporal(stats) and dset.get_dim_name(stats, "time") in dims:
        weights = dset.compute_time_measures(stats).fillna(0)
        return stats.weighted(weights).mean(dim=dims, keep_attrs=True)
    return stats.mean(dim=dims, keep_attrs=True)


def _paired(stats: xr.Dataset) -> xr.Dataset:
    """Mask every statistic where any is missing, so they cover the same cells."""
    valid = reduce(operator.and_, (da.notnull() for da in stats.data_vars.values()))
    return stats.where(valid)


def _with_bounds(stats: xr.Dataset, grid: xr.Dataset | None) -> xr.Dataset:
    """Copy the cell bounds of the dimensions of the statistics from the grid."""
    if grid is None:
        return stats
    for name in stats.dims:
        bounds = grid[name].attrs.get("bounds") if name in grid.coords else None
        if bounds in grid.variables:
            stats = stats.assign_coords({bounds: grid[bounds]})
            stats[name].attrs["bounds"] = bounds
    return stats
