"""
Scores computed from additive statistics.

A score maps a relative error onto the unit interval with a nonlinear function, so a
map of scores must not be averaged. Each score is instead computed in three steps:

1. `statistics` reduces the error in each cell to two additive statistics, the mean
   squared error ``mse`` and the mean squared error of the reference ``mse_ref``,
   which is the square of the normalizer.
2. `aggregate` takes the mean of the statistics over a region or over dimensions.
3. `score` maps the relative error ``sqrt(mse / mse_ref)`` onto a score.

Scoring statistics that were not aggregated gives a local score, such as a map, which
is flagged so that `ilamb3` refuses to aggregate it. `add_score` stores a score map
with the statistics it was computed from, and `get_statistics` recovers them, so that
regional scores can be recomputed exactly from an analysis output.

See Also
--------
ilamb3.dataset.mark_local_score : Flag a local score.
"""

import operator
from functools import reduce
from types import EllipsisType
from typing import Literal, get_args

import numpy as np
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.analysis.base import integrate_or_mean
from ilamb3.exceptions import MissingVariable

Method = Literal["Collier2018", "RegionalQuantiles"]

# Statistics stored with a score map are named after it with this suffix
STATISTICS_SUFFIX = "_statistics"


def statistics(
    error: xr.DataArray,
    norm: xr.DataArray | None = None,
    grid: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Return the additive statistics of an error.

    Parameters
    ----------
    error : xr.DataArray
        The error in each cell, such as a bias or a centralized RMSE.
    norm : xr.DataArray, optional
        The normalizer in each cell, in the units of `error`, such as the reference
        standard deviation. Omit it to get only the mean squared error.
    grid : xr.Dataset, optional
        The dataset on whose grid the error is defined. The cell bounds of that grid
        are kept, so that `aggregate` weights each cell by its true area.

    Returns
    -------
    xr.Dataset
        The mean squared error ``mse`` and, if `norm` is given, the mean squared error
        of the reference ``mse_ref``, on the cells where every statistic is defined.
    """
    stats = {"mse": _square(error)}
    if norm is not None:
        stats["mse_ref"] = _square(norm)
    return _with_bounds(_paired(xr.Dataset(stats)), grid)


def aggregate(
    stats: xr.Dataset,
    region: str | None = None,
    *,
    dim: str | list[str] | EllipsisType | None = None,
    weight: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Return the mean of each statistic over a region or over dimensions.

    The statistics are additive, so their mean over many cells is the statistic of
    those cells pooled. Only cells where every statistic is defined are used.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics, see `statistics`.
    region : str, optional
        The region label over which to take the area-weighted mean, or `None` for the
        whole domain.
    dim : str, list of str, or ..., optional
        Dimensions over which to take the mean instead of a region, such as years, or
        ``...`` for all dimensions.
    weight : xr.DataArray, optional
        An additional weight for each cell in a region, used when mass weighting.

    Returns
    -------
    xr.Dataset
        The aggregated statistics.
    """
    if dim is not None and (region is not None or weight is not None):
        raise ValueError("Aggregate over a region or over dimensions, not both.")
    dset.check_aggregable(stats)
    stats = _paired(stats)
    if dim is not None:
        return stats.mean(dim=dim)
    return xr.Dataset(
        {
            name: integrate_or_mean(stats, str(name), region, mean=True, weight=weight)
            for name in stats.data_vars
        }
    )


def score(stats: xr.Dataset, method: Method = "Collier2018") -> xr.DataArray:
    """
    Score the relative error of the statistics.

    Parameters
    ----------
    stats : xr.Dataset
        The statistics, aggregated for a regional score or not for a local score.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.

    Returns
    -------
    xr.DataArray
        The score of ``sqrt(mse / mse_ref)``. A score that still varies over cells is
        flagged as a local score. Where ``mse_ref`` is zero the score is 0, or NaN if
        ``mse`` is also zero.
    """
    if "mse_ref" not in stats:
        raise MissingVariable(
            "The statistics lack 'mse_ref', the squared normalizer, so the error "
            "cannot be made relative. Compute them with scoring.statistics(error, norm), "
            "giving the normalizer."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = np.sqrt(stats["mse"] / stats["mse_ref"])
    match method:
        case "Collier2018":
            out = np.exp(-relative_error)
        case "RegionalQuantiles":
            out = np.clip(1.0 - relative_error, 0, 1)
        case _:
            methods = ", ".join(f"'{m}'" for m in get_args(Method))
            raise ValueError(f"Unknown method '{method}'. Use one of {methods}.")
    out.attrs = {"units": "1"}
    return dset.mark_local_score(out) if out.ndim else out


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
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.

    Returns
    -------
    xr.Dataset
        The dataset with the score map `name`, and its statistics stacked along a
        ``statistic`` dimension in `name` followed by `STATISTICS_SUFFIX`.
    """
    ds = ds.assign(
        {
            name: score(stats, method),
            name + STATISTICS_SUFFIX: stats.to_dataarray(dim="statistic").assign_attrs(
                units=stats["mse"].attrs["units"]
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
        da.attrs = dict(stacked.attrs)
    return _with_bounds(stats, ds)


def _square(da: xr.DataArray) -> xr.DataArray:
    """Square the input, labelling the units as squared without converting them."""
    out = da**2
    out.attrs = {"units": f"({da.attrs['units']})**2" if "units" in da.attrs else "1"}
    return out


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
