"""Analysis functions used in ILAMB.

These functions are designed to work on their own, called outside of an analysis run,
but also be efficient when many are called in sequence.

"""
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ilamb3 import compare as cmp
from ilamb3 import dataset as dset


def bookkeeping(dsa: xr.DataArray, dsb: xr.DataArray, mean: bool = True):
    """."""
    dsa, dsb = cmp.pick_grid_aligned(dsa, dsb)
    for logic in [
        dsa.notnull() * dsb.notnull(),  # a and b
        dsa.notnull() * dsb.isnull(),  # a not b
        dsa.isnull() * dsb.notnull(),  # b not a
    ]:
        print(dset.integrate_space(dsa * logic, mean=mean))


def bias_collier2018(
    ref: xr.Dataset,
    com: xr.Dataset,
    varname: str,
    regions: Union[None, list[str]] = None,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Score the bias between two variables.

    Analyze the bias of the given variable between the reference and comparison
    datasets following the methodology explained in
    [Collier2018](https://doi.org/10.1029/2018MS001354).

    Parameters
    ----------
    ref
        The reference dataset.
    com
        The comparison dataset.
    varname
        The name of the variable to compare.
    regions
        The list of regions over which to perform the analysis.

    Returns
    -------
    df
        A dataframe containing scalars from the analysis.
    ds
        A dataset containing data to plot.

    """
    aname = "Bias"
    dfs = []

    # Ideally this will already be done once for all metrics outside this
    # routine, but so that the function can be used on any two datasets, we need
    # to ensure that the two objects are comparable.
    ref, com = cmp.make_comparable(ref, com, varname)

    # Temporal means across the time period
    ref_mean = (
        dset.integrate_time(ref, varname, mean=True) if "time" in ref.dims else ref
    )
    com_mean = (
        dset.integrate_time(com, varname, mean=True) if "time" in com.dims else com
    )

    # If temporal information is available, we normalize the error by the
    # standard deviation of the reference. If not, we revert to the traditional
    # definition of relative error.
    norm = ref_mean
    if "time" in ref.dims and ref["time"].size > 1:
        norm = dset.std_time(ref, varname)

    # Nest the grids, we postpend composite grid variables with "_"
    ref_, com_, norm_ = cmp.nest_spatial_grids(ref_mean, com_mean, norm)
    bias = com_ - ref_
    score = np.exp(-np.abs(bias) / norm_)

    # Reference period mean
    refw = dset.integrate_space(ref_mean, varname, mean=True)

    dfs.append(
        [
            "Reference",
            str(None),
            aname,
            "Period Mean",
            "scalar",
            f"{refw.pint.units:~cf}",
            float(refw.pint.dequantify()),
        ]
    )
    return dfs, score
