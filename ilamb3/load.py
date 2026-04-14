"""
Functions encapsulating the logic of how ilamb3 loads data during a run.
"""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
from ilamb3.exceptions import VarNotInModel


def fix_pint_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Return the dataset with the units adjusted.

    Note
    ----
    The `pint` package is a great python-only package for handling units, but
    some things that are commonly used in ESMs are not handled and so we
    implement a hack here.
    """

    def _fix(units: str) -> str:
        """
        Modify units that pint cannot handle.
        """
        try:
            val_units = float(units)
        except ValueError:
            return units
        if np.allclose(val_units, 1):
            return "dimensionless"
        if np.allclose(val_units, 1e-3):
            return "psu"
        return units

    for var, da in ds.items():
        if "units" not in da.attrs:
            continue
        ds[var].attrs["units"] = _fix(da.attrs["units"])
    return ds


def fix_lndgrid_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Return a dataset with coordinates properly assigned.

    Note
    ----
    E3SM/CESM2 raw land model output comes as if it were run over sites in the
    `lndgrid` dimension. Some of the variables that are listed in these files
    are only of this dimension and really belong in the coordinates. These tend
    to be things like `lat`, `lon`, etc. and ilamb3 needs them to be associated
    with the dataset coordinates to work.
    """
    return ds.assign_coords({v: ds[v] for v in ds if ds[v].dims == ("lndgrid",)})


def _lookup(df: xr.Dataset, key: str) -> list[str]:
    """
    Lookup the key in the dataframe.

    Note
    ----
    This function also allows that the key may be a regular expression or a path
    to a file, either absolute or relative to ILAMB_ROOT.
    """
    try:
        return [df.loc[key, "path"]]
    except KeyError:
        pass
    out = sorted(df[df.index.str.contains(key)]["path"].to_list())
    if out:
        return out
    # The key could rather be an absolute/relative path
    path = Path(key)
    if path.is_file():
        return [key]
    if "ILAMB_ROOT" in os.environ:
        path = Path(os.environ["ILAMB_ROOT"]) / path
        if path.is_file():
            return [str(path)]
    raise ValueError(
        f"Could not find {key} in the reference dataframe or locate it as a data file."
    )


def _is_uniform(
    condition: Callable[[xr.DataArray], bool], dsd: dict[str, xr.Dataset]
) -> bool:
    """
    Given a condition, return whether true for all items in the dataset
    dictionary.

    Note
    ----
    This function could live in ilamb3.compare, but we are making an assumption
    that is valid here for our reference data--the keys of the dataset
    dictionary are also found in the contained datasets.
    """
    return all([condition(ds[key]) for key, ds in dsd.items()])


def load_reference_data(
    reference_data: pd.DataFrame,
    variable_id: str,
    sources: dict[str, str],
    relationships: dict[str, str] | None = None,
    transforms: list | None = None,
) -> xr.Dataset:
    """
    Load the reference data into containers and merge if more than 1 variable is
    used.
    """
    # First load all variables defined as `sources` or in `relationships`.
    if relationships is not None:
        sources = sources | relationships
    ref = {
        key: xr.open_mfdataset(_lookup(reference_data, str(filename)))
        for key, filename in sources.items()
    }
    # Sometimes there is a bounds variable but it isn't in the attributes
    ref = {key: dset.fix_missing_bounds_attrs(ds) for key, ds in ref.items()}
    # Merge all the data together
    if len(ref) > 1:
        if _is_uniform(dset.is_spatial, ref):
            grid_variable = variable_id if variable_id in ref else next(iter(ref))
            ref = cmp.same_spatial_grid(ref[grid_variable], **ref)
        if _is_uniform(dset.is_temporal, ref):
            ref = {
                var: cmp.convert_calendar_monthly_noleap(ds) for var, ds in ref.items()
            }
            ref = cmp.trim_time(**ref)
        ds_ref = xr.merge(
            [
                ds if varname == variable_id else ds[varname]
                for varname, ds in ref.items()
            ],
            compat="override",
        )
    else:
        ds_ref = ref[variable_id]
    ds_ref = fix_pint_units(ds_ref)
    # Finally apply transforms
    for transform in transforms:
        ds_ref = transform(ds_ref)
    if variable_id not in ds_ref:
        raise VarNotInModel(
            f"Could not find or create '{variable_id}' from reference data:\n{ds_ref}"
        )
    return ds_ref


def load_comparison_data(
    df: pd.DataFrame,
    variable_id: str,
    alternate_vars: list[str] | None = None,
    transforms: list | None = None,
) -> xr.Dataset:
    """
    Load the comparison (model) data into containers and merge if more than 1
    variable is used.

    Parameters
    ----------
    df: pd.DataFrame
        The database of all possible variables and where to load them.
    variable_id: str
        The name of the variable that is the focus in the comparison.
    alternate_vars: list[str], optional
        A list of acceptable synonyms to be used if `variable_id` is not found.
    transforms: list, optional
        A list of functions that operate on the combined dataset.
    """
    # First load all variables passed into the input dataframe. This will
    # include all relationship variables as well as alternates.
    com = {
        var: xr.open_mfdataset(
            sorted((df[df["variable_id"] == var]["path"]).to_list()),
            preprocess=fix_lndgrid_coords,
            data_vars="minimal",
        )
        for var in df["variable_id"].unique()
    }
    # If the variable_id is not present, it may be called something else
    if alternate_vars is not None and variable_id not in com:
        found = [v for v in alternate_vars if v in com]
        if found:
            found = found[0]
            com[variable_id] = (
                com[found].rename_vars({found: variable_id})
                if found in com[found]
                else com[found]
            )
            com.pop(found)
    # Fix bounds attributes (there is a bounds variable but it isn't in the
    # attributes)
    com = {var: dset.fix_missing_bounds_attrs(ds) for var, ds in com.items()}
    # Merge all the data together
    if len(com) > 1:
        # The grids should be the same, but sometimes models generate output
        # with very small differences in lat/lon
        try:
            com = cmp.same_spatial_grid(com[next(iter(com))], **com)
        except KeyError:
            pass
        ds_com = xr.merge([v for _, v in com.items()], compat="override")
    else:
        ds_com = com[next(iter(com))]
    # pint can't handle some units like `0.001`, so we have to intercept and fix
    ds_com = fix_pint_units(ds_com)
    # Finally apply transforms. These may create the needed variable.
    for transform in transforms:
        ds_com = transform(ds_com)
    if variable_id not in ds_com:
        raise VarNotInModel(
            f"Could not find or create '{variable_id}' from model variables {list(df['variable_id'].unique())}"
        )
    ds_com = dset.cmip_cell_measures(ds_com, variable_id)
    ds_com = ds_com[["gpp", "time_bounds"]]
    print(ds_com)
    return ds_com
