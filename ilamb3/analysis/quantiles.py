"""Functions for use in scoring methods using regional quantiles."""

import pandas as pd
import xarray as xr

import ilamb3.dataset as ild
import ilamb3.regions as ilr
from ilamb3.exceptions import MissingRegion, NoDatabaseEntry


def check_quantile_database(dbase: pd.DataFrame | None) -> None:
    if dbase is None:
        raise ValueError("Need a quantile database")
    missing = set(dbase["region"].unique()) - set(ilr.Regions().regions)
    if missing:
        raise MissingRegion(
            "Regional quantile database uses regions with no definition in ilamb3 regions"
        )


def create_quantile_map(
    dbase: pd.DataFrame,
    quantile_variable: str,
    quantile_type: str,
    quantile_threshold: int,
) -> xr.DataArray:
    # query the database
    q = f"(quantile=={quantile_threshold})"
    q += f" & (type=='{quantile_type}')"
    q += f" & (variable=='{quantile_variable}')"
    q = dbase.query(q)
    if not len(q):
        raise NoDatabaseEntry

    # build a map
    scalars = {region: value for region, value in zip(q["region"], q["value"])}
    scalar_map = ilr.Regions().region_scalars_to_map(
        ild.ones_grid(resolution=0.5), scalars
    )
    scalar_map.attrs["units"] = q.iloc[0]["unit"]
    scalar_map.name = "quantile"
    return scalar_map
