"""Functions for use in scoring methods using regional quantiles."""

import pandas as pd
import xarray as xr

from ilamb3.exceptions import MissingRegion, NoDatabaseEntry
from ilamb3.regions import Regions


def check_quantile_database(dbase: pd.DataFrame) -> None:
    if dbase is None:
        raise ValueError("Need a quantile database")
    missing = set(dbase["region"].unique()) - set(Regions().regions)
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
    scalar_map = Regions().region_scalars_to_map(
        {row["region"]: row["value"] for _, row in q.iterrows()}
    )
    scalar_map.attrs["units"] = q.iloc[0]["unit"]
    return scalar_map
