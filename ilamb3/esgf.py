from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from intake_esgf import ESGFCatalog
from rich.progress import track

import ilamb3.dataset as ild
import ilamb3.load as ill
import ilamb3.run as ilr
from ilamb3.cache import dataframe_cache


def get_setup_variable_info(setup: dict[str, Any]) -> pd.DataFrame:
    """
    Given a setup block, return the variables used, their frequencies, and time extents.
    """
    df = []
    for var, key in setup["sources"].items():
        ds = ill.load_key_or_filename(key)
        freq_lbl = ild.get_frequency_label(ds)
        # Some data will be yearly (like biomass) but that isn't in ESGF, just
        # punt and overwrite the frequency label for now.
        freq_lbl = "mon" if freq_lbl == "yr" else freq_lbl
        try:
            tmin, tmax = ild.get_time_extent(ds)
            tmin = tmin.dt.strftime("%Y-%m-%d").values
            tmax = tmax.dt.strftime("%Y-%m-%d").values
        except KeyError:
            continue
        df.append(
            {
                "variable_id": var,
                "frequency": freq_lbl,
                "tmin": tmin,
                "tmax": tmax,
            }
        )
    if not df:
        return pd.DataFrame()

    # There may be other variables involved, expand the dataframe with the most
    # limiting conditions for all other variables.
    freq_label = min(
        [row["frequency"] for row in df],
        key=lambda r: ild.CMIP_TIME_FREQUENCY.get(r, float("inf")),
    )
    tmin = max([row["tmin"] for row in df if row["tmin"] != ""])
    tmax = min([row["tmax"] for row in df if row["tmax"] != ""])

    related = ilr.find_related_variables(
        ilr.setup_analyses(setup, Path("/tmp/")),
        ilr.setup_transforms(setup),
        setup.get("alternate_vars", []),
    )
    df += [
        {"variable_id": var, "frequency": freq_label, "tmin": tmin, "tmax": tmax}
        for var in related
    ]
    df = pd.DataFrame(df).drop_duplicates(subset=["variable_id", "frequency"])
    return df


def get_configure_variables(configure_yaml: Path) -> pd.DataFrame:
    """
    Return a dataframe of which variables this confrontation uses along with their frequencies and time extents.
    """
    timestamp = str(configure_yaml.stat().st_mtime)
    df = get_configure_variables_cached(str(configure_yaml.absolute()), timestamp)
    return df


@dataframe_cache
def get_configure_variables_cached(configure_yaml: str, timestamp: str) -> pd.DataFrame:
    """
    A cached version of `get_configure_variables` that will only recompute if the configure file has been modified.
    """

    def _freq_min(series: pd.Series) -> str:
        min_freq = min(
            series, key=lambda f: ild.CMIP_TIME_FREQUENCY.get(f, float("inf"))
        )
        return min_freq

    cfg = ilr.parse_benchmark_setup(configure_yaml)
    cfg = ilr._flatten_dict(cfg)
    df = []
    for _, setup in track(cfg.items(), description="Examining reference data"):
        df += [get_setup_variable_info(setup)]
    df = pd.concat(df, ignore_index=True)
    df = (
        df.groupby(["variable_id"])
        .agg(
            {
                "frequency": _freq_min,
                "tmin": "min",
                "tmax": "max",
            }
        )
        .reset_index()
    )
    df = df.sort_values(["variable_id", "frequency"]).astype(str)
    return df


def esgf_remove_duplicate_tables(cat: ESGFCatalog) -> ESGFCatalog:
    """
    Remove duplicate tables from the catalog.

    Note
    ----
    Sometimes a variable is published twice in possibly multiple tables (like ImonAnt, ImonGre).
    These will come up in our search but we do not want them if the regular table (Amon, Lmon)
    entries are available.
    """
    PREFERRED_TABLES = ["Amon", "Lmon", "LImon"]
    to_remove = []
    for _, grp in cat.df.groupby(
        cat.project.modelgroup_facets()
        + [
            cat.project.variable_facet(),
        ]
    ):
        if len(grp) > 1:
            if grp["table_id"].isin(PREFERRED_TABLES).any():
                to_remove += grp[~grp["table_id"].isin(PREFERRED_TABLES)].index.tolist()
            else:
                to_remove += grp.index[1:].tolist()
    cat.df = cat.df.drop(to_remove, axis=0)
    return cat


def esgf_remove_nonmax(cat: ESGFCatalog) -> ESGFCatalog:
    """
    Remove non-maximum ensemble members from the catalog.

    Note
    ----
    Each model will have possibly many ensemble members but they do not always
    have all the variables. This will find the per ensemble member maximum and
    remove all members that are less than this maximum.
    """

    def _remove_nonmax(max_ensembles: pd.Series, df: pd.DataFrame) -> bool:
        max_count = max_ensembles.loc[df.iloc[0]["source_id"]]
        if len(df) < max_count:
            return False
        return True

    max_ensembles = cat.model_groups().groupby(level=0).max()
    cat.remove_incomplete(partial(_remove_nonmax, max_ensembles))
    return cat


def get_esgf_catalog(
    df: pd.DataFrame, source_ids: list[str] | None = None
) -> ESGFCatalog:
    """
    Given the dataframe of variable info populated from the configure file, use
    intake-esgf to get a ESGFCatalog of what is available.

    Note
    ----
    This will search through all ensembles, but then return only the smallest
    ensemble which contains as much of the data as we can find. It will also
    fix problems like variables existing in multiple tables.
    """
    kwargs = dict(
        experiment_id="historical",
        variable_id=list(df["variable_id"].unique()),
        frequency=list(df["frequency"].unique()),
        file_start=str(df["tmin"].min()),
        file_end=str(df["tmax"].max()),
    )
    if source_ids is not None:
        kwargs["source_id"] = source_ids
    cat = ESGFCatalog().search(**kwargs)
    cat = esgf_remove_duplicate_tables(cat)
    cat = esgf_remove_nonmax(cat)
    cat.remove_ensembles()
    return cat


def download_esgf_catalog(
    df: pd.DataFrame, cat: ESGFCatalog, threshold_date: str = "1960-01-01"
) -> dict[str, list[str]]:
    """
    Download the catalog and return a dictionary of lists of paths per dataset.

    Note
    ----
    Most reference data will require model data at earliest around the 1960's.
    However, there are some estimates that extend back to 1850. In order to avoid
    downloading needless files, we will apply a filter and then repeat the download
    for any files with earlier start dates.
    """
    path_dict: dict[str, list[str]] = {}
    grps = cat.model_groups()

    # Downloads everything that starts beyond the threshold date
    rem_tmin = cat.file_start
    if df["tmin"].min() < threshold_date:
        cat.file_start = pd.Timestamp(df[df["tmin"] >= threshold_date]["tmin"].min())
    path_dict.update(cat.to_path_dict(minimal_keys=False))

    # Now download the rest, if present
    cat.file_start = rem_tmin
    downloaded_vars = df[df["tmin"] >= threshold_date]["variable_id"]
    cat.df = cat.df.drop(cat.df[cat.df["variable_id"].isin(downloaded_vars)].index)
    try:
        path_dict.update(cat.to_path_dict(minimal_keys=False))
    except ValueError:
        pass

    for source_id, _, grid_label in grps.index:
        for var in ["areacella", "sftlf"]:
            tmp = cat.clone()
            tmp.search(
                source_id=source_id,
                grid_label=grid_label,
                variable=var,
                frequency="fx",
            )
            tmp.df = tmp.df.iloc[:1]
            path_dict.update(tmp.to_path_dict(minimal_keys=False))

    return path_dict
