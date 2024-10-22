import os
import warnings
from pathlib import Path

import pandas as pd
import xarray as xr

import ilamb3
import ilamb3.analysis as anl
import ilamb3.dataset as dset
from ilamb3.exceptions import AnalysisFailure
from ilamb3.models.base import Model

DEFAULT_ANALYSES = {"bias": anl.bias_analysis}


def open_reference_data(key_or_path: str | Path) -> xr.Dataset:
    """
    Load the reference data.

    Parameters
    ----------
    key_or_path: str or Path
        The key or path to the reference data. First we check if it is a key in the
        `ilamb3.ilamb_catalog()` and then if it is a path to a file. That path may be
        absolute or relative to an environment variable `ILAMB_ROOT`.

    Returns
    -------
    xr.Dataset
        The reference data.
    """
    key_or_path = Path(key_or_path)
    if key_or_path.is_file():
        return xr.open_dataset(key_or_path)
    root = Path(os.environ["ILAMB_ROOT"])
    if (root / key_or_path).is_file():
        return xr.open_dataset(root / key_or_path)
    cat = ilamb3.ilamb_catalog()
    key_or_path = str(key_or_path)
    if key_or_path in cat:
        return cat[key_or_path].read()
    raise FileNotFoundError(
        f"'{key_or_path}' is not a key in the ilamb3 catalog, nor is it a valid file as an absolute path or relative to ILAMB_ROOT={root}"
    )


def work_model_data(work):

    # Unpack the work
    model, analysis_setup = work

    # We will catch all warnings and then put them into the log file
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")

        # If a specialized analysis will be required, then it should be given in the
        # analysis_setup.
        if "analysis" in analysis_setup:
            raise NotImplementedError()
        else:
            df, ds_ref, ds_com = work_default_analysis(model, **analysis_setup)

    return df, ds_ref, ds_com


def work_default_analysis(model: Model, **analysis_setup):

    # Check on inputs
    sources = analysis_setup.get("sources", {})
    relationships = analysis_setup.get("relationships", {})
    if len(sources) != 1:
        raise ValueError(
            f"The default ILAMB analysis requires a single variable and source, but I found: {sources}"
        )
    variable = list(sources.keys())[0]

    # Setup the default analysis
    analyses = {
        name: a(variable)
        for name, a in DEFAULT_ANALYSES.items()
        if analysis_setup.get(f"skip_{name}", False) is False
    }
    analyses.update(
        {
            f"rel_{ind_variable}": anl.relationship_analysis(variable, ind_variable)
            for ind_variable in relationships
        }
    )

    # Get reference data
    ref = {v: open_reference_data(s) for v, s in (sources | relationships).items()}
    if relationships:
        # Interpolate relationships to the reference grid
        lat = ref[variable][dset.get_dim_name(ref[variable], "lat")]
        lon = ref[variable][dset.get_dim_name(ref[variable], "lon")]
        for v in relationships:
            ref[v] = ref[v].interp(
                {
                    dset.get_dim_name(ref[v], "lat"): lat,
                    dset.get_dim_name(ref[v], "lon"): lon,
                },
                method="nearest",
            )
    ref = xr.merge([v for _, v in ref.items()])

    # Get the model data, only use the measures from the primary variables
    com = [
        model.get_variable(v).drop_vars("cell_measures", errors="ignore")
        for v in set([v for _, a in analyses.items() for v in a.required_variables()])
    ]
    com = xr.merge(com)

    # Run the analyses
    dfs = []
    ds_refs = []
    ds_coms = []
    for name, a in analyses.items():
        try:
            df, ds_ref, ds_com = a(ref, com)
        except Exception:
            raise AnalysisFailure(name, variable, "?", model.name)
        dfs.append(df)
        ds_refs.append(ds_ref)
        ds_coms.append(ds_com)

    # Merge results
    dfs = pd.concat(dfs)
    dfs["source"] = dfs["source"].str.replace("Comparison", model.name)

    return dfs, ds_refs, ds_coms


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and "sources" not in v:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
