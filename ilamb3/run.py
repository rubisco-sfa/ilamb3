"""Functions for rendering ilamb3 output."""

import importlib
import re
from itertools import chain
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import xarray as xr
import yaml
from jinja2 import Template
from loguru import logger

import ilamb3
import ilamb3.analysis as anl
import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as ilp
import ilamb3.regions as ilr
from ilamb3.analysis.base import ILAMBAnalysis, add_overall_score
from ilamb3.exceptions import AnalysisNotAppropriate, VarNotInModel
from ilamb3.transform import ALL_TRANSFORMS
from ilamb3.transform.base import ILAMBTransform


def fix_pint_units(ds: xr.Dataset) -> xr.Dataset:
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


def select_analysis_variable(setup: dict[str, Any]) -> str:
    """
    Return the main variable to be used in this analysis.

    We will try to guess it from what the user has specified, but to avoid
    ambiguity please specify `analysis_variable` in your configure block.

    Parameters
    ----------
    setup: dict
        A dictionary of keywords parsed from the configure files.

    Returns
    -------
    str
        The main setup variable.

    """
    variable = setup.get("analysis_variable", None)
    if variable is None and "sources" in setup:
        if len(setup["sources"]) > 1:
            raise ValueError(
                "Ambiguous definition of the main variable of this analysis. "
                "If you have multiple sources, you should specify `analysis_variable` "
                f"in your configure file. This is the problematic portion:\n{yaml.dump(setup)}"
            )
        variable = next(iter(setup["sources"]))
    return variable


def setup_analyses(
    setup: dict[str, Any], output_path: Path | None
) -> dict[ILAMBAnalysis]:
    """
    Return the initialized analysis components to be used for this block.

    You may list individual analyses to run in a `analyses` line in the
    configure block or we will run the ilamb3.analysis.DEFAULT_ANALYSES by
    default. Note that we have special rules for invoking the `relationship`
    analysis. It is run by including a configure line similar to `sources` but
    call `relationships` for each independent variable you wish to compare to
    the main variable of this block.

    Parameters
    ----------
    setup: dict
        A dictionary of keywords parsed from the configure files.

    Returns
    -------
    dict
        A dictionary of initialized ilamb3.analysis.base.ILAMBAnalysis objects.

    """
    main_variable = select_analysis_variable(setup)
    analyses = setup.get("analyses", list(anl.DEFAULT_ANALYSES.keys()))
    invalid = set(analyses) - set(anl.ALL_ANALYSES)
    if invalid:
        raise ValueError(
            f"Invalid analyses given, {invalid} not in {list(anl.ALL_ANALYSES.keys())}. "
            f"This is the problematic portion:\n{yaml.dump(setup)}"
        )
    analyses = {
        a: anl.ALL_ANALYSES[a](
            **(
                setup
                | {
                    "required_variable": main_variable,
                    "output_path": (
                        None
                        if ilamb3.conf["run_mode"] == "interactive"
                        else output_path
                    ),
                }
            )
        )
        for a in analyses
    }
    if "relationships" in setup:
        analyses.update(
            {
                f"Relationship {ind_variable}": anl.ALL_ANALYSES["relationship"](
                    main_variable, ind_variable, **setup
                )
                for ind_variable in setup["relationships"]
            }
        )
    return analyses


def setup_transforms(setup: dict[str, Any]) -> list[ILAMBTransform]:
    """
    Return the initialized transforms to be used for this block.

    Parameters
    ----------
    setup: dict
        A dictionary of keywords parsed from the configure files.

    Returns
    -------
    list
        A list of initialized ilamb3.transform.base.ILAMBTransform objects.

    """
    transforms = setup.get("transforms", [])
    transform_names = [
        next(iter(t.keys())) if isinstance(t, dict) else t for t in transforms
    ]
    invalid = set(transform_names) - set(ALL_TRANSFORMS)
    if invalid:
        raise ValueError(
            f"Invalid transform{'s' if len(invalid) > 1 else ''} given. "
            f"{list(invalid)} not in {list(ALL_TRANSFORMS.keys())}. "
            f"This is the problematic portion:\n{yaml.dump(setup)}"
        )
    transforms = [
        ALL_TRANSFORMS[t](**args[t]) if isinstance(args, dict) else ALL_TRANSFORMS[t]()
        for t, args in zip(transform_names, transforms)
    ]
    return transforms


def find_related_variables(
    analyses: dict[ILAMBAnalysis],
    transforms: list[ILAMBTransform],
    alternate_vars: list[str] | None,
) -> list[str]:
    """
    Return the list of required variables for the given analyses/transforms.

    Parameters
    ----------
    analyses: dict
        A dictionary of initialized ilamb3.analysis.base.ILAMBAnalysis objects.
    transforms: list
        A list of ilamb3.transform.base.ILAMBTransform to apply.
    alternate_vars: list, optional
        A list of alternate variables that this analysis will accept.

    Returns
    -------
    list
        A list of the related variables to be used in query the model database.

    """
    related = list(
        chain(
            *[a.required_variables() for _, a in analyses.items()],
            *[t.required_variables() for t in transforms],
            [] if alternate_vars is None else alternate_vars,
        )
    )
    related = list(set(related))
    return related


def augment_setup_with_options(
    setup: dict[str, Any], reference_data: pd.DataFrame
) -> dict[str, Any]:
    """
    Augment the configure block with global ilamb3 options.
    """
    # Augment options with things in the global options
    if "regions" not in setup:
        setup["regions"] = ilamb3.conf["regions"]
    if "method" not in setup:
        if ilamb3.conf["prefer_regional_quantiles"]:
            setup["method"] = "RegionalQuantiles"
            try:
                setup["quantile_database"] = pd.read_parquet(
                    reference_data.loc[ilamb3.conf["quantile_database"], "path"]
                )
                setup["quantile_threshold"] = ilamb3.conf["quantile_threshold"]
                ilr.Regions().add_netcdf(
                    xr.load_dataset(reference_data.loc["regions/Whittaker.nc", "path"])
                )
            except Exception:
                setup["method"] = "Collier2018"
        else:
            setup["method"] = "Collier2018"
    if "use_uncertainty" not in setup:
        setup["use_uncertainty"] = ilamb3.conf["use_uncertainty"]
    return setup


def _lookup(df: xr.Dataset, key: str) -> list[str]:
    """
    Lookup the key in the dataframe, allowing that it may be a regular expression.
    """
    try:
        return [df.loc[key, "path"]]
    except KeyError:
        pass
    out = sorted(df[df.index.str.contains(key)]["path"].to_list())
    if not out:
        raise ValueError(f"Could not find {key} in the reference dataframe.")
    return out


def _load_reference_data(
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
        ref = cmp.trim_time(**ref)
        ref = cmp.same_spatial_grid(ref[variable_id], **ref)
        ds_ref = xr.merge([v for _, v in ref.items()], compat="override")
    else:
        ds_ref = ref[variable_id]
    # pint can't handle some units like `0.001`, so we have to intercept and fix
    ds_ref = fix_pint_units(ds_ref)
    # Finally apply transforms
    for transform in transforms:
        ds_ref = transform(ds_ref)
    if variable_id not in ds_ref:
        raise VarNotInModel(
            f"Could not find or create '{variable_id}' from reference data:\n{ds_ref}"
        )
    return ds_ref


def cmip_cell_measures(ds: xr.Dataset, varname: str) -> xr.Dataset:
    """
    Add a DataArray for the `cell_measures` built from CMIP variables if present.
    """
    da = ds[varname]
    if "cell_measures" not in da.attrs:
        return ds
    m = re.search(r"area:\s(.*)", da.attrs["cell_measures"])
    if not m:
        return ds
    msr_name = m.group(1)
    if msr_name not in ds:
        return ds
    msr = ds[msr_name]
    ds = ds.drop_vars(msr_name)
    if "cell_methods" in da.attrs:
        if "where land" in da.attrs["cell_methods"] and "sftlf" in ds:
            msr *= ds["sftlf"] * 0.01
            ds = ds.drop_vars("sftlf")
        elif "where sea" in da.attrs["cell_methods"] and "sftof" in ds:
            msr *= ds["sftof"] * 0.01
            ds = ds.drop_vars("sftof")
    msr = xr.where(msr > 0, msr, np.nan)
    ds["cell_measures"] = msr
    return ds


def _load_comparison_data(
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
            data_vars=None,
            compat="no_conflicts",
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
    ds_com = cmip_cell_measures(ds_com, variable_id)
    return ds_com


def registry_to_dataframe(registry: pooch.Pooch) -> pd.DataFrame:
    """
    Convert a ILAMB/IOMB registry to a DatasetCollection for use in REF.

    Parameters
    ----------
    registry : pooch.Pooch
        The pooch registry.

    Returns
    -------
    DatasetCollection
        The converted collection.
    """
    df = pd.DataFrame(
        [
            {
                "key": key,
                "path": registry.abspath / Path(key),
            }
            for key in registry.registry.keys()
        ]
    )
    return df.set_index("key")


def remove_irrelevant_variables(df: pd.DataFrame, **setup: Any) -> pd.DataFrame:
    """
    Remove unused variables from the dataframe.
    """
    reduce = df[
        df["variable_id"].isin(
            list(setup["sources"].keys())
            + list(setup.get("relationships", {}).keys())
            + setup.get("alternate_vars", [])
            + setup.get("related_vars", [])
        )
    ]
    return reduce


def _load_local_assets(
    csv_file: Path, ref_file: Path, com_file: Path
) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
    if not (csv_file.is_file() and ref_file.is_file() and com_file.is_file()):
        raise ValueError()
    df = pd.read_csv(str(csv_file))
    df["region"] = df["region"].astype(str).str.replace("nan", "None")
    ds_ref = xr.open_dataset(str(ref_file))
    ds_com = xr.open_dataset(str(com_file))
    return df, ds_ref, ds_com


def run_single_block(
    block_name: str,
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    output_path: Path,
    **setup: Any,
):
    """
    Run a configuration block.

    Parameters
    ----------

    """
    # Initialize
    if reference_data.index.name != "key":
        reference_data = reference_data.set_index("key")
    setup = augment_setup_with_options(setup, reference_data)
    variable = select_analysis_variable(setup)
    analyses = setup_analyses(setup, output_path)
    transforms = setup_transforms(setup)

    # Thin out the dataframe to only contain variables we need for this block.
    comparison_data = comparison_data[
        comparison_data["variable_id"].isin(
            find_related_variables(
                analyses, transforms, setup.get("alternate_vars", [])
            )
            + ["areacella", "sftlf", "areacello", "sftof"]
        )
    ]

    # Phase I: loop over each model in the group and run an analysis function
    df_all = []
    ds_com = {}
    ds_ref = None
    for _, grp in comparison_data.groupby(ilamb3.conf["comparison_groupby"]):
        # Define what we will call the output artifacts
        source_name = "-".join(
            [grp.iloc[0][f] for f in ilamb3.conf["model_name_facets"]]
        )
        csv_file = output_path / f"{source_name}.csv"
        ref_file = output_path / "Reference.nc"
        com_file = output_path / f"{source_name}.nc"
        log_file = output_path / f"{source_name}.log"
        log_id = logger.add(log_file, backtrace=True, diagnose=True)

        # Attempt to load local assets if preferred
        if ilamb3.conf["use_cached_results"]:
            try:
                dfs, ds_ref, ds_com[source_name] = _load_local_assets(
                    csv_file, ref_file, com_file
                )
                df_all.append(dfs)
                logger.info(f"Using cached information {com_file}")
                continue
            except Exception:
                pass

        try:
            # Load data and run comparison
            ref = _load_reference_data(
                reference_data,
                variable,
                setup["sources"],
                setup["relationships"] if "relationships" in setup else {},
                transforms=transforms,
            )
            com = _load_comparison_data(
                grp,
                variable,
                alternate_vars=setup.get("alternate_vars", []),
                transforms=transforms,
            )
            dfs, ds_ref, ds_com[source_name] = run_analyses(ref, com, analyses)
            dfs["source"] = dfs["source"].str.replace("Comparison", source_name)

            # Set a group name optionally, if facets were specified
            if ilamb3.conf["group_name_facets"] is not None:
                if not set(ilamb3.conf["group_name_facets"]).issubset(grp.columns):
                    raise ValueError(
                        f"Could not set model group name. You gave these facets {ilamb3.conf['group_name_facets']} but I am not finding them in the comparison dataset dataframe {grp.columns}."
                    )
                group_name = grp[ilamb3.conf["group_name_facets"]].apply(
                    lambda row: "-".join(row), axis=1
                )
                assert all(group_name == group_name.iloc[0])
                dfs["group"] = str(group_name.iloc[0])

            # Write out artifacts
            dfs.to_csv(csv_file, index=False)
            if not ref_file.is_file():  # pragma: no cover
                ds_ref.to_netcdf(ref_file)
            ds_com[source_name].to_netcdf(com_file)
            df_all.append(dfs)
        except Exception:  # pragma: no cover
            logger.exception(
                f"ILAMB analysis '{block_name}' failed for '{source_name}'."
            )
            continue

        # Pop log and remove zero size files
        logger.remove(log_id)
        if log_file.stat().st_size == 0:  # pragma: no cover
            log_file.unlink()

    # Check that the reference intermediate data really was generated.
    if ds_ref is None:
        raise ValueError(
            "Reference intermediate data was not generated."
        )  # pragma: no cover

    log_file = output_path / "post.log"
    log_id = logger.add(log_file, backtrace=True, diagnose=True)

    # Phase 2: get plots and combine scalars and save
    try:
        plt.rcParams.update({"figure.max_open_warning": 0})
        df = pd.concat(df_all).drop_duplicates(
            subset=["source", "region", "analysis", "name"]
        )
        df = add_overall_score(df)
        df_plots = plot_analyses(df, ds_ref, ds_com, analyses, output_path)
    except Exception:
        logger.exception(f"ILAMB analysis '{block_name}' failed in plotting.")
        return

    # Generate an output page
    try:
        if ilamb3.conf["debug_mode"] and (output_path / "index.html").is_file():
            logger.remove(log_id)
            return
        ds_ref.attrs["header"] = block_name
        html = generate_html_page(df, ds_ref, ds_com, df_plots)
        with open(output_path / "index.html", mode="w") as out:
            out.write(html)
    except Exception:
        logger.exception(f"ILAMB analysis '{block_name}' failed in generating html.")
        return
    logger.remove(log_id)


def run_analyses(
    ref: xr.Dataset, com: xr.Dataset, analyses: dict[str, ILAMBAnalysis]
) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
    """
    Run the input analyses on the given reference and comparison datasets.

    Parameters
    ----------
    ref : xr.Dataset
        The dataset which will be considered as the reference.
    com : xr.Dataset
        The dataset which will be considered as the comparison.
    analyses: dict[str, ILAMBAnalysis]
        A dictionary of analyses to

    Returns
    -------
    pd.DataFrame, xr.Dataset, xr.Dataset
        Analysis output, dataframe with scalar information and datasets with
        reference and comparison information for plotting.
    """
    dfs = []
    ds_refs = []
    ds_coms = []
    for aname, a in analyses.items():
        try:
            df, ds_ref, ds_com = a(ref, com)
        except AnalysisNotAppropriate:
            continue
        dfs.append(df)
        ds_refs.append(ds_ref)
        ds_coms.append(ds_com)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs["name"] = dfs["name"] + " [" + dfs["units"] + "]"
    ds_ref = xr.merge(ds_refs, compat="override")
    ds_com = xr.merge(ds_coms, compat="override")
    return dfs, ds_ref, ds_com


def regenerate_figs(path: Path) -> bool:
    """
    Do we need to regenerate the figures?
    """
    path.mkdir(exist_ok=True, parents=True)
    png_files = list(path.glob("*.png"))
    if not png_files:
        return True
    first_png_time = min([p.stat().st_mtime for p in png_files])
    nc_files = list(path.glob("*.nc"))
    if not nc_files:
        return True
    last_nc_time = max([p.stat().st_mtime for p in nc_files])
    if last_nc_time > first_png_time:
        return True
    return False


def plot_analyses(
    df: pd.DataFrame,
    ref: xr.Dataset,
    com: dict[str, xr.Dataset],
    analyses: dict[str, ILAMBAnalysis],
    plot_path: Path,
) -> pd.DataFrame:
    """
    Plot analysis output encoded in each analysis.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of all scalars from the analyses.
    ref : xr.Dataset
        A dataset containing reference data for plotting.
    com : dict[str,xr.Dataset]
        A dictionary of the comparison datasets whose keys are the model names.
    analyses : dict[str, ILAMBAnalysis]
        A dictionary of analyses to
    plot_path : Path
        A path to prepend all filenames.

    Returns
    -------
    pd.DataFrame
        A dataframe containing plot information and matplotlib axes.
    """
    if ilamb3.conf["debug_mode"] and not regenerate_figs(plot_path):
        return pd.DataFrame([])
    plot_path.mkdir(exist_ok=True, parents=True)
    df_plots = []
    for name, a in analyses.items():
        dfp = a.plots(
            df,
            ref,
            com,
        )
        for _, row in dfp.iterrows():
            if not row["axis"]:
                continue
            row["axis"].get_figure().savefig(
                plot_path / f"{row['source']}_{row['region']}_{row['name']}.png"
            )
        plt.close("all")
        if "analysis" not in dfp.columns:
            dfp["analysis"] = name
        df_plots.append(dfp)
    df_plots = pd.concat(df_plots)
    return df_plots


def generate_html_page(
    df: pd.DataFrame,
    ref: xr.Dataset,
    com: dict[str, xr.Dataset],
    df_plots: pd.DataFrame,
) -> str:
    """
    Generate an html page encoding all analysis data.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of all scalars from the analyses.
    ref : xr.Dataset
        A dataset containing reference data for plotting.
    com : dict[str,xr.Dataset]
        A dictionary of the comparison datasets whose keys are the model names.
    df_plots : pd.DataFrame
        A dataframe containing plot information and matplotlib axes.

    Returns
    -------
    str
        The html page.
    """
    ilamb_regions = ilr.Regions()
    # Setup template analyses and plots
    analyses = {analysis: {} for analysis in df["analysis"].dropna().unique()}
    for (aname, pname), df_grp in df_plots.groupby(["analysis", "name"], sort=False):
        analyses[aname][pname] = []
        if len(df_grp["source"].unique()) == 1 and None in df_grp["source"].unique():
            analyses[aname][pname] += [{"None": f"None_RNAME_{pname}.png"}]
            continue
        if "Reference" in df_grp["source"].unique():
            analyses[aname][pname] += [{"Reference": f"Reference_RNAME_{pname}.png"}]
        analyses[aname][pname] += [{"Model": f"MNAME_RNAME_{pname}.png"}]
    ref_plots = list(df_plots[df_plots["source"] == "Reference"]["name"].unique())
    mod_plots = list(
        df_plots[~df_plots["source"].isin(["Reference", None])]["name"].unique()
    )
    all_plots = sorted(list(set(ref_plots) | set(mod_plots)))
    if not all_plots:
        all_plots = [""]

    # Setup template dictionary
    df = df.reset_index(drop=True)
    df["id"] = df.index
    data = {
        "page_header": ref.attrs["header"] if "header" in ref.attrs else "",
        "analysis_list": list(analyses.keys()),
        "model_names": [m for m in df["source"].unique() if m != "Reference"],
        "ref_plots": ref_plots,
        "mod_plots": mod_plots,
        "all_plots": all_plots,
        "regions": {
            (None if key == "None" else key): (
                "All Data" if key == "None" else ilamb_regions.get_name(key)
            )
            for key in df["region"].unique()
        },
        "analyses": analyses,
        "data_information": {
            key.capitalize(): ref.attrs[key]
            for key in ["title", "institution", "version", "doi"]
            if key in ref.attrs
        },
        "table_data": str(
            [row.to_dict() for _, row in df.drop(columns="units").iterrows()]
        ).replace("nan", "NaN"),
    }

    # Generate the html from the template
    template = importlib.resources.open_text(
        "ilamb3.templates", "dataset_page.html"
    ).read()
    html = Template(template).render(data)
    return html


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and "sources" not in v:
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _clean_pathname(filename: str) -> str:
    """Removes characters we do not want in our paths."""
    invalid_chars = r'[\\/:*?"<>|\s]'
    cleaned_filename = re.sub(invalid_chars, "", filename)
    return cleaned_filename


def _is_leaf(current: dict) -> bool:
    """Is the current item in the nested dictionary a leaf?"""
    if not isinstance(current, dict):
        return False
    if "sources" in current:
        return True
    return False


def _add_path(current: dict, path: Path | None = None) -> dict:
    """Recursively add the nested dictionary headings as a `path` in the leaves."""
    path = Path() if path is None else path
    for key, val in current.items():
        if not isinstance(val, dict):
            continue
        key_path = path / Path(_clean_pathname(key))
        if _is_leaf(val):
            val["path"] = str(key_path)
        else:
            current[key] = _add_path(val, key_path)
    return current


def _to_leaf_list(current: dict, leaf_list: list | None = None) -> list:
    """Recursively flatten the nested dictionary only returning the leaves."""
    leaf_list = [] if leaf_list is None else leaf_list
    for _, val in current.items():
        if not isinstance(val, dict):
            continue
        if _is_leaf(val):
            leaf_list.append(val)
        else:
            _to_leaf_list(val, leaf_list)
    return leaf_list


def _create_paths(current: dict, root: Path):
    """Recursively ensure paths in the leaves are created."""
    for _, val in current.items():
        if not isinstance(val, dict):
            continue
        if _is_leaf(val):
            if "path" in val:
                (root / Path(val["path"])).mkdir(parents=True, exist_ok=True)
        else:
            _create_paths(val, root)


def parse_benchmark_setup(yaml_file: str | Path) -> dict:
    """Parse the file which is analagous to the old configure file."""
    yaml_file = Path(yaml_file)
    with open(yaml_file) as fin:
        analyses = yaml.safe_load(fin)
    assert isinstance(analyses, dict)
    return analyses


def set_model_colors(df_datasets: pd.DataFrame):
    """
    Set model colors, some hard coded.
    """
    ilamb3.conf.set(
        label_colors={
            "Reference": [0.0, 0.0, 0.0, 1.0],
            "CMIP5": [0.19215, 0.35294, 0.81176, 1.0],
            "CMIP6": [0.81568, 0.21176, 0.21176, 1.0],
        }
    )
    model_names = sorted(
        df_datasets[ilamb3.conf["model_name_facets"]]
        .apply(lambda row: "-".join(row), axis=1)
        .unique(),
        key=lambda m: m.lower(),
    )
    ilamb3.conf.set(label_colors=ilp.set_label_colors(model_names))


def run_study(
    study_setup: str,
    df_datasets: pd.DataFrame,
    ref_datasets: pd.DataFrame | None = None,
    output_path: str | Path = "_build",
):
    ilamb3.conf["run_mode"] = "batch"
    output_path = Path(output_path)
    # Some yaml text that would get parsed like a dictionary.
    analyses = parse_benchmark_setup(study_setup)
    registry = analyses.pop("registry") if "registry" in analyses else "ilamb.txt"
    if registry == "ilamb.txt":
        reg = ilamb3.ilamb_catalog()
    elif registry == "iomb.txt":
        reg = ilamb3.iomb_catalog()
    else:
        raise ValueError("Unsupported registry.")
    set_model_colors(df_datasets)

    # The yaml analysis setup can be as structured as the user needs. We are no longer
    # limited to the `h1` and `h2` headers from ILAMB 2.x. We will detect leaf nodes by
    # the presence of a `sources` dictionary.
    analyses = _add_path(analyses)

    # Various traversal actions
    _create_paths(analyses, output_path)

    # Create a list of just the leaves to use in creation all work combinations
    analyses_list = _to_leaf_list(analyses)

    # Run the confrontations
    for analysis in analyses_list:
        path = analysis.pop("path")
        try:
            run_single_block(
                path.replace("/", " | "),
                (
                    ref_datasets
                    if ref_datasets is not None
                    else registry_to_dataframe(reg)
                ),
                df_datasets,
                output_path / path,
                **analysis,
            )
        except Exception:
            continue
