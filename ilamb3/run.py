"""Functions for rendering ilamb3 output."""

import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pooch
import xarray as xr
from jinja2 import Template

import ilamb3
import ilamb3.analysis as anl
import ilamb3.regions as ilr
from ilamb3.analysis.base import ILAMBAnalysis


def setup_analyses(
    registry: pooch.Pooch, **analysis_setup: Any
) -> tuple[str, dict[str, ILAMBAnalysis]]:
    """.

    sources
    relationships

    variable_cmap
    skip_XXX

    """
    # Check on sources
    sources = analysis_setup.get("sources", {})
    relationships = analysis_setup.get("relationships", {})
    if len(sources) != 1:
        raise ValueError(
            f"The default ILAMB analysis requires a single variable and source, but I found: {sources}"
        )
    variable = list(sources.keys())[0]

    # If specialized analyses are given, setup those and return
    if "analyses" in analysis_setup:
        analyses = {
            a: anl.ALL_ANALYSES[a](variable, **analysis_setup)
            for a in analysis_setup.pop("analyses", [])
            if a in anl.ALL_ANALYSES
        }
        return variable, analyses

    # Augment options with things in the global options
    if "regions" not in analysis_setup:
        analysis_setup["regions"] = ilamb3.conf["regions"]
    if "method" not in analysis_setup:
        if ilamb3.conf["prefer_regional_quantiles"]:
            analysis_setup["method"] = "RegionalQuantiles"
            analysis_setup["quantile_database"] = pd.read_parquet(
                registry.fetch(ilamb3.conf["quantile_database"])
            )
            analysis_setup["quantile_threshold"] = ilamb3.conf["quantile_threshold"]
            ilr.Regions().add_netcdf(
                xr.load_dataset(registry.fetch("regions/Whittaker.nc"))
            )
        else:
            analysis_setup["method"] = "Collier2018"
    if "use_uncertainty" not in analysis_setup:
        analysis_setup["use_uncertainty"] = ilamb3.conf["use_uncertainty"]

    # Setup the default analysis
    analyses = {
        name: a(variable, **analysis_setup)
        for name, a in anl.DEFAULT_ANALYSES.items()
        if analysis_setup.get(f"skip_{name.lower()}", False) is False
    }
    analyses.update(
        {
            f"Relationship {ind_variable}": anl.relationship_analysis(
                variable, ind_variable, **analysis_setup
            )
            for ind_variable in relationships
        }
    )
    return variable, analyses


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
        A dictionary of analyses to run.

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
        df, ds_ref, ds_com = a(ref, com)
        dfs.append(df)
        ds_refs.append(ds_ref)
        ds_coms.append(ds_com)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs["name"] = dfs["name"] + " [" + dfs["units"] + "]"
    ds_ref = xr.merge(ds_refs)
    ds_com = xr.merge(ds_coms)
    return dfs, ds_ref, ds_com


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
        A dictionary of analyses to run.
    plot_path : Path
        A path to prepend all filenames.

    Returns
    -------
    pd.DataFrame
        A dataframe containing plot information and matplotlib axes.
    """
    plot_path.mkdir(exist_ok=True, parents=True)
    df_plots = []
    for name, a in analyses.items():
        dfp = a.plots(df, ref, com)
        dfp["analysis"] = name
        df_plots.append(dfp)
    df_plots = pd.concat(df_plots)
    for _, row in df_plots.iterrows():
        row["axis"].get_figure().savefig(
            plot_path / f"{row['source']}_{row['region']}_{row['name']}.png"
        )
    plt.close("all")
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
        if "Reference" in df_grp["source"].unique():
            analyses[aname][pname] += [{"Reference": f"Reference_RNAME_{pname}.png"}]
        analyses[aname][pname] += [{"Model": f"MNAME_RNAME_{pname}.png"}]
    ref_plots = list(df_plots[df_plots["source"] == "Reference"]["name"].unique())
    mod_plots = list(df_plots[df_plots["source"] != "Reference"]["name"].unique())
    all_plots = sorted(list(set(ref_plots) | set(mod_plots)))

    # Setup template dictionary
    df = df.reset_index(drop=True)  # ?
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
            for key in ["title", "institutions", "version"]
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
