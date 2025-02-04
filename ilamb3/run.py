"""Functions for rendering ilamb3 output."""

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from jinja2 import Template

import ilamb3
import ilamb3.regions as ilr
from ilamb3.analysis.base import ILAMBAnalysis


def run_analyses(
    ref: xr.Dataset, com: xr.Dataset, analyses: dict[str, ILAMBAnalysis]
) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
    dfs = []
    ds_refs = []
    ds_coms = []
    for _, a in analyses.items():
        df, ds_ref, ds_com = a(ref, com, regions=ilamb3.conf["regions"])
        dfs.append(df)
        ds_refs.append(ds_ref)
        ds_coms.append(ds_com)
    dfs = pd.concat(dfs)
    dfs["name"] = dfs["name"] + " [" + df["units"] + "]"
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
    """."""
    ilamb_regions = ilr.Regions()

    # Setup template analyses and plots
    analyses = {analysis: {} for analysis in df["analysis"].unique()}
    for (aname, pname), df_grp in df_plots.groupby(["analysis", "name"], sort=False):
        analyses[aname][pname] = []
        if "Reference" in df_grp["source"].unique():
            analyses[aname][pname] += [{"Reference": f"Reference_RNAME_{pname}.png"}]
        analyses[aname][pname] += [{"Model": f"MNAME_RNAME_{pname}.png"}]
    ref_plots = list(df_plots[df_plots["source"] == "Reference"]["name"].unique())
    mod_plots = list(df_plots[df_plots["source"] != "Reference"]["name"].unique())
    all_plots = list(set(ref_plots) | set(mod_plots))

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
