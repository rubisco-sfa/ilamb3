"""Functions for synthesizing ilamb3 output."""

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

import ilamb3.regions as ilr
from ilamb3.analysis import add_overall_score


def dataframe_to_cmec(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a CMEC boundle from information in the dataframe.
    """
    df = df.drop(columns=["analysis", "type", "units"])
    df["h2"] = df["section"] + "::" + df["variable"]
    df["h3"] = df["h2"] + "!!" + df["dataset"]
    ilamb_regions = ilr.Regions()
    cmec_regions = {
        r: {
            "LongName": "None" if r == "None" else ilamb_regions.get_name(r),
            "Description": (
                "Reference data extents" if r == "None" else ilamb_regions.get_name(r)
            ),
            "Generator": "N/A" if r == "None" else ilamb_regions.get_source(r),
        }
        for r in df["region"].unique()
    }
    cmec_models = {
        m: {"Description": m, "Source": m}
        for m in df["source"].unique()
        if m != "Reference"
    }
    cmec_h1 = {
        name: {
            "name": name,
            "Abstract": "composite score",
            "URI": [
                "https://www.osti.gov/biblio/1330803",
                "https://doi.org/10.1029/2018MS001354",
            ],
            "Contact": "forrest AT climatemodeling.org",
        }
        for name in df["section"].unique()
    }
    cmec_h2 = {
        name: {
            "name": name,
            "Abstract": "composite score",
            "URI": [
                "https://www.osti.gov/biblio/1330803",
                "https://doi.org/10.1029/2018MS001354",
            ],
            "Contact": "forrest AT climatemodeling.org",
        }
        for name in df["h2"].unique()
    }
    cmec_h3 = {
        name: {
            "name": name,
            "Abstract": "benchmark score",
            "URI": [
                "https://www.osti.gov/biblio/1330803",
                "https://doi.org/10.1029/2018MS001354",
            ],
            "Contact": "forrest AT climatemodeling.org",
        }
        for name in df["h3"].unique()
    }
    cmec_statistics = {
        "indices": [s.replace(" [1]", "") for s in df["name"].unique()],
        "short_names": [s.replace(" [1]", "") for s in df["name"].unique()],
    }
    cmec_results = {}
    for region in df["region"].unique():
        cmec_results[region] = {}
        for model in df["source"].unique():
            cmec_results[region][model] = {}

            for sec in ["section", "h2", "h3"]:
                for name in df[sec].unique():
                    q = df[
                        (df["region"] == region)
                        & (df["source"] == model)
                        & (df[sec] == name)
                    ]
                    if not len(q):
                        continue
                    q.groupby("name").mean(numeric_only=True)
                    q = (
                        q[["name", "dataset", "value"]]
                        .pivot(columns="name", index="dataset")
                        .mean()
                    )
                    q.index = [i.replace(" [1]", "") for i in q.index.levels[1]]
                    cmec_results[region][model][name] = q.to_dict()

    bundle = {
        "SCHEMA": {"name": "CMEC", "version": "v1", "package": "ILAMB"},
        "DIMENSIONS": {
            "json_structure": ["region", "model", "metric", "statistic"],
            "dimensions": {
                "region": cmec_regions,
                "model": cmec_models,
                "metric": cmec_h1 | cmec_h2 | cmec_h3,
                "statistic": cmec_statistics,
            },
        },
        "RESULTS": cmec_results,
    }
    return bundle


def generate_dashboard_page(
    output_path: Path, page_title: str = "ILAMB Results"
) -> None:
    """
    Create the required CMEC assets for a Unified Dashboard page.
    """
    df = build_global_dataframe(output_path)
    bundle = dataframe_to_cmec(df)
    with open(output_path / "scalar_database.json", "w") as out:
        out.write(json.dumps(bundle))
    with open(output_path / "_lmtUDConfig.json", "w") as out:
        out.write(
            json.dumps(
                {
                    "udcJsonLoc": "scalar_database.json",
                    "udcDimSets": {
                        "x_dim": "model",
                        "y_dim": "metric",
                        "fxdim": {
                            "region": df["region"].value_counts().idxmax(),
                            "statistic": "Overall Score",
                        },
                    },
                    "udcScreenHeight": 0,
                    "udcCellValue": 1,
                    "udcNormType": "standarized",
                    "udcNormAxis": "row",
                    "logofile": "None",
                }
            )
        )
    template = importlib.resources.open_text(
        "ilamb3.templates", "unified_dashboard.html"
    ).read()
    html = Template(template).render({"page_title": page_title})
    with open(output_path / "index.html", "w") as out:
        out.write(html)


def _load_local_csvs(csv_files: list[Path]) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(str(csv_file)) for csv_file in csv_files]
    ).drop_duplicates(subset=["source", "region", "analysis", "name"])
    df["region"] = df["region"].astype(str).str.replace("nan", "None")
    df = add_overall_score(df)
    return df


def build_global_dataframe(root: Path):
    dfs = []
    for parent, _, files in root.walk():
        # the dashboard needs html files to be {DATASET}.html
        for html_file in [parent / f for f in files if f.endswith(".html")]:
            html_file.rename(html_file.parent / f"{html_file.parent.name}.html")

        csv_files = [parent / f for f in files if f.endswith(".csv")]
        if not csv_files:
            continue
        df = _load_local_csvs(csv_files)
        df = df[df.type == "score"]

        # if at this point the depth is not 3, this function will not work
        heading = parent.relative_to(root)
        if len(heading.parents) != 3:
            raise ValueError("This function requires analysis runs of depth level = 3")
        heading = str(heading).split("/")

        # add metadata to this part of the dataframe
        df["section"] = heading[0]
        df["variable"] = heading[1]
        df["dataset"] = heading[2]
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs
