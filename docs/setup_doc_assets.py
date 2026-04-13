"""
A list of functions that should be run before myst is started so that all the
content for `literalinclude` is pre-generated. For consistency, keep content
generation for a doc file in a function bearing the file's name.
"""

import importlib
import inspect

import ilamb3
import pandas as pd
import yaml
from intake_esgf import ESGFCatalog


def basic_genyaml():
    cat = ilamb3.ilamb3_catalog()
    wecann_gpp_key = [key for key in cat.registry if "WECANN" in key and "gpp" in key]
    if len(wecann_gpp_key) != 1:
        raise ValueError(f"Ambiguity in finding WECANN gpp {wecann_gpp_key=}")
    wecann_gpp_key = wecann_gpp_key[0]
    out = {
        "Ecosystem and Carbon Cycle": {
            "Gross Primary Productivity": {
                "WECANN-1-0": {
                    "sources": {"gpp": wecann_gpp_key},
                    "variable_cmap": "Greens",
                }
            }
        }
    }
    with open("my_benchmark_study.yaml", "w") as f:
        f.write(yaml.dump(out))
    cat.fetch(wecann_gpp_key)


def basic_canesm5():
    KEY_PATTERN = [
        "mip_era",
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
    ]
    cat = (
        ESGFCatalog()
        .search(
            experiment_id="historical",
            source_id="CanESM5",
            variable_id=["gpp", "areacella", "sftlf"],
            table_id=["Lmon", "fx"],
            file_start="1980-01",
            file_end="2016-01",
        )
        .remove_ensembles()
    )
    dpd = cat.to_path_dict(minimal_keys=False)
    df = []
    for key, paths in dpd.items():
        row = {col: value for col, value in zip(KEY_PATTERN, key.split("."))}
        for path in paths:
            row["path"] = str(path)
            df.append(row)
    df = pd.DataFrame(df)
    df.to_csv("CanESM5.csv")


def datasets_ilamb3():
    cat = ilamb3.ilamb3_catalog()
    df = (
        pd.DataFrame(
            [
                {
                    "source_id": key.split("/")[0],
                    "variable_id": key.split("/")[1].split("_")[4],
                    "key": key,
                    # "download": f"<a href='{cat.get_url(key)}'>'⤓'</a>",
                }
                for key in cat.registry
            ]
        )
        .sort_values(["source_id", "variable_id"])
        .set_index(["source_id", "variable_id"])
    )
    styles = [
        {"selector": "th.col0, td.col0", "props": [("width", "5%")]},
        {"selector": "th.col1, td.col1", "props": [("width", "5%")]},
        {"selector": "th.col2, td.col2", "props": [("width", "90%")]},
    ]
    with open("catalog_ilamb3.html", "w") as fout:
        fout.write(df.style.set_table_styles(styles).to_html())


if __name__ == "__main__":
    # Run all the functions defined in this module
    mod = importlib.import_module("setup_doc_assets")
    for _, fnc in inspect.getmembers(mod, inspect.isfunction):
        fnc()
