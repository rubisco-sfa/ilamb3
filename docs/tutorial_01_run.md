---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{warning}
We are still developing the user experience and so while the information here is not expected to change drastically, you may find some differences as we continue to hone our approach.
```

# Run an Analysis

This tutorial addresses how to run a benchmarking analysis in `ilamb3`. While some of the particulars have changed in `ilamb3`, the basic flow is the same. First, we will prepare a configure file which details which reference data will be used along with options that control the structure and flow of the analyses. Second, we specify model data by creating a CSV file which `ilamb3` which points to local data. Finally, we will use a run script to execute the study.

In order to run this tutorial, you will need to have `ilamb3` installed. While there are many options, the simplest is to use `pip`:

```bash
pip install ilamb3
```

## Configuration

```{code-cell}
---
tags: [remove-cell]
---
import yaml
out = {
    "Ecosystem and Carbon Cycle": {
        "Gross Primary Productivity": {
            "WECANN-1-0": {
                "sources": {
                    "gpp": "WECANN-1-0/obs4MIPs_ColumbiaU_WECANN-1-0_mon_gpp_gn_v20250902.nc"
                }
            }
        }
    }
}
with open("sample.yaml", "w") as f:
    f.write(yaml.dump(out))
```

The configuration of an ILAMB study uses a [YAML](https://yaml.org) file to describe and organize the reference data and analysis options. Consider the contents below of a file called `sample.yaml`.

```{literalinclude} sample.yaml
:language: yaml
```

Copy the contents above and paste it into a file with that name. The `ilamb3` system will scan this nested dictionary file looking for the `sources` keyword and initialize a comparison for each instance, the title of which comes from the containing dictionary. In this example, `ilamb3` will find a single comparison which it will call `WECANN-1-0`. The `sources` dictionary then details which variables are expected to be found in which sources. These sources can be:

1. Keys in a `ilamb3` registry. See [this](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/registry/ilamb3.txt) registry for a sample. The text `WECANN-1-0/obs4MIPs_ColumbiaU_WECANN-1-0_mon_gpp_gn_v20250902.nc` is such a key and can be used to automatically download the reference data. We will do this in the next section.
2. An absolute path. `ilamb3` will at first assume that you are specifying a key from our registries. However if this fails, then we will treat your key as an absolute path and see if the file can be found.
3. If no file is found and you have set the `ILAMB_ROOT` environment variable, we will also treat your key as a relative path with respect to `ILAMB_ROOT` and look for the file's existence.

The remaining dictionary keys are used to organize the study and are completely arbitrary in `ilamb3`. You may nest and deep or shallow as you wish, but in this case we use a nesting that is familiar to previous ILAMB users.

## Reference data

Since we specified the reference data using keys in the registry, we can fetch the data using a built-in command. From your command-line type:

```bash
ilamb fetch sample.yaml
```
```{code-cell}
---
tags: [remove-cell]
---
from ilamb3.cli import fetch
fetch("sample.yaml")
```

The `ilamb` command is provided when you install the python package. Inside it are several subcommands and one is `fetch`. By passing the configuration file, `ilamb3` will extract the registry keys and download/verify the reference data that the study uses. This data is put into a cache which by default will be located in `${HOME}/.cache/ilamb3/` but can be changed by setting the `ILAMB_ROOT` environment variable.

## Model Data

In ilamb3, we use pandas DataFrames internally to represent model data. If you have model files stored as tabular data in a CSV file, this can be ingested by `ilamb3` and used to find model data. In this tutorial, we will use [intake-esgf](https://intake-esgf.readthedocs.io/en/latest/) to query ESGF servers and download CMIP model data.

```{code-cell}
from intake_esgf import ESGFCatalog

cat = (
    ESGFCatalog()
    .search(
        experiment_id="historical",
        source_id="CanESM5",
        variable_id="gpp",
        frequency="mon",
        file_start="1980-01",
        file_end="2016-01",
    )
    .remove_ensembles()
)
dpd = cat.to_path_dict(minimal_keys=False)
```

This will download the model data and load it into a dictionary of file paths. We then write a simple function which splits the dictionary keys into column headings for a pandas DataFrame.

```{code-cell}
from pathlib import Path
import pandas as pd

def path_dict_to_pandas(path_dict: dict[str, Path]) -> pd.DataFrame:
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
    df = []
    for key, paths in path_dict.items():
        row = {col: value for col, value in zip(KEY_PATTERN, key.split("."))}
        for path in paths:
            row["path"] = str(path)
            df.append(row)
    df = pd.DataFrame(df)
    return df
```

Finally we call this function on our path dictionary and save to a CSV file for use in the `ilamb3` system.

```{code-cell}
df = path_dict_to_pandas(dpd)
df.to_csv("CanESM5.csv")
```

## Running the Study

In addition to the `fetch` subcommand, we have implemented a `run` subcommand, analagous to `ilamb-run` from the previous ILAMB version. As before, it will run benchmarking study and have a lot of options you can use to control the run.

```bash
> ilamb run --help

 Usage: ilamb run [OPTIONS] CONFIG

 Run a benchmarking analysis

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    config      PATH  [required]                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --regions                            TEXT                                                 │
│ --region-sources                     TEXT                                                 │
│ --df-comparison                      PATH                                                 │
│ --output-path                        PATH   [default: _build]                             │
│ --cache                --no-cache           [default: cache]                              │
│ --central-longitude                  FLOAT  [default: 0.0]                                │
│ --title                              TEXT   [default: Benchmarking Results]               │
│ --global-region                      TEXT                                                 │
│ --help                                      Show this message and exit.                   │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

To run the study we have setup, pass in the configure file and specify the comparison (model) data with `--df-comparison`. This can be a comma delimited list of files and internally we will concatenate them together. Finally, we send the output to a `_tutorial` direcory.

```bash
ilamb run sample.yaml --df-comparison CanESM5.csv --output-path _tutorial
```

Unless we have a problem, this incarnation of `ilamb run` is relatively quiet with respect to previous versions. It may seem that nothing happened, but you should find that a `_tutorial` directory was generated and populated with files and directories. In order to view the output, change directories to the `_tutorial` and run a simple http server.

```bash
cd _tutorial
python -m http.server
```

This command will give a local link of the form `http://0.0.0.0:8000/`. Open this link and the ILAMB output page will display.

## Adding a reference dataset

Expanding the reference data is as simple as adding another section to the configure file. Looking through this [registry](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/registry/ilamb.txt) we find the key `gpp/FLUXNET2015/gpp.nc` which we can add to our file.

```{code-cell}
---
tags: [remove-cell]
---
import yaml
out = {
    "Ecosystem and Carbon Cycle": {
        "Gross Primary Productivity": {
            "WECANN-1-0": {
                "sources": {
                    "gpp": "WECANN-1-0/obs4MIPs_ColumbiaU_WECANN-1-0_mon_gpp_gn_v20250902.nc"
                }
            },
            "FLUXNET2015": {
                "sources": {
                    "gpp": "gpp/FLUXNET2015/gpp.nc"
                }
            }
        }
    }
}
with open("expand.yaml", "w") as f:
    f.write(yaml.dump(out))
```

```{literalinclude} expand.yaml
:language: yaml
```

Then if we re-run:

```bash
ilamb run sample.yaml --df-comparison CanESM5.csv --output-path _tutorial
```

you will see a message:

```bash
2025-12-11 11:59:06.928 | INFO     | ilamb3.run:run_single_block:512 - Using cached information _tutorial/EcosystemandCarbonCycleGrossPrimaryProductivity/WECANN-1-0/CanESM5.nc
```

This is to alert you that we already had computed the `WECANN-1-0` comparison and are instead reading in the cached information. This is the default behavior of `ilamb run`. If you wish to force a recompute, run with `--no-cache`.
