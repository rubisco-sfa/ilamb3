---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# ILAMB Basics

## Introduction

The Internaional Land Model Benchmarking (ILAMB) project is a on-going effort to be more holistic and systematic about how we confront Earth system models with reference data products. We compare model to reference data using a consistent methodology that includes aspects of bias, RMSE, annual cycle, spatial distribution and optionally relationships (variable to variable comparisons). There are a great many choices that we made in this process, all of which will affect how you interpret the results. If you are unfamiliar with our methodology, we recommend that you start with our paper [*Collier, et al. 2018*](https://doi.org/10.1029/2018MS001354) which details these decisions and provides our philosophy of how ILAMB results should be used.

While our specific aims have historically been tied to biogeochemical cycles on land, our methods and software is meant to be generic and can be applied to any domain. This tutorial will teach you the basic steps required to apply the ILAMB methodology to any reference/model data pairs of your choosing. You will learn how to:

1. Set up which reference data will be used in a comparison,
2. Set up model data to which the comparison will apply,
3. Run the ILAMB analysis and view the results.

## Reference Data

You will setup your benchmarking study by specifying which reference datasets define the comparison of interest. These go into a [yaml](https://yaml.org) file which needs to look something like the following. Using a text editor, create this file and copy/paste in the content.

```{literalinclude} basic_step1.yaml
:language: yaml
```

This yaml file contains a nested set of dictionaries which are used to organize your study. In this case, we are using the top level, called `Ecosystem and Carbon Cycle`, to reflect a grouping of variables. Beneath this we have another layer used to denote the variable to be compared, `Gross Primary Productivity` followed by another layer denoting the dataset we are using `WECANN-1-0`. Note that while we will use this nesting in this tutorial, the organization of this yaml file can now be completely [arbitrary](TODO).

We parse the `yaml` file and look for the `sources` keyword, defining a reference as the containing dictionary. In this first example, there is a single comparison which will be known as `WECANN-1-0`, contains pointers to the WECANN data source, and also provides some options specific to this block `variable_cmap: Greens`. While ILAMB uses consistent colormaps for quantities like bias, RMSE, etc., map plots of the main variable itself can be configured using this keyword. To see what other configuration keywords are possible, check out this [tutorial](configure_yaml). To get a sense of what all can be defined in these configure files or if you wish to run the studies that we support, check out the
[land](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/configure/ilamb.yaml) and/or [ocean](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/configure/iomb.yaml) configurations.

The `sources` keyword expects to find at least one `VARIABLE: DATASET` pair, where the `VARIABLE` is expected to be found inside the `DATASET`. Datasets may be defined in a number of ways:

1. Keys in our registry of reference data products. If you are wanting to use data that we have formatted and provide, this is the best option. To see what is available, check out our datasets [page](TODO).
2. An absolute path. If we fail to find the text you provide as the `DATASET` in our registries, we will treat it like an absolute path to a file.
3. A path relative to the `ILAMB_ROOT` environment variable. This allows you to move your data around on your system and just change an environment variable to inform the system of the change.

Another advantage to using our registry keys when possible, is that the reference data may be downloaded automatically with a tool that is installed along with the `ilamb3` package. From a terminal window, execute the following command:

```{code} bash
ilamb fetch basic_step1.yaml
```

This will download any keys it finds that are not already downloaded. It is a great way for you to develop a set of comparisons that are portable to other systems or shareable with your colleagues.

## Model Data

While we have written `ilamb3` expecting that the reference and model datafiles follow [CF-Conventions](https://cf-convention.github.io/), we are not pedantic about these conventions being followed. Your model data will need to be able to be read by `xarray` and have a minimal subset of dimensions encoded which represent time and space (latitude and longitude). We understand that many times benchmarking is most useful early in model development, but then requires some flexibility as post-processing model output is time consuming and often an unfamiliar process to developers.

You will need to tell `ilamb3` where your model data is located. For the moment, we expect that this data be provided in tabular form as a CSV file that looks something like the following:

```{code-cell} python
:tags: [remove-input]
import pandas as pd
df = pd.read_csv("CanESM5.csv").drop(columns=["Unnamed: 0","mip_era","activity_id","institution_id","table_id"])
df
```

```{aside} Obtaining CMIP Data
If you intend to use CMIP data in your comparison and also need to download it, these CSV files can be generated using [intake-esgf](https://doi.org/10.5281/zenodo.18378994). See this [tutorial](intake) for details.
```

You have some flexibility in how this data appears, but at minimum it should:

1. Contain one row per unique variable/file. If you have a variable split into multiple files, as some CMIP models do, then you would still have a row per unique file. If your model data is closer to raw output where single files contain many variables, you would create a row for each unique variable and file even if you reference the same file multiple times.
2. Contain a `path` column which provides the absolute path to the file location on your system.
3. Contain the columns `source_id`, `member_id`, and `grid_label`, unless you have [configured](TODO) `ilamb3` otherwise. If you are using your own model data that has not been standardized, you may have to create these columns manually and fill with data you invent.

While generating these CSV files is the user's burden, we do have some functionality to assist if your model data follows expected patterns. For example:

```{code} bash
ilamb model-csv PATH_TO_MODEL_DATA FILENAME.csv --file-regex
```

## Running the study

We now have all the ingredients we need to run the test study.

```{code} bash
ilamb run basic_step1.yaml --model-csv CanESM5.csv
```

Internally, `ilamb3` will use the benchmark definitions found in `basic_step1.yaml` to query the model data given in `CanESM5.csv` for relevant variables. For each unique combination of `source_id`, `member_id`, and `grid_label`, we will run the block generating output that is saved in the `_build` directory as shown below.

```{code} bash
_build/
├── EcosystemandCarbonCycle
│   └── GrossPrimaryProductivity
│       └── WECANN-1-0
│           ├── CanESM5.csv
│           ├── CanESM5.nc
│           ├── CanESM5_None_bias.png
│           ├── CanESM5_None_biasscore.png
│           ├── CanESM5_None_cycle.png
│           ├── CanESM5_None_cyclescore.png
│           ├── CanESM5_None_mean.png
│           ├── CanESM5_None_rmse.png
│           ├── CanESM5_None_rmsescore.png
│           ├── CanESM5_None_shift.png
│           ├── CanESM5_None_tmax.png
│           ├── CanESM5_None_trace.png
│           ├── None_None_taylor.png
│           ├── post.log
│           ├── Reference.nc
│           ├── Reference_None_mean.png
│           ├── Reference_None_tmax.png
│           └── WECANN-1-0.html
├── index.html
├── _lmtUDConfig.json
├── run.yaml
└── scalar_database.json
```

Let's unpack what `ilamb3` has done.

1. Note that the output has been stored in subdirectories that mirror the organization found in `basic_step1.yaml`.
2. Inside the `EcosystemandCarbonCycle/GrossPrimaryProductivity/WECANN-1-0` directory, we find the results of the single benchmark block that was defined.
3. As `ilamb3` works, it writes out intermediate files for the analysis artifacts. `CanESM5.csv` stores all scalars that were generated and `CanESM5.nc` contains the maps and time series that are plot. If any errors were encountered during the run, they would be logged in `CanESM5.log`.
4. Plots are generated following a naming convention of `{MODEL}_{REGION}_{PLOT}.png`, where `MODEL` could be `Reference` for the reference data. If a plot is composite across all models, then `MODEL` may appear as `None`.
5. If any errors were encountered during the post-processing phase of the `ilamb3` run, you will find details in `post.log`.
6. While you are free to navigate the output manually, opening the HTML page `WECANN-1-0.html` in a browser will load a data dashboard that has been designed to assist the scientist in discovery. Your page should look something like this [comparison](https://www.ilamb.org/dev/Land/EcosystemandCarbonCycle/GrossPrimaryProductivity/WECANN-1-0/WECANN-1-0.html) of several CMIP6 models.

In addition, note that there are several files located at the root of the `_build` directory. The file `run.yaml` is a copy of the benchmark configuration file. This is for reproducibility so that you can always tell what produced any given `ilamb3` build. The remaining files were produced as a meta analysis of the full `ilamb3` run.

1. We harvested all the scalars from all the blocks and models and put them into a single file `scalar_database.json`. This file is compatible with the [Unified Dashboard](https://github.com/climatemodeling/unified-dashboard) and is used to create an interactive portrait plot.
2. A version of the Unified Dashboard is saved with the run in `index.html` along with dashboard options `_lmtUDConfig.json`. Since the dashboard will load a datafile, it will not work if you simply open `index.html` in a browser as it violates security policy. You need to either move it to a web-visible location or emulate a HTTP server locally with `python -m http.server` and then open the link that python provides.

```{warning} Unified Dashboard Meta Analysis Limitation
At the time of this writing requires, the unified dashboard meta analysis requires that the benchmarks be found in a 2-nesting (`EcosystemandCarbonCycle` and `GrossPrimaryProductivity` in this case). While you are free to organize your benchmarks as suits your needs, the meta analyses may have requirements that limit their applicability to your study.
```

## Next steps

In order to solidify these basic `ilamb3` concepts, we recommend that you attempt the following expansions on your own.

1. **Add another model:** This consists of creating another (or expanding the current) CSV file. To keep downloads small, you might choose to add `UKESM1-0-LL` as it is also relatively coarse. If you create a second CSV file, when running `ilamb3` you can just add it to the `--model-csv` option separated by a comma. We will concatenate these files together internally.
```{code} bash
ilamb run basic_step1.yaml --model-csv CanESM5.csv,UKESM1-0-LL.csv
```
2. **Add another benchmark:** Locate the `ilamb3` registry key for the `Fluxnet-2015` `gpp` data from the [datasets](datasets) page. Add another benchmark block to `basic_step1.yaml` which is also under the `Gross Primary Productivity` heading. Essentially you need to duplicate the `WECANN-1-0` block and then replace the WECANN specific information with Fluxnet2015.
