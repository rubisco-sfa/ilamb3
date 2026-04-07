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

We parse the `yaml` file and look for the `sources` keyword, defining a reference as the containing dictionary. In this first example, there is a single comparison which will be known as `WECANN-1-0`, contains pointers to the WECANN data source, and also provides some options specific to this block `variable_cmap: Greens`. While ILAMB uses consistent colormaps for quantities like bias, RMSE, etc., map plots of the main variable itself can be configured using this keyword. To see what other configuration keywords are possible, check out this [tutorial](TODO). To get a sense of what all can be defined in these configure files or if you wish to run the studies that we support, check out the
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

You need to tell `ilamb3` where your model data is located. For the moment, we expect that this data be provided in tabular form as a CSV file that looks something like the following:

```{code-cell} python
:tags: [remove-input]
import pandas as pd
df = pd.read_csv("CanESM5.csv").drop(columns=["Unnamed: 0","mip_era","activity_id","institution_id","table_id"])
df
```

You have some flexibility in how this data appears, but at minimum it should:

1. Contain one row per unique variable/file. If you have a variable split into multiple files, as some CMIP models do, then you would still have a row per unique file. If your model data is closer to raw output where single files contain many variables, you would create a row for each unique variable and file even if you reference the same file multiple times.
2. Contain a `path` column which provides the absolute path to the file location on your system.
3. Contain the columns `source_id`, `member_id`, and `grid_label`, unless you have [configured](TODO) `ilamb3` otherwise. If you are using your own model data that has not been standardized, you may have to create these columns manually and fill with data you invent.

```{code} bash
ilamb model PATH_TO_MODEL_DATA {--by-cmip6-dirs|--by-cmip6-attrs|--by-cmip6-filename}
```

[intake-esgf](https://doi.org/10.5281/zenodo.18378994)

## Running the study
