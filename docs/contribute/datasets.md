# Contribute Reference Data to ILAMB

## Table of contents

- [Data file types](#data-file-types)
- [Format a reference dataset for ILAMB using Python](#format-a-reference-dataset-for-ilamb-using-python)
  - [1. Clone the ilamb3-data repository and create a branch](#1-clone-the-ilamb3-data-repository-and-create-a-branch)
    - [1.a. For rubisco-sfa members: create a branch](#1a-for-rubisco-sfa-members-create-a-branch)
    - [1.b. For non-members: fork the repository and create a branch](#1b-for-non-members-fork-the-repository-and-create-a-branch)
    - [1.c. Create a directory for the dataset](#1c-create-a-directory-for-the-dataset)
  - [2. Write a data conversion Python script](#2-write-a-data-conversion-python-script)
    - [2.a. Import useful libraries](#2a-import-useful-libraries)
    - [2.b. Download the dataset](#2b-download-the-dataset)
    - [2.c. Load the dataset](#2c-load-the-dataset)
    - [2.d. Standardize the dimensions and attributes](#2d-standardize-the-dimensions-and-attributes)
    - [2.e. Define variable-specific attributes](#2e-define-variable-specific-attributes)
    - [2.f. Create one NetCDF per variable/uncertainty pair](#2f-create-one-netcdf-per-variableuncertainty-pair)
    - [2.g. Add variable-specific attributes](#2g-add-variable-specific-attributes)
    - [2.h. Add global attributes](#2h-add-global-attributes)
    - [2.i. Export the NetCDF](#2i-export-the-netcdf)
  - [3. Validate the dataset](#3-validate-the-dataset)
  - [4. Contribute the dataset to the ILAMB registry](#4-contribute-the-dataset-to-the-ilamb-registry)
  - [5. Use an LLM to help write a conversion script](#5-use-an-llm-to-help-write-a-conversion-script)
    - [5.a. Prepare the context](#5a-prepare-the-context)
    - [5.b. General guidelines](#5b-general-guidelines)
    - [5.c. Example prompt](#5c-example-prompt)

ILAMB maintains a publicly accessible registry of reference data that can be downloaded using `ilamb fetch`---see the registry page at {doc}`../reference/datasets.md` to see what is currently available. If there is a dataset that isn't in the registry that is useful to the greater ILAMB community, consider formatting it and contributing it to the registry. This tutorial walks through an example of how to format a dataset for ILAMB, and how to submit it to the registry.

## Data file types

ILAMB loads data using xarray, which can read a [variety of file formats](https://docs.xarray.dev/en/latest/user-guide/io.html#). The most heavily tested and thus recommended data format for ILAMB is a [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) file. NetCDFs are self-describing, portable, scalable, appendable, sharable, and archivable. They are also the most common format for Earth System Model output. As part of the ILAMB version 3 release, we have changed our data standards to better align with [CF Conventions](http://cfconventions.org/) and obs4MIPs Data Specifications (ODS), which is a community standard for formatting observational datasets often used to benchmark Earth System Model outputs.

ILAMB-ready data most often have one data variable (plus any ancillary variables such as uncertainty), contain relevant coordinates (e.g., time, latitude, longitude, and depth), have descriptive global attributes, and are gridded or site-level.

As of summer 2026, 60% of ILAMB data are global in coverage, 25% are site-level, and the remaining data are regional, i.e., pan-Arctic, Tropics, Northern Hemisphere, or mid-Latitudes. 80% of the gridded datasets have 0.5-degree spatial resolution, and the rest have 1-degree resolution. The temporal resolutions of the datasets are annual, monthly, or fixed-period. However, ILAMB has no restrictions on spatiotemporal resolution, variables, or data source. We are always looking to expand the benchmarking data we have available in ILAMB, so if you have a dataset that you think would be useful for benchmarking, please consider formatting it and contributing it to the registry.

## Format a reference dataset for ILAMB using Python

Below is a tutorial for how to format a contributable ILAMB dataset. We will use the [Hoffman et al. (2014)](https://doi.org/10.1002/2013JG002381) Land and Ocean Anthropogenic Carbon Flux Estimates observational dataset as an example. This dataset includes Surface Downward Mass Flux of Carbon as CO2 (fgco2), Carbon Mass Flux out of Atmosphere due to Net Biospheric Production on Land (nbp), and the 90% uncertainty range for fgco2 and nbp.

### 1. Clone the ilamb3-data repository and create a branch

To contribute a dataset to ILAMB, you need a free GitHub account and a local clone of the `ilamb3-data` repository. Follow the instructions below based on whether you are a member of the `rubisco-sfa` GitHub organization. You can clone the repository using either SSH or HTTPS. If you use SSH, you will need to [set up an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and add it to your GitHub account. If you use HTTPS, consult the [authentication methods for command-line Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line) before pushing your changes.

#### 1.a. For rubisco-sfa members: create a branch

If you are a member of `rubisco-sfa`, clone the main repository directly:

```bash
git clone git@github.com:rubisco-sfa/ilamb3-data.git
cd ilamb3-data
```

Ensure that your local `main` branch is up to date, then create a contribution branch. Avoid making changes directly to `main`. Use a descriptive branch name for your dataset contribution, such as `hoffman-example`:

```bash
git switch main
git pull --ff-only
git switch -c hoffman-example
```

#### 1.b. For non-members: fork the repository and create a branch

If you are not a member of `rubisco-sfa`, create a fork under your GitHub account:

1. Navigate to the [ilamb3-data GitHub page](https://github.com/rubisco-sfa/ilamb3-data).
2. Click **Fork** in the top-right corner of the page and keep the default settings.
3. Click **Create fork**.
4. Clone your fork, replacing `YOUR-USERNAME` with your GitHub username:

```bash
git clone git@github.com:YOUR-USERNAME/ilamb3-data.git
cd ilamb3-data
```

Add the main repository as the `upstream` remote and verify that it was added correctly:

```bash
git remote add upstream https://github.com/rubisco-sfa/ilamb3-data.git
git remote -v
```

Ensure that your local `main` branch is up to date with the upstream repository, then create a contribution branch:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git switch -c hoffman-example
```

#### 1.c. Create a directory for the dataset

Create a new directory for your dataset in the `data` folder of the repository:

The directory should be the PRODUCT-VERSION or AUTHORS-VERSION of the dataset with no special characters. For this example, Hoffman et al. (2014) do not give their product a name, so we will name it `Hoffman` with an arbitrary version `1`. However, since `Hoffman-1` already exists in the ILAMB repository, we'll create a new directory for this example dataset called `Hoffman-1-example`:

```bash
mkdir -p data/Hoffman-1-example
touch data/Hoffman-1-example/convert.py
```

Then, you can open the Python script in your favorite text editor and start writing the code to format the dataset for ILAMB. Below is an example of how you might structure the `convert.py` script to read in the Hoffman dataset, process it, and save it in a format compatible with ILAMB.

### 2. Write a data conversion Python script
Once you have cloned the `ilamb3-data` repository and created a contribution branch, you can activate your Python environment and install the required dependencies. We recommend using `uv` to manage your Python environment. See [here](https://docs.astral.sh/uv/getting-started/installation/) for more information about installing `uv` on your system. You can install the required dependencies from the `ilamb3-data` directory and activate the environment using the following commands:

```bash
uv sync
source .venv/bin/activate
```

After activating the environment, you will have access to several useful libraries that are `ilamb3-data` dependencies, including `dask` for parallel computing on large datasets, `s3fs` for accessing data stored in Amazon S3, `cf-units` for handling unit conversions, `cftime` for handling time, `requests` and `bs4` for accessing and parsing HTML, `earthaccess` to access data from NASA Earthdata servers, as well as `xarray`, `rioxarray`, and `geopandas` for working with geospatial data. You can install additional dependencies as needed using `pip install <package-name>`. If there are libraries that you think other users would benefit from, you can run `uv add <package-name>`, which adds the package to the `ilamb3-data` environment and updates the `pyproject.toml` file so that other users can install it when they run `uv sync`.

Next, you can start writing the code to format the dataset for ILAMB. Below is an example of how you might structure the `convert.py` script to read in the Hoffman dataset, process it, and save it in a format compatible with ILAMB.

#### 2.a. Import useful libraries
At the top of your `convert.py` script, import the necessary libraries:

```python
import time
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3_data as ild
```

The ilamb3_data (`ild`) package provides a set of utilities for preparing ILAMB datasets so that they are CF and ODS-compliant. We will walk through some of the modules below, but for the most up-to-date information on the modules and functions available in ilamb3_data, please refer to the [ilamb3-data/ilamb3_data repository](https://github.com/rubisco-sfa/ilamb3-data/tree/main/ilamb3_data).

#### 2.b. Download the dataset
If possible, the `convert.py` script should download the dataset from a public source. This ensures that the dataset can be easily accessed and reproduced by others. If this isn't possible, please leave a note at the top of your script with instructions on how you accessed the data.

```python
# Download the data
url = "https://www.ilamb.org/ILAMB-Data/DATA/nbp/HOFFMAN/nbp_1850-2010.nc"
netcdf = ild.download.from_html(url)
download_stamp = ild.output.utc_timestamp(netcdf.stat().st_mtime)
```

The `download` module contains functions for downloading datasets from various sources, such as `from_thredds`, `from_html`, `from_s3`, `from_arcgis_rest`, `from_zenodo`, `from_figshare`, `from_wcs`, etc. We also include the `earthaccess` package as a dependency for querying and downloading data from NASA Earthdata servers.

#### 2.c. Load the dataset

```python
# Load the dataset
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_dataset(netcdf, decode_times=time_coder)
```

We use `xarray` to open the NetCDF file and decode the time coordinate using `cftime`. This ensures that the time coordinate is properly interpreted and legible by `ild` functions.

#### 2.d. Standardize the dimensions and attributes

```python
# Standardize the dimensions
ds = ild.time.standardize(ds, bounds_frequency="YS")
```

The Hoffman dataset does not contain latitude, longitude, or depth coordinates, so we will not standardize those dimensions. However, if your dataset contains these dimensions, you can use the `lat.standardize`, `lon.standardize`, and `depth.standardize` functions to ensure that they are CF-compliant. Here, we only use `time.standardize`, and the `bounds_frequency` argument specifies the frequency of the time bounds, which can be `"D"`, `"MS"`, `"QS-Jan"`, or `"YS"`, which are `pandas` frequency strings. If your data are fixed time intervals (e.g., the data represent broadly between 2000-2010), you should use `time.create_time_axis` with `frequency="fx"` and manually define the start and end dates (`sdate`, `edate`) of the representative fixed time period.

#### 2.e. Define variable-specific attributes

Often, data sources contain more than one variable. Instead of writing one convert script per variable, we should write one convert script per data source/variant, and loop through the variables inside the convert script. This ensures that all variables in the dataset are formatted consistently. The Hoffman dataset contains two variables, `fgco2` and `nbp`, as well as their associated uncertainties. When writing a convert script for a dataset with multiple variables, we recommend defining variable-specific attributes in one place, such as a dictionary. This ensures that all variables are formatted consistently and makes it easier to update the attributes if needed. It will become clear later in this tutorial why these particular attributes were isolated into a dictionary.

```python
# Describe variable-specific metadata in one place
bounds = {"lbnd": "Lower", "ubnd": "Upper"}
variable_specs = {
    "nbp": {
        "realm": "land",
        "uncertainty_comment": "Uncertainty in land uptake calculated by carbon-budget mass balance, using the uncertainty envelope of the adjusted ocean-uptake trajectory",
        "source_description": "derived as a global carbon-budget residual using historical anthropogenic emissions, atmospheric CO₂ observations from ice cores, Mauna Loa, and NOAA, and an fgco2 ocean-uptake estimate",
    },
    "fgco2": {
        "realm": "ocean",
        "uncertainty_comment": "Uncertainty envelope of the Khatiwala et al. ocean-uptake trajectory after scaling it to the 2010 ocean-inventory estimate and adjusting it to an 1850 baseline",
        "source_description": "derived from in situ ship-based ocean carbon, tracer, and salinity measurements using a maximum-entropy-constrained Green's-function ocean-transport inversion",
    },
}
```

ODS provides guidance on how to label uncertainty variables. As of ODS 2.6.1, uncertainty variables can be appended with `nobs` (number of observations), `ustd` (standard uncertainty as 1 standard deviation), `uind` (uncertainty due to independent effects as 1 standard deviation), `ustr` (uncertainty due to
structured effects as 1 standard deviation), `ucom` (uncertainty due to common effects as 1 standard deviation), `lbnd` (lower bound of asymmetrical uncertainty), or `ubnd` (upper bound of asymmetrical uncertainty). See [ODS for up-to-date details](https://doi.org/10.5281/zenodo.11500473). ILAMB also includes `coefficient_of_variation` (coefficient of variation as a percentage). If you feel that your data's uncertainty does not fit into one of these categories, create a new variable name that is descriptive of the uncertainty type, and include a description of the uncertainty in the variable's `comment` attribute.

#### 2.f. Create one NetCDF per variable/uncertainty pair

Unlike `ODS`, which does not allow uncertainty to be stored in the same NetCDF file as the variable, ILAMB does, and CF conventions allow it as an `ancillary_variable`. For the Hoffman dataset, the NetCDF encodes uncertainty as `nbp_bnds` and `fgco2_bnds`, which contain 2D coordinates with the lower and upper bounds of the uncertainty. First, we need to break the `_bnds` into separate lower and upper ancillary variables to be CF-compliant.

```python
# Create one NetCDF per variable
for var, spec in variable_specs.items():
    source_bounds_name = f"{var}_bnds"  # How lower and upper uncertainty bounds are currently encoded
    bound_names = [f"{var}{suffix}" for suffix in bounds]  # What we want to rename the lower and upper uncertainty bounds to
    out = ds[[var, source_bounds_name, "time_bnds"]].copy()  # The output dataset with the variable, uncertainty, and dimensional bounds

    # Extract the lower and upper uncertainty bounds from {var}_bnds and rename them to {var}lbnd and {var}ubnd
    source_bounds = out[source_bounds_name]
    out = out.assign(
        {
            name: source_bounds.isel(bnds=index, drop=True)
            for index, name in enumerate(bound_names)
        }
    ).drop_vars(source_bounds_name)
```

#### 2.g. Add variable-specific attributes

Inside the loop, we can now set the variable- and uncertainty-specific attributes using built-in `ild` functions.

```python
    # Get CMIP6 variable information and standardize the primary variable
    var_info = ild.variable.lookup_cmip6(var, var)
    out = ild.variable.standardize(
        out,
        var,
        units=var_info["variable_units"],  # Required: units of the variable, e.g., "kg m-2 s-1"
        standard_name=var_info["cf_standard_name"],  # Optional: CF standard name of the variable, e.g., "surface_downward_mass_flux_of_carbon_as_carbon_dioxide"
        long_name=var_info["variable_long_name"],  # Required: long name of the variable, e.g., "Surface Downward Mass Flux of Carbon as CO2"
        ancillary_variables=" ".join(bound_names),  # Space-separated list of ancillary variables: "{var}lbnd {var}ubnd"
        target_dtype="float32",  # Optional: target data type of the variable
        convert=False  # Optional: if target units are different from the current units, convert the variable to the target units (default: False)
    )
```

The `ild.variable.standardize` function standardizes the variable's attributes to be CF-compliant, and it adds any additional attributes specified in the function call. The `lookup_cmip6` function searches the variable's standard name, long name, and units from the CMIP6 variable registry using `intake-esgf`. If your variable does not have an accepted MIP variable name, you can manually specify the `units` and `long_name` in the `ild.variable.standardize` function. CF requires that `ancillary_variables` should be a space-separated list of the ancillary variable names (if more than one). In this case, our uncertainty values are the ancillary variables. If a user provides `units` but sets `convert=False`, the old units attribute will be retained and the variable will not be converted. In the case of `Hoffman-1`, we choose to keep them as Pg yr-1. If a user provides `units` and sets `convert=True`, the variable will be converted to the new units using `cf-units` and the units attribute will be updated. The `ild.variable.standardize` function also accepts `cell_methods`, `flag_values`, `flag_meanings`, `nodata_value`, `compression`, and `extra_attrs`.

```python
    # Standardize the uncertainty variables
    for suffix, label in bounds.items():
        out = ild.variable.standardize(
            out,
            f"{var}{suffix}",
            units=var_info["variable_units"],
            standard_name=f"{out[var].attrs['standard_name']} {label.lower()}_bound",
            long_name=f"{label} Bound of Predicted {var_info['variable_long_name']}",
            target_dtype="float32",
            extra_attrs={"comment": spec["uncertainty_comment"]},  # Extra attributes to add to the variable
        )

    # Clean up straggling attributes
    out[var].encoding.pop("missing_value", None)
    out[var].attrs.pop("bounds", None)
```

We do the same thing for the ancillary uncertainty variables, but we also add a `comment` attribute to describe the uncertainty. The `extra_attrs` argument is a dictionary of additional attributes to add to the variable. The `ild.variable.standardize` function will not overwrite any existing attributes unless they are specified in the function call. Here, you can see that we use the `bounds` dictionary that we created earlier to easily set some of these attributes. Lastly, we clean up any straggling attributes that are not needed.

#### 2.h. Add global attributes

The final part of the convert script is to add global attributes to the dataset. If you want to contribute a dataset to ILAMB, you will need to add the following global attributes to the NetCDF file, which are modeled after those required in ODS:

```python
    # Set global attributes
    out = ild.global_attrs.set_ods26(
        out,
        activity_id="ILAMB",
        aux_uncertainty_id=", ".join(bounds),
        contact="Forrest Hoffman (forrest@climatemodeling.org)",
        creation_date=ild.output.utc_timestamp(),
        dataset_contributor="Morgan Steckler",
        frequency="yr",
        grid_label="gm",
        has_aux_unc="TRUE",
        history=f"Downloaded on {download_stamp}. Converted to CF-compliant and ODS-aligned product by ILAMB on {ild.output.utc_timestamp()}.",
        institution="University of California at Irvine, Irvine, CA, USA; Oak Ridge National Laboratory, Oak Ridge, TN, USA",
        institution_id="UCI-ORNL",
        license="Creative Commons Attribution 4.0 International License (CC BY 4.0)",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/Hoffman-1-0/convert.py",
        product="derived",
        realm=spec["realm"],
        references="Hoffman, F.M., et al. (2014): Causes and Implications of Persistent Atmospheric Carbon Dioxide Biases in Earth System Models. J. Geophys Res. Biogeosci., 119(2):141-162. https://doi.org/10.1002/2013JG002381.",
        region=f"global_{spec['realm']}",
        source=f"{out[var].attrs['long_name']} {spec['source_description']}",
        source_id="Hoffman-1-0",
        source_data_retrieval_date=download_stamp,
        source_data_url=url,
        source_label="Hoffman",
        source_type="insitu",
        source_version_number="1.0",
        title=f"{spec['realm'].capitalize()} Anthropogenic Carbon Flux Estimates",
        variable_id=var,
        variant_label="ILAMB",
        variant_info="Formatted product prepared by ILAMB",
        version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
    )
```

The `ild.global_attrs.set_ods26` function sets the global attributes of the dataset to be ODS-compliant. Some of these variables are part of the ODS controlled vocabulary, such as `frequency`, `grid_label`, `has_aux_unc`, `institution`, `institution_id`, `product`, `realm`, `region`, `source_id`, `source_label`, `source_type`, and `variable_id`. The function docstrings contain links to the controlled vocabulary for your reference. If you set a global attribute that is not part of the ODS controlled vocabulary, the function will raise a warning but will not fail. When possible, choose from the ODS controlled vocabulary. However, for some attributes that you want to set, especially `institution`, `institution_id`, `source_id`, `source_label`, and `variable_id`, you should ignore the warnings if your particular institution, source, or variable is not in the controlled vocabulary. In some cases, ODS might be too restrictive for your dataset, so if you feel that your dataset is not well represented by the controlled vocabulary, feel free to make up your own. ILAMB data are modeled after but not restricted to ODS requirements.

#### 2.i. Export the NetCDF

Finally, we prepare and export the output NetCDF file. The `ild.output.order_dimensions` function orders the dimensions in a standardized way, which isn't required but is recommended for legibility. The `ild.output.filename_from_attrs` function generates a filename based on the global attributes of the dataset, which is required for ILAMB datasets. The filename will be in the format of `{source_id}_{variable_id}_{frequency}_{grid_label}_{realm}_{region}_{variant_label}_{version}.nc`. A `variant_label` is only added when it differs from the `activity_id`. The `version` is the date of the conversion in `YYYYMMDD` format. The output NetCDF file will be saved in the same directory as the `convert.py` script.

```python
    # Write output
    out = ild.output.order_dimensions(out)
    out_path = ild.output.filename_from_attrs(out.attrs)
    out.to_netcdf(out_path)
```

### 3. Validate the dataset

Currently, ILAMB does not have a formal validation tool for contributed datasets. However, you can use the IOOS `compliance-checker` CLI tool to check for CF compliance. In the future, ILAMB will have a validation tool that checks for both CF-compliance and ODS-modeled ILAMB compliance.

### 4. Contribute the dataset to the ILAMB registry

To add your dataset to the ILAMB registry, you will need to create a pull request (PR) on the `ilamb3-data` repository. Before creating the PR, ensure that your local branch is up to date with the upstream `main` branch and that your changes are committed. Then, push your branch to your forked repository and create a PR on GitHub. In the PR description, we recommend labelling it like:

```markdown
NEWDATA(Hoffman-1-example): fgco2 (fgco2lbnd, fgco2ubnd), nbp (nbplbnd, nbpubnd)
```

In the extended description, it is helpful to describe the data, copy/paste the output of `ncdump`, attach screenshots of `ncview` maps, and/or include a summary of any changes you made to the `ild` code. Once the PR is submitted, it will be reviewed by the ILAMB team, and upon approval, it will be merged into the main repository, making your dataset available for others to use in ILAMB.

### 5. Use an LLM to help write a conversion script

Large Language Models (LLMs), including ChatGPT and Claude, can help write a `convert.py` script. However, they need current information about both the input dataset and the `ilamb3-data` repository. Treat the code in the [ilamb3-data repository](https://github.com/rubisco-sfa/ilamb3-data) as authoritative because the [ILAMB documentation](https://ilamb3.readthedocs.io/en/latest/) and examples may not reflect the latest API.

Some coding agents can inspect a repository and execute code directly, while an LLM in an ordinary chat may only be able to use files or text that you provide. Supplying a repository link does not guarantee that the LLM can open it. Ask the LLM to tell you if it cannot access a linked resource rather than relying on remembered APIs.

#### 5.a. Prepare the context

Before requesting a conversion script, provide the following information or ask the LLM to inspect it:

- The current `ilamb3_data` modules and public functions.
- `pyproject.toml`, including the existing dependencies.
- One or two existing conversion scripts for similar datasets.
- The input dataset structure, preferably the output of printing an `xarray.Dataset` or running `ncdump -h`.
- The source URL, DOI, citation, license, and retrieval method.
- The variables to export and any known requirements for their units, dimensions, uncertainties, and missing values.
- The intended global attributes and output files, if known.

If the LLM cannot access the repository, paste the relevant material into the conversation. Avoid pasting complete datasets or large portions of the ILAMB repository. A dataset summary, some API definitions, and closely related examples should provide enough context.

#### 5.b. General guidelines

Ask the LLM to follow these principles:

1. Prefer existing `ilamb3_data` functions for standardizing dimensions, variables, attributes, filenames, and output.
2. Confirm functions and arguments against the current repository. Do not invent APIs based on documentation, examples, or memory.
3. Keep `convert.py` short, flat, and specific to the data source. Avoid defining functions or classes where possible, and do not add an `if __name__ == "__main__":` block.
4. A conversion script does not need to be generalizable. If genuinely reusable behavior is needed, add it to `ilamb3_data` rather than embedding a general-purpose utility in one conversion script.
5. Make only the smallest necessary changes to `ilamb3_data`, and avoid unrelated refactoring.
6. Check `pyproject.toml` and use existing dependencies whenever possible. Do not introduce a dependency unless the person prompting the LLM explicitly approves it.
7. Do not guess scientific metadata. Identify unclear units, coordinate meanings, uncertainty definitions, licenses, citations, and controlled-vocabulary values for human review.
8. Preserve provenance and reproducibility, including source URLs, retrieval dates, citations, processing history, and deterministic output.
9. Preserve important characteristics of the source data, including calendars, time and spatial bounds, fill values, uncertainties, coordinate orientation, and data types where appropriate.
10. Validate the script as far as the available tools permit. Never claim that a command or check passed unless it was actually run. If execution is unavailable, provide the exact commands for the user to run locally and request the results for a second review.

You can tell the LLM that [CF Conventions](https://cfconventions.org/) and [Obs4MIPs Data Specifications](https://doi.org/10.5281/zenodo.11500473) are useful references when interpreting metadata and formatting the output.

#### 5.c. Example prompt

The following prompt is designed for a two-pass interaction: the LLM first proposes a plan, then writes the script after you review the plan. To complete the work in one pass, replace the instruction to stop after the plan with an instruction to continue directly to implementation.

```text
Help me write a short, flat convert.py script for this dataset.

Authoritative repository:
https://github.com/rubisco-sfa/ilamb3-data

Documentation:
https://ilamb3.readthedocs.io/en/latest/

Inspect the current repository before coding. Treat its code as authoritative because the documentation may be out of date. Check pyproject.toml and do not add a dependency without my approval. Confirm every ilamb3_data function and argument you use, review one or two similar conversion scripts, and prefer existing ilamb3_data utilities for dimensions, metadata, units, filenames, and output. If you cannot access a linked resource, say so and ask me to paste the relevant files; do not guess the available APIs.

Keep the script specific to this data source. Avoid functions, classes, and an `if __name__ == "__main__":` block. If reusable functionality is missing, propose the smallest possible library change separately and avoid unrelated refactoring. Do not guess scientific metadata; identify anything that needs human review.

First give me a brief conversion plan and stop so I can review it. After I approve the plan, write the script and validate it as far as your tools permit. Never claim a command or check passed unless you actually ran it. If you cannot run the conversion, give me the exact local commands to run and tell me which outputs to paste back for a second review.

Dataset source, citation, license, and documentation:
[PASTE OR LINK HERE]

Input dataset structure (paste xr.Dataset output or ncdump -h):
[PASTE HERE]

Required output variables and any known metadata requirements:
[PASTE HERE]
```

Always review the code and output NetCDFs thoroughly before submitting a PR. LLMs never produce a perfectly written code or perfectly formatted dataset on the first try. You especially want to ensure that the global attributes in the NetCDF are accurate and aligned with CF and ODS standards. If you are unsure about any of the metadata, please ask for help from the ILAMB team or consult the CF Conventions and Obs4MIPs Data Specifications.