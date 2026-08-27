# Contribute Reference Data to ILAMB

ILAMB maintains a publicly accessible registry of reference data that can be downloaded using `ilamb fetch`---see the registry page at {doc}`./reference/datasets.md` to see what is currently available. If there is a dataset that isn't in the registry that is useful to the greater ILAMB community, consider formatting it and contributing it to the registry. This tutorial walks through an example of how to format a dataset for ILAMB, and how to submit it to the registry.

## Data file types

ILAMB loads data using xarray, which can read a [variety of file formats](https://docs.xarray.dev/en/latest/user-guide/io.html#). The most heavily tested and thus recommended data format for ILAMB is a [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) file. NetCDFs are self-describing, portable, scalable, appendable, sharable, and archivable. They are also the most common format for Earth System Model output. As part of the ILAMB version 3 release, we have changed our data standards to better align with [CF Conventions](http://cfconventions.org/) and obs4MIPs Data Specifications (ODS), which is a community standard for formatting observational datasets often used to benchmark Earth System Model outputs.

ILAMB-ready data most often have one data variable (plus any ancillary variables such as uncertainty), contain relevant coordinates (e.g., time, latitude, longitude, and depth), have descriptive global attributes, and are gridded or site-level.

Currently, 60% of ILAMB-ready data are global coverage, 25% are site-level, and the remaining are regional, i.e., pan-Arctic, Tropics, Northern Hemisphere, or mid-Latitudes. 80% of the gridded datasets are 0.5 degree spatial resolution, and the rest are 1 degree. The temporal resolution of the data are annual, monthly, or represent a fixed period of time. However, ILAMB has no restrictions on spatiotemporal resolution, variables, or data source. We are always looking to expand the benchmarking data we have available in ILAMB, so if you have a dataset that you think would be useful for benchmarking, please consider formatting it and contributing it to the registry.

## Format a reference dataset for ILAMB using Python

Let us walk through an example of how to format a dataset for ILAMB. We will use the [Conserving Land-Atmosphere Synthesis Suite (CLASS)](https://doi.org/10.25914/5c872258dc183) dataset as an example. The CLASS dataset is a global gridded dataset of land-atmosphere fluxes and states, which is available in NetCDF format. The dataset contains multiple variables, including net primary productivity (NPP), gross primary productivity (GPP), and evapotranspiration (ET), among others. You can visit the NCI Australia [THREDDS Server](https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/CLASS/v1-1/catalog.html) to see the data that we will download.

### 1. Fork and clone ilamb3-data
To contribute a dataset to ILAMB, you will first need to fork and clone the `ilamb3-data` repository hosted on GitHub, which requires at least a free GitHub account. You can use either SSH or HTTPS to clone the repository. If you are using SSH, you will need to [set up an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and add it to your GitHub account. If you are using HTTPS, you will need to consult [authentication methods for command-line Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line) to push to your fork without error.

1. Navigate to the [ilamb3-data GitHub page](https://github.com/rubisco-sfa/ilamb3-data)
2. Click the "Fork" button in the top right corner of the page to create a copy of the repository under your GitHub account (keep the default settings)
3. Click "Create fork" to create the fork
4. Clone the forked repository to your local machine using either SSH or HTTPS:

```bash
git clone git@github.com:YOUR-USERNAME/ilamb3-data.git  # fill with your GitHub username; you could also use HTTPS instead of SSH
cd ilamb3-data
```

5. Ensure that ilamb3-data is set up to track the upstream repository by adding the upstream remote, which will allow you to pull changes from the main repository:

```bash
git remote add upstream https://github.com/rubisco-sfa/ilamb3-data.git
```

6. Verify that the upstream remote has been added correctly:

```bash
git remote -v
```

7. Ensure your local `main` branch is up to date with the upstream repository:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
```

8. Create a contribution branch to work on your dataset:

Avoid making changes directly to the `main` branch. Instead, create a new branch with a descriptive name for your dataset contribution. For example, if you are contributing the CLASS dataset, you could name your branch `add-class-dataset`.

```bash
git switch -c add-class-dataset
```

9. Create a new directory for your dataset in the `datasets` folder of the repository:

The directory should be the PRODUCT-VERSION of the dataset with no special characters. For example, for the CLASS dataset version 1.1, you would create a directory named `CLASS-1-1` inside the `data` folder. Since `CLASS-1-1` already exists in the ILAMB repository, we'll create a new directory for this example dataset called `CLASS-1-1-example`:

```bash
mkdir -p data/CLASS-1-1-example
touch data/CLASS-1-1-example/convert.py
```

Then, you can open the Python script in your favorite text editor and start writing the code to format the dataset for ILAMB. Below is an example of how you might structure the `convert.py` script to read in the CLASS dataset, process it, and save it in a format compatible with ILAMB.

### 2. Write a data conversion Python script
Once you have forked and cloned the `ilamb3-data` repository, you can activate your Python environment and install the required dependencies. We recommend using `uv` to manage your Python environment. See [here](https://docs.astral.sh/uv/getting-started/installation/) for more information about installing `uv` on your system. You can install the required dependencies from the `ilamb3-data` directory and activate the environment using the following commands:

```bash
uv sync
source .venv/bin/activate
```

After activating the environment, you will have access to several useful libraries that are `ilamb3-data` dependencies, including `dask` for parallel computing on large datasets, `s3fs` for accessing data stored in Amazon S3, `cf-units` for handling unit conversions, `cftime` for handling time, `requests` and `bs4` for accessing and parsing HTML, `earthaccess` to access data from NASA Earthdata servers, as well as `xarray`, `rioxarray`, and `geopandas` for working with geospatial data. You can install additional dependencies as needed using `pip install <package-name>`. If there are libraries that you think many other users would be interested, you can run `uv add <package-name>` which adds the package to the `ilamb3-data` environment and updates the `pyproject.toml` file so that other users can install it when they run `uv sync`.

Next, you can start writing the code to format the dataset for ILAMB. Below is an example of how you might structure the `convert.py` script to read in the CLASS dataset, process it, and save it in a format compatible with ILAMB.

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

The ilamb3_data package provides a set of utilities for preparing ILAMB datasets so that they are CF and ODS-compliant. We will walk through some of the modules below, but for the most up-to-date information on the modules and functions available in ilamb3_data, please refer to the [ilamb3-data/ilamb3_data repository](https://github.com/rubisco-sfa/ilamb3-data/tree/main/ilamb3_data).

#### 2.b. Download the dataset
If possible, the `convert.py` script should download the dataset from a public source. This ensures that the dataset can be easily accessed and reproduced by others. If this isn't possible, please leave a note at the top of your script with instructions on how you accessed the data.

```python
url = "https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/CLASS/v1-1/catalog.xml"
input_netcdfs = ild.download.from_thredds(url, pattern="CLASS_v1-1_*.nc")
download_stamp = ild.output.utc_timestamp(input_netcdfs[0].stat().st_mtime)
```

The `download` module contains functions for downloading datasets from various sources, such as `from_thredds`, `from_html`, `from_s3`, `from_arcgis_rest`, `from_zenodo`, and `from_figshare`. We also include the `earthaccess` package as a dependency for querying and downloading data from NASA Earthdata servers.

#### 2.c. Load the dataset

```python
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)  # use to convert relative time values (days since ...) to cftime objects
ds = xr.open_mfdataset(input_netcdfs, decode_times=time_coder)
ds = ds.rename({"hfds": "hfdsl", "hfds_sd": "hfdsl_sd", "rs": "rns", "rs_sd": "rns_sd"})
```

We use `xarray` to open several NetCDF files at once using `open_mfdataset`. The `rename` method is used to rename variables in the dataset to accepted CMIP6 variable names. This is not required, because users can set `alternate_vars` in their benchmark study `yaml` file to specify the variable names in their dataset. However, we recommend using accepted CMIP6 variable names when possible for clarity. As CMIP7 outputs are released, it is natural that users should use the CMIP7 variable names when they become available. You can browse CMIP5 and CMIP6 variable names on [CEDA](https://clipc-services.ceda.ac.uk/dreq/mipVars.html) or programatically through [intake-esgf](https://github.com/esgf2-us/intake-esgf). We will use `intake-esgf` to automatically gather attributes for the dataset in a later step.

#### 2.d. Format time, lat, lon, and bounds
Next, we can use built-in `ilamb3-data` functions to format the time, latitude, and longitude coordinates, as well as their bounds. We will start with the time coordinate:

```python
ds = ild.time.standardize(ds, bounds_frequency="MS")
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.add_rectilinear_bounds(ds)
```

Since the dataset already contains time, lat, and lon bounds, all we need to do is ensure that they are CF-compliant. We can use the `standardize` functions to ensure compliance. For `time.standardize`, you must provide the frequency of your time data as `"D"`, `"MS"`, `"QS-Jan"`, or `"YS"`. Other frequencies are not currently supported. If your data are fixed time intervals (e.g., the data represent somewhere between 2000-2010), you should use `time.create_time_axis` with `frequency="fx"` and manually define the start and end dates (`sdate`, `edate`) of the representative fixed time period. The `add_rectilinear_bounds` function will add bounds to the lat and lon coordinates if they are not present.

#### 2.e. Format the data variables
Next, we can format the data variables in the dataset. CLASS contains `dw`, `hfdsl`, `hfls` `hfss`, `mrro`, `pr`, and `rns`, as well as their associated uncertainties labeled with `_sd`. We will loop through each variable and format them.

```python




