# Contribute Reference Data to ILAMB

ILAMB maintains a publicly accessible registry of reference data that can be downloaded using `ilamb fetch` (see the registry page at {doc}`./reference/datasets.md` to see what is currently available). If there is a dataset that you think could be useful to the ILAMB community, consider formatting it and contributing it to the registry. This tutorial walks through an example of how to format a dataset for ILAMB, and how to submit it to the registry.

## Data File Types

ILAMB loads data using xarray, which can read a [variety of file formats](https://docs.xarray.dev/en/latest/user-guide/io.html#). The most heavily tested and thus recommended data format for ILAMB is a [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) file. NetCDFs are self-describing, portable, scalable, appendable, sharable, and archivable. They are also the most common format for Earth System Model output. As part of the ILAMB version 3 release, we have changed our data standards to better align with [CF Conventions](http://cfconventions.org/) and obs4MIPs Data Specifications (ODS), which is a community standard for formatting observational datasets often used to benchmark Earth System Model outputs.

ILAMB-ready data most often have one data variable (plus any ancillary variables such as uncertainty), contain relevant coordinates (e.g., time, latitude, longitude, and depth), have descriptive global attributes, and are gridded or site-level.

Currently, 60% of ILAMB-ready data are global coverage, 25% are site-level, and the remaining are regional, i.e., pan-Arctic, Tropics, Northern Hemisphere, or mid-Latitudes. 80% of the gridded datasets are 0.5 degree spatial resolution, and the rest are 1 degree. The temporal resolution of the data are annual, monthly, or represent a fixed period of time. However, ILAMB has no restrictions on spatiotemporal resolution, variables, or data source. We are always looking to expand the benchmarking data we have available in ILAMB, so if you have a dataset that you think would be useful for benchmarking, please consider formatting it and contributing it to the registry.

## Format a Dataset for ILAMB using Python

Let us walk through an example of how to format a dataset for ILAMB. We will use the [Conserving Land-Atmosphere Synthesis Suite (CLASS)](https://doi.org/10.25914/5c872258dc183) dataset as an example. The CLASS dataset is a global gridded dataset of land-atmosphere fluxes and states, which is available in NetCDF format. The dataset contains multiple variables, including net primary productivity (NPP), gross primary productivity (GPP), and evapotranspiration (ET), among others. You can visit the NCI Australia [THREDDS Server](https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/CLASS/v1-1/catalog.html) to see the data that we will download.

### 1. Fork and Clone ilamb3-data
To contribute a dataset to ILAMB, you will first need to fork and clone the `ilamb3-data` repository hosted on GitHub, which requires at least a free account. You can use either SSH or HTTPS to clone the repository. If you are using SSH, you will need to [set up an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and add it to your GitHub account. If you are using HTTPS, you will need to consult [authentication methods for command-line Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line) to push to your fork without error.

1. Navigate to the [ilamb3-data GitHub page](https://github.com/rubisco-sfa/ilamb3-data)
2. Click the "Fork" button in the top right corner of the page to create a copy of the repository under your GitHub account (keep the default settings)
3. Click "Create fork" to create the fork
4. Clone the forked repository to your local machine using either SSH or HTTPS

```bash
git clone git@github.com:YOUR-USERNAME/ilamb3-data.git  # fill with your GitHub username; you could also use HTTPS instead of SSH
cd ilamb3-data
```

5. Ensure that ilamb3-data is set up to track the upstream repository by adding the upstream remote, which will allow you to pull changes from the main repository

```bash
git remote add upstream https://github.com/rubisco-sfa/ilamb3-data.git
```

6. Verify that the upstream remote has been added correctly

```bash
git remote -v
```

7. Ensure your local `main` branch is up to date with the upstream repository

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
```

8. Create a contribution branch to work on your dataset

Avoid making changes directly to the `main` branch. Instead, create a new branch with a descriptive name for your dataset contribution. For example, if you are contributing the CLASS dataset, you could name your branch `add-class-dataset`.

```bash
git switch -c add-class-dataset
```

9. Create a new directory for your dataset in the `datasets` folder of the repository

The directory should be the PRODUCT-VERSION of the dataset with no special characters. For example, for the CLASS dataset version 1.1, you would create a directory named `CLASS-1-1` inside the `data` folder. Since CLASS-1-1 already exists, we'll create a new directory for this example dataset called `CLASS-1-1-example`:

```bash
mkdir -p data/CLASS-1-1-example
touch data/CLASS-1-1-example/convert.py
```

Then, you can open the Python script in your favorite text editor and start writing the code to format the dataset for ILAMB. Below is an example of how you might structure the `convert.py` script to read in the CLASS dataset, process it, and save it in a format compatible with ILAMB.

### 2. Write the Data Conversion Python Script

```python
import time
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3_data as ild
```
