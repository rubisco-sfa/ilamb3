---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Run Analysis in a Notebook

`ilamb3` has been redesigned to allow you to import our analysis functions and run them locally on your own datasets. This means that you can apply our analysis methods in your own Jupyter notebooks and python scripts. First, we import the functionality that we will need.

```{code-cell}
import matplotlib.pyplot as plt
import xarray as xr
import ilamb3
from ilamb3.analysis import bias_analysis
```

ILAMB analysis functions are available in the `ilamb3.analysis` package. You can import just this package and browse the member functions to see what is available. In this example, we will run the ILAMB bias methodology and so we import only this function. The ILAMB analysis functions have been redesigned to take as inputs two xarray datasets, a reference and a comparison. In this example, we will load two of our biomass reference data products and use the ILAMB bias methodology to compare them.

ILAMB reference datasets are available through an [pooch](https://github.com/fatiando/pooch) registry available via `ilamb3.open_catalog()`. We will use the catalog to load the biomass products from [Xu & Saatchi, 2021](https://zenodo.org/records/4161694) and [ESACCI](https://climate.esa.int/en/projects/biomass/).

```{code-cell}
cat = ilamb3.ilamb_catalog()
ds_xusaatchi = xr.open_dataset(cat.fetch("biomass/XuSaatchi2021/XuSaatchi.nc"))
ds_esacci = xr.open_dataset(cat.fetch("biomass/ESACCI/biomass.nc"))
```

Now that this data is loaded into memory as xarray datasets, we can initialize an ILAMB bias analysis.

```{code-cell}
analysis = bias_analysis("biomass")
```

```{code-cell}
df, out_xusaatchi, out_esacci = analysis(ds_xusaatchi, ds_esacci)
```

## Scalar Output

Each ILAMB analysis function returns the same 3 things: a pandas dataframe, and 2 xarray datasets. The returned dataframe contains scalar and score information from the analysis. When ILAMB is run via the application `ilamb-run`, this is the information that will be placed tables in the output html pages.

```{code-cell}
df
```

Note that the dataframe contains rows labeled "Reference" and "Comparison". In our case, the reference refers to the XuSaatchi product and the comparison to the ESACCI product. You can rename these easily with the following code.

```{code-cell}
df["source"] = df["source"].apply(lambda c: "XuSaatchi" if c == "Reference" else "ESACCI")
df
```

You may also notice that the bias does not seem to agree with the difference in mean values. This is a nuance in the ILAMB methodology that can cause confusion. The mean values returned are presented on their original grids. We found that model centers would complain if our analysis did not present the same numeric value that they produced with their in-house scripts. However the bias presented is only over the portion of the globe that both sources contain data. In the case of biomass here, there is great variance in land representation especially in highly vegetated areas around the islands in the tropics. This leads to a seemingly large difference in mean values and the bias reported.

## Gridded Output

In addition to the scalar information you will also obtain two datasets with other output that ILAMB will render as plots in the html output pages. You may do what you will with this information. The first of these is the reference intermediate output.

```{code-cell}
out_xusaatchi
```

The second is the comparison dataset intermediate output. This is where you will find bias (difference in this case) plots as well as maps of the ILAMB scores developed as part of the analysis.

```{code-cell}
out_esacci
```

Here we will plot the bias from the `out_esacci` dataset.

```{code-cell}
fig, ax = plt.subplots(tight_layout=True)
out_esacci["bias"].plot(ax=ax)
ax.set_title("Biomass Differences: XuSaatchi - ESACCI");
```
