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
import intake
import matplotlib.pyplot as plt

from ilamb3.analysis import bias_analysis
```

ILAMB analysis functions are available in the `ilamb3.analysis` package. You can import just this package and browser the member functions to see what is available. In this example, we will run the ILAMB bias methodology and so we import only this function. The ILAMB analysis functions have be redesigned to take as inputs two xarray datasets, a reference and a comparison. In this example, we will load two of our biomass reference data products and use the ILAMB bias methodology to compare them.

ILAMB reference datasets are available through an [intake](https://github.com/intake/intake) catalog. To use it, you only need to install the `intake` package and then add the following call to `open_catalog()`. We will use the catalog to load the biomass products from [Xu & Saatchi, 2021](https://zenodo.org/records/4161694) and [ESACCI](https://climate.esa.int/en/projects/biomass/).

```{code-cell}
cat = intake.open_catalog(
    "https://raw.githubusercontent.com/nocollier/intake-ilamb/main/ilamb.yaml"
)
ds_xusaatchi = cat["biomass | XuSaatchi2021"].read()
ds_esacci = cat["biomass | ESACCI"].read()
```

Now that this data is loaded into memory as xarray datasets, we can initialize an ILAMB bias analysis.

```{code-cell}
bias = bias_analysis("biomass")
df, out_xusaatchi, out_esacci = bias(ds_xusaatchi, ds_esacci)
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
