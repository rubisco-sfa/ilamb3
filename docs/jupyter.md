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

ILAMB

```{code-cell}
cat = intake.open_catalog(
    "https://raw.githubusercontent.com/nocollier/intake-ilamb/main/ilamb.yaml"
)
ds_xusaatchi = cat["biomass | XuSaatchi2021"].read()
ds_esacci = cat["biomass | ESACCI"].read()
```

Now that we ha

```{code-cell}
bias = bias_analysis("biomass")
df, out_xusaatchi, out_esacci = bias(ds_xusaatchi, ds_esacci)
```

The returned dataframe contains scalar and score information from the analysis. When ILAMB is run via the application `ilamb-run`, this is the information that will be placed in the output html pages.

```{code-cell}
df
```

Note that the dataframe contains rows labeled "Reference" and "Comparison". In our case, the reference refers to the XuSaatchi product and the comparison to the ESACCI product. You can rename these easily with the following code.

```{code-cell}
df["source"] = df["source"].apply(lambda c: "XuSaatchi" if c == "Reference" else "ESACCI")
df
```

```{code-cell}
out_esacci
```

```{code-cell}
out_esacci["bias"].plot()
```


```{code-cell}
out_esacci["bias_score"].plot(cmap="plasma")
```
