---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Add an Analysis

In `ilamb3`, the analyses are being rewritten to function on `xarray` datasets. If you have some analysis function that takes in two datasets (and possibly other arguments) and does something to evaluate or analyze performance, then you can adapt it for use in the `ilamb3` system.

You will need to understand a little bit of more advanced python. ILAMB analysis functions are implemented to derive from an [abstract base class](https://docs.python.org/3/glossary.html#term-abstract-base-class). An abstract base class (ABC) in python is the way we define the structure of an object that can be used in other parts of the ILAMB system. This means that in order for your analysis to be compatible in the ILAMB system, you need to write a class that derives from our base class [ILAMBAnalysis](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/analysis/base.py). If you follow that link, you will find the class and functions that you will need to implement in your analysis. We will also explain them here.

## The `ILAMBAnalysis` Base Class

In order to demonstrate what you will need to implement, we will create a custom analysis which implements a naive measure of model performance for the sake of simplicity. We will compare the arithmetic mean, {math}`v`, of a reference and comparison variable. We will just use the traditional measure of relative error,

```{math}
\varepsilon =  \frac{v_{\mathrm{com}} - v_{\mathrm{ref}} }{ v_{\mathrm{ref}}}
```

and then a score,

```{math}
S = 1 - \left|\varepsilon\right|
```

restricted to be positive. To implement this, we first import the `ILAMBAnalysis` base class and create a new object we will call `MyAnalysis`. We will also import `pandas` and `xarray` because we will use them later.

```{code-cell}
:tags: [skip-execution]
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ilamb3.analysis.base import ILAMBAnalysis

class MyAnalysis(ILAMBAnalysis):
    ...
```

### `required_variables()`

ILAMB uses this function to query model results for which variables will be used in this analysis. If you inspect the function as shown in the base [class](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/analysis/base.py#L19), you will see that this function accepts no arguments and returns either a list of strings (the variable names) or a dictionary of lists of strings.

- If you choose to return a list of strings, ILAMB will interpret this as variables that are needed from a `historical`-like experiment. Technically, this means that ILAMB will simply query the model for variables with no information what experiment was intended. This is the traditional function of ILAMB--to compare model output run over the contemporary era to reference data.
- However, if your analysis requires variables from multiple experiments, you will want to to include lists in a dictionary whose keys are the experiment names to be used. This preserves ILAMB's original function while also opening the door for perturbation-style experiments.

In our case, we are writing a generic analysis function that could work on any variable. For this reason, we will add a custom constructor function (`__init__`) which accepts the variable name as an argument. Then we store the variable name in the class member data and return it in a list in the `required_variables()` function.

```{code-cell}
:tags: [skip-execution]
    def __init__(self, variable: str):
        self.variable = variable

    def required_variables(self) -> Union[list[str], dict[str, list[str]]]:
        return [self.variable]
```

This also represents an important principle of writing `ILAMBAnalysis` functions: you need to implement at minimum the functions listed in the base class, but can add any number of additional functions and data.

If, for example, we were writing a more specific analysis function which requires certain variables, then we could skip the `__init__` function and simply return the hard-coded variables. Say your analysis is only meant for temperature, `tas`, then you would implement instead,

```{code-cell}
:tags: [skip-execution]
    def required_variables(self) -> Union[list[str], dict[str, list[str]]]:
        return ["tas"]
```

### `__call__()`

If you are new to python classes, you may find these methods with double underscores (so-called *dunder methods*) strange. Python uses these for special function names so that they are highly unlikely to clash with any name that you may choose in your own codes. The [`__call__`](https://www.geeksforgeeks.org/__call__-in-python/) method can be used to call your class as if it were a function. This is where you will put the guts of your analysis. Here we will present the function in its entirety with more explanation below.

```{code-cell}
:tags: [skip-execution]
    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset, **kwargs
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:

        # naive means just for demonstration
        vref = ref[self.variable].mean()
        vcom = com[self.variable].mean()

        # we use pint to handle units, make sure both are in the same units
        vref = vref.pint.quantify()
        vcom = vcom.pint.quantify().pint.to(vref.pint.units)

        # the analysis method
        eps = (vcom - vref) / vref
        score = (1 - np.abs(eps)).clip(0, 1)

        # populate a pandas dataframe with scalars and scores
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "MyNewStuff",
                    "name": "Mean",
                    "type": "scalar",
                    "units": f"{vref.pint.units:~cf}",
                    "value": float(vref.pint.dequantify()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "MyNewStuff",
                    "name": "Mean",
                    "type": "scalar",
                    "units": f"{vcom.pint.units:~cf}",
                    "value": float(vcom.pint.dequantify()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "MyNewStuff",
                    "name": "Bias Score",
                    "type": "score",
                    "units": "1",
                    "value": float(score.pint.dequantify()),
                },
            ]
        )
        return df, xr.Dataset(), xr.Dataset()
```

The `__call__()` function itself requires two arguments at minimum: the reference and comparison `xarray` datasets. However, we provide unlimited flexibility for your analysis needs in the way of open-ended keyword arguments (`**kwargs`). In this case our analysis is simple and therefore we do not need to specify additional keywords.

Then we take the arithmetic mean of the input datasets. We note again here that this is for simplicity of this demonstration. In a serious analysis you should be taking area-weighted means and considering trimming time and space to the maximal overlap in each dataset.

In previous versions of ILAMB, we utilized a python wrapper around the UDUNITS2 library for unit conversions. We are now using [pint](https://pint.readthedocs.io/en/stable/) for conversions. `pint` is now integrated into `xarray` objects, but you must manually *quantify* your datasets via the `pint` accessor. Once the variables are in the same units, we perform the analysis.

The `__call__()` function is expected to return 3 things. The first of which is a `pandas` dataframe containing all scalar information you wish to report, one row per scalar. Each row should contain the following columns at minimum:

- `source`: the source attached to the scalar, either "Reference" or "Comparison". For scalars that involve both, such as a bias or difference, we associate it with the comparison dataset.
- `region`: the region associated with the scalar. In this case we use "None", but if you were making your analysis applicable to regions then you would include the ILAMB region label here.
- `analysis`: an identifier for this analysis. When run in the ILAMB system, we will merge the scalar information produced here with that from other analyses. This string will be used to place scalars in sections allowing users to tease out only those that they wish to examine.
- `name`: the name of the scalar.
- `type`: used to distinguish scalar types, either "scalar" or "score". Scores will be included in the overall score computation.
- `units`: the unit of the scalar.
- `value`: the numerical value of the scalar. Note that to ensure compatibility outside of the `xarray` ecosystem, we convert them to floats after de-quantifying the values.

In the code above, we include 3 scalars in this dataframe: the mean of the reference and comparison along with the score we developed as part of the analysis. This is completely open-ended and you could add as many scores as is relevant to your target application.

The other 2 arguments that are returned are `xarray` datasets of other intermediate information that we want to plot in the ILAMB analysis and present in the output webpages. As this example is simple, we will return empty datasets indicating that there is nothing else to plot.
