---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Add an Analysis

In `ilamb3`, the analyses are being rewritten to function on `xarray` datasets. If you have some analysis function that takes in two datasets (and possibly other arguments) and does something to evaluate or analze performance, then you can adapt it for use in the `ilamb3` system.

You will need to understand a little bit of more advanced python. ILAMB analysis functions are implemented to derive from an [abstract base class](https://docs.python.org/3/glossary.html#term-abstract-base-class). An abstract base class (ABC) in python is the way we define the structure of an object that can be used in other parts of the ILAMB system. This means that in order for your analysis to be compatible in the ILAMB system, you need to write a class that derives from our base class [ILAMBAnalysis](https://github.com/rubisco-sfa/ilamb3/blob/main/ilamb3/analysis/base.py). If you follow that link, you will find the class and functions that you will need to implement in your analysis. We will also explain them here.

## The ILAMBAnalysis Base Class

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

### __call__

blah
