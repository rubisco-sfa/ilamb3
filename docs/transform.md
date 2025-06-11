---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Add a Transform

The ILAMB paradigm is to apply consistent analysis methods to reference data whose variables map directly to model output as part of experimental runs. While this applies to many datasets, it is not always possible. For example, the Hoffman global net biome production reference dataset is a global integral and yet exists in models as a a function of `time`, `lat`, and `lon`. We could write a specialized analysis that embeds this global integration when loading the model data, but this makes the analysis routines specialized to this variable and not reusable elsewhere.

In `ilamb3` we have abstracted the preprocessing step into a concept we call an ILAMB transform. In short, a transform is just a function which takes in a dataset and if it can operate on it, does so and returns the dataset. In this example, we will implement a transform for the needs expressed above--we will take a spatial integral of the input dataset. Once the transform is written, it enables the following configure block:

```yaml
Hoffman:
  sources:
    nbp: HOFFMAN/nbp_1850-2010.nc
  transforms:
  - spatial_integral
  analyses:
  - timeseries
```

When ILAMB encounters this block it will:

1. Look for `nbp` in the reference data `HOFFMAN/nbp_1850-2010.nc`.
2. Pass the reference data into the `spatial_integral` transform, but because `nbp` in the reference is already integrated, the transform simply returns the input dataset.
3. For each comparison (model) set of results, we will also look for `nbp` but in this case it will find a spatio-temporal dataset.
4. The comparison dataset is passed in to the `spatial_integral` transform and this time returns the spatially integrated dataset.

This means that we can focus on implementing abstract analysis methods that can work on a range of reference data and let the transforms take care of preprocessing.

## Implementing the Transform

The main body of the transform is just a simple function:

```python
import xarray as xr
import ilamb3.dataset as dset

def spatial_integral(ds: xr.Dataset) -> xr.Dataset:
    # If not a spatial variable, just return
    if not dset.is_spatial(ds):
        return ds
    # Now we just integrate each DataArray using ilamb3 functions
    for var, _ in ds.items():
        ds[var] = dset.integrate_space(ds, var)
    return ds
```

You do not even need to use `ilamb3` functions. In principle, we only need your function to return the modified Dataset and not fail if your transform was not appropriate.

## A Slight Expansion

While the main body of the transform is a simple function, `ilamb3` needs a bit more information. Internally, we look at all the analyses and transforms to determine which variables we should look for in the models. This means that we need to be able to query your function and learn what variables you might be using in the input dataset.

This means that we need to make our function and small class.

```python
from ilamb3.transform.base import ILAMBTransform

class spatial_integral(ILAMBTransform):

    def __init__(self, **kwargs: Any):
        pass

    def required_variables(self) -> list[str]:
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        # If not a spatial variable, just return
        if not dset.is_spatial(ds):
            return ds
        # Now we just integrate each DataArray using ilamb3 functions
        for var, _ in ds.items():
            ds[var] = dset.integrate_space(ds, var)
        return ds
```

If you are unfamiliar with python classes, we will explain the pieces of this one by one.

1. We make the class name `spatial_integral` and it *inherits* from `ILAMBTransform`. This is a general class that allows us to check at runtime that we are passing functions that should behave correctly.
2. There is a `__init__` method which takes as a first argument `self`. In fact, all the functions here take a `self` first argument. This is pythons way to pass along the member data along with intances of the class. This is where we would store any options that your transform might have, but in this case, we do not have any.
3. There is a `required_variables` function. This should return a list of variables that you are using as part of the transformation code. In this case, our transform is general and requires nothing specific.
4. The body of our transform is now in a member function named `__call__`. This makes the class instance callable and look like you are applying a function.

More generally, we need your transform to be able to tell `ilamb3` which variables are being used and store the options you may want to apply.
