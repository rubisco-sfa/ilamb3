"""
An ILAMB transform to compute quantiles.
"""

import xarray as xr

import ilamb3.dataset as dset
from ilamb3.transform.base import ILAMBTransform


class quantile(ILAMBTransform):
    """
    Compute quantiles of a variable over one or more dimensions.

    If `dims` is a list/tuple, the quantile is computed jointly across all
    listed dimensions in a single reduction.

    Parameters
    ----------
    quantiles: float or list
        The quantiles in [0,1].
    dims: str of list
        The meta dimension names, one of {`time`, `lat`, `lon`, `depth`, `site`}.

    """

    def __init__(
        self,
        quantiles: float | list[float],
        dims: str | list[str],
    ):
        self.quantiles = quantiles if isinstance(quantiles, list) else [quantiles]
        self.dims = dims if isinstance(dims, list) else [dims]

    def required_variables(self) -> list[str]:
        """
        This transform does not require anything in particular.
        """
        return []

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Compute quantiles of all varaibles in dataset along dimensions.
        """
        # dim names as they are known in the dataset
        reduce_dims = [dset.get_dim_name(ds, dim) for dim in self.dims]
        bounds_vars = dset.get_all_bounds_vars(ds)
        for name, da in ds.data_vars.items():
            if not set(reduce_dims).issubset(da.dims) or name in bounds_vars:
                continue
            ds[name] = da.quantile(q=self.quantiles, dim=reduce_dims)
            ds[name].attrs["transform"] = "quantiles"
            ds[name].attrs["reduced_dims"] = reduce_dims
        return ds


"""
- I have removed Sequence because strings are also sequences...
- We also cast tihngs as lists, let's just make the types lists especially since they will be imported from yamls
- I moved the checking if things are a list to the constructor and added one for quantiles also
- I added a empty list to `required_variables()` instead of pass
- I added calls to ilamb3.dataset.get_dim_name for each dim
- It isn't necessary to store the quantiles because they will be a dimension in the returned dataset
- I dropped the 's' from quantiles so it matches the name of the file. Eventually we may want to load transforms programmatically and this may be helpful.
- We want the transform to work slightly differently, only working on arrays that have all the dimensions. This will make it skip bounds variables but also leave them there.
- Let's detect if the data variables are actually bounds of some dims and also skip these.
- We will just overwrite the dataset array if it qualifies for quantiles
- In the first version of the tutorial, I didn't have the bit about how to add this transform to the library. This has been added in __init__.
- I added a basic test
"""
