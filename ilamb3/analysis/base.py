"""
An abstract class for implementing analysis functions used in ILAMB.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import xarray as xr

import ilamb3.dataset as dset
import ilamb3.regions as ilr


class ILAMBAnalysis(ABC):
    """
    The ILAMB analysis base class.

    An abstract base class (ABC) in python is a way to define the structure of
    an object that can be used in other parts of the system. In our case this
    means that in order for your analysis to be compatible in the ILAMB system,
    you need to write a class that has *at minimum* the following member
    functions with their arguments and return values.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any):
        """
        Initialize the analysis.

        This is run when you initialize the analysis. Any options that your
        analysis method will require should be input and stored here as
        keywords.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def required_variables(self) -> list[str] | dict[str, list[str]]:
        """
        Return the variables used in this analysis.

        Your analysis must be able to provide a list of variables that are to be
        used. This is so we can query models with your function and learn which
        variables are to be used. The assumption is that the variables you
        require are part of a historical-like experiment. If your analysis
        requires variables from different experiments, this function can also
        return a dictionary of lists where they keys are the experiment names
        from which the lists of variables are required.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        The function which performs the analysis.

        The work of your analysis must accept a reference and comparison
        dataset. The user will need to package everything that each needs for
        the analysis upon submission. Inside this function you can use whatever
        you want to generate scalars and maps/curves for plotting. Once
        complete, ILAMB would like results in a pandas dataframe (for scalar
        information) as well as a dataset for the reference and comparison.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.

        Returns
        -------
        pd.DataFrame
            The scalars and scores to be returned.
        xr.Dataset
            The reference intermediate maps and curves to plot
        xr.Dataset
            The comparison intermediate maps and curves to plot
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset]
    ) -> pd.DataFrame:
        """
        The function which creates plots from analysis intermediate results.

        ILAMB will run your __call__ method over all comparisons and build up a
        dataframe of scalars and a dictionary of intermediate result datasets.
        We will pass these into this function where you can generate plots.
        Instead of saving them directly here, place the into a pandas dataframe
        which is returned to the user.

        Parameters
        ----------
        df : pd.DataFrame
            The scalars and scores from all comparisons.
        ref : xr.Dataset
            The reference intermediate maps and curves to plot
        com : dict[str,xr.Dataset]
            A dictionary of comparison intermediate maps and curves to plot,
            whose keys are the model names.

        Returns
        -------
        pd.DataFrame
            The dataframe of plots.
        """
        raise NotImplementedError()  # pragma: no cover


def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize an average 'Overall' score from all scores in the dataframe.
    """
    add = (
        df[df["type"] == "score"]
        .groupby(["source", "region"])
        .mean(numeric_only=True)
        .reset_index()
    )
    add["name"] = "Overall Score [1]"
    add["type"] = "score"
    add["units"] = "1"
    df = pd.concat([df, add]).reset_index(drop=True)
    return df


def integrate_or_mean(
    var: xr.DataArray | xr.Dataset, varname: str, region: str | None, mean: bool
) -> xr.DataArray:
    """
    Integration/average the input dataarray/dataset to reduce in space/site.
    """
    da = var
    if isinstance(var, xr.Dataset):
        da = var[varname]
    if dset.is_spatial(da):
        da = dset.integrate_space(
            da,
            varname,
            region=region,
            mean=mean,
        )
    elif dset.is_site(da):
        da = ilr.Regions().restrict_to_region(da, region)
        da = da.mean(dim=dset.get_dim_name(da, "site"))
    else:
        raise ValueError(f"Input is neither spatial nor site: {da}")
    return da


def scalarify(
    var: xr.DataArray | xr.Dataset, varname: str, region: str | None, mean: bool
) -> tuple[float, str]:
    """
    Integration/average the input dataarray/dataset to generate a scalar.
    """
    da = integrate_or_mean(var, varname, region, mean)
    da = da.pint.quantify()
    return float(da.pint.dequantify()), f"{da.pint.units:~cf}"
