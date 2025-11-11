"""
The ILAMB area comparison methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as ilp
from ilamb3.analysis.base import ILAMBAnalysis


def _neither(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    out = (np.isclose(a, 0) | (a.isnull())) * (np.isclose(b, 0) | (b.isnull()))
    out.attrs = {"units": "1"}
    return out


def _both(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    out = ((a > 0) * (~a.isnull())) * ((b > 0) * (~b.isnull()))
    out.attrs = {"units": "1"}
    return out


def _complement(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    out = ((a > 0) | (~a.isnull())) * (np.isclose(b, 0) | (b.isnull()))
    out.attrs = {"units": "1"}
    return out


class area_analysis(ILAMBAnalysis):
    """
    The ILAMB area comparison methodology.

    Parameters
    ----------
    required_variable : str
        The name of the variable to be used in this analysis.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        required_variable: str,
        **kwargs: Any,  # this is so we can pass extra arguments without failure
    ):
        self.analysis_name = "Area Comparison"
        self.req_variable = required_variable
        self.kwargs = kwargs

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.req_variable]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.

        Returns
        -------
        pd.DataFrame
            A dataframe with scalar and score information from the comparison.
        xr.Dataset
            A dataset containing reference grided information from the comparison.
        xr.Dataset
            A dataset containing comparison grided information from the comparison.
        """
        # Initialize
        varname = self.req_variable
        ref.load()
        com.load()
        ref, com = cmp.adjust_lon(ref, com)

        # This analysis is not time-dependent
        if dset.is_temporal(ref[varname]):
            ref[varname] = dset.integrate_time(ref, varname, mean=True)
        if dset.is_temporal(com[varname]):
            com[varname] = dset.integrate_time(com, varname, mean=True)

        # Enforce we are dealing with numbers on the same grid
        ref_, com_ = cmp.nest_spatial_grids(ref, com)

        # Compare areas
        r = ref_[varname]
        c = com_[varname]
        bias = (
            xr.where(_neither(r, c) | np.isclose(r, 0), np.nan, 0)
            + _both(r, c) * 0
            - _complement(r, c)
            + _complement(c, r)
        )
        bias.attrs["units"] = "1"
        com_["bias"] = bias

        # scalars
        area_ref = float(
            dset.convert(dset.integrate_space(ref, "permafrost_extent"), "megameter**2")
        )
        area_com = float(
            dset.convert(dset.integrate_space(com, "permafrost_extent"), "megameter**2")
        )
        overlap = float(
            dset.convert(dset.integrate_space(_both(r, c), "overlap"), "megameter**2")
        )
        missed = float(
            dset.convert(
                dset.integrate_space(_complement(r, c), "missed"), "megameter**2"
            )
        )
        excess = float(
            dset.convert(
                dset.integrate_space(_complement(c, r), "excess"), "megameter**2"
            )
        )
        score_missed = overlap / (overlap + missed)
        score_excess = overlap / (overlap + excess)

        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": self.analysis_name,
                    "name": "Total Area",
                    "type": "scalar",
                    "units": "megameter**2",
                    "value": area_ref,
                },
            ]
            + [
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": self.analysis_name,
                    "name": f"{name} Area",
                    "type": "scalar",
                    "units": "megameter**2",
                    "value": val,
                }
                for val, name in zip(
                    [area_com, overlap, missed, excess],
                    ["Total", "Overlap", "Missed", "Excess"],
                )
            ]
            + [
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": self.analysis_name,
                    "name": f"{name} Score",
                    "type": "score",
                    "units": "1",
                    "value": val,
                }
                for val, name in zip(
                    [score_missed, score_excess],
                    ["Missed", "Excess"],
                )
            ]
        )
        return df, ref_.rename({varname: "extent"}), com_.rename({varname: "extent"})

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        if self.analysis_name not in df["analysis"].unique():
            return pd.DataFrame()
        com["Reference"] = ref

        # Extent plot
        axs = [
            {
                "name": "extent",
                "title": self.req_variable.replace("_", " ").title(),
                "region": None,
                "source": source,
                "axis": ilp.plot_map(
                    ds["extent"],
                    cmap="Blues",
                    title=source + " " + self.req_variable.replace("_", " ").title(),
                    ncolors=(
                        len(ds["extent"].attrs["labels"])
                        if "labels" in ds["extent"].attrs
                        else 1
                    ),
                    ticks=np.unique(ds["extent"].values[~np.isnan(ds["extent"])]),
                    ticklabels=(
                        ds["extent"].attrs["labels"]
                        if "labels" in ds["extent"].attrs
                        else ["Permafrost"]
                    ),
                    cbar_kwargs={"label": ""},
                ),
            }
            for source, ds in com.items()
            if "extent" in ds
        ]

        # Bias plot
        axs += [
            {
                "name": "bias",
                "title": "Bias",
                "region": None,
                "source": source,
                "axis": ilp.plot_map(
                    ds["bias"],
                    cmap="bwr",
                    title=source + " Bias",
                    ncolors=3,
                    ticks=[-1, 0, 1],
                    ticklabels=["Missed", "Overlap", "Excess"],
                    cbar_kwargs={"label": ""},
                ),
            }
            for source, ds in com.items()
            if "bias" in ds
        ]

        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
