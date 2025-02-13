"""
The ILAMB relationship (variable to variable) methodology.

There is a lot of data to manage for relationships. So we build a relationship class and
then later create a ILAMBAnalysis which uses it.
"""

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as plt
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.regions import Regions


@dataclass
class Relationship:
    """
    A class for developing and comparing relationships from gridded data.
    """

    dep: xr.DataArray
    ind: xr.DataArray
    color: xr.DataArray = None
    dep_log: bool = False
    ind_log: bool = False
    dep_var: str = "dep"
    ind_var: str = "ind"
    dep_label: str = "dep"
    ind_label: str = "ind"
    order: int = 1
    _dep_limits: list[float] = field(init=False, default_factory=lambda: None)
    _ind_limits: list[float] = field(init=False, default_factory=lambda: None)
    _dist2d: np.ndarray = field(init=False, default_factory=lambda: None)
    _ind_edges: np.ndarray = field(init=False, default_factory=lambda: None)
    _dep_edges: np.ndarray = field(init=False, default_factory=lambda: None)
    _response_mean: np.ndarray = field(init=False, default_factory=lambda: None)
    _response_std: np.ndarray = field(init=False, default_factory=lambda: None)

    def __post_init__(self):
        """After initialization, perform checks on input data."""
        # check input dataarrays for compatibility
        assert isinstance(self.dep, xr.DataArray)
        assert isinstance(self.ind, xr.DataArray)
        self.dep = self.dep.sortby(list(self.dep.sizes.keys()))
        self.ind = self.ind.sortby(list(self.ind.sizes.keys()))
        self.dep, self.ind = xr.align(self.dep, self.ind, join="exact")
        if self.color is not None:
            assert isinstance(self.color, xr.DataArray)
            self.color = self.color.sortby(list(self.color.sizes.keys()))
            self.dep, self.ind, self.color = xr.align(
                self.dep, self.ind, self.color, join="exact"
            )

        # only consider where both are valid and finite
        keep = self.dep.notnull() * self.ind.notnull()
        keep *= np.isfinite(self.dep)
        keep *= np.isfinite(self.ind)
        self.dep = xr.where(keep, self.dep, np.nan)
        self.ind = xr.where(keep, self.ind, np.nan)
        if self.dep_log:
            assert self.dep.min() > 0
        if self.ind_log:
            assert self.ind.min() > 0

    def compute_limits(
        self, rel: Union["Relationship", None] = None
    ) -> Union["Relationship", None]:
        """
        Compute the limits of the dependent and independent variables.

        Parameters
        ----------
        rel : Relationship, optional
            An optional and additional relationship who limits you also would like to
            include in the check.

        Returns
        -------
        Relationship or None
            If a relationship was passed in, this will return it also with its limits
            set.
        """

        def _singlelimit(var, limit=None):  # numpydoc ignore=GL08
            lim = [var.min(), var.max()]
            delta = 1e-8 * (lim[1] - lim[0])
            lim[0] -= delta
            lim[1] += delta
            if limit is None:
                limit = lim
            else:
                limit[0] = min(limit[0], lim[0])
                limit[1] = max(limit[1], lim[1])
            return limit

        dep_lim = _singlelimit(self.dep)
        ind_lim = _singlelimit(self.ind)
        if rel is not None:
            dep_lim = _singlelimit(self.dep, limit=dep_lim)
            ind_lim = _singlelimit(self.ind, limit=ind_lim)
            rel._dep_limits = dep_lim
            rel._ind_limits = ind_lim
        self._dep_limits = dep_lim
        self._ind_limits = ind_lim
        return rel

    def build_response(self, nbin: int = 25, eps: float = 3e-3):
        """
        Create the 2D distribution and functional response.

        Parameters
        ----------
        nbin : int
            The number of bins to use in both dimensions.
        eps : float
            The fraction of points required for a bin in the
            independent variable be included in the funcitonal responses.
        """
        # if no limits have been created, make them now
        if self._dep_limits is None or self._ind_limits is None:
            self.compute_limits(None)

        # compute the 2d distribution
        ind = np.ma.masked_invalid(self.ind.values).compressed()
        dep = np.ma.masked_invalid(self.dep.values).compressed()
        xedges = nbin
        yedges = nbin
        if self.ind_log:
            xedges = 10 ** np.linspace(
                np.log10(self.ind_limits[0]), np.log10(self.ind_limits[1]), nbin + 1
            )
        if self.dep_log:
            yedges = 10 ** np.linspace(
                np.log10(self.dep_limits[0]), np.log10(self.dep_limits[1]), nbin + 1
            )
        dist, xedges, yedges = np.histogram2d(
            ind,
            dep,
            bins=[xedges, yedges],
            range=[
                [v.values for v in self._ind_limits],
                [v.values for v in self._dep_limits],
            ],
        )
        dist = np.ma.masked_values(dist.T, 0).astype(float)
        dist /= dist.sum()
        self._dist2d = dist
        self._ind_edges = xedges
        self._dep_edges = yedges

        # compute a binned functional response
        which_bin = np.digitize(ind, xedges).clip(1, xedges.size - 1) - 1
        mean = np.ma.zeros(xedges.size - 1)
        std = np.ma.zeros(xedges.size - 1)
        cnt = np.ma.zeros(xedges.size - 1)
        with np.errstate(under="ignore"):
            for i in range(mean.size):
                depi = dep[which_bin == i]
                cnt[i] = depi.size
                if cnt[i] == 0:  # will get masked out later
                    mean[i] = 0
                    std[i] = 0
                else:
                    if self.dep_log:
                        depi = np.log10(depi)
                        mean[i] = 10 ** depi.mean()
                        std[i] = 10 ** depi.std()
                    else:
                        mean[i] = depi.mean()
                        std[i] = depi.std()
            mean = np.ma.masked_array(mean, mask=(cnt / cnt.sum()) < eps)
            std = np.ma.masked_array(std, mask=(cnt / cnt.sum()) < eps)
        self._response_mean = mean
        self._response_std = std

    def score_response(self, rel: "Relationship") -> float:
        """
        Score the functional response of the relationships.

        Parameters
        ----------
        rel : Relationship
            The other relationship to compare.

        Returns
        -------
        float
            The response score.
        """
        rel_error = np.linalg.norm(
            self._response_mean - rel._response_mean
        ) / np.linalg.norm(self._response_mean)
        score = np.exp(-rel_error)
        return score

    def to_dataset(self) -> xr.Dataset:
        """
        Convert internal relationship representation to a dataset.
        """
        ds = xr.Dataset(
            data_vars={
                f"distribution_{self.ind_var}": (
                    [self.dep_var, self.ind_var],
                    self._dist2d,
                ),
                f"response_{self.ind_var}": ([self.ind_var], self._response_mean),
                f"response_{self.ind_var}_variability": (
                    [self.ind_var],
                    self._response_std,
                ),
            },
            coords={
                self.ind_var: (
                    self.ind_var,
                    0.5 * (self._ind_edges[1:] + self._ind_edges[:-1]),
                ),
                self.dep_var: (
                    self.dep_var,
                    0.5 * (self._dep_edges[1:] + self._dep_edges[:-1]),
                ),
            },
        )
        ds[f"response_{self.ind_var}"].attrs = {
            "ancillary_variables": f"response_{self.ind_var}_variability"
        }
        ds[self.ind_var].attrs = {"standard_name": self.ind_label}
        ds[self.dep_var].attrs = {"standard_name": self.dep_label}
        return ds


class relationship_analysis(ILAMBAnalysis):
    """
    The ILAMB relationship methodology.

    Parameters
    ----------
    dep_variable : str
        The name of the dependent variable to be used in this analysis.
    ind_variable : str
        The name of the independent variable to be used in this analysis.
    regions: list[str | None] = [None],
        The regions overwhich to perform the analysis.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        dep_variable: str,
        ind_variable: str,
        regions: list[str | None] = [None],
        **kwargs: Any,
    ):  # numpydoc ignore=GL08
        self.dep_variable = dep_variable
        self.ind_variable = ind_variable
        self.regions = regions

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.dep_variable, self.ind_variable]

    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB relationship methodology on the given datasets.

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

        def _regionify(ds: xr.Dataset, region: str | None):
            """Postpend the region name to the variables/dimenions"""
            ds = ds.rename({v: f"{v}_{region}" for v in list(ds.variables)})
            for key, da in ds.items():
                if "ancillary_variables" in da.attrs:
                    ds[key].attrs[
                        "ancillary_variables"
                    ] = f"{da.attrs['ancillary_variables']}_{region}"
            return ds

        # Initialize and make comparable
        analysis_name = f"Relationship {self.ind_variable}"
        var_ind = self.ind_variable
        var_dep = self.dep_variable
        for var in self.required_variables():
            ref, com = cmp.make_comparable(ref, com, var)
            if dset.is_temporal(ref[var]):
                ref[var] = dset.integrate_time(ref[var], mean=True)
            if dset.is_temporal(com[var]):
                com[var] = dset.integrate_time(com[var], mean=True)

        # Create and score relationships per region
        dfs = []
        ds_ref = []
        ds_com = []
        ilamb_regions = Regions()
        for region in self.regions:
            refr = ilamb_regions.restrict_to_region(ref, region)
            comr = ilamb_regions.restrict_to_region(com, region)
            rel_ref = Relationship(
                refr[var_dep],
                refr[var_ind],
                dep_var=var_dep,
                ind_var=var_ind,
                dep_label=var_dep,
                ind_label=var_ind,
            )
            rel_com = Relationship(
                comr[var_dep],
                comr[var_ind],
                dep_var=var_dep,
                ind_var=var_ind,
                dep_label=var_dep,
                ind_label=var_ind,
            )
            rel_com = rel_ref.compute_limits(rel_com)
            rel_ref.build_response()
            rel_com.build_response()
            score = rel_ref.score_response(rel_com)
            ds_ref.append(_regionify(rel_ref.to_dataset(), region))
            ds_com.append(_regionify(rel_com.to_dataset(), region))
            dfs.append(
                [
                    "Comparison",
                    str(region),
                    analysis_name,
                    f"Relationship Score {var_ind}",
                    "score",
                    "1",
                    score,
                ]
            )

        # Conversions and output
        dfs = pd.DataFrame(
            dfs,
            columns=[
                "source",
                "region",
                "analysis",
                "name",
                "type",
                "units",
                "value",
            ],
        )
        ds_ref = xr.merge(ds_ref)
        ds_com = xr.merge(ds_com)
        return dfs, ds_ref, ds_com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        # Some initialization
        com["Reference"] = ref

        # Build up a dataframe of matplotlib axes, first the distribution plots
        axs = [
            {
                "name": f"distribution_{self.ind_variable}",
                "title": f"{self.dep_variable} vs. {self.ind_variable}",
                "region": region,
                "source": source,
                "axis": (
                    plt.plot_distribution(
                        ds[f"distribution_{self.ind_variable}_{region}"],
                        title=f"{source} {self.dep_variable} vs. {self.ind_variable}",
                    )
                    if f"distribution_{self.ind_variable}_{region}" in ds
                    else pd.NA
                ),
            }
            for region in self.regions
            for source, ds in com.items()
        ]
        com.pop("Reference")
        axs += [
            {
                "name": f"response_{self.ind_variable}",
                "title": f"{self.dep_variable} vs. {self.ind_variable}",
                "region": region,
                "source": source,
                "axis": (
                    plt.plot_response(
                        ref[f"response_{self.ind_variable}_{region}"],
                        ref[f"response_{self.ind_variable}_variability_{region}"],
                        ds[f"response_{self.ind_variable}_{region}"],
                        ds[f"response_{self.ind_variable}_variability_{region}"],
                        source,
                        title=f"{source} {self.dep_variable} vs. {self.ind_variable}",
                    )
                    if f"response_{self.ind_variable}_{region}" in ds
                    else pd.NA
                ),
            }
            for region in self.regions
            for source, ds in com.items()
        ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])

        return axs
