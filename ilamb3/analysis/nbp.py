"""
The ILAMB net biome production scoring methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from typing import Any

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import xarray as xr

import ilamb3
import ilamb3.plot as ilplt
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import NoUncertainty, TemporalOverlapIssue


def _convert_carbon_units(ds):
    for var, da in ds.items():
        if "units" not in da.attrs:
            continue
        if ilamb3.units(da.attrs["units"]).check("[mass] / [time]"):
            ds = dset.convert(ds, "Pg yr-1", varname=var)
            ds[var] = -ds[var]
    return ds


def _carbon_accumulation(ds):
    for var, da in ds.items():
        if "units" not in da.attrs:
            continue
        if ilamb3.units(da.attrs["units"]).check("[mass] / [time]"):
            ds[f"a{var}"] = da.cumsum(dim="year")
            ds[f"a{var}"].attrs["units"] = "Pg"
            if "ancillary_variables" in da.attrs:
                ds[f"a{var}"].attrs[
                    "ancillary_variables"
                ] = f"a{da.attrs['ancillary_variables']}"
            if "bounds" in da.attrs:
                ds[f"a{var}"].attrs["bounds"] = f"a{da.attrs['bounds']}"

    return ds


class nbp_analysis(ILAMBAnalysis):
    """
    The ILAMB net biome production scoring methodology.

    Parameters
    ----------
    evaluation_year : int, optional
        The year at which to report a difference and score. If not given, the
        last year of the reference dataset.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(self, evaluation_year: int | None = None, **kwargs: Any):
        self.evaluation_year = evaluation_year

    def required_variables(self) -> list[str]:
        """
        Return the variable names required in this analysis.

        Returns
        -------
        list
            A list of the required variables, here always [`nbp`].

        Notes
        -----
        This analysis also accepts the variable `netAtmosLandCO2Flux`. If you are
        running this routine inside an ILAMB analysis and need to use this variable,
        register it with the model as a synonym.
        """
        return ["nbp"]

    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset
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
        # We do not want to do the normal ILAMB maximal temporal overlap. Since
        # we are accumulating, the comparison needs to cover the reference but
        # we will use ILAMB functions and verify after the fact.
        t0, tf = dset.get_time_extent(ref, include_bounds=False)
        if self.evaluation_year is None:
            self.evaluation_year = int(tf.dt.year)
        ref, com = cmp.trim_time(ref, com)
        ref.load()
        com.load()

        # Coarsen to annual
        ref = dset.coarsen_annual(ref)
        com = dset.coarsen_annual(com)

        # Trim to the evaluation year
        ref = ref.sel(year=slice(None, self.evaluation_year))
        com = com.sel(year=slice(None, self.evaluation_year))

        # If after trimming we don't span the original time frame, we cannot do this analysis
        if com["year"].min() != t0.dt.year:
            raise TemporalOverlapIssue(
                "Comparison dataset not defined at reference beginning."
            )
        if com["year"].max() < self.evaluation_year:
            com_end_year = int(com["year"].max())
            raise TemporalOverlapIssue(
                f"Comparison dataset does not reach the evaluation year: {com_end_year=} < {self.evaluation_year=}"
            )

        # Convert flux units
        ref = _convert_carbon_units(ref)
        com = _convert_carbon_units(com)

        # Add in accumulations
        ref = _carbon_accumulation(ref)
        com = _carbon_accumulation(com)

        # Trajectory score
        uncert = dset.get_scalar_uncertainty(ref, "nbp")
        bounds = dset.get_interval_uncertainty(ref, "nbp")
        bnd_dim = bounds.dims[-1]
        # Only count errors outside of the envelope for scoring
        eps = (com["nbp"] - bounds.min(dim=bnd_dim)).clip(0) + (
            bounds.max(dim=bnd_dim) - com["nbp"]
        ).clip(0)
        traj_score = np.exp(-eps / uncert)
        traj_score = float(traj_score.mean())

        # Difference score
        ref_val = float(ref["nbp"].sel(year=self.evaluation_year))
        com_val = float(com["nbp"].sel(year=self.evaluation_year))
        try:
            uncert = dset.get_scalar_uncertainty(ref, "anbp")
        except NoUncertainty:
            uncert = xr.zeros_like(ref["anbp"])
        uncert_val = float(uncert.sel(year=self.evaluation_year))
        scale = -np.log(0.5) / 1  # outside of the uncertainty window? score < 50%
        diff_score = np.exp(
            -scale
            * np.abs(com_val - ref_val)
            / (np.abs(ref_val) if np.abs(uncert_val) < 1e-8 else uncert_val)
        )

        # Scores and scalars
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({self.evaluation_year})",
                    "type": "scalar",
                    "units": "Pg",
                    "value": ref_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({self.evaluation_year})",
                    "type": "scalar",
                    "units": "Pg",
                    "value": com_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"diff({self.evaluation_year})",
                    "type": "scalar",
                    "units": "Pg",
                    "value": com_val - ref_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": "Difference Score",
                    "type": "score",
                    "units": "1",
                    "value": diff_score,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": "Trajectory Score",
                    "type": "score",
                    "units": "1",
                    "value": traj_score,
                },
            ]
        )
        return df, ref, com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        # plot over the reference limits
        uncert = dset.get_interval_uncertainty(ref, "anbp")
        com["Reference"] = ref
        axs = [
            {
                "name": "accumulation",
                "title": "nbp_accumulation",
                "region": None,
                "source": None,
                "axis": plot_accumulated_nbp(
                    com, ref, vmin=float(uncert.min()), vmax=float(uncert.max())
                ),
            }
        ]
        lim = ilplt.determine_plot_limits(com, percent_pad=0).set_index("name")
        axs += [
            {
                "name": plot,
                "title": "nbp",
                "region": None,
                "source": source,
                "axis": (
                    ilplt.plot_curve(
                        {source: ds} | {"Reference": ref},
                        plot,
                        vmin=lim.loc[plot, "low"]
                        - 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
                        vmax=lim.loc[plot, "high"]
                        + 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
                        title=source + " - nbp fluxes",
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in ["nbp"]
            for source, ds in com.items()
            if source != "Reference"
        ]
        return pd.DataFrame(axs)


def _space_labels(
    dsd: dict[str, xr.Dataset], ymin: float, maxit: int = 10
) -> dict[str, float]:
    """
    Space out the model labels for the anbp plot using a modified Laplacian
    smoothing.
    """
    sorted_dsd = {key: float(ds.isel(year=-1)["anbp"]) for key, ds in dsd.items()}
    sorted_dsd = dict(sorted(sorted_dsd.items(), key=lambda item: item[1]))
    y = np.array([v for _, v in sorted_dsd.items()])
    for j in range(maxit):
        dy = np.abs(np.diff(y))
        if dy.min() > ymin:
            break
        update = (dy[:-1] < ymin) + (dy[1:] < ymin)
        y[1:-1] = (~update) * y[1:-1] + update * 0.5 * (y[2:] + y[:-2])
    sorted_dsd = {key: v for key, v in zip(sorted_dsd.keys(), y)}
    return sorted_dsd


def plot_accumulated_nbp(
    dsd: dict[str, xr.Dataset], ref: xr.Dataset, vmin: float, vmax: float
):
    FONT_SIZE = 16
    with mpl.rc_context({"font.size": FONT_SIZE}):
        fig = mpl.figure(figsize=(12.8, 5.8))
        ax = fig.add_subplot(1, 1, 1, position=[0.1, 0.1, 0.7, 0.85])
        data_range = vmax - vmin
        fig_height = fig.get_figheight()
        pad = 0.05 * data_range

        try:
            da = dset.get_interval_uncertainty(ref, "anbp")
        except NoUncertainty:
            da = None
        if da is not None:
            ax.fill_between(
                ref["year"],
                da.values[:, 0],
                da.values[:, 1],
                color="k",
                alpha=0.1,
                lw=0,
            )
        y_text = _space_labels(dsd, data_range / fig_height * FONT_SIZE / 50.0)
        for key, ds in dsd.items():
            ds["anbp"].plot(
                ax=ax,
                lw=2,
                color=ilamb3.conf["label_colors"].get(key, "k"),
            )
            ax.text(
                ds.year[-1] + 2,
                y_text[key],
                key,
                color=ilamb3.conf["label_colors"].get(key, "k"),
                va="center",
                size=FONT_SIZE,
            )
        ax.text(
            0.02,
            0.95,
            "Land Source",
            transform=ax.transAxes,
            size=FONT_SIZE,
            alpha=0.5,
            va="top",
        )
        ax.text(
            0.02, 0.05, "Land Sink", transform=ax.transAxes, size=FONT_SIZE, alpha=0.5
        )
        ax.set_ylabel("[Pg]")
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.spines[["top", "right"]].set_visible(False)
    return ax
