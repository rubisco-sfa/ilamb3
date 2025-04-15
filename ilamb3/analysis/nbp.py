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

import ilamb3.plot as ilplt
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingVariable, TemporalOverlapIssue


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
        # Default year to evaluate if none given
        if self.evaluation_year is None:
            self.evaluation_year = int(ref["time"][-1].dt.year)

        # Check that the comparison starts in the appropriate year
        if com["time"][0].dt.year > ref["time"][0].dt.year:
            raise TemporalOverlapIssue()
        tstart = min([t for t in com["time"] if t.dt.year == ref["time"][0].dt.year])
        tend = max([t for t in com["time"] if t.dt.year == self.evaluation_year])
        com = com.sel({"time": slice(tstart, tend)})

        # Fixes to data names and checks for required variables
        if "netAtmosLandCO2Flux" in com:
            com = com.rename_vars(dict(netAtmosLandCO2Flux="nbp"))
        if "nbp" not in com:
            msg = "`nbp` or `netAtmosLandCO2Flux` needs to be in the `com` Dataset."
            raise MissingVariable(msg)
        for var in ref:
            if not var.startswith("nbp"):
                ref = ref.drop_vars(var)
            else:
                ref[var] = -ref[var]
        com["nbp"] = -com["nbp"]

        # Integrate globally
        if dset.is_spatial(com):
            com["nbp"] = dset.integrate_space(com, "nbp")
        com.load()

        # Accumulate fluxes
        def _cumsum(ds):
            for var, da in ds.items():
                da = da.pint.quantify()
                if da.pint.units is None:
                    continue
                if not (1.0 * da.pint.units).check("[mass] / [time]"):
                    continue
                da = dset.accumulate_time(ds, var)
                da = dset.convert(da, "Pg")
                ds[var] = da
            return ds

        ref = _cumsum(ref)
        com = _cumsum(com)

        # Coarsen to annual
        ref = dset.coarsen_annual(ref)
        com = dset.coarsen_annual(com)

        # Trajectory score
        ref = ref.pint.quantify()
        out_units = f"{ref['nbp'].pint.units:~cf}"
        ref = ref.pint.dequantify()
        nbp_low = ref["nbp"]
        nbp_high = ref["nbp"]
        if "bounds" in ref["nbp"].attrs and ref["nbp"].attrs["bounds"] in ref:
            nbp_low = ref[ref["nbp"].attrs["bounds"]][:, 0]
            nbp_high = ref[ref["nbp"].attrs["bounds"]][:, 1]
        uncert = xr.DataArray(
            np.sqrt(
                (ref["nbp"] - nbp_low).values ** 2, (nbp_high - ref["nbp"]).values ** 2
            ),
            coords={"year": ref["year"]},
        )
        eps = (com["nbp"] - nbp_low).clip(0) + (nbp_high - com["nbp"]).clip(0)
        traj_score = np.exp(-eps / uncert)
        traj_score = float(traj_score.mean())

        # Difference score
        ref_val = float(ref["nbp"].sel(year=self.evaluation_year))
        try:
            com_val = float(com["nbp"].sel(year=self.evaluation_year))
        except KeyError:
            com_val = np.nan
        uncert_val = float(uncert.sel(year=self.evaluation_year))
        scale = -np.log(0.5) / 1  # outside of the uncertainty window? score < 50%
        diff_score = np.exp(-scale * np.abs(com_val - ref_val) / uncert_val)

        # Scores and scalars
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({self.evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
                    "value": ref_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"nbp({self.evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
                    "value": com_val,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "nbp",
                    "name": f"diff({self.evaluation_year})",
                    "type": "scalar",
                    "units": out_units,
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
        vmin = ref["nbp"].min()
        vmax = ref["nbp"].max()
        if "bounds" in ref["nbp"].attrs and ref["nbp"].attrs["bounds"] in ref:
            bnd_name = ref["nbp"].attrs["bounds"]
            vmin = ref[bnd_name].min()
            vmax = ref[bnd_name].max()
        vmin = float(vmin)
        vmax = float(vmax)
        com["Reference"] = ref
        return pd.DataFrame(
            [
                {
                    "name": "accumulation",
                    "title": "nbp_accumulation",
                    "region": None,
                    "source": None,
                    "axis": plot_accumulated_nbp(com, ref, vmin=vmin, vmax=vmax),
                }
            ]
        )


def _space_labels(
    dsd: dict[str, xr.Dataset], ymin: float, maxit: int = 10
) -> dict[str, float]:
    """
    Space out the model labels for the nbp plot using a modified Laplacian
    smoothing.
    """
    sorted_dsd = {key: float(ds.isel(year=-1)["nbp"]) for key, ds in dsd.items()}
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
        if "bounds" in ref["nbp"].attrs and ref["nbp"].attrs["bounds"] in ref:
            da = ref[ref["nbp"].attrs["bounds"]]
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
            ds["nbp"].plot(
                ax=ax,
                lw=2,
                color=ilplt.get_model_color(key),
            )
            ax.text(
                ds.year[-1] + 2,
                y_text[key],
                key,
                color=ilplt.get_model_color(key),
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
