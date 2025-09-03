"""
Runoff sensitivity to temperature and precipitation per river basin.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingVariable
from ilamb3.regions import Regions


def _nc_obs_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert Hanjun's dataset into a dataframe.
    """
    dfs = [
        (
            xr.DataArray(
                ds[dsvar].values,
                dims=("for", "ens", "basin", "sens_type"),
                coords={
                    "for": ds["foc_names"].values,
                    "basin": ds["basin_names"].values,
                    "sens_type": ds["sens_type_names"].values,
                },
                name="value",
            )
            .to_dataframe()
            .reset_index()
            .set_index(["for", "ens", "basin"])
            .pivot_table(
                columns="sens_type", values="value", index=["for", "ens", "basin"]
            )
            .rename(
                columns={
                    b"lower bound (95% confidence interval of reg. coeff.)": f"{var} Low",
                    b"upper bound (95% confidence interval of reg. coeff.)": f"{var} High",
                    b"sensitivity value": f"{var} Sensitivity",
                }
            )
        )
        for dsvar, var in [["psens_obs", "pr"], ["tsens_obs", "tas"]]
    ]
    df = pd.merge(*dfs, left_index=True, right_index=True)
    return df


def _fix_ds_dims(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(
        sens_type=ds["sens_type_names"].astype(str),
        foc=ds["foc_names"].astype(str),
        basin=ds["basin_names"].astype(str),
    ).drop_vars(["basin_names", "sens_type_names", "foc_names"])
    ds = ds.drop_duplicates(dim="basin")  # because there are 2 COLORADO and FITZROY's
    return ds


class runoff_sensitivity_analysis(ILAMBAnalysis):
    """
    Runoff sensitivity to temperature and precipitation per river basin.

    Parameters
    ----------
    basin_source : str
        The source file for the basins to use in the analysis.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        basin_source: str | Path | None = None,
        **kwargs: Any,
    ):
        # Register basins in the ILAMB region system
        assert basin_source is not None
        ilamb_regions = Regions()
        self.basins = ilamb_regions.add_netcdf(basin_source)

    def required_variables(self) -> list[str]:
        """
        Return the variable names required in this analysis.
        """
        return ["psens_obs", "tsens_obs"]

    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset, basins: list[str] | None = None
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """"""
        # Ensure we have the variables we need for this confrontation
        missing = set(self.required_variables()) - set(com)
        if missing:
            raise MissingVariable(f"Comparison dataset is lacking variables: {missing}")
        basins = self.basins if basins is None else basins

        # These are small so lets load them
        ref = _fix_ds_dims(ref)
        ref.load()
        com.load()

        # Compute the bias
        bias = com - ref

        # A per basin/forcing/ensemble scalar estimate of uncertainty
        uncert = np.sqrt(
            (ref.isel(sens_type=0) - ref.isel(sens_type=1)) ** 2
            + (ref.isel(sens_type=2) - ref.isel(sens_type=0)) ** 2
        )

        # A score which discounts error in the uncertainty window, not
        # normalized in this case because sensitivities tend to be of consistent
        # order of magnitude?
        score = np.exp(-((np.abs(bias.isel(sens_type=0)) - uncert).clip(0)))

        # Drop scores and scalars into the dataframe
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Mean Temperature Sensitivity",
                    "type": "scalar",
                    "units": ref["tsens_obs"].attrs["units"],
                    "value": float(ref["tsens_obs"].mean()),
                },
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Mean Precipitation Sensitivity",
                    "type": "scalar",
                    "units": ref["psens_obs"].attrs["units"],
                    "value": float(ref["psens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Mean Temperature Sensitivity",
                    "type": "scalar",
                    "units": ref["tsens_obs"].attrs["units"],
                    "value": float(com["tsens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Mean Precipitation Sensitivity",
                    "type": "scalar",
                    "units": ref["psens_obs"].attrs["units"],
                    "value": float(com["psens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Bias Temperature Sensitivity",
                    "type": "scalar",
                    "units": ref["tsens_obs"].attrs["units"],
                    "value": float(bias["tsens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Bias Precipitation Sensitivity",
                    "type": "scalar",
                    "units": ref["psens_obs"].attrs["units"],
                    "value": float(bias["psens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Score Temperature Sensitivity",
                    "type": "score",
                    "units": ref["tsens_obs"].attrs["units"],
                    "value": float(score["tsens_obs"].mean()),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "Runoff Sensitivity",
                    "name": "Score Precipitation Sensitivity",
                    "type": "score",
                    "units": ref["psens_obs"].attrs["units"],
                    "value": float(score["psens_obs"].mean()),
                },
            ]
        )

        # Load these values into the datasets for use later in plotting
        for var, da in uncert.items():
            ref[f"uncert_{var}"] = da
        for var, da in bias.items():
            com[f"bias_{var}"] = da
        for var, da in score.items():
            com[f"score_{var}"] = da

        return df, ref, com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        """
        Return figures of the reference and comparison data.
        """
        # basin time series
        # tanked bar plot, how does a given model
        # bottom left

        # The first plot is meant to help us understand if our scoring methodology makes sense.
        for model, ds in com.items():
            for basin in ref["basin"]:
                debug_plot(ref, ds, "psens_obs", basin)

        return pd.DataFrame(
            [
                {
                    "name": "",
                    "title": "",
                    "region": None,
                    "source": None,
                    "axis": False,
                }
            ]
        )


def debug_plot(ref: xr.Dataset, com: xr.Dataset, vname: str, basin: str):
    import matplotlib.patches as mpatches

    score = com[f"score_{vname}"].sel(basin=basin).mean().values
    fig, ax = plt.subplots(tight_layout=True)
    ind = np.linspace(0, 1, 100).reshape((4, 25))
    for i, c in zip(range(3), ["r", "g", "b"]):
        ax.plot(ind, ref[vname].sel(basin=basin).isel(sens_type=i), ".", color=c)
    ax.plot([1.123] * 3, com[vname].sel(basin=basin).values, "^", color="k")
    ax.set_xticks([0.125, 0.375, 0.625, 0.875], ref["foc"].astype(str).values)
    ax.set_ylabel(vname)
    ax.set_title(
        f"{vname} | {ref['basin'].sel(basin=basin).astype(str).values} | {score:.2f}"
    )
    fig.legend(
        handles=[
            mpatches.Patch(color="b", label="Upper"),
            mpatches.Patch(color="r", label="Sensitivity"),
            mpatches.Patch(color="g", label="Lower"),
        ],
        loc=2,
    )
    name = f"{score:.4f}"[2:]
    fig.savefig(f"_build_runoff/{name}.png")
    plt.close()
