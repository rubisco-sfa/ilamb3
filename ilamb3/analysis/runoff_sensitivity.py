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

import ilamb3
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
        output_path: str | Path | None = None,
        **kwargs: Any,
    ):
        # Register basins in the ILAMB region system
        assert basin_source is not None
        ilamb_regions = Regions()
        self.basins = ilamb_regions.add_netcdf(basin_source)
        self.output_path = Path(output_path) if output_path is not None else None

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

        rows = []
        for basin in sorted(ref["basin"]):
            if rows:
                continue
            for model in com:
                _basin_plots(df, ref, com, basin, model, self.output_path)
                rows.append(
                    {
                        "name": str(basin.values),
                        "title": f"Basin plot for {str(basin.values)}",
                        "region": None,
                        "source": model,
                        "axis": False,
                    }
                )
        return pd.DataFrame(rows)


def _basin_plots(
    df: pd.DataFrame,
    ref: xr.Dataset,
    com: dict[str, xr.Dataset],
    basin: str,
    model: str,
    output_path: Path | None,
):
    def _errorbar(ax, mod: xr.DataArray, loc: int, color: str):
        ax.errorbar(
            [loc],
            mod.isel(sens_type=0).values,
            yerr=np.abs(
                mod.isel(sens_type=0) - mod.isel(sens_type=[1, 2])
            ).values.reshape((-1, 1)),
            color=color,
            marker="o",
            markersize=2,
            capsize=2,
        )

    def _errorbarxy(ax, mod: xr.DataArray, color: str):
        ax.errorbar(
            mod["psens_obs"].isel(sens_type=0).values,
            mod["tsens_obs"].isel(sens_type=0).values,
            xerr=np.abs(
                mod["psens_obs"].isel(sens_type=0)
                - mod["psens_obs"].isel(sens_type=[1, 2])
            ).values.reshape((-1, 1)),
            yerr=np.abs(
                mod["tsens_obs"].isel(sens_type=0)
                - mod["tsens_obs"].isel(sens_type=[1, 2])
            ).values.reshape((-1, 1)),
            color=color,
            marker="o",
            markersize=2,
            capsize=2,
        )

    NAME_CLEANUP = {
        "psens_obs": r"$\Delta Q / \Delta P$",
        "tsens_obs": r"$\Delta Q / \Delta T$",
    }
    # Define the unique groups, if more than 1 models is available
    groups = (
        list(df["group"].dropna().unique()) if ("group" in df and len(com) > 1) else []
    )

    fig, axs = plt.subplots(
        figsize=(6.4, 2.0),
        ncols=3,
        width_ratios=[1, 1, 2],
        tight_layout=True,
        dpi=ilamb3.conf["figure_dpi"],
    )
    for i, vname in enumerate(["psens_obs", "tsens_obs"]):
        _errorbar(axs[i], ref[vname].sel(basin=basin).mean(dim=["foc", "ens"]), 1, "k")
        _errorbar(axs[i], com[model][vname].sel(basin=basin), 2, "r")
        for loc, group in enumerate(groups):
            grp = xr.concat(
                [
                    ds
                    for name, ds in com.items()
                    if name in df[df["group"] == group]["source"].unique()
                ],
                dim="model",
            ).mean(dim="model")
            _errorbar(axs[i], grp[vname].sel(basin=basin), 3 + loc, "b")
        axs[i].set_ylabel(f"{NAME_CLEANUP[vname]} [{ref[vname].attrs['units']}]")
        axs[i].set_xticks(
            range(1, (3 + len(groups))),
            [
                "Reference",
                model,
            ]
            + groups,
        )
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].tick_params(axis="x", labelrotation=45)

    # Scatter plots for all models
    _errorbarxy(axs[2], ref.sel(basin=basin).mean(dim=["foc", "ens"]), color="k")
    for _, cm in com.items():
        _errorbarxy(axs[2], cm.sel(basin=basin), color="r")
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].set_xlabel(
        f"{NAME_CLEANUP['psens_obs']} [{ref['psens_obs'].attrs['units']}]"
    )
    axs[2].set_ylabel(
        f"{NAME_CLEANUP['tsens_obs']} [{ref['tsens_obs'].attrs['units']}]"
    )
    fig.suptitle(f"{str(basin.values)}")
    if output_path is None:
        return fig
    else:
        fig.savefig(output_path / f"{model}_None_{str(basin.values)}.png")
        plt.close()
