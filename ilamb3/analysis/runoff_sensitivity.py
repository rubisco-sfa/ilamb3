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
import ilamb3.plot as ilp
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingVariable
from ilamb3.regions import Regions

# Map the variable names to something more aesthetic
NAME_CLEANUP = {
    "psens_obs": r"$\Delta Q / \Delta P$",
    "tsens_obs": r"$\Delta Q / \Delta T$",
}


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
        output_path: str | Path | None = None,
        **kwargs: Any,
    ):
        # Register basins in the ILAMB region system
        cat = ilamb3.ilamb_catalog()
        self.basins = list(
            set(Regions().add_netcdf(xr.open_dataset(cat.fetch("G-RUN/mrb_basins.nc"))))
        )
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
        rows = []

        # Create score maps
        ilamb_regions = Regions()
        for model, mod in com.items():
            for var in ["score_psens_obs", "score_tsens_obs"]:
                title = f"{model} {NAME_CLEANUP['_'.join(var.split('_')[1:])]} Score"
                da = ilamb_regions.region_scalars_to_map(
                    mod[var].mean(dim=["foc", "ens"]).to_pandas().to_dict()
                )
                da.attrs["units"] = 1
                ax = ilp.plot_map(da, vmin=0, vmax=1, cmap="plasma")
                rows.append(
                    {
                        "name": var.replace("_", ""),
                        "title": title,
                        "region": None,
                        "source": model,
                        "axis": False,
                    }
                )
                ax.set_title(title)
                fig = ax.get_figure()
                if self.output_path is None:
                    return fig
                else:
                    fig.savefig(
                        self.output_path / f"{model}_None_{var.replace('_', '')}.png"
                    )
                    plt.close()

        for basin in sorted(ref["basin"]):
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


def _get_group_membership(df: pd.DataFrame, model: str) -> str | None:
    """Return the group to which the model belongs if any."""
    if "group" not in df.columns:
        return None
    return df[df["source"] == model].dropna(subset="group")["group"].unique()[0]


def _basin_plots(
    df: pd.DataFrame,
    ref: xr.Dataset,
    com: dict[str, xr.Dataset],
    basin: str,
    model: str,
    output_path: Path | None,
):
    """Create a basin/model specific plot."""

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

    # Define the unique groups, if more than 1 models is available
    groups = (
        list(df["group"].dropna().unique()) if ("group" in df and len(com) > 1) else []
    )

    # Scores
    score = xr.concat(
        [da for var, da in com[model].items() if "score" in var], dim="score"
    ).mean(dim=["foc", "ens", "score"])

    # Create the plot
    fig, axs = plt.subplots(
        figsize=(6.4, 3.0),
        ncols=3,
        width_ratios=[1, 1, 2],
        tight_layout=True,
        dpi=ilamb3.conf["figure_dpi"],
    )
    # The first 2 panels...
    for i, vname in enumerate(["psens_obs", "tsens_obs"]):
        _errorbar(axs[i], ref[vname].sel(basin=basin).mean(dim=["foc", "ens"]), 1, "k")
        _errorbar(
            axs[i],
            com[model][vname].sel(basin=basin),
            2,
            ilamb3.conf["label_colors"].get(model, "k"),
        )
        for loc, group in enumerate(groups):
            grp = xr.concat(
                [
                    ds
                    for name, ds in com.items()
                    if name in df[df["group"] == group]["source"].unique()
                ],
                dim="model",
            ).mean(dim="model")
            _errorbar(
                axs[i],
                grp[vname].sel(basin=basin),
                3 + loc,
                ilamb3.conf["label_colors"].get(group, "k"),
            )
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

    # ... and the far right panel
    for model_name, cm in com.items():
        if model_name == model:
            continue
        label_name = _get_group_membership(df, model_name)
        if label_name is None:
            label_name = model_name
        _errorbarxy(
            axs[2],
            cm.sel(basin=basin),
            ilamb3.conf["label_colors"].get(label_name, "k"),
        )
    _errorbarxy(axs[2], ref.sel(basin=basin).mean(dim=["foc", "ens"]), "k")
    _errorbarxy(
        axs[2], com[model].sel(basin=basin), ilamb3.conf["label_colors"].get(model, "k")
    )
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].set_xlabel(
        f"{NAME_CLEANUP['psens_obs']} [{ref['psens_obs'].attrs['units']}]"
    )
    axs[2].set_ylabel(
        f"{NAME_CLEANUP['tsens_obs']} [{ref['tsens_obs'].attrs['units']}]"
    )
    fig.suptitle(
        f"{model} {str(basin.values)} ($S={float(score.sel(basin=basin)):.3f}$)",
        x=0.01,
        y=0.98,
        horizontalalignment="left",
    )
    if output_path is None:
        return fig
    else:
        fig.savefig(output_path / f"{model}_None_{str(basin.values)}.png")
        plt.close()
