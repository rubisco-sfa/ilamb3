from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as ilplt
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name


class timeseries_analysis(ILAMBAnalysis):
    def __init__(self, required_variable: str, **kwargs: Any):
        self.required_variable = required_variable
        self.description = kwargs.get("description", "Period Mean")
        self.kwargs = kwargs

    def required_variables(self):
        return [self.required_variable]

    def __call__(self, ref: xr.Dataset, com: xr.Dataset):
        varname = self.required_variable
        unit = ref[varname].attrs["units"]

        # make time series comparable
        ref, com = cmp.trim_time(ref, com)
        com = dset.convert(com, unit, varname=varname)

        # find period mean
        ref_mean = dset.integrate_time(ref, varname, mean=True)
        com_mean = dset.integrate_time(com, varname, mean=True)

        # score the difference as a bias
        bias_score = np.exp(-np.abs((com_mean - ref_mean) / ref_mean))

        # taylor score the time series
        ref_std = float(dset.std_time(ref, varname))
        com_std = float(dset.std_time(com, varname))
        norm_std = com_std / ref_std
        corr = float(np.corrcoef(ref[varname], com[varname].squeeze())[0, 1])
        taylor_score = 4 * (1 + corr) / ((norm_std + 1 / norm_std) ** 2 * 2)

        # create a dataframe of scalars
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": self.description,
                    "type": "scalar",
                    "units": unit,
                    "value": float(ref_mean),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": self.description,
                    "type": "scalar",
                    "units": unit,
                    "value": float(com_mean),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": "Bias Score",
                    "type": "score",
                    "units": "1",
                    "value": float(bias_score),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": "Normalized Standard Deviation",
                    "type": "scalar",
                    "units": "1",
                    "value": norm_std,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": "Correlation",
                    "type": "scalar",
                    "units": "1",
                    "value": corr,
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "timeseries",
                    "name": "Taylor Score",
                    "type": "score",
                    "units": "1",
                    "value": taylor_score,
                },
            ]
        )

        # When writing intermediate outputs, make sure dims are a consistent name
        ref, com = cmp.rename_dims(ref, com)

        return df, ref, com

    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        path.mkdir(parents=True, exist_ok=True)
        # Setup plot data
        com["Reference"] = ref
        lim = ilplt.determine_plot_limits(com, percent_pad=0).set_index("name")

        # Build up a dataframe of matplotlib axes
        plot = self.required_variable
        df_plots = []
        for source, ds in com.items():
            if plot not in ds or source == "Reference":
                continue
            row = {
                "name": plot,
                "title": lim.loc[plot, "title"],
                "region": None,
                "source": source,
                "path": get_plot_name(source, None, plot, path),
            }
            ax = ilplt.plot_curve(
                {source: ds} | {"Reference": ref},
                plot,
                vmin=lim.loc[plot, "low"]
                - 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
                vmax=lim.loc[plot, "high"]
                + 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
                title=f"{source} - {self.description}",
            )
            ax.get_figure().savefig(row["path"])
            plt.close()
            df_plots.append(row)

        # Plot all curves on one
        row = {
            "name": f"all{plot}",
            "title": self.description,
            "region": None,
            "source": None,
            "path": get_plot_name(None, None, f"all{plot}", path),
        }
        ax = ilplt.plot_curve(
            com,
            plot,
            vmin=lim.loc[plot, "low"]
            - 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
            vmax=lim.loc[plot, "high"]
            + 0.05 * (lim.loc[plot, "high"] - lim.loc[plot, "low"]),
            title=f"Combined {self.description}",
        )
        ax.get_figure().savefig(row["path"])
        plt.close()
        df_plots.append(row)

        # Add the Taylor diagram
        row = {
            "name": "taylor",
            "title": "Taylor Diagram",
            "region": None,
            "source": None,
            "path": get_plot_name(None, None, "taylor", path),
        }
        ax = ilplt.plot_taylor_diagram(df)
        ax.get_figure().savefig(row["path"])
        plt.close()
        df_plots.append(row)

        axs = pd.DataFrame(df_plots)
        return axs
