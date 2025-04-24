from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as ilplt
from ilamb3.analysis.base import ILAMBAnalysis


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

        return df, ref, com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        # Setup plot data
        varname = self.required_variable
        com["Reference"] = ref
        lim = ilplt.determine_plot_limits(com, percent_pad=0).set_index("name")
        lim.loc[varname, ["cmap", "title"]] = [None, self.description]

        # Build up a dataframe of matplotlib axes
        axs = [
            {
                "name": plot,
                "title": lim.loc[plot, "title"],
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
                        title=source + " - " + lim.loc[plot, "title"],
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in [varname]
            for source, ds in com.items()
            if source != "Reference"
        ]

        # Plot all curves on one
        axs += [
            {
                "name": f"all{varname}",
                "title": self.description,
                "region": None,
                "source": None,
                "axis": (
                    ilplt.plot_curve(
                        com,
                        varname,
                        vmin=lim.loc[varname, "low"]
                        - 0.05 * (lim.loc[varname, "high"] - lim.loc[varname, "low"]),
                        vmax=lim.loc[varname, "high"]
                        + 0.05 * (lim.loc[varname, "high"] - lim.loc[varname, "low"]),
                        title=f"Combined {self.description}",
                    )
                ),
            }
        ]

        # Add the Taylor diagram
        axs += [
            {
                "name": "taylor",
                "title": "Taylor Diagram",
                "region": None,
                "source": None,
                "axis": ilplt.plot_taylor_diagram(df),
            }
        ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
