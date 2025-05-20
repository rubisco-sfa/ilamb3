from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as ilplt
from ilamb3.analysis.base import ILAMBAnalysis


class accumulate_analysis(ILAMBAnalysis):
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
        bias = com_mean - ref_mean
        bias_score = np.exp(-np.abs((bias) / ref_mean))

        # create a dataframe of scalars
        df = pd.DataFrame(
            [
                {
                    "source": "Reference",
                    "region": "None",
                    "analysis": "accumulate",
                    "name": self.description,
                    "type": "scalar",
                    "units": unit,
                    "value": float(ref_mean),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "accumulate",
                    "name": self.description,
                    "type": "scalar",
                    "units": unit,
                    "value": float(com_mean),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "accumulate",
                    "name": "Bias",
                    "type": "scalar",
                    "units": unit,
                    "value": float(bias),
                },
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": "accumulate",
                    "name": "Bias Score",
                    "type": "score",
                    "units": "1",
                    "value": float(bias_score),
                },
            ]
        )

        # When writing intermediate outputs, make sure dims are a consistent name
        ref, com = cmp.rename_dims(ref, com)

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
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs
