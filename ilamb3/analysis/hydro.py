from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
import ilamb3.plot as plt
import ilamb3.regions as ilr
from ilamb3.analysis.base import ILAMBAnalysis


def metric_maps(
    da: xr.Dataset | xr.DataArray, varname: str | None = None
) -> xr.Dataset:
    """
    Return a dataset containing Deeksha's metrics request for the ILAMB Hydro project.

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        The dataset containing the variable
    varname: str, optional
        The name of the variable if a xr.Dataset is passed.

    Returns
    -------
    xr.Dataset
        The metrics derived from the input dataset.

    """
    if isinstance(da, xr.Dataset):
        assert varname is not None
        da = da[varname]
    out = {}

    # annual
    grp = da.groupby("time.year")
    out["annual_mean"] = grp.mean().mean(dim="year")
    out["annual_std"] = grp.mean().std(dim="year")
    amp = grp.max() - grp.min()
    out["amplitude_mean"] = amp.mean(dim="year")
    out["amplitude_std"] = amp.std(dim="year")

    # seasons
    grp = da.groupby("time.season")
    mean = grp.mean()
    out.update(
        {
            f"seasonal_mean_{str(s)}": mean.sel(season=s).drop_vars("season")
            for s in mean["season"].values
        }
    )
    std = grp.std("time")
    out.update(
        {
            f"seasonal_std_{str(s)}": std.sel(season=s).drop_vars("season")
            for s in std["season"].values
        }
    )

    # cycle
    cycle = da.groupby("time.month").mean()
    out["peak_timing"] = xr.where(
        ~cycle.isnull().all("month"),
        cycle.fillna(0).argmax(dim="month").astype(float),
        np.nan,
    )
    out["peak_timing"].attrs["units"] = "month"
    return xr.Dataset(out)


def scalarify(
    var: xr.DataArray | xr.Dataset, varname: str, region: str | None, mean: bool
) -> tuple[float, str]:
    """
    Integration/average the input dataarray/dataset to generate a scalar.
    """
    da = var
    if isinstance(var, xr.Dataset):
        da = var[varname]
    if dset.is_spatial(da):
        da = dset.integrate_space(
            da,
            varname,
            region=region,
            mean=mean,
        )
    elif dset.is_site(da):
        da = ilr.Regions().restrict_to_region(da, region)
        da = da.mean(dim=dset.get_dim_name(da, "site"))
    else:
        raise ValueError(f"Input is neither spatial nor site: {da}")
    da = da.pint.quantify()
    return float(da.pint.dequantify()), f"{da.pint.units:~cf}"


class hydro_analysis(ILAMBAnalysis):
    def __init__(self, required_variable: str, **kwargs: Any):
        self.req_variable = required_variable
        self.kwargs = kwargs

        # This analysis will split plots/scalars into sections as organized below
        self.sections = {
            "Annual": [
                "annual_mean",
                "annual_std",
            ],
            "Amplitude": [
                "amplitude_mean",
                "amplitude_std",
            ],
            "Seasonal DJF": [
                "seasonal_mean_DJF",
                "seasonal_std_DJF",
            ],
            "Seasonal MAM": [
                "seasonal_mean_MAM",
                "seasonal_std_MAM",
            ],
            "Seasonal JJA": [
                "seasonal_mean_JJA",
                "seasonal_std_JJA",
            ],
            "Seasonal SON": [
                "seasonal_mean_SON",
                "seasonal_std_SON",
            ],
            "Cycle": [
                "peak_timing",
            ],
        }

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.req_variable]

    def _get_analysis_section(self, varname: str) -> str:
        """Given the plot/variable, return from which section it belongs."""
        section = [s for s, vs in self.sections.items() if varname in vs]
        if not section:
            raise ValueError(f"Could not find {varname} in {self.sections}.")
        return section[0]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.
        """
        # Initialize
        varname = self.req_variable

        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname)

        # Run the hydro metrics
        ref = metric_maps(ref, varname)
        com = metric_maps(com, varname)

        # Create scalars
        df = []
        for source, ds in {"Reference": ref, "Comparison": com}.items():
            for vname, da in ds.items():
                for region in [None]:
                    scalar, unit = scalarify(da, vname, region=region, mean=True)
                    df.append(
                        [
                            source,
                            str(region),
                            self._get_analysis_section(vname),
                            " ".join(
                                [
                                    v.capitalize() if v.islower() else v
                                    for v in vname.split("_")
                                ]
                            ),
                            "scalar",
                            unit,
                            scalar,
                        ]
                    )

        # Convert to dataframe
        df = pd.DataFrame(
            df,
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
        df.attrs = self.__dict__.copy()
        return df, ref, com

    def plots(
        self,
        df: pd.DataFrame,
        ref: xr.Dataset,
        com: dict[str, xr.Dataset],
    ) -> pd.DataFrame:
        com["Reference"] = ref

        # Which plots are we handling in here?
        plots = list(chain(*[vs for _, vs in self.sections.items()]))
        com = {key: ds[plots] for key, ds in com.items()}

        # Setup plots
        df = plt.determine_plot_limits(com).set_index("name")
        df["title"] = [
            " ".join([v.capitalize() if v.islower() else v for v in plot.split("_")])
            for plot in df.index
        ]

        # Build up a dataframe of matplotlib axes
        axs = [
            {
                "name": plot,
                "title": df.loc[plot, "title"],
                "region": region,
                "source": source,
                "analysis": self._get_analysis_section(plot),
                "axis": (
                    plt.plot_map(
                        ds[plot],
                        region=region,
                        vmin=df.loc[plot, "low"],
                        vmax=df.loc[plot, "high"],
                        cmap="viridis",
                        title=source + " " + df.loc[plot, "title"],
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in plots
            for source, ds in com.items()
            for region in [None]
        ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs


if __name__ == "__main__":
    import ilamb3

    ref = xr.open_dataset(ilamb3.ilamb_catalog().fetch("tas/CRU4.02/tas.nc"))
    com = xr.open_dataset(
        "/home/nate/.esgf/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
    )
    anl = hydro_analysis("tas")
    ds_com = {}
    df, ds_ref, ds_com["Comparison"] = anl(ref, com)
    dfp = anl.plots(df, ds_ref, ds_com)
