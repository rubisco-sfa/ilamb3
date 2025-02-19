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


def score_difference(ref: xr.Dataset, com: xr.Dataset) -> xr.Dataset:
    # Compute differences and scores
    ref_, com_ = cmp.nest_spatial_grids(ref, com)
    diff = com_ - ref_
    diff = diff.rename_vars({v: f"{v}_difference" for v in diff})
    # Add scores to the means that also have std's
    diff = diff.merge(
        {
            v.replace("_difference", "_score"): np.exp(
                -np.abs(diff[v])
                / ref_[v.replace("_mean", "_std").replace("_difference", "")]
            )
            for v in diff
            if "mean" in v and v.replace("_mean_", "_std_") in diff
        }
    )
    # Rename the lat dimension for merging with the comparison on return
    lat_name = dset.get_dim_name(diff, "lat")
    lon_name = dset.get_dim_name(diff, "lon")
    com = com.merge(diff.rename({lat_name: f"{lat_name}_", lon_name: f"{lon_name}_"}))
    return com


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
    def __init__(
        self, required_variable: str, regions: list[str | None] = [None], **kwargs: Any
    ):
        self.req_variable = required_variable
        self.regions = regions
        self.kwargs = kwargs

        # This analysis will split plots/scalars into sections as organized below
        self.sections = {
            "Annual": [
                "annual_mean",
                "annual_mean_difference",
                "annual_mean_score",
                "annual_std",
                "annual_std_difference",
            ],
            "Amplitude": [
                "amplitude_mean",
                "amplitude_mean_difference",
                "amplitude_mean_score",
                "amplitude_std",
                "amplitude_std_difference",
            ],
            "Seasonal DJF": [
                "seasonal_mean_DJF",
                "seasonal_mean_DJF_difference",
                "seasonal_mean_DJF_score",
                "seasonal_std_DJF",
                "seasonal_std_DJF_difference",
            ],
            "Seasonal MAM": [
                "seasonal_mean_MAM",
                "seasonal_mean_MAM_difference",
                "seasonal_mean_MAM_score",
                "seasonal_std_MAM",
                "seasonal_std_MAM_difference",
            ],
            "Seasonal JJA": [
                "seasonal_mean_JJA",
                "seasonal_mean_JJA_difference",
                "seasonal_mean_JJA_score",
                "seasonal_std_JJA",
                "seasonal_std_JJA_difference",
            ],
            "Seasonal SON": [
                "seasonal_mean_SON",
                "seasonal_mean_SON_difference",
                "seasonal_mean_SON_score",
                "seasonal_std_SON",
                "seasonal_std_SON_difference",
            ],
            "Cycle": [
                "peak_timing",
                "peak_timing_difference",
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
        com = score_difference(ref, com)

        # Create scalars
        df = []
        for source, ds in {"Reference": ref, "Comparison": com}.items():
            for vname, da in ds.items():
                for region in self.regions:
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
                            "score" if "score" in vname else "scalar",
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

        def _choose_cmap(plot_name):
            if "score" in plot_name:
                return "plasma"
            if "difference" in plot_name:
                return "bwr"
            return "viridis"

        # Which plots are we handling in here?
        plots = list(chain(*[vs for _, vs in self.sections.items()]))
        com = {key: ds[set(ds) & set(plots)] for key, ds in com.items()}

        # Setup plots
        df = plt.determine_plot_limits(com, symmetrize=["difference"]).set_index("name")
        df["title"] = [
            " ".join([v.capitalize() if v.islower() else v for v in plot.split("_")])
            for plot in df.index
        ]
        df["cmap"] = df.index.map(_choose_cmap)

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
                        cmap=df.loc[plot, "cmap"],
                        title=source + " " + df.loc[plot, "title"],
                    )
                    if plot in ds
                    else pd.NA
                ),
            }
            for plot in plots
            for source, ds in com.items()
            for region in self.regions
        ]
        axs = pd.DataFrame(axs).dropna(subset=["axis"])
        return axs


if __name__ == "__main__":
    """
    [x] Separate it out into sections (Annual, Amplitude, Season, Cycle) maybe?
    [ ] Fix the timing plot to read months on the labels
    [x] Compute differences in model-reference
    [x] Add more models, especially the models you want to show
    [x] Generate synthesis scores? I would suggest for each aspect we compute a relative error as eps = (reference_mean - model_mean) / reference_std and then score = exp(-|eps|) but we can chat.
    [x] Use the US regions we developed
    """
    import ilamb3

    ref = xr.open_dataset(ilamb3.ilamb_catalog().fetch("tas/CRU4.02/tas.nc"))
    com = xr.open_dataset(
        "/home/nate/.esgf/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
    )
    anl = hydro_analysis("tas")
    ds_com = {}
    df, ds_ref, ds_com["Comparison"] = anl(ref, com)
    dfp = anl.plots(df, ds_ref, ds_com)
