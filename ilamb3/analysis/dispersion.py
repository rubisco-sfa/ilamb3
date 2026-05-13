"""
The ILAMB dispersion methodology.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""


from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.plot as ilp
import ilamb3.regions as ilr
from ilamb3 import compare as cmp
from ilamb3 import dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis, get_plot_name, scalarify
from ilamb3.exceptions import NoDatabaseEntry, NoUncertainty

class dispersion_analysis(ILAMBAnalysis):
    """
    The ILAMB bias methodology.

    Parameters
    ----------
    required_variable : str
        The name of the variable to be used in this analysis.
    variable_cmap : str
        The colormap to use in plots of the comparison variable, optional.
    method : str
        The name of the scoring methodology to use, either `Collier2018` or
        `RegionalQuantiles`.
    regions : list
        A list of region labels over which to apply the analysis.
    use_uncertainty : bool
        Enable to utilize uncertainty information from the reference product if
        present.
    spatial_sum : bool
        Enable to report a spatial sum in the period mean as opposed to a
        spatial mean. This is often preferred in carbon variables where the
        total global carbon is of interest.
    mass_weighting : bool
        Enable to weight the score map integrals by the temporal mean of the
        reference dataset.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        required_variable: str,
        score_basis: Literal["series", "cycle"] = "series",
        regions: list[str | None] = [None],
        use_uncertainty: bool = True,
        table_unit: str | None = None,
        plot_unit: str | None = None,
        **kwargs: Any,
    ):
        self.req_variable = required_variable
        self.regions = regions
        self.use_uncertainty = use_uncertainty
        self.table_unit = table_unit
        self.plot_unit = plot_unit
        self.kwargs = kwargs
    )

    def required_variables(self) -> list[str]:
        """
        Return the list of variables required for this analysis.

        Returns
        -------
        list
            The variable names used in this analysis.
        """
        return [self.req_variable]

    def __call__(
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
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
        # Initialize
        ANALYSIS_NAME = "Dispersion"
        var_name = self.req_variable

        if not (dset.is_temporal(ref[varname]) and dset.is_temporal(com[varname])):
            raise AnalysisNotAppropriate()


        # Make the variables comparable and force loading into memory
        ref, com = cmp.make_comparable(ref, com, varname, **self.kwargs)

        # Is the time series long enough for this to be meaningful?
        # 24 is enough?
        if len(ref[dset.get_dim_name(ref, "time")]) < 24:
            raise AnalysisNotAppropriate()
        if len(com[dset.get_dim_name(com, "time")]) < 24:
            raise AnalysisNotAppropriate()

        results = {}
        for name, ds in {"ref": ref, "com": com}.items():
            data = ds[var_name]
            quantiles = ds[f"{var_name}_quantile"]

            reduced_dims = quantiles.attrs["reduced_dims"]

            # Quantiles
            q25 = quantiles.sel(quantile=0.25)
            q50 = quantiles.sel(quantile=0.50)
            q75 = quantiles.sel(quantile=0.75)

            # Statistics over the same reduced dimensions
            mean = data.mean(dim=reduced_dims)
            stdev = data.std(dim=reduced_dims)

            # Interquartile range
            iqr = q75 - q25

            # skewness coefficient
            skewness = (mean - q50) / stdev

            # Kurtosis
            kurtosis = (
                ((data - mean) ** 4).mean(dim=reduced_dims)
                / (stdev ** 4)
            )
 
            results[name] = {
                f"iqr": iqr,
                f"skewness": skewness,
                f"kurtosis": kurtosis,
            }
        ds_ref = results["ref"]
        ds_com = results["com"]

        for name, value in ds_ref.items:
            ds_com[f"{name}_bias"] = ds_com[name] - value


        df = []
        for region in self.regions:
            for name in ds_ref.keys():
                val, unit = scalarify(ds_com[f"{name}_bias"], name, region, mean=True, unit=self.plot_unit)
                df += [
                    {
                        "source": "Comparison",
                        "region": str(region),
                        "analysis": ANALYSIS_NAME,
                        "name": name,
                        "type": "scalar",
                        "units": unit,
                        "value": val,
                    },
                ]

        # spatial distribution

        for name, xref in ds_ref.items():
            xcom = ds_com[name]

            xref, xcom = cmp.rename_dims(*cmp.nest_spatial_grids(xref, xcom))

            # Compute scalars over all regions
            ilamb_regions = ilr.Regions()
            for region in self.regions:
                # Get regional versions
                rref = ilamb_regions.restrict_to_region(xref, region)
                rcom = ilamb_regions.restrict_to_region(xcom, region)

                # Spatial standard deviation
                ref_std = float(rref.std())
                com_std = float(rcom.std())
                if np.allclose(ref_std, 0):
                    # There is no spatial variance for this region and we should skip
                    continue
                norm_std = com_std / ref_std

                # Correlation
                isnan = rref.isnull() | rcom.isnull()
                rref = np.ma.masked_invalid(
                    xr.where(isnan, np.nan, rref).values
                ).compressed()
                rcom = np.ma.masked_invalid(
                    xr.where(isnan, np.nan, rcom).values
                ).compressed()
                corr = float(np.corrcoef(rref, rcom)[0, 1])
                taylor_score = 4 * (1 + corr) / ((norm_std + 1 / norm_std) ** 2 * 2)

                df += [
                    {
                        "source": "Comparison",
                        "region": str(region),
                        "analysis": ANALYSIS_NAME,
                        "name": f"Normalized Standard Deviation ({name})",
                        "type": "scalar",
                        "units": "1",
                        "value": norm_std,
                    },
                    {
                        "source": "Comparison",
                        "region": str(region),
                        "analysis": ANALYSIS_NAME,
                        "name": f"Correlation ({name})",
                        "type": "scalar",
                        "units": "1",
                        "value": corr,
                    },
                    {
                        "source": "Comparison",
                        "region": str(region),
                        "analysis": ANALYSIS_NAME,
                        "name": f"Spatial Distribution Score ({name})",
                        "type": "score",
                        "units": "1",
                        "value": taylor_score,
                    },
                ]

        df = pd.DataFrame(df)

        ds_ref = xr.merge([ds_ref], compat="override")
        ds_com = xr.merge([ds_com], compat="override")

        return df, ds_ref, ds_com


    def plots(
        self, df: pd.DataFrame, ref: xr.Dataset, com: dict[str, xr.Dataset], path: Path
    ) -> pd.DataFrame:

        if "Dispersion" not in df["analysis"].unique():
            return pd.DataFrame()

        path.mkdir(parents=True, exist_ok=True)

        regions = [None if r == "None" else r for r in df["region"].unique()]

        com["Reference"] = ref
        for source, ds in com.items():
            for plot in ["iql_bias", "skewness_bias", "kurtosis_bias"]:
                if plot in ds:
                    com[source][plot] = dset.convert(
                        ds[plot], 
                        ds[plot].attrs["units"] if self.plot_unit is None else self.plot_unit,
                    )

        # Setup a dataframe with the information we will need for each plot in
        # this analysis.
        df_meta = pd.DataFrame(
            [
                {"name": "iql_bias", "cmap": "Oranges", "title": "RMSE"},
                {"name": "skewness_bias", "cmap": "Oranges", "title": "Skewness"},
                {"name": "kurtosis_bias", "cmap": "Oranges", "title": "Kurtosis"},
            ]
        ).set_index("name")

        df_limits = ilp.determine_plot_limits(com)
        df = pd.merge(df_meta, df_limits, left_index=True, right_index=True)
        df["analysis"] = "Dispersion"

        # Create each plot for each source if present in the dataset
        df_plots = []
        for plot, row in df.iterrows():
            for source, ds in com.items():
                if plot not in ds:
                    continue
                # Maps are plot over each region
                for region in regions:
                    out = row.to_dict()
                    out["name"] = plot
                    out["source"] = source
                    out["region"] = region
                    out["path"] = get_plot_name(source, region, plot, path)
                    ax = ilp.plot_map(
                        ds[plot],
                        region=region,
                        vmin=row["low"],
                        vmax=row["high"],
                        cmap=row["cmap"],
                        title=f"{source} {row['title']}",
                    )
                    ax.get_figure().savefig(out["path"])
                    plt.close()
                    df_plots.append(out)

        df_plots = pd.DataFrame(df_plots)
        return df_plots
