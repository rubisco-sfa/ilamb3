"""
Runoff sensitivity to temperature and precipitation per river basin.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr

import ilamb3
import ilamb3.dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingRegion, MissingVariable
from ilamb3.regions import Regions


def compute_runoff_sensitivity(
    ds: xr.Dataset,
    basins: list[str],
    window_size: int = 9,
    use_cross: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Compute the runoff sensitivites in the basins provided.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing `mrro` (runoff), `pr` (precip), and `tas` (temperature).
    basins : list[str]
        The basins in which to compute sensitivities. These should refer to labels that
        have been registered in the ilamb3.regions.Regions class.
    window_size : int
        The number of water years to use in the windowed average.
    use_cross : bool
        Enable to use the cross term `1 + pr + tas + pr * tas` sensitivities, otherwise
        use those from a `1 + pr + tas` least squares fit.
    quiet : bool, default False
        Enable to suppress the progress bar.

    Returns
    -------
    pd.DataFrame
        A dataframe with the sensitivities over each basin.
    """
    # Check that the required variables are present
    required_vars = ["mrro", "tas", "pr"]
    missing = set(required_vars) - set(ds)
    if missing:
        raise MissingVariable(f"Input dataset is lacking variables: {missing}")

    # Check that the basins are in fact registed in the ilamb system
    ilamb_regions = Regions()
    missing = set(basins) - set(ilamb_regions.regions)
    if missing:
        raise MissingRegion(f"Input basins are not registered as regions: {missing}")

    # Associate cell/time measures with the dataset if not present
    ds = dset.shift_lon(ds)
    space = [dset.get_dim_name(ds, "lat"), dset.get_dim_name(ds, "lon")]
    if "cell_measures" not in ds:
        ds["cell_measures"] = dset.compute_cell_measures(ds).pint.dequantify()
    if "time_measures" not in ds:
        ds["time_measures"] = dset.compute_time_measures(ds).pint.dequantify()
        if "bounds" in ds[dset.get_dim_name(ds, "time")].attrs:
            bnds = ds[dset.get_dim_name(ds, "time")].attrs["bounds"]
            if bnds in ds:
                ds = ds.drop_vars(bnds)

    # Compute the water year for the time series (beginning of year is Oct)
    t = ds[dset.get_dim_name(ds, "time")]
    ds["water_year"] = xr.where(t.dt.month > 9, t.dt.year, t.dt.year - 1)
    ds = ds.set_coords("water_year")

    # Averages per water year. This is weighted by the number of days per month in each
    # water year. The `groupby` does not work with the `weighted` accessor and so we
    # take the weighted mean by hand.
    ds = (
        (ds * ds["time_measures"]).groupby("water_year").sum()
        / ds["time_measures"].groupby("water_year").sum()
    ).drop_vars("time_measures")

    # Loop over basins and build up the dataframe of sensitivities
    df = []
    for basin in basins:
        # Compute the regional mean values per basin
        dsb = ilamb_regions.restrict_to_region(ds, basin)
        msr = dsb["cell_measures"].fillna(0)
        dsb = dsb.drop_vars([v for v in ds.data_vars if v not in required_vars])
        dsb = dsb.weighted(msr).mean(dim=space)

        # Take a windowed (decadal) average over the water years
        mean = dsb.rolling(water_year=window_size, center=True).mean()

        # Compute the anomalies
        anomaly = mean - mean.mean()
        anomaly["mrro"] = anomaly["mrro"] / mean["mrro"].mean() * 100.0
        anomaly["pr"] = anomaly["pr"] / mean["pr"].mean() * 100.0

        # Fit a linear model (with and without the cross term) and compute stats
        model = smf.ols("mrro ~ tas + pr", data=anomaly.to_dataframe()).fit()
        model_cross = smf.ols("mrro ~ tas * pr", data=anomaly.to_dataframe()).fit()
        focus = model_cross if use_cross else model
        out = {
            "basin": basin,
            "tas Sensitivity": focus.params.to_dict()["tas"],
            "tas Low": focus.conf_int()[0].to_dict()["tas"],
            "tas High": focus.conf_int()[1].to_dict()["tas"],
            "pr Sensitivity": focus.params.to_dict()["pr"],
            "pr Low": focus.conf_int()[0].to_dict()["pr"],
            "pr High": focus.conf_int()[1].to_dict()["pr"],
            "R2": model.rsquared,
            "R2 Cross": model_cross.rsquared,
            "Cond": focus.condition_number,
        }
        df.append(out)
    df = pd.DataFrame(df).set_index("basin")
    return df


class runoff_sensitivity_analysis(ILAMBAnalysis):
    """
    Runoff sensitivity to temperature and precipitation per river basin.

    Parameters
    ----------
    basin_source : str
        The source file for the basins to use in the analysis.
    sensitivity_frame : pd.DataFrame, optional
        The reference sensitivities. If not provided, we will compute them from the
        sources provided.
    mrro_source : str
        The source of runoff (mrro), required if no sensitivities are provided.
    tas_source : str
        The source of temperature (tas), required if no sensitivities are provided.
    pr_source : str
        The source of precipitation (pr), required if no sensitivities are provided.
    timestamp_start : str
        The initial time over which we compute comparison sensitivities.
    timestamp_end : str
        The final time over which we compute comparison sensitivities.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

    def __init__(
        self,
        basin_source: str | Path,
        sensitivity_frame: None | str | pd.DataFrame = None,
        mrro_source: str | Path = None,
        tas_source: str | Path = None,
        pr_source: str | Path = None,
        timestamp_start: str = "1940-10-01",
        timestamp_end: str = "2014-10-01",
    ):  # numpydoc ignore=GL08
        # Consistency checks
        cat = ilamb3.ilamb_catalog()
        self.sources = []
        if sensitivity_frame is None:
            if [src for src in [mrro_source, tas_source, pr_source] if src is None]:
                raise ValueError(
                    f"If sensitivities are not provided, you must give a source for each variable. Found {mrro_source=} {tas_source=} {pr_source=}"
                )
            for src in [mrro_source, tas_source, pr_source]:
                if Path(src).is_file():
                    self.sources.append(xr.open_dataset(src))
                elif src in cat:
                    self.sources.append(cat[src].read())
                else:
                    raise ValueError(f"Could not load source file {src=}")

        # Load/store the sensitivities if present
        self.sensitivity_frame = sensitivity_frame
        if sensitivity_frame is not None:
            read = Path(sensitivity_frame).suffix.replace(".", "read_")
            if read not in dir(pd):
                raise OSError(f"Cannot read sensitivty file: {sensitivity_frame}")
            read = pd.__dict__[read]
            self.sensitivity_frame = read(sensitivity_frame)

        # Register basins in the ILAMB region system
        ilamb_regions = Regions()
        if Path(basin_source).is_file():
            self.basins = ilamb_regions.add_netcdf(basin_source)
        elif isinstance(basin_source, str) and basin_source in cat:
            self.basins = ilamb_regions(cat[basin_source].read())
        else:
            raise ValueError(f"Unable to ingest {basin_source=} as ILAMB regions.")

        # Record timestamp variables
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end

    def required_variables(self) -> list[str]:
        """
        Return the variable names required in this analysis.

        Returns
        -------
        list
            A list of the required variables, here always [pr, tas, mrro].
        """
        return ["pr", "tas", "mrro"]

    def __call__(
        self, ref: xr.Dataset, com: xr.Dataset, basins: list[str] | None = None
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            Not used in this analysis object.
        com : xr.Dataset
            The comparison dataset.
        basins : list[str], optional
            The basins in which to compute sensitivities. By default will use all those
            present in the provided `basin_source` but a subset can be provided here if
            desired.

        Returns
        -------
        pd.DataFrame
            A dataframe with scalar and score information from the comparison.
        xr.Dataset
            A dataset containing reference grided information from the comparison.
        xr.Dataset
            A dataset containing comparison grided information from the comparison.
        """
        # Ensure we have the variables we need for this confrontation
        missing = set(self.required_variables()) - set(com)
        if missing:
            raise MissingVariable(f"Comparison dataset is lacking variables: {missing}")
        basins = self.basins if basins is None else basins

        # Optionally compute reference sensitivities if not given
        if self.sensitivity_frame is None:
            ref = xr.merge(self.sources)
            self.sensitivity_frame = compute_runoff_sensitivity(ref, basins)
        df_ref = self.sensitivity_frame

        # Compute comparisons sensitivities
        com = com.sel({"time": slice(self.timestamp_start, self.timestamp_end)})
        df_com = compute_runoff_sensitivity(com, basins)

        # Paint by the numbers to create maps of scalars
        ilamb_regions = Regions()

        def _df_to_ds(df):  # numpydoc ignore=GL08
            ds = xr.Dataset(
                {
                    key: ilamb_regions.region_scalars_to_map(df[key].to_dict())
                    for key in df.columns
                }
            )
            return ds

        ds_ref = _df_to_ds(df_ref)
        ds_com = _df_to_ds(df_com)

        return df_com, ds_ref, ds_com

    def plots(
        self, ds_ref: xr.Dataset, dsd_com: dict[str, xr.Dataset]
    ) -> dict[str, dict[str, plt.Figure]]:
        """
        Return figures of the reference and comparison data.

        Parameters
        ----------
        ds_ref : xr.Dataset
            The reference dataset.
        dsd_com : dictionary of xr.Dataset
            A dictionary of xr.Datasets whose keys are the comparisons (models).

        Returns
        -------
        dict[str,dict[str,plt.Figure]]
            A nested dictionary of matplotlib figures where the first level corresponds
            to the source name and the second level the plot name.
        """
        pass


if __name__ == "__main__":
    from ilamb3.models import ModelESGF

    # Initialize the analysis
    ref_file = Path("ref.parquet")
    analysis = runoff_sensitivity_analysis(
        basin_source="mrb_basins.nc",
        sensitivity_frame=ref_file if ref_file.is_file() else None,
        mrro_source="mrro | LORA",
        tas_source="tas | CRU4.02",
        pr_source="pr | GPCPv2.3",
    )

    # Test against CanESM5
    m = ModelESGF("CanESM5", "r1i1p1f1", "gn")
    com = xr.merge(
        [m.get_variable(v) for v in analysis.required_variables()], compat="override"
    )

    # Run the analysis
    df, ds_ref, ds_com = analysis(xr.Dataset(), com)
    if not ref_file.is_file():
        analysis.sensitivity_frame.to_parquet(ref_file)

    # Make plots
    figs = analysis.plots(ds_ref, {"CanESM5": ds_com})

    # Hanjun's numbers for reference:
    #   tas: -0.61410
    #   pr: 1.233
