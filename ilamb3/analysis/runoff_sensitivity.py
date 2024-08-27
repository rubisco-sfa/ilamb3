"""
Runoff sensitivity to temperature and precipitation per river basin.

See Also
--------
ILAMBAnalysis : The abstract base class from which this derives.
"""

import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr
from tqdm import tqdm

import ilamb3.dataset as dset
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import MissingRegion, MissingVariable


def compute_runoff_sensitivity(
    ds: xr.Dataset, basins: list[str], window_size: int = 9, quiet: bool = False
) -> pd.DataFrame:
    """
    Compute the runoff sensitivites in the basins provided.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing `mrro` (runoff), `pr` (precip), and `tas` (temperature).
    basins : list[str]
        The basins in which to compute sensitivities. These should refer to labels
        that have been registered in the ilamb3.regions.Regions class.
    window_size : int
        The number of water years to use in the windowed average.
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
    space = [dset.get_dim_name(ds, "lat"), dset.get_dim_name(ds, "lon")]
    if "cell_measures" not in ds:
        ds["cell_measures"] = dset.compute_cell_measures(ds).pint.dequantify()
    if "time_measures" not in ds:
        ds["time_measures"] = dset.compute_time_measures(ds).pint.dequantify()
        if "time_bnds" in ds:
            ds = ds.drop_vars("time_bnds")

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
    for basin in tqdm(
        basins, desc="Computing basin sensitivities", unit="basins", disable=quiet
    ):

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
        out = {
            "basin": basin,
            "tas Sensitivity": model.params.to_dict()["tas"],
            "tas Low": model.conf_int()[0].to_dict()["tas"],
            "tas High": model.conf_int()[1].to_dict()["tas"],
            "pr Sensitivity": model.params.to_dict()["pr"],
            "pr Low": model.conf_int()[0].to_dict()["pr"],
            "pr High": model.conf_int()[1].to_dict()["pr"],
            "R2": model.rsquared,
            "R2 Cross": model_cross.rsquared,
            "Cond": model.condition_number,
        }
        df.append(out)
    df = pd.DataFrame(df).set_index("basin")
    return df


class runoff_sensitivty_analysis(ILAMBAnalysis):
    """
    Runoff sensitivity to temperature and precipitation per river basin.

    Methods
    -------
    required_variables
        What variables are required.
    __call__
        The method
    """

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
        self,
        ref: xr.Dataset,
        com: xr.Dataset,
        basins: list[str],
        timestamp_start: str = "1905-01-01",
        timestamp_end: str = "2005-01-01",
    ) -> tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
        """
        Apply the ILAMB bias methodology on the given datasets.

        Parameters
        ----------
        ref : xr.Dataset
            The reference dataset.
        com : xr.Dataset
            The comparison dataset.
        basins : list[str]
            The basins in which to compute sensitivities. These should refer to labels
            that have been registered in the ilamb3.regions.Regions class.
        timestamp_start : str
            The initial time for which to compute comparision sensitivities.
        timestamp_end : str
            The final time for which to compute comparision sensitivities.

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

        com = com.sel({"time": slice(timestamp_start, timestamp_end)})
        df_com = compute_runoff_sensitivity(com, basins)

        return df_com, xr.Dataset(), xr.Dataset()


if __name__ == "__main__":
    # Temporary for some simple testing, using a coarse model for speed of execution
    from ilamb3.models import ModelESGF
    from ilamb3.regions import Regions

    # Register the MRB basins with the ilamb system
    ilamb_regions = Regions()
    basins = ilamb_regions.add_netcdf("mrb_basins.nc")

    # Test again CanESM5
    m = ModelESGF("CanESM5", "r1i1p1f1", "gn")
    ds = xr.merge([m.get_variable(v) for v in ["mrro", "tas", "pr"]], compat="override")

    # Select a time span for the models
    analysis = runoff_sensitivty_analysis()
    df, _, _ = analysis(xr.Dataset(), ds, basins)
    print(df)
