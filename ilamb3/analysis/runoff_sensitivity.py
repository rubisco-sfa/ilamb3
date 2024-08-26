"""
Blah.

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

"""
Next Steps:
- implement your def of anomaly
x harvest scalars from statsmodels
- register your region definitions in ilamb_regions (obs_data/mrb_*)
- do we get the same thing that the other code does?

To Think About:
- masking issues for coarse models / small basins

Optimization Ideas:
- swap water year and basin avg order?
- threaded?

"""


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

    # Compute the water year for the time series (beginning of year is Oct)
    t = ds[dset.get_dim_name(ds, "time")]
    ds["water_year"] = xr.where(t.dt.month > 9, t.dt.year, t.dt.year - 1)
    ds = ds.set_coords("water_year")

    # Associate cell/time measures with the dataset if not present
    if "cell_measures" not in ds:
        ds["cell_measures"] = dset.compute_cell_measures(ds).pint.dequantify()
    if "time_measures" not in ds:
        ds["time_measures"] = dset.compute_time_measures(ds).pint.dequantify()

    # Loop over basins and build up the dataframe of sensitivities
    df = []
    for basin in tqdm(
        basins, desc="Computing basin sensitivities", unit="basins", disable=quiet
    ):

        # Compute the regional mean values per basin
        dsb = ilamb_regions.restrict_to_region(ds, basin)
        msr = dsb["cell_measures"].fillna(0)
        dsb = dsb.drop_vars(
            [v for v in ds.data_vars if v not in required_vars + ["time_measures"]]
        )
        dsb = dsb.weighted(msr).mean(dim=space)

        # Averages per water year. This is weighted by the number of days per month in
        # each water year. The `groupby` does not work with the `weighted` accessor and
        # so we take the weighted mean by hand.
        dsb = (
            (dsb * dsb["time_measures"]).groupby("water_year").sum()
            / dsb["time_measures"].groupby("water_year").sum()
        ).drop_vars("time_measures")

        # Take a windowed (decadal) average over the water years
        mean = dsb.rolling(water_year=window_size, center=True).mean()
        std = dsb.rolling(water_year=window_size, center=True).std()
        anomaly = (dsb - mean) / std

        # Fit a linear model and compute stats
        results = smf.ols("mrro ~ tas * pr", data=anomaly.to_dataframe()).fit()
        out = {
            f"{key} Sensitivity": val
            for key, val in results.params.to_dict().items()
            if "Intercept" not in key
        }
        out.update(
            {
                f"{key} Low": val
                for key, val in results.conf_int()[0].to_dict().items()
                if "Intercept" not in key and ":" not in key
            }
        )
        out.update(
            {
                f"{key} High": val
                for key, val in results.conf_int()[1].to_dict().items()
                if "Intercept" not in key and ":" not in key
            }
        )
        out["R2"] = results.rsquared
        out["Cond"] = results.condition_number
        out["basin"] = str(basin)
        df.append(out)
    df = pd.DataFrame(df).set_index("basin")
    return df


class runoff_sensitivty_analysis(ILAMBAnalysis):
    """
    Blah.

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
            A list of the required variables, here always [pr, tas, hfls, mrro].
        """
        return ["pr", "tas", "hfls", "mrro"]

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
        # Ensure we have the variables we need for this confrontation
        missing = set(self.required_variables()) - set(com)
        if missing:
            raise MissingVariable(f"Comparison dataset is lacking variables: {missing}")


if __name__ == "__main__":
    # Temporary for some simple testing, using a coarse model for speed of execution
    import ilamb3
    from ilamb3.models import ModelESGF
    from ilamb3.regions import Regions

    # grab some sample basins, still need to encode the right ones. Also have to fix
    # ilamb to read this without fixes you see here.
    ilamb_regions = Regions()
    basins = ilamb3.ilamb_catalog()["river_basins | Dai"].read()
    basins = basins.rename(dict(basin_index="ids", label="name"))
    basins["label"] = basins["name"].str.lower()
    basins["ids"].attrs = {"names": "name", "labels": "label"}
    basins = ilamb_regions.add_netcdf(basins)

    m = ModelESGF("CanESM5", "r1i1p1f1", "gn")
    ds = xr.merge([m.get_variable(v) for v in ["mrro", "tas", "pr"]], compat="override")
    space = [dset.get_dim_name(ds, "lat"), dset.get_dim_name(ds, "lon")]

    # select a time span for the models
    ds = ds.sel({"time": slice("1905-01-01", "2005-01-01")})

    df = compute_runoff_sensitivity(ds, basins[:2])
