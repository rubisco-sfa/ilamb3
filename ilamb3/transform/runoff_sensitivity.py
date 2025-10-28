from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr
from tqdm import tqdm

import ilamb3
import ilamb3.dataset as dset
from ilamb3.cache import dataframe_cache
from ilamb3.exceptions import MissingRegion, MissingVariable
from ilamb3.regions import Regions
from ilamb3.transform.base import ILAMBTransform


class runoff_sensitivity(ILAMBTransform):
    def __init__(self):
        cat = ilamb3.ilamb_catalog()
        self.basins = list(
            set(Regions().add_netcdf(xr.open_dataset(cat.fetch("G-RUN/mrb_basins.nc"))))
        )

    def required_variables(self) -> list[str]:
        """
        Return the variables this transform uses.
        """
        return ["mrro", "tas", "pr"]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if "psens_obs" in ds and "tsens_obs" in ds:
            return ds
        df = compute_runoff_sensitivity(ds, self.basins)
        out = _df_to_ds(df)
        return out


def _df_to_ds(df: pd.DataFrame) -> xr.Dataset:
    ds = df.to_xarray()
    concat_dim = pd.Index(
        [
            "sensitivity value",
            "lower bound (95% confidence interval of reg. coeff.)",
            "upper bound (95% confidence interval of reg. coeff.)",
        ],
        name="sens_type",
    )
    out = xr.Dataset(
        data_vars={
            "tsens_obs": xr.concat(
                [ds["tas Sensitivity"], ds["tas Low"], ds["tas High"]], dim=concat_dim
            ),
            "psens_obs": xr.concat(
                [ds["pr Sensitivity"], ds["pr Low"], ds["pr High"]], dim=concat_dim
            ),
        }
    )
    return out


@dataframe_cache
def compute_runoff_sensitivity(
    ds: xr.Dataset,
    basins: list[str],
    window_size: int = 5,
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

    def _compute_one_basin(basin: str) -> dict[str, float | str]:
        # Compute the regional mean values per basin
        dsb = ilamb_regions.restrict_to_region(ds, basin)
        msr = dsb["cell_measures"].fillna(0)
        dsb = dsb.drop_vars(
            [v for v in ds.data_vars if v not in required_vars], errors="ignore"
        )
        dsb = dsb.weighted(msr).mean(dim=space)

        # Take a windowed (decadal) average over the water years
        mean = dsb.rolling(water_year=window_size, center=True).mean()

        # Compute the anomalies
        anomaly = mean - mean.mean()
        anomaly["mrro"] = anomaly["mrro"] / mean["mrro"].mean() * 100.0
        anomaly["pr"] = anomaly["pr"] / mean["pr"].mean() * 100.0
        anomaly.load()

        # Fit a linear model and compute stats
        try:
            model = smf.ols("mrro ~ tas * pr", data=anomaly.to_dataframe()).fit()
        except Exception:
            return {
                "basin": basin,
                "tas Sensitivity": np.nan,
                "tas Low": np.nan,
                "tas High": np.nan,
                "pr Sensitivity": np.nan,
                "pr Low": np.nan,
                "pr High": np.nan,
                "R2": np.nan,
                "Cond": np.nan,
            }

        return {
            "basin": basin,
            "tas Sensitivity": model.params.to_dict()["tas"],
            "tas Low": model.conf_int()[0].to_dict()["tas"],
            "tas High": model.conf_int()[1].to_dict()["tas"],
            "pr Sensitivity": model.params.to_dict()["pr"],
            "pr Low": model.conf_int()[0].to_dict()["pr"],
            "pr High": model.conf_int()[1].to_dict()["pr"],
            "R2": model.rsquared,
            "Cond": model.condition_number,
        }

    with ThreadPool(3) as pool:
        compute_basin_stats = pool.imap_unordered(_compute_one_basin, basins)
        rows = list(
            tqdm(
                compute_basin_stats,
                unit="basin",
                desc="Compute basin statistics",
                total=len(basins),
            )
        )
    df = pd.DataFrame(rows).set_index("basin")
    return df
