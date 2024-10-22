import xarray as xr
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

from ilamb3.exceptions import VarNotInModel
from ilamb3.models.base import Model


def _create_cell_measures(ds: xr.Dataset, variable_id: str) -> xr.Dataset:
    """Create cell measures from the areacella/sftlf or areacello/sftof information."""
    da = ds[variable_id]
    if "cell_measures" not in da.attrs:
        return ds
    cm = da.attrs["cell_measures"].replace("area: ", "")
    ds = ds.rename({cm: "cell_measures"})
    if "cell_methods" not in da.attrs:
        return ds
    for domain, frac in zip(["land", "ocean"], ["sftlf", "sftof"]):
        if domain in da.attrs["cell_methods"] and frac in ds:
            fr = ds["sftlf"]
            if fr.max() > 50.0:
                fr = fr * 0.01
            ds["cell_measures"] *= fr
            ds = ds.drop_vars(frac)
    return ds


class ModelESGF(Model):
    def __init__(self, source_id: str, member_id: str, grid_label: str):
        self.name = source_id
        self.search = dict(
            source_id=source_id,
            member_id=member_id,
            grid_label=grid_label,
        )

    def __repr__(self):
        return f"{self.search['source_id']}|{self.search['member_id']}|{self.search['grid_label']}"

    def get_variable(
        self,
        variable_id: str,
        experiment_id: str = "historical",
        frequency: str = "mon",
        quiet: bool = True,
    ) -> xr.Dataset:
        search = self.search.copy()
        search.update(
            {
                "experiment_id": experiment_id,
                "variable_id": variable_id,
                "frequency": frequency,
            }
        )
        try:
            cat = ESGFCatalog().search(quiet=quiet, **search)
        except NoSearchResults:
            raise VarNotInModel(f"{variable_id} not in {self.__repr__()}")
        if len(cat.df) > 1:
            cat.df = cat.df.iloc[0]
        ds = cat.to_dataset_dict(quiet=quiet)[variable_id]
        ds = _create_cell_measures(ds, variable_id)
        return ds
