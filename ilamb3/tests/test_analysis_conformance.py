"""
Checks that every analysis in `ilamb3.analysis.ALL_ANALYSES` must pass.

Each analysis needs a case in `CASES` that builds it with a valid reference and
comparison. The checks enforce the scoring rules in docs/deep-dive/scoring.md: scores lie
in [0, 1], score maps are flagged as local scores so that ilamb3 refuses to aggregate
them, and the statistics saved with a score map reproduce the regional score reported for
it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ilamb3
import ilamb3.dataset as dset
import ilamb3.regions as ilr
from ilamb3.analysis import ALL_ANALYSES, scoring
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.tests.test_compare import generate_test_dset

COLUMNS = ["source", "region", "analysis", "name", "type", "units", "value"]
GLOBAL_REGION = str(None)


def _gridded(seed: int = 1, **grid) -> xr.Dataset:
    # Values of order one, as spatial_distribution skips a region whose spatial
    # standard deviation is below 1e-8 in absolute terms
    ds = generate_test_dset(seed=seed, **grid)
    ds["da"] = (ds["da"] * 1e8).assign_attrs(units="g m-2 d-1")
    return ds


def _gridded_pair() -> tuple[xr.Dataset, xr.Dataset]:
    # The grids differ, so that scores are computed on the nested grid
    return _gridded(ntime=36, nlat=10, nlon=20), _gridded(
        seed=2, ntime=36, nlat=7, nlon=13
    )


def _gridded_case(analysis: ILAMBAnalysis):
    return analysis, *_gridded_pair()


def _area_case():
    def extent(nlat: int, nlon: int, lat: float) -> xr.Dataset:
        da = generate_test_dset(nlat=nlat, nlon=nlon)["da"]
        da = xr.ones_like(da).where(da["lat"] > lat).assign_attrs(units="1")
        return da.to_dataset(name="permafrost_extent")

    analysis = ALL_ANALYSES["area"]("permafrost_extent")
    return analysis, extent(10, 20, 40), extent(7, 13, 50)


def _nbp_case():
    ref = generate_test_dset(ntime=36).rename_vars({"da": "nbp"})
    uncert = generate_test_dset(seed=2, ntime=36)["da"] * 1e-2
    ref["nbp_bnds"] = xr.DataArray(
        np.array([ref["nbp"] - uncert, ref["nbp"] + uncert]).T,
        coords={"time": ref["time"]},
        dims=["time", "nb"],
        attrs={"units": "Pg yr-1"},
    )
    ref["nbp"].attrs.update(bounds="nbp_bnds", units="Pg yr-1")
    com = generate_test_dset(seed=3, ntime=36).rename_vars({"da": "nbp"})
    com["nbp"].attrs["units"] = "Pg yr-1"
    return ALL_ANALYSES["nbp"](), ref, com


def _relationship_case():
    def pair(seed: int) -> xr.Dataset:
        return xr.Dataset(
            {
                "gpp": _gridded(seed=seed, nlat=10, nlon=20)["da"],
                "tas": _gridded(seed=seed + 1, nlat=10, nlon=20)["da"],
            }
        )

    return ALL_ANALYSES["relationship"]("gpp", "tas"), pair(1), pair(3)


def _runoff_sensitivity_case():
    from ilamb3.analysis.runoff_sensitivity import _fix_ds_dims

    analysis = ALL_ANALYSES["runoff_sensitivity"]()
    ref = xr.open_dataset(
        ilamb3.ilamb_catalog().fetch("G-RUN/runoff_sensitivity_obs.nc")
    )
    sens = _fix_ds_dims(ref.copy())[["psens_obs", "tsens_obs"]]
    return analysis, ref, sens.mean(dim=["foc", "ens"]) * 1.3


def _timeseries_case():
    ref, com = (
        dset.integrate_space(ds, "da", mean=True).to_dataset(name="da")
        for ds in _gridded_pair()
    )
    return ALL_ANALYSES["timeseries"]("da"), ref, com


CASES = {
    "area": _area_case,
    "bias": lambda: _gridded_case(ALL_ANALYSES["bias"]("da")),
    "cycle": lambda: _gridded_case(ALL_ANALYSES["cycle"]("da")),
    "dispersion": lambda: _gridded_case(
        ALL_ANALYSES["dispersion"]("da", required_num_years=0, nbins=3)
    ),
    "hydro": lambda: _gridded_case(ALL_ANALYSES["hydro"]("da")),
    "nbp": _nbp_case,
    "relationship": _relationship_case,
    "rmse": lambda: _gridded_case(ALL_ANALYSES["rmse"]("da")),
    "runoff_sensitivity": _runoff_sensitivity_case,
    "spatial_distribution": lambda: _gridded_case(
        ALL_ANALYSES["spatial_distribution"]("da")
    ),
    "timeseries": _timeseries_case,
}

KNOWN_FAILURES = {
    "runoff_sensitivity": pytest.mark.xfail(
        raises=ValueError,
        strict=True,
        reason="Registering the basins fails on the invalid region label "
        "'kunene (also cunene)'. Remove this mark once the label is fixed.",
    ),
}


def check_conformance(
    key: str, analysis: ILAMBAnalysis, ref: xr.Dataset, com: xr.Dataset, path: Path
) -> None:
    """
    Run an analysis and check its output against the rules every analysis follows.

    Parameters
    ----------
    key : str
        The name of the analysis, used in the failure messages.
    analysis : ILAMBAnalysis
        The analysis to run.
    ref, com : xr.Dataset
        A valid reference and comparison for the analysis.
    path : Path
        A directory to write the output datasets to.
    """
    df, ds_ref, ds_com = analysis(ref, com)
    scores = _check_table(key, df)
    for source, ds in [("Reference", ds_ref), ("Comparison", ds_com)]:
        # Check the output as it is read back from disk
        ds.to_netcdf(path / f"{key}_{source}.nc")
        ds = xr.load_dataset(path / f"{key}_{source}.nc")
        for name in _score_maps(ds):
            _check_score_map(key, ds, name, scores)


def _check_table(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Check the table of scalars and scores, and return its scores."""
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, (
        f"{key}: the result table lacks the columns {sorted(missing)}. Return one row "
        f"per scalar or score, with the columns {COLUMNS}."
    )
    types = set(df["type"]) - {"scalar", "score"}
    assert not types, (
        f"{key}: the result table has rows of type {sorted(types)}. Set the type of "
        "each row to 'scalar' or 'score'."
    )
    scores = df[df["type"] == "score"]
    assert len(scores), (
        f"{key}: the analysis reported no scores, so the score checks cannot run. "
        "Give its case in CASES data for which the analysis reports scores."
    )
    units = set(scores["units"]) - {"1"}
    assert not units, (
        f"{key}: scores have the units {sorted(units)}. Scores are dimensionless, so "
        "set their units to '1'."
    )
    outside = scores[~scores["value"].astype(float).between(0, 1)]
    assert outside.empty, (
        f"{key}: these scores are not in [0, 1]:\n"
        f"{outside[['region', 'name', 'value']].to_string(index=False)}\n"
        "A score maps an error onto [0, 1], where 1 is perfect. See "
        "docs/deep-dive/scoring.md."
    )
    return scores


def _score_maps(ds: xr.Dataset) -> list[str]:
    """Return the names of the score maps in an output dataset."""
    return [
        str(name)
        for name, da in ds.data_vars.items()
        if "score" in str(name).lower()
        and not str(name).endswith(scoring.STATISTICS_SUFFIX)
        and da.ndim
    ]


def _check_score_map(key: str, ds: xr.Dataset, name: str, scores: pd.DataFrame):
    """Check that a score map cannot be averaged and matches its regional score."""
    assert ds[name].attrs.get(dset.LOCAL_SCORE), (
        f"{key}: '{name}' is a score map, but it is not flagged as a local score, so "
        "ilamb3 would let it be averaged into a regional score. Build it with "
        "ilamb3.analysis.scoring.add_score, or flag it with "
        "ilamb3.dataset.mark_local_score."
    )
    assert 0 <= float(ds[name].min()) and float(ds[name].max()) <= 1, (
        f"{key}: '{name}' has values outside [0, 1]. A score maps an error onto "
        "[0, 1], where 1 is perfect."
    )
    if name + scoring.STATISTICS_SUFFIX not in ds:
        return
    value = float(scoring.score(scoring.aggregate(scoring.get_statistics(ds, name))))
    reported = scores[scores["region"].astype(str) == GLOBAL_REGION]["value"]
    assert np.isclose(reported.astype(float), value).any(), (
        f"{key}: the statistics saved with '{name}' give a global score of {value:.6g}, "
        f"but the analysis reported the global scores {reported.tolist()}. Compute "
        "each regional score from the statistics passed to scoring.add_score, as "
        "scoring.score(scoring.aggregate(stats, region)), not from the score map."
    )


@pytest.fixture(autouse=True)
def _restore_regions(monkeypatch):
    # Analyses such as runoff_sensitivity register regions, which must not leak into
    # other tests
    monkeypatch.setattr(ilr.Regions, "_regions", dict(ilr.Regions._regions))


def test_every_analysis_has_a_case():
    missing = set(ALL_ANALYSES) - set(CASES)
    assert not missing, (
        f"Add a case for {sorted(missing)} to CASES in {Path(__file__).name}, so "
        "that the checks every analysis must pass run on it."
    )
    unknown = set(CASES) - set(ALL_ANALYSES)
    assert not unknown, (
        f"CASES has {sorted(unknown)}, which are not in "
        "ilamb3.analysis.ALL_ANALYSES. ALL_ANALYSES silently leaves out a module "
        "that fails to import, so import the module directly to see the error."
    )


@pytest.mark.parametrize(
    "key",
    [pytest.param(key, marks=KNOWN_FAILURES.get(key, ())) for key in sorted(CASES)],
)
def test_analysis_conforms(key, tmp_path):
    analysis, ref, com = CASES[key]()
    check_conformance(key, analysis, ref, com, tmp_path)


class _mean_of_score_map(ILAMBAnalysis):
    """An analysis that breaks the rules by averaging its score map."""

    def __init__(self, flag: bool = True):
        self.flag = flag

    def name(self) -> str:
        return "Mean of Score Map"

    def required_variables(self) -> list[str]:
        return ["da"]

    def __call__(self, ref: xr.Dataset, com: xr.Dataset):
        stats = scoring.statistics(com["da"] - ref["da"], ref["da"], grid=com)
        out = scoring.add_score(xr.Dataset(), "dascore", stats)
        if not self.flag:
            out["dascore"].attrs.pop(dset.LOCAL_SCORE)
        df = pd.DataFrame(
            [
                {
                    "source": "Comparison",
                    "region": "None",
                    "analysis": self.name(),
                    "name": "Score",
                    "type": "score",
                    "units": "1",
                    "value": float(out["dascore"].mean()),
                }
            ]
        )
        return df, xr.Dataset(), out

    def plots(self, df, ref, com, path):  # pragma: no cover
        return pd.DataFrame()


@pytest.mark.parametrize(
    "flag,message",
    [
        (False, "mark_local_score"),
        (True, r"scoring\.score\(scoring\.aggregate"),
    ],
)
def test_checks_catch_a_score_map_that_is_averaged(flag, message, tmp_path):
    ref = generate_test_dset(nlat=10, nlon=20)
    com = generate_test_dset(seed=2, nlat=10, nlon=20)
    with pytest.raises(AssertionError, match=message):
        check_conformance("mean", _mean_of_score_map(flag), ref, com, tmp_path)
