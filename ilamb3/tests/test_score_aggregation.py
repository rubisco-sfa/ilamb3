import numpy as np
import pytest
import xarray as xr

import ilamb3.compare as cmp
import ilamb3.dataset as dset
from ilamb3.analysis.base import integrate_or_mean
from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.cycle import cycle_analysis
from ilamb3.analysis.hydro import hydro_analysis
from ilamb3.analysis.rmse import rmse_analysis
from ilamb3.exceptions import ScoreAggregationError
from ilamb3.tests.test_compare import generate_test_dset
from ilamb3.tests.test_dataset import generate_test_site_dset


@pytest.mark.parametrize(
    "ds",
    [
        generate_test_dset(nlat=10, nlon=20),
        generate_test_site_dset().isel(time=0, drop=True),
    ],
)
def test_local_score_cannot_be_aggregated(ds):
    ds["da"] = dset.mark_local_score(ds["da"])
    with pytest.raises(ScoreAggregationError):
        integrate_or_mean(ds["da"], "da", None, mean=True)


def test_local_score_cannot_be_integrated():
    ds = generate_test_dset(ntime=12, nlat=10, nlon=20)
    ds["da"] = dset.mark_local_score(ds["da"])
    with pytest.raises(ScoreAggregationError):
        dset.integrate_space(ds, "da", mean=True)
    with pytest.raises(ScoreAggregationError):
        dset.integrate_time(ds, "da", mean=True)


def test_local_score_flag_survives_arithmetic():
    ds = generate_test_dset(nlat=10, nlon=20)
    ds["da"] = dset.mark_local_score(ds["da"])
    with pytest.raises(ScoreAggregationError):
        dset.integrate_space(1 - ds["da"], "da", mean=True)


def _offset(ref: xr.Dataset, offset: xr.DataArray) -> xr.Dataset:
    com = ref.copy(deep=True)
    com["da"] = (ref["da"] + offset).assign_attrs(ref["da"].attrs)
    return com


def _score(analysis, ref: xr.Dataset, com: xr.Dataset, name: str) -> float:
    df, _, _ = analysis(ref.copy(deep=True), com)
    return df[df["name"] == name].iloc[0]["value"]


def test_bias_score_pools_errors_before_scoring():
    # Two models with the same pooled squared bias, spread evenly or concentrated
    # in one site, must score the same. Averaging per-site scores does not.
    ref = generate_test_site_dset(ntime=36)
    b = 3e-9
    scores = [
        _score(
            bias_analysis("da", use_uncertainty=False),
            ref,
            _offset(ref, xr.DataArray(offsets, dims="site")),
            "Bias Score",
        )
        for offsets in [[b, b, b, b], [2 * b, 0, 0, 0]]
    ]
    sigma = dset.std_time(ref, "da")
    assert np.allclose(scores, np.exp(-b / np.sqrt((sigma**2).mean())))


def test_rmse_pools_errors_before_scoring():
    ref = generate_test_site_dset(ntime=36)
    pattern = xr.DataArray(np.resize([1.0, -1.0], 36), coords={"time": ref["time"]})
    r = 3e-9
    out = []
    for amplitude in [[r, r, r, r], [2 * r, 0, 0, 0]]:
        com = _offset(ref, pattern * xr.DataArray(amplitude, dims="site"))
        df, _, _ = rmse_analysis("da", use_uncertainty=False)(ref.copy(deep=True), com)
        out.append(df.set_index("name")["value"])
    # Both the RMSE (mean of the MSE, then the root) and its score are pooled
    for name in ["RMSE", "RMSE Score"]:
        assert np.isclose(out[0][name], out[1][name], rtol=1e-10)
    assert 0 < out[0]["RMSE Score"] < 1


def test_hydro_scores_pool_errors_before_scoring():
    ref = generate_test_dset(ntime=36, nlat=2, nlon=3)
    area = dset.compute_cell_measures(ref, "da").reset_coords(drop=True)
    b = 3e-9
    # Equal area-weighted squared offsets, spread evenly or put in one cell
    concentrated = xr.zeros_like(area)
    concentrated[0, 0] = b * np.sqrt(area.sum() / area[0, 0])
    out = []
    for offset in [xr.full_like(area, b), concentrated]:
        df, _, _ = hydro_analysis("da")(ref.copy(deep=True), _offset(ref, offset))
        out.append(df[df["type"] == "score"].set_index(["analysis", "name"])["value"])
    assert len(out[0]) == 5
    assert np.allclose(out[0], out[1])


@pytest.mark.parametrize(
    "com_grid,score",
    [
        (dict(nlat=10, nlon=20), 0.48879032920754284),
        # The nested grid is irregular, so its cell bounds must not be estimated
        (dict(nlat=7, nlon=13), 0.4781192193293507),
    ],
)
def test_cycle_score_is_affine_in_the_regional_shift(com_grid, score):
    # Scoring the regional mean shift must match the old mean of the score map
    ref = generate_test_dset(ntime=36, nlat=10, nlon=20)
    com = generate_test_dset(seed=2, ntime=36, **com_grid)
    df, _, _ = cycle_analysis("da")(ref, com)
    df = df.set_index("name")["value"]
    assert np.isclose(df["Seasonal Cycle Score"], score)


def test_bias_score_weights_nested_cells_by_their_true_area():
    # The nested grid of mismatched grids is irregular, so cell areas must come from
    # its bounds rather than be estimated from the cell centers
    ref = generate_test_dset(nlat=10, nlon=20)
    com = generate_test_dset(seed=3, nlat=7, nlon=13)
    df, _, ds_com = bias_analysis("da", use_uncertainty=False)(ref.copy(), com.copy())
    nested, _ = cmp.nest_spatial_grids(ref, com)
    latb = np.deg2rad(nested["lat_nested_bounds"].values)
    lonb = np.deg2rad(nested["lon_nested_bounds"].values)
    area = np.outer(np.diff(np.sin(latb)), np.diff(lonb))
    mse = (area * ds_com["bias"].values ** 2).sum()
    mse_ref = (area * nested["da"].values ** 2).sum()
    expected = np.exp(-np.sqrt(mse / mse_ref))
    assert np.isclose(df[df["name"] == "Bias Score"].iloc[0]["value"], expected)


def test_refusal_says_how_to_score_a_region():
    ds = generate_test_dset(nlat=10, nlon=20).rename_vars({"da": "biasscore"})
    ds["biasscore"] = dset.mark_local_score(ds["biasscore"])
    with pytest.raises(ScoreAggregationError) as err:
        dset.integrate_space(ds, "biasscore", mean=True)
    message = str(err.value)
    assert "scoring.get_statistics(ds, 'biasscore')" in message
    assert "scoring.score(scoring.aggregate(" in message
    assert "aggregate the error map" in message


@pytest.mark.parametrize(
    "function,kwargs",
    [
        ("accumulate_time", dict(varname="da")),
        ("coarsen_annual", {}),
        ("coarsen_dataset", dict(varname="da", res=36)),
        ("compute_annual_mean", {}),
        ("compute_monthly_mean", {}),
        ("compute_seasonal_climatology", dict(varname="da")),
        ("compute_seasonal_mean", dict(varname="da")),
        ("integrate_depth", dict(varname="da")),
        ("std_time", dict(varname="da")),
    ],
)
def test_every_reduction_refuses_a_local_score(function, kwargs):
    # A score of each time step, which must not be averaged over time either
    ds = generate_test_dset(ntime=24, nlat=10, nlon=20)
    ds["da"] = dset.mark_local_score(ds["da"])
    with pytest.raises(ScoreAggregationError):
        getattr(dset, function)(ds, **kwargs)
