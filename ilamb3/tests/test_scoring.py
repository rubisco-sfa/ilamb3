import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.analysis import scoring
from ilamb3.exceptions import MissingVariable, ScoreAggregationError
from ilamb3.tests.test_dataset import generate_test_site_dset


def _site_array(values, units="1"):
    da = generate_test_site_dset()["da"].isel(time=0, drop=True)
    return da.copy(data=np.asarray(values, dtype=float)).assign_attrs(units=units)


def _relative_error(stats: xr.Dataset) -> float:
    return float(np.sqrt(stats["mse"] / stats["mse_ref"]))


def test_statistics_square_the_error_and_normalizer():
    stats = scoring.statistics(
        _site_array([1.0, -2.0, 0, 0], "m"), _site_array([3.0, 1, 1, 1], "m")
    )
    assert stats["mse"].values.tolist() == [1.0, 4.0, 0.0, 0.0]
    assert stats["mse_ref"].values.tolist() == [9.0, 1.0, 1.0, 1.0]
    assert stats["mse"].attrs["units"] == "(m)**2"


def test_aggregate_pools_the_statistics():
    error = _site_array([1.0, 0.0, 2.0, 0.0])
    norm = _site_array([1.0, 1.0, 2.0, 2.0])
    pooled = scoring.aggregate(scoring.statistics(error, norm))
    assert np.isclose(_relative_error(pooled), np.sqrt(5 / 10))
    assert np.isclose(float(scoring.score(pooled)), np.exp(-np.sqrt(5 / 10)))


def test_aggregate_pairs_missing_values():
    # The NaN error must drop its normalizer from the denominator too
    error = _site_array([1.0, np.nan, 2.0, 0.0])
    norm = _site_array([1.0, 100.0, 2.0, 2.0])
    pooled = scoring.aggregate(scoring.statistics(error, norm))
    assert np.isclose(_relative_error(pooled), np.sqrt(5 / 9))


def test_aggregate_over_dimensions():
    # Equal pooled squared error, spread evenly or put in one cell
    b = 0.5
    dims = ["basin", "foc"]
    ones = xr.DataArray(np.ones((2, 2)), dims=dims)
    concentrated = xr.DataArray([[2 * b, 0], [0, 0]], dims=dims)
    for error in [ones * b, concentrated]:
        pooled = scoring.aggregate(scoring.statistics(error, ones), dim=dims)
        assert np.isclose(float(scoring.score(pooled)), np.exp(-b))
    # Pooling over some dimensions leaves a local score for each basin
    per_basin = scoring.score(
        scoring.aggregate(scoring.statistics(concentrated, ones), dim="foc")
    )
    assert np.allclose(per_basin, np.exp(-b * np.array([np.sqrt(2), 0])))
    assert per_basin.attrs[dset.LOCAL_SCORE]


def test_aggregate_over_region_or_dimensions_not_both():
    stats = scoring.statistics(_site_array([1.0, 0, 0, 0]), _site_array([1.0] * 4))
    with pytest.raises(ValueError):
        scoring.aggregate(stats, "global", dim="site")


def test_score_of_zero_normalizer():
    # A region without variability scores 0 or NaN rather than raising
    norm = _site_array([0.0, 0.0, 0.0, 0.0])
    for error, expected in [([1.0, 0, 0, 0], 0.0), ([0.0, 0, 0, 0], np.nan)]:
        stats = scoring.statistics(_site_array(error), norm)
        value = float(scoring.score(scoring.aggregate(stats)))
        assert np.isclose(value, expected, equal_nan=True)


def test_score_of_unaggregated_statistics_is_local():
    stats = scoring.statistics(_site_array([1.0, 0, 2, 0]), _site_array([1.0] * 4))
    local = scoring.score(stats)
    assert np.allclose(local, np.exp(-np.array([1.0, 0, 2, 0])))
    with pytest.raises(ScoreAggregationError):
        dset.integrate_space(local.to_dataset(name="s"), "s", mean=True)
    assert dset.LOCAL_SCORE not in scoring.score(scoring.aggregate(stats)).attrs


def test_score_methods():
    pooled = scoring.aggregate(
        scoring.statistics(_site_array([1.0] * 4), _site_array([2.0] * 4))
    )
    assert np.isclose(float(scoring.score(pooled, "RegionalQuantiles")), 0.5)


def test_get_statistics_of_a_map_saved_without_them():
    ds = xr.Dataset({"cyclescore": dset.mark_local_score(_site_array([1.0, 0, 0, 0]))})
    with pytest.raises(MissingVariable, match="scoring.add_score"):
        scoring.get_statistics(ds, "cyclescore")


def test_score_without_a_normalizer():
    stats = scoring.statistics(_site_array([1.0, 0, 0, 0]))
    with pytest.raises(MissingVariable, match=r"scoring\.statistics\(error, norm"):
        scoring.score(stats)


def test_score_of_an_unknown_method_lists_the_methods():
    stats = scoring.statistics(_site_array([1.0] * 4), _site_array([2.0] * 4))
    with pytest.raises(ValueError, match="'Collier2018', 'RegionalQuantiles'"):
        scoring.score(stats, "unknown")


def test_aggregate_refuses_a_score_map():
    local = scoring.score(
        scoring.statistics(_site_array([1.0, 0, 2, 0]), _site_array([1.0] * 4))
    )
    with pytest.raises(ScoreAggregationError):
        scoring.aggregate(local.to_dataset(name="s"), dim="site")


def test_statistics_of_the_absolute_error():
    # A score linear in the error scores the mean error, the mean of the local scores
    error = _site_array([1.0, -3.0, 0, 2.0], "month")
    stats = scoring.statistics(error, 6.0, power=1)
    assert stats["mae"].values.tolist() == [1.0, 3.0, 0.0, 2.0]
    assert stats["mae_ref"].attrs["units"] == "month"
    pooled = float(scoring.score(scoring.aggregate(stats), "Linear"))
    local = scoring.score(stats, "Linear")
    assert np.isclose(pooled, 1 - 6 / 24)
    assert np.isclose(pooled, float(local.mean()))


def test_statistics_with_one_normalizer():
    error = _site_array([1.0, np.nan, 2.0, 0.0])
    stats = scoring.statistics(error, 2.0)
    assert np.isnan(stats["mse_ref"].values[1])
    assert np.isclose(float(scoring.nmse(scoring.aggregate(stats))), 5 / 3 / 4)


def test_weighted_statistics_give_the_weighted_ratio():
    error = _site_array([1.0, 0.0, 2.0, 0.0])
    norm = _site_array([1.0, 1.0, 1.0, 1.0])
    weight = _site_array([3.0, 1.0, 0.0, 1.0])
    stats = scoring.weighted(scoring.statistics(error, norm), weight)
    assert stats["mse"].attrs["units"] == "(1)**2"
    assert np.isclose(float(scoring.nmse(scoring.aggregate(stats))), 3 / 5)


def test_aggregate_weights_time_steps_by_length():
    # One step of 31 days with error 1 and one of 28 days with error 0
    time = xr.DataArray(
        np.array(["2000-01-16", "2000-02-15"], dtype="datetime64[ns]"), dims="time"
    )
    bounds = np.array(
        [["2000-01-01", "2000-02-01"], ["2000-02-01", "2000-02-29"]],
        dtype="datetime64[ns]",
    )
    grid = xr.Dataset(
        coords={"time": time, "time_bnds": (("time", "nb"), bounds)}
    )
    grid["time"].attrs["bounds"] = "time_bnds"
    error = xr.DataArray([1.0, 0.0], coords={"time": time}, dims="time")
    stats = scoring.statistics(error, xr.ones_like(error), grid=grid)
    pooled = scoring.aggregate(stats, dim="time")
    assert np.isclose(float(scoring.nmse(pooled)), 31 / 59)


def test_regional_score_uses_the_saved_method():
    stats = scoring.statistics(_site_array([1.0] * 4), _site_array([2.0] * 4))
    ds = scoring.add_score(xr.Dataset(), "biasscore", stats, "RegionalQuantiles")
    assert np.isclose(scoring.regional_score(ds, "biasscore"), 0.5)


def test_score_row_carries_the_nmse_of_a_component():
    pooled = scoring.aggregate(
        scoring.statistics(_site_array([1.0] * 4), _site_array([2.0] * 4))
    )
    row = scoring.score_row(pooled, normalizer="D")
    assert np.isclose(row["nmse"], 0.25)
    assert np.isclose(row["value"], float(scoring.score_nmse(0.25)))
    assert "nmse" not in scoring.score_row(pooled)


def _rows(*rows):
    columns = ["source", "name", "type", "value", "nmse", "normalizer"]
    return pd.DataFrame([dict(zip(columns, r)) for r in rows]).assign(
        region="None", analysis="a", units="1"
    )


def test_overall_score_adds_components_that_share_a_normalizer():
    nan = np.nan
    df = _rows(
        ("m", "Bias Score", "score", 0.0, 0.2, "D"),
        ("m", "RMSE Score", "score", 0.0, 0.3, "D"),
        ("m", "Other Score", "score", 0.0, 0.1, "E"),
        ("m", "Seasonal Cycle Score", "score", 0.01, nan, nan),
        ("m", "Bias", "scalar", 5.0, nan, nan),
        ("d", "Seasonal Cycle Score", "score", 0.9, nan, nan),
    )
    out = scoring.add_overall_score(df)
    overall = out[out["name"] == scoring.OVERALL_SCORE].set_index("source")
    # The shares of D add; the groups D and E are averaged; diagnostics are left out
    assert np.isclose(overall.loc["m", "nmse"], (0.5 + 0.1) / 2)
    assert np.isclose(overall.loc["m", "value"], scoring.score_nmse(0.3))
    # A source with only diagnostics has no Overall Score
    assert "d" not in overall.index


def test_overall_score_of_a_table_without_nmse():
    df = _rows(("m", "Bias Score", "score", 0.5, np.nan, np.nan)).drop(
        columns=["nmse", "normalizer"]
    )
    assert scoring.add_overall_score(df).equals(df)
