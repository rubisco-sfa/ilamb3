import numpy as np
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
    ones = xr.ones_like(xr.DataArray(np.ones((2, 2)), dims=dims))
    for error in [ones * b, xr.DataArray([[2 * b, 0], [0, 0]], dims=dims)]:
        pooled = scoring.aggregate(scoring.statistics(error, ones), dim=dims)
        assert np.isclose(float(scoring.score(pooled)), np.exp(-b))
    # Pooling over some dimensions leaves a local score for each basin
    error = xr.DataArray([[2 * b, 0], [0, 0]], dims=dims)
    per_basin = scoring.score(
        scoring.aggregate(scoring.statistics(error, ones), dim="foc")
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
    with pytest.raises(ValueError):
        scoring.score(pooled, "unknown")


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
