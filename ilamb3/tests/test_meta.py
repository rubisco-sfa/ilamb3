import numpy as np
import pandas as pd

from ilamb3.analysis import scoring
from ilamb3.meta import dataframe_to_cmec


def test_roll_up_scores_the_mean_error_of_the_datasets():
    rows = [
        ("A", scoring.OVERALL_SCORE, 0.1),
        ("B", scoring.OVERALL_SCORE, 0.5),
        ("A", "Seasonal Cycle Score", np.nan),
        ("B", "Seasonal Cycle Score", np.nan),
    ]
    df = pd.DataFrame(
        [
            {
                "section": "Carbon",
                "variable": "GPP",
                "dataset": dataset,
                "source": "m",
                "region": "None",
                "analysis": "a",
                "type": "score",
                "units": "1",
                "name": name,
                "value": 0.8 if np.isnan(nmse) else float(scoring.score_nmse(nmse)),
                "nmse": nmse,
            }
            for dataset, name, nmse in rows
        ]
    )
    results = dataframe_to_cmec(df)["RESULTS"]["None"]["m"]
    # A single dataset is shown as is
    assert np.isclose(results["Carbon::GPP!!A"]["Overall Score"], scoring.score_nmse(0.1))
    # The variable scores the mean NMSE of its datasets once
    assert np.isclose(results["Carbon::GPP"]["Overall Score"], scoring.score_nmse(0.3))
    assert np.isclose(results["Carbon"]["Overall Score"], scoring.score_nmse(0.3))
    # A diagnostic is shown for each dataset but not rolled up
    assert results["Carbon::GPP!!A"]["Seasonal Cycle Score"] == 0.8
    assert results["Carbon::GPP"]["Seasonal Cycle Score"] is None
