import pytest

from ilamb3.exceptions import AnalysisFailure


def test_exceptions():
    with pytest.raises(AnalysisFailure, match=r".*my_analysis.*my_model.*"):
        raise AnalysisFailure("my_analysis", "my_model")
