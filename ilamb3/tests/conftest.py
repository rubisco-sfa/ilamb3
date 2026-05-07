import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_path() -> Path:
    path = Path(tempfile.gettempdir()) / "test-ilamb3"
    path.mkdir(parents=True, exist_ok=True)
    return path
