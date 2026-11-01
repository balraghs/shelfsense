import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelfsense.data import generate_sales, prepare  # noqa: E402


@pytest.fixture(scope="session")
def sales():
    return prepare(generate_sales(days=400, end_date="2026-10-31", seed=1))
