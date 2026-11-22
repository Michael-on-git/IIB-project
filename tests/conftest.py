
# tests/conftest.py
from pathlib import Path
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
CIRCUITS_DIR = PROJECT_ROOT / "circuits"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def circuits_dir() -> Path:
    return CIRCUITS_DIR


def _circuit_path(name: str) -> Path:
    path = CIRCUITS_DIR / name / "design_gate.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} not found")
    return path


@pytest.fixture
def mux4_json() -> Path:
    return _circuit_path("mux4")


@pytest.fixture
def and4_json() -> Path:
    return _circuit_path("and4")


@pytest.fixture
def add4_json() -> Path:
    return _circuit_path("add4")


@pytest.fixture
def chained_mux_json() -> Path:
    return _circuit_path("chained_mux")