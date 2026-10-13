"""Regression tests for package version metadata and CalVer configuration."""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

import cpax

ROOT = Path(__file__).resolve().parents[1]


def test_single_source_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]
    assert Version(cpax.__version__) == Version(project["version"])
    assert Version(version("CPax")) == Version(project["version"])


def test_calver_configuration_accepts_historical_and_new_versions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)["tool"]["bumpversion"]

    assert "current_version" not in config
    assert config["parts"]["release"]["calver_format"] == "{YYYY}.{MM}"
    assert config["files"] == [
        {
            "filename": "pyproject.toml",
            "search": 'version = "{current_version}"',
            "replace": 'version = "{new_version}"',
        }
    ]

    for valid in ("2025.05.0", "2026.10.0", "2026.10.1", "2026.11.0", "2027.1.0"):
        assert re.fullmatch(config["parse"], valid), valid

    for invalid in ("2026.00.0", "2026.13.0", "2026.010.1", "26.10.0", "2026.10.-1"):
        assert re.fullmatch(config["parse"], invalid) is None, invalid
