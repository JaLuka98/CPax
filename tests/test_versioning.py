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

def _bump_in_temporary_repo(tmp_path: Path, current: str, part: str) -> str:
    """Run the installed bump-my-version CLI against a disposable copy of CPax."""
    import subprocess

    with (ROOT / "pyproject.toml").open("rb") as file:
        configured_version = tomllib.load(file)["project"]["version"]
    contents = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    search = f'version = "{configured_version}"'
    assert contents.count(search) == 1
    (tmp_path / "pyproject.toml").write_text(contents.replace(search, f'version = "{current}"'), encoding="utf-8")

    # Keep the test isolated from the CPax checkout, release tags, and user configuration.
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "pyproject.toml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CPax Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "Initial"],
        cwd=tmp_path,
        check=True,
    )
    initial = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    subprocess.run(
        ["bump-my-version", "bump", part, "--no-commit", "--no-tag"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    final = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert final == initial, "Version bump should not create Git commits."
    assert not subprocess.check_output(["git", "tag", "--list"], cwd=tmp_path, text=True).strip()

    with (tmp_path / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def test_bump_my_version_same_month(tmp_path: Path) -> None:
    """The actual CLI increments N without altering the current UTC release month."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    assert _bump_in_temporary_repo(tmp_path, f"{now.year}.{now.month}.3", "patch") == (f"{now.year}.{now.month}.4")


def test_bump_my_version_month_rollover(tmp_path: Path) -> None:
    """A new month resets N to zero, including across a year boundary."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    previous_year, previous_month = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    assert _bump_in_temporary_repo(tmp_path, f"{previous_year}.{previous_month}.7", "release") == (
        f"{now.year}.{now.month}.0"
    )


def test_bump_my_version_year_rollover(tmp_path: Path) -> None:
    """Jumping from a previous year updates the year and resets N."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    assert _bump_in_temporary_repo(tmp_path, f"{now.year - 1}.12.7", "release") == (f"{now.year}.{now.month}.0")


def test_bump_my_version_legacy_zero_padded_month(tmp_path: Path) -> None:
    """The historical YYYY.05.N spelling remains parseable by the CLI."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    assert _bump_in_temporary_repo(tmp_path, "2025.05.0", "release") == f"{now.year}.{now.month}.0"
