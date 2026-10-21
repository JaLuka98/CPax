"""Release-tag validation can be tested without PyPI or GitHub permissions."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_pypi_tag.py"


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture
def release_repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "CPax Tests")
    git(tmp_path, "config", "user.email", "tests@example.invalid")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "CPax"\nversion = "2026.10.0"\n')
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "cpax"\nversion = "2026.10.0"\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Initial release")
    initial = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", initial)
    git(tmp_path, "tag", "-a", "v2026.10.0", "-m", "CPax v2026.10.0")
    return tmp_path


def check(repo: Path, tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), tag],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_annotated_tag(release_repository: Path) -> None:
    result = check(release_repository, "v2026.10.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == git(release_repository, "rev-parse", "HEAD")


def test_malformed_or_missing_tags_are_rejected(release_repository: Path) -> None:
    for tag in ("main", "v2026.13.0", "v2026.10.01", "v2026.10.1;echo bad", "v2026.10.2"):
        assert check(release_repository, tag).returncode != 0


def test_lightweight_tag_is_rejected(release_repository: Path) -> None:
    git(release_repository, "tag", "v2026.10.1")
    result = check(release_repository, "v2026.10.1")
    assert result.returncode != 0
    assert "annotated" in result.stderr


def test_version_mismatch_is_rejected(release_repository: Path) -> None:
    git(release_repository, "tag", "-a", "v2026.10.1", "-m", "Incorrect version")
    result = check(release_repository, "v2026.10.1")
    assert result.returncode != 0
    assert "version" in result.stderr


def test_unreachable_tag_is_rejected(release_repository: Path) -> None:
    (release_repository / "notes.txt").write_text("New unmerged commit")
    git(release_repository, "add", ".")
    git(release_repository, "commit", "-qm", "Unmerged changes")
    git(release_repository, "tag", "-a", "v2026.10.1", "-m", "Unmerged tag")
    result = check(release_repository, "v2026.10.1")
    assert result.returncode != 0
    assert "merge-base" in result.stderr


def test_lockfile_mismatch_is_rejected(release_repository: Path) -> None:
    git(release_repository, "tag", "-d", "v2026.10.0")
    (release_repository / "uv.lock").write_text('[[package]]\nname = "cpax"\nversion = "2026.9.0"\n')
    git(release_repository, "add", ".")
    git(release_repository, "commit", "-qm", "Stale uv lock")
    git(release_repository, "update-ref", "refs/remotes/origin/main", git(release_repository, "rev-parse", "HEAD"))
    git(release_repository, "tag", "-a", "v2026.10.0", "-m", "Release with stale lock")
    result = check(release_repository, "v2026.10.0")
    assert result.returncode != 0
    assert "uv.lock" in result.stderr
