"""Regression tests for wheel and sdist metadata validation."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.validate_pypi_dist import ReleaseArtifactError, validate_artifacts


def _metadata(version: str, *, name: str = "CPax") -> bytes:
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode()


def _build_artifacts(tmp_path: Path, *, version: str = "2026.10.0", sdist_version: str | None = None) -> None:
    wheel = tmp_path / f"cpax-{version}-py3-none-any.whl"
    sdist = tmp_path / f"cpax-{version}.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"cpax-{version}.dist-info/METADATA", _metadata(version))
    with tarfile.open(sdist, "w:gz") as archive:
        for path, data in (
            (f"cpax-{version}/PKG-INFO", _metadata(sdist_version or version)),
            (f"cpax-{version}/src/CPax.egg-info/PKG-INFO", _metadata("0.0.0")),
        ):
            member = tarfile.TarInfo(path)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_validates_canonical_sdist_metadata_with_egg_info_copy(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    (tmp_path / ".gitignore").write_text("Not a distribution")
    wheel, sdist = validate_artifacts("2026.10.0", tmp_path)
    assert wheel.suffix == ".whl"
    assert sdist.name.endswith(".tar.gz")


def test_rejects_mismatched_top_level_sdist_metadata(tmp_path: Path) -> None:
    _build_artifacts(tmp_path, sdist_version="2026.10.1")
    with pytest.raises(ReleaseArtifactError, match="Sdist metadata mismatch"):
        validate_artifacts("2026.10.0", tmp_path)


def test_rejects_wrong_requested_version(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    with pytest.raises(ReleaseArtifactError, match="Wheel filename"):
        validate_artifacts("2026.10.1", tmp_path)


def test_rejects_duplicate_wheels(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    (tmp_path / "extra.whl").write_bytes(b"unexpected")
    with pytest.raises(ReleaseArtifactError, match="Expected one wheel and one sdist"):
        validate_artifacts("2026.10.0", tmp_path)


def test_rejects_missing_sdist_metadata(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    sdist = tmp_path / "cpax-2026.10.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = _metadata("2026.10.0")
        member = tarfile.TarInfo("cpax-2026.10.0/src/CPax.egg-info/PKG-INFO")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ReleaseArtifactError, match="Sdist is missing"):
        validate_artifacts("2026.10.0", tmp_path)
