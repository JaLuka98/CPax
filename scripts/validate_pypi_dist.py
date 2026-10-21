"""Check release distribution filenames and their canonical package metadata.

The wheel uses dist-info/METADATA. A setuptools sdist can contain two
PKG-INFO files: the top-level file and an egg-info copy. Only the
top-level file represents the sdist's published metadata.
"""

import argparse
import email
import tarfile
import zipfile
from pathlib import Path


class ReleaseArtifactError(ValueError):
    """A built release artifact does not match the requested version."""


def validate_artifacts(version: str, directory: Path) -> tuple[Path, Path]:
    """Check a CPax wheel and sdist, tolerating unrelated build-directory files."""
    artifacts = list(directory.iterdir())
    wheels = [path for path in artifacts if path.is_file() and path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.is_file() and path.name.endswith(".tar.gz")]

    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseArtifactError(
            f"Expected one wheel and one sdist in {directory}; found wheels={wheels}, sdists={sdists}."
        )

    wheel, sdist = wheels[0], sdists[0]
    expected_wheel = f"cpax-{version}-py3-none-any.whl"
    expected_sdist = f"cpax-{version}.tar.gz"
    if wheel.name != expected_wheel:
        raise ReleaseArtifactError(f"Wheel filename {wheel.name!r} does not match {expected_wheel!r}.")
    if sdist.name != expected_sdist:
        raise ReleaseArtifactError(f"Sdist filename {sdist.name!r} does not match {expected_sdist!r}.")

    with zipfile.ZipFile(wheel) as archive:
        wheel_metadata_path = f"cpax-{version}.dist-info/METADATA"
        try:
            wheel_metadata = email.message_from_bytes(archive.read(wheel_metadata_path))
        except KeyError as exc:
            raise ReleaseArtifactError(f"Wheel is missing {wheel_metadata_path}.") from exc

    with tarfile.open(sdist, "r:gz") as archive:
        # Do not count all /PKG-INFO files: setuptools also bundles
        # src/CPax.egg-info/PKG-INFO in source distributions.
        sdist_metadata_path = f"cpax-{version}/PKG-INFO"
        try:
            member = archive.getmember(sdist_metadata_path)
        except KeyError as exc:
            raise ReleaseArtifactError(f"Sdist is missing {sdist_metadata_path}.") from exc
        stream = archive.extractfile(member)
        if stream is None:
            raise ReleaseArtifactError(f"Sdist entry {sdist_metadata_path} is not a file.")
        sdist_metadata = email.message_from_bytes(stream.read())

    for label, metadata in (("Wheel", wheel_metadata), ("Sdist", sdist_metadata)):
        name = metadata.get("Name")
        actual_version = metadata.get("Version")
        if name is None or name.lower() != "cpax" or actual_version != version:
            raise ReleaseArtifactError(
                f"{label} metadata mismatch: expected CPax {version}, got {name!r} {actual_version!r}."
            )

    return wheel, sdist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Expected package version, without v prefix")
    parser.add_argument("dist_dir", type=Path, help="Directory containing built artifacts")
    args = parser.parse_args()
    try:
        wheel, sdist = validate_artifacts(args.version, args.dist_dir)
    except ReleaseArtifactError as exc:
        parser.exit(1, f"Release artifact validation failed: {exc}\n")
    print(f"Validated {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
