"""Validate a pre-existing annotated CPax release tag against main and metadata.

Run from a checkout with refs/remotes/origin/main and the requested tag
already fetched. Print the verified target commit SHA for checkout.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

TAG_RE = re.compile(r"v[1-9][0-9]{3}\.(?:0?[1-9]|1[0-2])\.(?:0|[1-9][0-9]*)\Z")


class ReleaseValidationError(Exception):
    """Requested Git tag cannot be used to publish a CPax release."""


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReleaseValidationError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def validate(tag: str, *, cwd: Path | None = None) -> str:
    """Return the release commit SHA after checking tag provenance and metadata."""
    if not TAG_RE.fullmatch(tag):
        raise ReleaseValidationError("Expected a CalVer tag such as v2026.10.0.")

    tag_ref = f"refs/tags/{tag}"
    if git("cat-file", "-t", tag_ref, cwd=cwd) != "tag":
        raise ReleaseValidationError(f"{tag} is not an annotated Git tag.")

    commit = git("rev-parse", "--verify", f"{tag_ref}^{{commit}}", cwd=cwd)
    git("merge-base", "--is-ancestor", commit, "refs/remotes/origin/main", cwd=cwd)

    try:
        pyproject = tomllib.loads(git("show", f"{commit}:pyproject.toml", cwd=cwd))
        lock = tomllib.loads(git("show", f"{commit}:uv.lock", cwd=cwd))
    except (ValueError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseValidationError("The tagged source has invalid release metadata.") from exc

    project = pyproject["project"]
    version = tag[1:]
    if project["name"].lower() != "cpax" or project["version"] != version:
        raise ReleaseValidationError("The package name or pyproject.toml version does not match the tag.")

    packages = [pkg for pkg in lock["package"] if pkg["name"].lower() == "cpax"]
    if len(packages) != 1 or packages[0]["version"] != version:
        raise ReleaseValidationError("uv.lock must contain exactly one CPax entry matching the release tag.")
    return commit


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/validate_pypi_tag.py vYYYY.M.N")
    try:
        print(validate(sys.argv[1]))
    except (ReleaseValidationError, KeyError, TypeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
